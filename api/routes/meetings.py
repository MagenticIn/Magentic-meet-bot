from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import (
    ActionItem,
    MeetingListItem,
    MeetingOut,
    MeetingRecord,
    MeetingStatus,
    PushToPMResponse,
    TranscriptOut,
    TriggerMeetingRequest,
    TriggerMeetingResponse,
)
from integrations.pm_client import PMClient

router = APIRouter(prefix="/meetings", tags=["meetings"])

AUDIO_BASE_PATH = Path("/data")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_QUEUE_KEY = "meetbot_join_queue"


def _extract_meeting_code(url: str) -> str | None:
    parts = url.rstrip("/").split("/")
    if not parts:
        return None
    candidate = parts[-1]
    return candidate if len(candidate) >= 10 else None


def _enqueue_bot_job(meeting_id: uuid.UUID, meeting_url: str) -> None:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.rpush(
        REDIS_QUEUE_KEY,
        f'{{"meeting_id":"{meeting_id}","meeting_url":"{meeting_url}"}}',
    )


async def _get_meeting_or_404(db: AsyncSession, meeting_id: uuid.UUID) -> MeetingRecord:
    query = await db.execute(select(MeetingRecord).where(MeetingRecord.id == meeting_id))
    meeting = query.scalar_one_or_none()
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.post("/trigger", response_model=TriggerMeetingResponse)
async def trigger_meeting(
    body: TriggerMeetingRequest,
    db: AsyncSession = Depends(get_db),
) -> TriggerMeetingResponse:
    meeting_id = uuid.uuid4()
    audio_path = AUDIO_BASE_PATH / str(meeting_id) / "recording.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    record = MeetingRecord(
        id=meeting_id,
        meeting_url=body.meeting_url,
        meeting_code=_extract_meeting_code(body.meeting_url),
        title=body.title,
        date=datetime.now(timezone.utc),
        status=MeetingStatus.JOINING,
        audio_path=str(audio_path),
    )
    db.add(record)
    await db.commit()

    # Direct subprocess trigger removed in favor of queue-based worker path.
    _enqueue_bot_job(meeting_id, body.meeting_url)

    return TriggerMeetingResponse(meeting_id=meeting_id, status="started")


@router.get("", response_model=list[MeetingListItem])
async def list_meetings(db: AsyncSession = Depends(get_db)) -> list[MeetingListItem]:
    rows = await db.execute(select(MeetingRecord).order_by(MeetingRecord.created_at.desc()))
    records = rows.scalars().all()
    return [
        MeetingListItem(
            id=record.id,
            title=record.title,
            date=record.date or record.created_at,
            status=record.status,
            duration=record.duration_minutes,
        )
        for record in records
    ]


@router.get("/{meeting_id}", response_model=MeetingOut)
async def get_meeting(meeting_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> MeetingOut:
    record = await _get_meeting_or_404(db, meeting_id)
    summary = record.summary or {}
    transcript = record.transcript or []

    action_items = summary.get("action_items", [])
    if not action_items:
        action_rows = await db.execute(select(ActionItem).where(ActionItem.meeting_id == meeting_id))
        action_items = [
            {"task": item.task, "owner": item.owner or "", "deadline": item.deadline}
            for item in action_rows.scalars().all()
        ]

    return MeetingOut(
        id=record.id,
        meeting_url=record.meeting_url,
        title=record.title,
        date=record.date or record.created_at,
        status=record.status,
        duration=record.duration_minutes,
        summary=summary.get("summary"),
        key_points=summary.get("key_points", []),
        action_items=action_items,
        decisions=summary.get("decisions", []),
        next_meeting=summary.get("next_meeting"),
        sentiment=summary.get("sentiment"),
        topics_discussed=summary.get("topics_discussed", []),
        transcript=transcript,
        raw_transcript=summary.get("raw_transcript"),
        translated_transcript=summary.get("translated_transcript"),
    )


@router.get("/{meeting_id}/transcript", response_model=TranscriptOut)
async def get_transcript(meeting_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> TranscriptOut:
    record = await _get_meeting_or_404(db, meeting_id)
    summary = record.summary or {}
    return TranscriptOut(
        raw_transcript=summary.get("raw_transcript", ""),
        translated_transcript=summary.get("translated_transcript", ""),
    )


@router.post("/{meeting_id}/push-to-pm", response_model=PushToPMResponse)
async def push_to_pm(meeting_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> PushToPMResponse:
    record = await _get_meeting_or_404(db, meeting_id)
    pm = PMClient()
    try:
        pm_response = await asyncio.to_thread(
            pm.post_meeting_notes,
            meeting_id=str(record.id),
            meeting_url=record.meeting_url,
            summary=record.summary or {},
            transcript=record.transcript or [],
        )
        record.pm_synced = "synced"
        await db.commit()
        return PushToPMResponse(success=True, pm_response=pm_response)
    except Exception as exc:
        record.pm_synced = "failed"
        await db.commit()
        return PushToPMResponse(success=False, pm_response={"error": str(exc)})
