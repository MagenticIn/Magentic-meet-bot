"""
Magentic-meetbot  ·  routes/webhook.py
──────────────────────────────────────
Internal webhook endpoints consumed by the bot and pipeline services.

POST  /recording-complete   — bot signals that recording finished
POST  /pipeline-complete    — Celery worker signals pipeline results
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from celery import Celery
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import (
    MeetingRecord,
    MeetingStatus,
    PipelineCompletePayload,
    RecordingCompletePayload,
)
from integrations.pm_client import PMClient

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

# Celery app reference (broker only — we don't run workers here)
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("meetbot_pipeline", broker=REDIS_URL)


async def _get_record(meeting_id: str, db: AsyncSession) -> MeetingRecord:
    q = await db.execute(
        select(MeetingRecord).where(MeetingRecord.id == uuid.UUID(meeting_id))
    )
    record = q.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return record


# ── POST /recording-complete ─────────────────────────────────────────────
@router.post("/recording-complete", status_code=200)
async def recording_complete(
    body: RecordingCompletePayload,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Called by the bot service after it stops recording.
    Updates the DB record and enqueues the Celery pipeline task.
    """
    record = await _get_record(body.meeting_id, db)

    record.audio_path = body.audio_path
    record.end_reason = body.end_reason
    record.status = MeetingStatus.PROCESSING

    await db.commit()

    # Fire off the pipeline task
    celery_app.send_task(
        "pipeline.process_meeting",
        args=[body.meeting_id, body.audio_path],
    )

    log.info(
        "webhook.recording_complete",
        meeting_id=body.meeting_id,
        audio_path=body.audio_path,
    )
    return {"status": "processing", "meeting_id": body.meeting_id}


# ── POST /pipeline-complete ──────────────────────────────────────────────
@router.post("/pipeline-complete", status_code=200)
async def pipeline_complete(
    body: PipelineCompletePayload,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Called by the Celery worker when the pipeline finishes (success or fail).
    Persists results and triggers POST to the external PM API.
    """
    record = await _get_record(body.meeting_id, db)

    if body.status == "completed":
        record.status = MeetingStatus.COMPLETED
        record.transcript = body.transcript
        record.summary = body.summary

        # ── Sync to external PM tool ────────────────────────────────────
        try:
            pm = PMClient()
            await asyncio.to_thread(
                pm.post_meeting_notes,
                meeting_id=body.meeting_id,
                meeting_url=record.meeting_url,
                summary=body.summary or {},
                transcript=body.transcript or [],
            )
            record.pm_synced = "synced"
            log.info("webhook.pm_synced", meeting_id=body.meeting_id)
        except Exception:
            record.pm_synced = "failed"
            log.exception("webhook.pm_sync_failed", meeting_id=body.meeting_id)

    else:
        record.status = MeetingStatus.FAILED
        record.error_message = body.error

    await db.commit()

    log.info(
        "webhook.pipeline_complete",
        meeting_id=body.meeting_id,
        status=body.status,
    )
    return {"status": record.status.value, "meeting_id": body.meeting_id}
