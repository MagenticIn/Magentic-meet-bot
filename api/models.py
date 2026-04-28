from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MeetingStatus(str, enum.Enum):
    PENDING = "pending"
    JOINING = "joining"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MeetingRecord(Base):
    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_url: Mapped[str] = mapped_column(String(512), nullable=False)
    meeting_code: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[MeetingStatus] = mapped_column(SAEnum(MeetingStatus), default=MeetingStatus.PENDING, nullable=False)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    transcript: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    end_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pm_synced: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    utterances: Mapped[list["Utterance"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    action_items: Mapped[list["ActionItem"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")


class Utterance(Base):
    __tablename__ = "utterances"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    speaker: Mapped[str] = mapped_column(String(128), nullable=False)
    start: Mapped[float] = mapped_column(nullable=False)
    end: Mapped[float] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_en: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    meeting: Mapped[MeetingRecord] = relationship(back_populates="utterances")


class ActionItem(Base):
    __tablename__ = "action_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deadline: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    meeting: Mapped[MeetingRecord] = relationship(back_populates="action_items")


class TriggerMeetingRequest(BaseModel):
    meeting_url: str
    title: str | None = None


class TriggerMeetingResponse(BaseModel):
    meeting_id: uuid.UUID
    status: str


class MeetingListItem(BaseModel):
    id: uuid.UUID
    title: str | None = None
    date: datetime | None = None
    status: MeetingStatus
    duration: int | None = None


class MeetingActionItemOut(BaseModel):
    task: str
    owner: str
    deadline: str | None = None


class MeetingOut(BaseModel):
    id: uuid.UUID
    meeting_url: str
    title: str | None = None
    date: datetime | None = None
    status: MeetingStatus
    duration: int | None = None
    summary: str | None = None
    key_points: list[str] = []
    action_items: list[MeetingActionItemOut] = []
    decisions: list[str] = []
    next_meeting: str | None = None
    sentiment: str | None = None
    topics_discussed: list[str] = []
    transcript: list[dict[str, Any]] = []
    raw_transcript: str | None = None
    translated_transcript: str | None = None


class TranscriptOut(BaseModel):
    raw_transcript: str
    translated_transcript: str


class PushToPMResponse(BaseModel):
    success: bool
    pm_response: dict[str, Any]


class RecordingCompletePayload(BaseModel):
    meeting_id: str
    audio_path: str
    end_reason: str = "unknown"


class PipelineCompletePayload(BaseModel):
    meeting_id: str
    status: str
    transcript: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
