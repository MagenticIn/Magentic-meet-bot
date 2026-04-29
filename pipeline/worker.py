"""
Magentic-meetbot  ·  worker.py
──────────────────────────────
Celery application + task for the audio-processing pipeline.

Flow
────
    audio  →  transcribe (+ diarize)  →  summarize  →  POST to API

    Default: OpenAI ``gpt-4o-transcribe-diarize``. Alternative: whisper → diarize.

The ``process_meeting`` task is enqueued by the API service when the bot
signals that recording is complete (via the ``/webhook/recording-complete``
endpoint).

CLI
───
    # Start the Celery worker:
    celery -A pipeline.worker worker --loglevel=info --concurrency=1

    # Or run a one-off pipeline directly:
    python -m pipeline.worker <meeting_id> <audio_path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import structlog
from celery import Celery

from pipeline.transcribe import transcribe
from pipeline.diarize import diarize, format_transcript
from pipeline.openai_transcribe_diarize import transcribe_diarize_openai
from pipeline.summarize import summarize

log = structlog.get_logger("worker")

# ─── Config ──────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
API_BASE: str = os.getenv("API_BASE_URL", "http://api:8000")
SHARED_DATA_DIR: str = os.getenv("SHARED_DATA_DIR", "/data")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
TRANSCRIPTION_BACKEND: str = os.getenv("TRANSCRIPTION_BACKEND", "openai").strip().lower()
POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://meetbot:meetbot@postgres:5432/meetbot",
)

# ─── Celery app ──────────────────────────────────────────────────────────
app = Celery("meetbot_pipeline", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Longer visibility timeout for large audio files
    broker_transport_options={"visibility_timeout": 7200},
)


# ═════════════════════════════════════════════════════════════════════════
#  Artifact persistence
# ═════════════════════════════════════════════════════════════════════════

def _save_artifact(meeting_id: str, name: str, data: Any) -> Path:
    """Persist a JSON artifact to the shared data directory."""
    out_dir = Path(SHARED_DATA_DIR) / meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_kb = path.stat().st_size / 1024
    log.info("artifact.saved", path=str(path), size_kb=f"{size_kb:.1f}")
    return path


def _save_text_artifact(meeting_id: str, name: str, text: str) -> Path:
    """Persist a plain-text artifact."""
    out_dir = Path(SHARED_DATA_DIR) / meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    log.info("artifact.saved", path=str(path))
    return path


# ═════════════════════════════════════════════════════════════════════════
#  API notification
# ═════════════════════════════════════════════════════════════════════════

def _notify_api(payload: dict[str, Any]) -> None:
    """POST pipeline results back to the API service."""
    url = f"{API_BASE}/api/v1/webhook/pipeline-complete"
    log.info("api.notify", url=url, meeting_id=payload.get("meeting_id"))

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
        log.info("api.notify.success", status=resp.status_code)
    except Exception as exc:
        log.exception("api.notify.failed", error=str(exc))


def _notify_api_failure(meeting_id: str, error: str) -> None:
    """Notify the API that the pipeline failed."""
    _notify_api({
        "meeting_id": meeting_id,
        "status": "failed",
        "error": error,
    })


def _asyncpg_dsn(url: str) -> str:
    """Convert SQLAlchemy asyncpg DSN to asyncpg-compatible DSN."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _fetch_meeting_meta_async(meeting_id: str) -> dict[str, Any]:
    """
    Load meeting metadata from Postgres meetings record.
    """
    import asyncpg

    conn = await asyncpg.connect(_asyncpg_dsn(POSTGRES_URL))
    try:
        row = await conn.fetchrow(
            "SELECT title, created_at, duration_minutes FROM meetings WHERE id = $1",
            meeting_id,
        )
    finally:
        await conn.close()

    if row is None:
        return {
            "title": "Untitled Meeting",
            "date": "Unknown date",
            "duration_minutes": 0,
            "attendees": [],
        }

    return {
        "title": row["title"] or "Untitled Meeting",
        "date": str(row["created_at"]) if row["created_at"] else "Unknown date",
        "duration_minutes": row["duration_minutes"] or 0,
        "attendees": [],
    }


def _fetch_meeting_meta(meeting_id: str) -> dict[str, Any]:
    try:
        return asyncio.run(_fetch_meeting_meta_async(meeting_id))
    except Exception as exc:
        log.warning("pipeline.meeting_meta_fetch_failed", meeting_id=meeting_id, error=str(exc))
        return {
            "title": "Untitled Meeting",
            "date": "Unknown date",
            "duration_minutes": 0,
            "attendees": [],
        }


# ═════════════════════════════════════════════════════════════════════════
#  Pipeline task
# ═════════════════════════════════════════════════════════════════════════

@app.task(name="pipeline.process_meeting", bind=True, max_retries=2)
def process_meeting(self, meeting_id: str, audio_path: str) -> dict[str, Any]:
    """
    End-to-end meeting processing pipeline.

    Steps:
        1. Transcribe (+ diarize): OpenAI ``gpt-4o-transcribe-diarize`` when
           ``TRANSCRIPTION_BACKEND=openai`` (default), else faster-whisper then
           whisperx/pyannote.
        2. (Whisper path only) Diarize with whisperx.
        3. Summarize with OpenAI (structured JSON)
        4. Save artifacts to shared volume
        5. POST results to API

    Parameters
    ----------
    meeting_id : str
        UUID of the meeting record.
    audio_path : str
        Path to the recorded WAV file.

    Returns
    -------
    dict
        Pipeline results including transcript and summary.
    """
    t0 = time.monotonic()

    try:
        log.info(
            "pipeline.start",
            meeting_id=meeting_id,
            audio=audio_path,
            transcription_backend=TRANSCRIPTION_BACKEND,
            hf_token_set=bool(HF_TOKEN),
        )

        # ── Validate audio file ──────────────────────────────────────
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size_mb = audio.stat().st_size / (1024 * 1024)
        log.info("pipeline.audio_info", size_mb=f"{size_mb:.1f}")

        # ── Step 1–2: Transcribe (+ diarize) ─────────────────────────
        if TRANSCRIPTION_BACKEND == "openai":
            log.info("pipeline.step", step="transcribe_diarize_openai", n=1)
            step_t0 = time.monotonic()

            utterances, raw_td = transcribe_diarize_openai(audio_path)
            _save_artifact(meeting_id, "01_transcript_raw", raw_td)
            _save_artifact(meeting_id, "02_transcript_diarized", utterances)
            formatted = format_transcript(utterances)
            _save_text_artifact(meeting_id, "02_transcript_readable", formatted)

            log.info(
                "pipeline.step_done",
                step="transcribe_diarize_openai",
                utterances=len(utterances),
                speakers=len({u.get("speaker") for u in utterances}),
                elapsed=f"{time.monotonic() - step_t0:.1f}s",
            )
        else:
            log.info("pipeline.step", step="transcribe", n=1)
            step_t0 = time.monotonic()

            segments = transcribe(audio_path)
            _save_artifact(meeting_id, "01_transcript_raw", segments)

            log.info(
                "pipeline.step_done",
                step="transcribe",
                segments=len(segments),
                elapsed=f"{time.monotonic() - step_t0:.1f}s",
            )

            if not segments:
                log.warning("pipeline.empty_transcript", meeting_id=meeting_id)
                result = {
                    "meeting_id": meeting_id,
                    "status": "completed",
                    "transcript": [],
                    "summary": {"executive_summary": "No speech detected in the recording."},
                }
                _notify_api(result)
                return result

            log.info("pipeline.step", step="diarize", n=2)
            step_t0 = time.monotonic()

            utterances = diarize(
                audio_path=audio_path,
                whisper_segments=segments,
                hf_token=HF_TOKEN,
            )
            _save_artifact(meeting_id, "02_transcript_diarized", utterances)

            formatted = format_transcript(utterances)
            _save_text_artifact(meeting_id, "02_transcript_readable", formatted)

            log.info(
                "pipeline.step_done",
                step="diarize",
                utterances=len(utterances),
                speakers=len({u.get("speaker") for u in utterances}),
                elapsed=f"{time.monotonic() - step_t0:.1f}s",
            )

        if not utterances:
            log.warning("pipeline.empty_transcript", meeting_id=meeting_id)
            result = {
                "meeting_id": meeting_id,
                "status": "completed",
                "transcript": [],
                "summary": {"executive_summary": "No speech detected in the recording."},
            }
            _notify_api(result)
            return result

        # ── Step 3: Summarize ────────────────────────────────────────
        log.info("pipeline.step", step="summarize", n=3)
        step_t0 = time.monotonic()

        meeting_meta = _fetch_meeting_meta(meeting_id)
        summary = summarize(utterances, meeting_meta)
        _save_artifact(meeting_id, "03_summary", summary)

        log.info(
            "pipeline.step_done",
            step="summarize",
            action_items=len(summary.get("action_items", [])),
            elapsed=f"{time.monotonic() - step_t0:.1f}s",
        )

        # ── Step 4: Notify API ───────────────────────────────────────
        total_elapsed = time.monotonic() - t0
        result = {
            "meeting_id": meeting_id,
            "status": "completed",
            "transcript": utterances,
            "summary": summary,
        }
        _notify_api(result)

        log.info(
            "pipeline.done",
            meeting_id=meeting_id,
            total_elapsed=f"{total_elapsed:.1f}s",
        )
        return result

    except Exception as exc:
        total_elapsed = time.monotonic() - t0
        log.exception(
            "pipeline.error",
            meeting_id=meeting_id,
            elapsed=f"{total_elapsed:.1f}s",
        )

        # Notify API of failure
        _notify_api_failure(meeting_id, str(exc))

        # Retry with backoff
        raise self.retry(exc=exc, countdown=60)


# ═════════════════════════════════════════════════════════════════════════
#  CLI — run pipeline directly (useful for testing / debugging)
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline_worker",
        description="Magentic-meetbot — Run the full pipeline directly (no Celery)",
    )
    parser.add_argument("meeting_id", help="Meeting UUID")
    parser.add_argument("audio_path", help="Path to the recorded WAV file")
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip API notification (useful for local testing)",
    )
    args = parser.parse_args()

    log.info("cli.starting", meeting_id=args.meeting_id, audio=args.audio_path)

    # Run the pipeline directly (not via Celery)
    t0 = time.monotonic()

    try:
        if TRANSCRIPTION_BACKEND == "openai":
            utterances, raw_td = transcribe_diarize_openai(args.audio_path)
            _save_artifact(args.meeting_id, "01_transcript_raw", raw_td)
            _save_artifact(args.meeting_id, "02_transcript_diarized", utterances)
            formatted = format_transcript(utterances)
            _save_text_artifact(args.meeting_id, "02_transcript_readable", formatted)
            speakers = {u.get("speaker") for u in utterances}
            print(
                f"✓ OpenAI transcribe+diarize: {len(utterances)} utterances, {len(speakers)} speakers",
                file=sys.stderr,
            )
        else:
            segments = transcribe(args.audio_path)
            _save_artifact(args.meeting_id, "01_transcript_raw", segments)
            print(f"✓ Transcribed: {len(segments)} segments", file=sys.stderr)

            if not segments:
                print("⚠ No segments found — empty audio?", file=sys.stderr)
                sys.exit(0)

            utterances = diarize(
                audio_path=args.audio_path,
                whisper_segments=segments,
                hf_token=HF_TOKEN,
            )
            _save_artifact(args.meeting_id, "02_transcript_diarized", utterances)
            formatted = format_transcript(utterances)
            _save_text_artifact(args.meeting_id, "02_transcript_readable", formatted)
            speakers = {u.get("speaker") for u in utterances}
            print(f"✓ Diarized: {len(utterances)} utterances, {len(speakers)} speakers", file=sys.stderr)

        if not utterances:
            print("⚠ No utterances — empty audio?", file=sys.stderr)
            sys.exit(0)

        # Summarize
        meeting_meta = _fetch_meeting_meta(args.meeting_id)
        summary = summarize(utterances, meeting_meta)
        _save_artifact(args.meeting_id, "03_summary", summary)
        print(f"✓ Summarized: {len(summary.get('action_items', []))} action items", file=sys.stderr)

        # Notify API (unless --no-notify)
        if not args.no_notify:
            _notify_api({
                "meeting_id": args.meeting_id,
                "status": "completed",
                "transcript": utterances,
                "summary": summary,
            })
            print("✓ API notified", file=sys.stderr)

        elapsed = time.monotonic() - t0
        print(f"\n✓ Pipeline complete in {elapsed:.1f}s", file=sys.stderr)

        # Print summary to stdout
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    except Exception as exc:
        print(f"✗ Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
