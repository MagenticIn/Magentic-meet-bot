"""
Magentic-meetbot  ·  bot/worker.py
──────────────────────────────────
Bot worker that polls a Redis queue for meeting join requests.
When a request is received, it runs the Playwright bot (meet_bot.py)
and manages audio capture directly via the AudioCapture class.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import redis
import structlog

from bot.audio_capture import AudioCapture
from bot.meet_bot import run_bot

log = structlog.get_logger("bot_worker")

# ─── Config ──────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_QUEUE_KEY = "meetbot_join_queue"
POLL_TIMEOUT_SEC = 30


class BotWorker:
    """Listens for meeting join requests and executes the bot + audio capture."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)
        self._running = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        log.info("bot_worker.received_exit_signal", signum=signum)
        self._running = False

    async def _process_meeting(self, meeting_id: str, meeting_url: str) -> None:
        """Process a single meeting: setup audio, join Meet, record, cleanup."""
        output_path = f"/data/{meeting_id}/recording.wav"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "bot_worker.processing_request",
            meeting_id=meeting_id,
            url=meeting_url,
            output_path=output_path,
        )

        capture = AudioCapture(output_path)
        try:
            capture.setup_pulse_sink()
        except Exception as exc:
            log.error("bot_worker.audio_setup_failed", error=str(exc))
            capture = None

        if capture:
            try:
                capture.start_recording()
                log.info("bot_worker.recording_started", path=output_path)
            except Exception as exc:
                log.error("bot_worker.recording_start_failed", error=str(exc))
                capture = None

        try:
            end_reason = "unknown"
            exit_code = await run_bot(
                meeting_url=meeting_url,
                recording_output_path=output_path,
                meeting_id=meeting_id,
                notify_api=False,
            )
            try:
                state_path = Path("/tmp/meetbot_state.json")
                if state_path.exists():
                    state_data = json.loads(state_path.read_text(encoding="utf-8"))
                    end_reason = str(state_data.get("end_reason") or "unknown")
            except Exception as exc:
                log.warning("bot_worker.state_read_failed", error=str(exc))

            log.info(
                "bot_worker.meeting_finished",
                meeting_id=meeting_id,
                exit_code=exit_code.name,
                end_reason=end_reason,
            )
        finally:
            if capture:
                try:
                    capture.stop_recording()
                    log.info("bot_worker.recording_stopped", path=output_path)
                except Exception as exc:
                    log.error("bot_worker.recording_stop_failed", error=str(exc))
                capture.cleanup()

            # Notify API only after audio is fully finalized
            from bot.meet_bot import _notify_api_recording_complete
            try:
                await _notify_api_recording_complete(
                    meeting_id=meeting_id,
                    audio_path=output_path,
                    end_reason=end_reason,
                )
                log.info("bot_worker.api_notified", meeting_id=meeting_id)
            except Exception as exc:
                log.warning("bot_worker.api_notify_failed", meeting_id=meeting_id, error=str(exc))

    async def run(self):
        log.info("bot_worker.starting", queue=REDIS_QUEUE_KEY, url=self.redis_url)
        
        while self._running:
            try:
                # Use BLPOP for efficient blocking wait
                result = self.client.blpop(REDIS_QUEUE_KEY, timeout=POLL_TIMEOUT_SEC)
                
                if not result:
                    continue
                
                _, raw_data = result
                request = json.loads(raw_data)
                
                meeting_id = request.get("meeting_id")
                meeting_url = request.get("meeting_url")
                
                if not meeting_url:
                    log.error("bot_worker.invalid_request", data=request)
                    continue

                await self._process_meeting(meeting_id, meeting_url)
                
            except json.JSONDecodeError as exc:
                log.error("bot_worker.json_error", raw=raw_data, error=str(exc))
            except Exception:
                log.exception("bot_worker.loop_error")
                await asyncio.sleep(5)  # Avoid tight error loop

        log.info("bot_worker.stopped")


if __name__ == "__main__":
    import asyncio
    worker = BotWorker(REDIS_URL)
    asyncio.run(worker.run())
