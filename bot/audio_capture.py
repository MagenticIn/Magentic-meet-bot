"""
Magentic-meetbot  ·  audio_capture.py
─────────────────────────────────────
Captures Google Meet audio on a headless Linux server using PulseAudio
+ ffmpeg.  Designed to run as a companion process alongside meet_bot.py.

Architecture
────────────
                             ┌─────────────┐
    Chromium ──audio──▶      │ PulseAudio  │
    (PULSE_SINK=meetbot_sink)│ null-sink   │
                             └──────┬──────┘
                                    │ .monitor
                             ┌──────▼──────┐
                             │   ffmpeg    │
                             │ -f pulse    │
                             │ -i monitor  │
                             │ -ar 16000   │
                             │ -ac 1       │
                             └──────┬──────┘
                                    │
                               output.wav
                          (16 kHz mono PCM s16le)

IPC contract
────────────
Polls /tmp/meetbot_state.json written by meet_bot.py:
    status = "recording_started"  →  start ffmpeg
    status = "recording_stopped"  →  SIGTERM ffmpeg, verify output

Usage
─────
    # Standalone (watches state file):
    python -m bot.audio_capture /data/recording.wav

    # Programmatic:
    from bot.audio_capture import AudioCapture
    cap = AudioCapture("/data/recording.wav")
    cap.setup_pulse_sink()
    cap.start_recording()
    ...
    path = cap.stop_recording()
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger("audio_capture")

# ─── Constants ───────────────────────────────────────────────────────────
SINK_NAME = "meetbot_sink"
SINK_MONITOR = f"{SINK_NAME}.monitor"
SAMPLE_RATE = 16_000          # exactly what Whisper expects
CHANNELS = 1                  # mono
CODEC = "pcm_s16le"           # 16-bit PCM
STATE_FILE = Path("/tmp/meetbot_state.json")
POLL_INTERVAL_SEC = 1.0
MIN_FILE_SIZE_BYTES = 10_240  # 10 KB — anything smaller is silence / corrupt
FFMPEG_GRACE_SEC = 10         # seconds to wait for ffmpeg after SIGTERM
FFMPEG_KILL_SEC = 5           # seconds to wait after SIGKILL


# ─── Exceptions ──────────────────────────────────────────────────────────
class PulseAudioError(Exception):
    """Raised when PulseAudio sink creation or configuration fails."""


class FFmpegError(Exception):
    """Raised when ffmpeg exits with an unexpected error code."""


class RecordingTooSmallError(Exception):
    """Raised when the output file is suspiciously small (< 10 KB)."""


# ─── PulseAudio sink management ──────────────────────────────────────────
def create_pulse_sink() -> int:
    """
    Create the PulseAudio null-sink 'meetbot_sink' and set it as default.

    Returns the module index for potential unloading later.
    Raises PulseAudioError on failure.
    """
    # ── Check if the sink already exists ─────────────────────────────
    try:
        probe = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True, text=True, timeout=10,
        )
        if SINK_NAME in probe.stdout:
            log.info("pulse.sink_already_exists", sink=SINK_NAME)
            # Extract the module index from existing modules
            mods = subprocess.run(
                ["pactl", "list", "short", "modules"],
                capture_output=True, text=True, timeout=10,
            )
            for line in mods.stdout.splitlines():
                if SINK_NAME in line:
                    parts = line.split()
                    if parts:
                        return int(parts[0])
            return 0
    except Exception:
        pass

    # ── Create the null-sink ─────────────────────────────────────────
    log.info("pulse.creating_sink", sink=SINK_NAME)
    result = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={SINK_NAME}",
            f"sink_properties=device.description=MagenticMeetbotCapture",
        ],
        capture_output=True, text=True, timeout=10,
    )

    if result.returncode != 0:
        raise PulseAudioError(
            f"Failed to create PulseAudio sink '{SINK_NAME}': "
            f"rc={result.returncode}, stderr={result.stderr.strip()}"
        )

    module_index = int(result.stdout.strip())
    log.info("pulse.sink_created", sink=SINK_NAME, module_index=module_index)

    # ── Set as default sink so Chromium routes audio here ────────────
    set_result = subprocess.run(
        ["pactl", "set-default-sink", SINK_NAME],
        capture_output=True, text=True, timeout=10,
    )
    if set_result.returncode != 0:
        log.warning(
            "pulse.set_default_failed",
            stderr=set_result.stderr.strip(),
        )
    else:
        log.info("pulse.default_sink_set", sink=SINK_NAME)

    return module_index


def verify_pulse_sink() -> bool:
    """Return True if the meetbot_sink exists and its monitor is available."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=10,
        )
        if SINK_MONITOR in result.stdout:
            log.info("pulse.monitor_verified", source=SINK_MONITOR)
            return True
        log.warning("pulse.monitor_not_found", source=SINK_MONITOR, sources=result.stdout.strip())
        return False
    except Exception as exc:
        log.error("pulse.verify_failed", error=str(exc))
        return False


def remove_pulse_sink(module_index: int) -> None:
    """Unload the PulseAudio null-sink module (cleanup)."""
    if module_index <= 0:
        return
    try:
        subprocess.run(
            ["pactl", "unload-module", str(module_index)],
            capture_output=True, text=True, timeout=10,
        )
        log.info("pulse.sink_removed", module_index=module_index)
    except Exception as exc:
        log.warning("pulse.sink_remove_failed", error=str(exc))


# ─── State file reader ──────────────────────────────────────────────────
def read_state() -> dict:
    """
    Read and parse /tmp/meetbot_state.json.
    Returns an empty dict if the file doesn't exist or is malformed.
    """
    try:
        if not STATE_FILE.exists():
            return {}
        raw = STATE_FILE.read_text(encoding="utf-8")
        return json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        log.debug("state.read_error", error=str(exc))
        return {}


# ─── AudioCapture class ─────────────────────────────────────────────────
@dataclass
class AudioCapture:
    """
    Manages the full lifecycle of audio capture:
        PulseAudio sink creation  →  ffmpeg recording  →  file verification.
    """

    output_path: str
    _ffmpeg_proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _module_index: int = field(default=0, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    # ── PulseAudio ───────────────────────────────────────────────────
    def setup_pulse_sink(self) -> None:
        """Create the virtual sink and verify the monitor source."""
        self._module_index = create_pulse_sink()
        if not verify_pulse_sink():
            raise PulseAudioError(
                f"Monitor source '{SINK_MONITOR}' not available after sink creation"
            )

    # ── Recording ────────────────────────────────────────────────────
    def start_recording(self) -> Path:
        """
        Launch ffmpeg to record from the PulseAudio monitor source.
        Returns the output path.

        Raises RuntimeError if already recording.
        Raises FFmpegError if ffmpeg fails to start.
        """
        if self._recording:
            raise RuntimeError("Recording already in progress")

        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",                       # overwrite output
            "-f", "pulse",              # PulseAudio input format
            "-i", SINK_MONITOR,         # monitor of our virtual sink
            "-ar", str(SAMPLE_RATE),    # 16 kHz
            "-ac", str(CHANNELS),       # mono
            "-acodec", CODEC,           # PCM s16le
            str(out),
        ]

        log.info("ffmpeg.starting", cmd=" ".join(cmd))

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise FFmpegError("ffmpeg binary not found — is it installed?")
        except OSError as exc:
            raise FFmpegError(f"Failed to start ffmpeg: {exc}")

        # Give ffmpeg a moment to start — check it hasn't crashed immediately
        time.sleep(0.5)
        if self._ffmpeg_proc.poll() is not None:
            stderr = self._drain_stderr()
            raise FFmpegError(
                f"ffmpeg exited immediately with rc={self._ffmpeg_proc.returncode}: {stderr}"
            )

        self._recording = True
        self._start_time = time.monotonic()
        log.info(
            "ffmpeg.started",
            pid=self._ffmpeg_proc.pid,
            output=str(out),
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )
        return out

    def stop_recording(self) -> Path:
        """
        Gracefully stop ffmpeg (SIGTERM → wait → SIGKILL if needed).
        Verifies the output file exists and is > 10 KB.

        Returns the output file path.

        Raises FFmpegError on unexpected exit.
        Raises RecordingTooSmallError if output < 10 KB.
        """
        if not self._recording or self._ffmpeg_proc is None:
            log.warning("ffmpeg.stop_called_but_not_recording")
            return Path(self.output_path)

        pid = self._ffmpeg_proc.pid
        duration = time.monotonic() - (self._start_time or time.monotonic())
        log.info("ffmpeg.stopping", pid=pid, recorded_seconds=f"{duration:.1f}")

        # ── Step 1: SIGTERM (graceful shutdown — flushes WAV header) ──
        try:
            self._ffmpeg_proc.send_signal(signal.SIGTERM)
            log.debug("ffmpeg.sigterm_sent", pid=pid)
        except OSError as exc:
            log.warning("ffmpeg.sigterm_failed", error=str(exc))

        # ── Step 2: Wait for graceful exit ───────────────────────────
        try:
            self._ffmpeg_proc.wait(timeout=FFMPEG_GRACE_SEC)
            log.debug("ffmpeg.exited_gracefully", pid=pid, rc=self._ffmpeg_proc.returncode)
        except subprocess.TimeoutExpired:
            # ── Step 3: SIGKILL as last resort ───────────────────────
            log.warning("ffmpeg.grace_period_expired_sending_sigkill", pid=pid)
            try:
                self._ffmpeg_proc.kill()
                self._ffmpeg_proc.wait(timeout=FFMPEG_KILL_SEC)
            except (OSError, subprocess.TimeoutExpired):
                log.error("ffmpeg.kill_failed", pid=pid)

        # ── Collect stderr for diagnostics ───────────────────────────
        stderr = self._drain_stderr()
        rc = self._ffmpeg_proc.returncode
        self._ffmpeg_proc = None
        self._recording = False

        # ffmpeg returns 255 on SIGTERM/SIGINT — that's expected
        if rc is not None and rc not in (0, -signal.SIGTERM, 255):
            log.error("ffmpeg.unexpected_exit", rc=rc, stderr=stderr[-500:])
            raise FFmpegError(f"ffmpeg exited with rc={rc}: {stderr[-300:]}")

        log.info("ffmpeg.stopped", rc=rc, recorded_seconds=f"{duration:.1f}")

        # ── Verify output ────────────────────────────────────────────
        return self._verify_output()

    def _verify_output(self) -> Path:
        """Check the output file exists and is large enough."""
        out = Path(self.output_path)

        if not out.exists():
            raise FileNotFoundError(f"Recording output file not found: {out}")

        size = out.stat().st_size
        size_kb = size / 1024
        log.info("ffmpeg.output_verified", path=str(out), size_kb=f"{size_kb:.1f}")

        if size < MIN_FILE_SIZE_BYTES:
            raise RecordingTooSmallError(
                f"Recording file too small ({size_kb:.1f} KB < {MIN_FILE_SIZE_BYTES // 1024} KB). "
                f"Likely silence or capture failure. Path: {out}"
            )

        return out

    def _drain_stderr(self) -> str:
        """Read any remaining stderr from ffmpeg."""
        if self._ffmpeg_proc and self._ffmpeg_proc.stderr:
            try:
                return self._ffmpeg_proc.stderr.read().decode(errors="replace")
            except Exception:
                pass
        return ""

    @property
    def is_recording(self) -> bool:
        """True if ffmpeg is currently running."""
        return self._recording and self._ffmpeg_proc is not None and self._ffmpeg_proc.poll() is None

    # ── Cleanup ──────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """Stop recording if active and remove the PulseAudio sink."""
        if self.is_recording:
            try:
                self.stop_recording()
            except Exception as exc:
                log.warning("cleanup.stop_failed", error=str(exc))
        remove_pulse_sink(self._module_index)


# ─── Standalone mode: poll state file ────────────────────────────────────
def run_state_watcher(output_path: str) -> int:
    """
    Run as a companion process to meet_bot.py.

    Polling loop:
        1. Wait for state = "recording_started"
        2. Start ffmpeg
        3. Wait for state = "recording_stopped" (or error / timeout)
        4. Stop ffmpeg, verify output
        5. Return 0 on success, 1 on failure
    """
    log.info("watcher.starting", output=output_path, state_file=str(STATE_FILE))

    capture = AudioCapture(output_path=output_path)

    # ── Setup PulseAudio sink ────────────────────────────────────────
    try:
        capture.setup_pulse_sink()
    except PulseAudioError as exc:
        log.error("watcher.pulse_setup_failed", error=str(exc))
        return 1

    # Export env so any Chromium instance launched in this shell uses our sink
    os.environ["PULSE_SINK"] = SINK_NAME
    log.info("watcher.env_set", PULSE_SINK=SINK_NAME)

    # ── Phase 1: Wait for "recording_started" ────────────────────────
    log.info("watcher.waiting_for_recording_start")
    while True:
        state = read_state()
        status = state.get("status", "")

        if status == "recording_started":
            log.info("watcher.detected_recording_start")
            break

        if status == "error":
            log.error("watcher.bot_errored_before_recording", state=state)
            capture.cleanup()
            return 1

        if status == "recording_stopped":
            # Bot never actually recorded — nothing to do
            log.warning("watcher.recording_stopped_before_start")
            capture.cleanup()
            return 0

        time.sleep(POLL_INTERVAL_SEC)

    # ── Phase 2: Start recording ─────────────────────────────────────
    try:
        capture.start_recording()
    except (FFmpegError, RuntimeError) as exc:
        log.error("watcher.start_recording_failed", error=str(exc))
        capture.cleanup()
        return 1

    # ── Phase 3: Wait for "recording_stopped" ────────────────────────
    log.info("watcher.recording_in_progress")
    while True:
        # Check ffmpeg is still healthy
        if not capture.is_recording:
            log.error("watcher.ffmpeg_died_unexpectedly")
            capture.cleanup()
            return 1

        state = read_state()
        status = state.get("status", "")

        if status in ("recording_stopped", "error"):
            log.info("watcher.detected_recording_stop", status=status)
            break

        time.sleep(POLL_INTERVAL_SEC)

    # ── Phase 4: Stop recording + verify ─────────────────────────────
    exit_code = 0
    try:
        final_path = capture.stop_recording()
        log.info("watcher.recording_complete", path=str(final_path))
    except RecordingTooSmallError as exc:
        log.error("watcher.recording_too_small", error=str(exc))
        exit_code = 1
    except (FFmpegError, FileNotFoundError) as exc:
        log.error("watcher.stop_failed", error=str(exc))
        exit_code = 1
    finally:
        capture.cleanup()

    return exit_code


# ─── CLI entrypoint ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audio_capture",
        description="Magentic-meetbot — PulseAudio + ffmpeg audio capture",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=os.getenv("RECORDING_OUTPUT_PATH", "/data/recording.wav"),
        help="Path for the output WAV file (or set RECORDING_OUTPUT_PATH env var)",
    )
    args = parser.parse_args()

    exit_code = run_state_watcher(args.output_path)
    log.info("audio_capture.exiting", code=exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
