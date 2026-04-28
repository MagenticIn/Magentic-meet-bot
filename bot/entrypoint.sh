#!/usr/bin/env bash
###############################################################################
# Magentic-meetbot  ·  Bot container entrypoint
#
# Starts all required services in the correct order:
#   1. D-Bus          (PulseAudio needs it)
#   2. Xvfb           (virtual framebuffer on :99 — Chromium needs a DISPLAY)
#   3. PulseAudio     (daemon mode — creates the meetbot_sink automatically)
#   4. audio_capture   (polls state file, records from the sink)
#   5. bot script      (either bot.worker or bot.meet_bot)
#
# The bot script is the "main" process — when it exits, the container exits.
###############################################################################
set -euo pipefail

echo "[entrypoint] Starting services..."

# ── 1. D-Bus daemon ─────────────────────────────────────────────────────
if [ ! -d /run/dbus ]; then
    mkdir -p /run/dbus
fi
dbus-daemon --system --fork 2>/dev/null || true
echo "[entrypoint] D-Bus started"

# ── 2. Xvfb (virtual framebuffer) ───────────────────────────────────────
export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 -ac -nolisten tcp &
XVFB_PID=$!
sleep 1

# Verify Xvfb is running
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "[entrypoint] ERROR: Xvfb failed to start" >&2
    exit 1
fi
echo "[entrypoint] Xvfb started on $DISPLAY (PID=$XVFB_PID)"

# ── 3. PulseAudio daemon ────────────────────────────────────────────────
# Start PulseAudio in the background with our config (which auto-loads meetbot_sink)
pulseaudio --daemonize --no-cpu-limit --disallow-exit \
    --log-target=stderr --log-level=notice 2>&1 &
sleep 2

# Verify PulseAudio is running and the sink exists
if ! pactl info >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: PulseAudio failed to start" >&2
    exit 1
fi
echo "[entrypoint] PulseAudio started"

# Verify meetbot_sink is available
if pactl list short sinks | grep -q meetbot_sink; then
    echo "[entrypoint] meetbot_sink is ready"
else
    echo "[entrypoint] WARNING: meetbot_sink not found in sinks, creating manually..."
    pactl load-module module-null-sink \
        sink_name=meetbot_sink \
        sink_properties=device.description=MagenticMeetbotCapture || true
    pactl set-default-sink meetbot_sink || true
fi

# Confirm the monitor source exists
if pactl list short sources | grep -q meetbot_sink.monitor; then
    echo "[entrypoint] meetbot_sink.monitor source confirmed"
else
    echo "[entrypoint] ERROR: meetbot_sink.monitor source not found" >&2
    pactl list short sources >&2
    exit 1
fi

# ── 4. Start bot process (foreground) ───────────────────────────────────
BOT_MODE="${BOT_MODE:-worker}"
export PULSE_SINK=meetbot_sink

if [ "$BOT_MODE" == "worker" ]; then
    echo "[entrypoint] Starting bot worker (polling Redis, audio managed inline)..."
    python -m bot.worker
    BOT_EXIT=$?
else
    # One-shot mode: run background audio capture + foreground meet_bot
    RECORDING_OUTPUT_PATH="${RECORDING_OUTPUT_PATH:-/data/recording.wav}"
    echo "[entrypoint] Starting audio_capture..."
    python -m bot.audio_capture "$RECORDING_OUTPUT_PATH" &
    CAPTURE_PID=$!

    MEET_URL="${MEET_URL:-}"
    if [ -z "$MEET_URL" ]; then
        echo "[entrypoint] ERROR: MEET_URL is required for one-shot mode" >&2
        exit 1
    fi
    echo "[entrypoint] Starting one-shot meet_bot → $MEET_URL"
    python -m bot.meet_bot "$MEET_URL" "$RECORDING_OUTPUT_PATH"
    BOT_EXIT=$?

    # Give audio_capture a moment to detect the state change and stop ffmpeg
    sleep 2

    # If audio_capture is still running, terminate it
    if kill -0 "$CAPTURE_PID" 2>/dev/null; then
        echo "[entrypoint] Stopping audio_capture (PID=$CAPTURE_PID)"
        kill -TERM "$CAPTURE_PID" 2>/dev/null || true
        wait "$CAPTURE_PID" 2>/dev/null || true
    fi
fi

# ── 5. Cleanup ───────────────────────────────────────────────────────────
echo "[entrypoint] Bot process exited with code $BOT_EXIT, cleaning up..."

# Stop Xvfb
kill "$XVFB_PID" 2>/dev/null || true

# Stop PulseAudio
pulseaudio --kill 2>/dev/null || true

echo "[entrypoint] All services stopped. Exiting with code $BOT_EXIT"
exit "$BOT_EXIT"
