###############################################################################
# Magentic-meetbot  ·  Bot service Dockerfile
#
# Ubuntu 22.04 with:
#   - PulseAudio   (virtual sink for capturing browser audio)
#   - ffmpeg        (records from the PulseAudio monitor source)
#   - Chromium      (headless browser for Google Meet)
#   - Xvfb          (virtual framebuffer — Chromium needs a DISPLAY)
#   - Python 3.11   (Playwright + bot logic)
#
# On container start:
#   1. Start PulseAudio daemon
#   2. Start Xvfb on :99
#   3. Create the meetbot_sink PulseAudio null-sink
#   4. Run the meet_bot + audio_capture Python processes
###############################################################################

FROM ubuntu:22.04

# ── Prevent interactive prompts ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Virtual display for headless Chromium
    DISPLAY=:99 \
    # PulseAudio config
    PULSE_SINK=meetbot_sink \
    # Playwright skips downloading browsers — we install Chromium via apt
    PLAYWRIGHT_BROWSERS_PATH=/usr/lib/chromium-browser

# ── System packages ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Audio
        pulseaudio \
        ffmpeg \
        # Browser
        chromium-browser \
        # Virtual framebuffer
        xvfb \
        # X11 libs that Chromium needs
        libnss3 \
        libatk-bridge2.0-0 \
        libdrm2 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpangocairo-1.0-0 \
        libgtk-3-0 \
        libxshmfence1 \
        libx11-xcb1 \
        libxcb-dri3-0 \
        # Fonts (so Meet renders text properly)
        fonts-liberation \
        fonts-noto-color-emoji \
        # Python 3.11
        software-properties-common \
        gnupg \
        gpg-agent \
        && add-apt-repository -y ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            python3.11 \
            python3.11-venv \
            python3.11-dev \
            python3-pip \
        # Misc
        dbus \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default ──────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── Upgrade pip ──────────────────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Python dependencies ─────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        playwright==1.52.0 \
        httpx==0.28.1 \
        python-dotenv==1.1.0 \
        redis==5.3.0 \
        structlog==25.3.0 \
        tenacity==9.1.2 \
    && playwright install --with-deps chromium

# ── PulseAudio daemon config (runs as root, no auth) ────────────────────
RUN mkdir -p /root/.config/pulse && \
    # default.pa: load the null-sink automatically on daemon start
    printf '%s\n' \
        "load-module module-native-protocol-unix" \
        "load-module module-null-sink sink_name=meetbot_sink sink_properties=device.description=MagenticMeetbotCapture" \
        "set-default-sink meetbot_sink" \
    > /root/.config/pulse/default.pa && \
    # client.conf: disable autospawn (we start it ourselves)
    printf '%s\n' \
        "autospawn = no" \
        "daemon-binary = /usr/bin/pulseaudio" \
    > /root/.config/pulse/client.conf && \
    # daemon.conf: run as root, no SHM (Docker)
    printf '%s\n' \
        "daemonize = no" \
        "exit-idle-time = -1" \
        "flat-volumes = no" \
        "system-instance = no" \
    > /root/.config/pulse/daemon.conf

# ── Copy source code ────────────────────────────────────────────────────
COPY bot/    /app/bot/
COPY integrations/ /app/integrations/

# ── Startup entrypoint ──────────────────────────────────────────────────
COPY bot/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
