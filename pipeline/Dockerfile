###############################################################################
# Magentic-meetbot  ·  Pipeline service (Celery worker)
#
# Runs: (default) OpenAI gpt-4o-transcribe-diarize, or faster-whisper + whisperx →
#       Helsinki-NLP translation for Hindi lines → OpenAI summarisation
#
# Single-concurrency worker to avoid OOM with large ML models.
###############################################################################
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is required by whisperx / torchaudio for audio decoding
# git is needed for pip installing whisperx from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install PyTorch CPU (swap for CUDA wheels if GPU available) ──
RUN pip install --no-cache-dir \
        torch==2.6.0+cpu \
        torchaudio==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install ML libraries ────────────────────────────────────────
RUN pip install --no-cache-dir \
        faster-whisper==1.2.1 \
        "whisperx @ git+https://github.com/m-bain/whisperX.git" \
        transformers==4.51.3 \
        sentencepiece==0.2.0

# ── Step 3: Install app dependencies ────────────────────────────────────
RUN pip install --no-cache-dir \
        openai>=1.0.0 \
        celery==5.5.2 \
        redis==5.3.0 \
        httpx==0.28.1 \
        python-dotenv==1.1.0 \
        structlog==25.3.0 \
        tenacity==9.1.2 \
        asyncpg==0.30.0

# ── Copy source ─────────────────────────────────────────────────────────
COPY pipeline/ /app/pipeline/
COPY integrations/ /app/integrations/

# ── Pre-download the translation model (avoids first-run latency) ───────
RUN python -c "\
from transformers import pipeline; \
pipeline('translation', model='Helsinki-NLP/opus-mt-hi-en'); \
print('opus-mt-hi-en downloaded')" || true

CMD ["celery", "-A", "pipeline.worker", "worker", "--loglevel=info", "--concurrency=1"]
