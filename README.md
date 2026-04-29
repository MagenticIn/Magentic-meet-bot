# Magentic-meetbot

Self-hosted AI meeting notetaker that auto-joins Google Meet, records audio, then runs a Celery pipeline: by default **OpenAI `gpt-4o-transcribe-diarize`** (one API call for transcript + speakers), optional **faster-whisper + whisperx** via `TRANSCRIPTION_BACKEND=whisper`, Hindi line translation (Helsinki-NLP when needed), **OpenAI `gpt-4o-mini`** summaries, and optional POST to an external PM API.

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI    │────▶│   Redis Queue   │────▶│   Meet Bot      │
│   (api)      │     │                 │     │  (Playwright)   │
│              │◀────│                 │◀────│  + PulseAudio   │
└──────┬───────┘     └─────────────────┘     └─────────────────┘
       │                     │
       │                     ▼
       │             ┌─────────────────┐
       │             │  Celery Worker  │
       │             │  (pipeline)     │
       │             │                 │
       │             │ ┌─────────────┐ │
       │             │ │ ASR+diarize │ │  OpenAI (default) or whisper+whisperx
       │             │ │ summarize   │ │  OpenAI API
       │             │ └─────────────┘ │
       │             └────────┬────────┘
       │                      │
       ▼                      ▼
┌──────────────┐     ┌─────────────────┐
│  PostgreSQL  │     │  PM API Client  │──▶ External PM Tool
└──────────────┘     └─────────────────┘
```

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url> && cd Magentic-meet-bot
cp .env.example .env
# Edit .env with your credentials
```

### 2. Start all services

```bash
docker compose up --build -d
```

### 3. Join a meeting

```bash
curl -X POST http://localhost:8000/api/v1/meetings/trigger \
  -H "Content-Type: application/json" \
  -d '{"meeting_url": "https://meet.google.com/abc-defg-hij"}'
```

### 4. Check status

```bash
curl http://localhost:8000/api/v1/meetings/<meeting_id>
```

## Organization Deployment (HTTPS)

For same-day org rollout on a cloud VM with public HTTPS:

1. Copy `deploy/.env.prod.example` to `.env.prod` and fill real secrets.
2. Follow `DEPLOY_ORG.md`.
3. Start:

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build
```

This deploys:
- API + dashboard
- Bot worker
- Pipeline worker (transcription + translation + summary)
- Postgres + Redis
- Caddy reverse proxy with TLS and optional basic auth

### 5. Get results

```bash
curl http://localhost:8000/api/v1/meetings/<meeting_id>
```

## Services

| Service    | Port | Description                                        |
|------------|------|----------------------------------------------------|
| `api`      | 8000 | FastAPI REST API — meeting CRUD, webhooks           |
| `bot`      | —    | Playwright bot — joins Meet, records audio          |
| `pipeline` | —    | Celery worker — transcribe (+ diarize) → summarize   |
| `postgres` | 5432 | Meeting records, transcripts, summaries             |
| `redis`    | 6379 | Task queue (bot join requests + Celery broker)      |

## Project Structure

```
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── bot/
│   ├── Dockerfile
│   ├── meet_bot.py          # Playwright-based Google Meet bot
│   └── audio_capture.py     # PulseAudio + ffmpeg audio recording
├── pipeline/
│   ├── Dockerfile
│   ├── worker.py            # Celery app + orchestration task
│   ├── transcribe.py              # faster-whisper (whisper backend)
│   ├── openai_transcribe_diarize.py  # OpenAI gpt-4o-transcribe-diarize
│   ├── diarize.py                 # whisperx (whisper backend)
│   ├── hi_en_translate.py         # Hindi → English (shared)
│   └── summarize.py               # OpenAI LLM summarisation
├── api/
│   ├── Dockerfile
│   ├── main.py              # FastAPI entrypoint
│   ├── database.py          # Async SQLAlchemy setup
│   ├── models.py            # Pydantic schemas + ORM models
│   └── routes/
│       ├── meetings.py      # Meeting CRUD + join
│       └── webhook.py       # Internal callbacks
└── integrations/
    └── pm_client.py         # POST results to external PM API
```

## Environment Variables

| Variable                     | Description                                |
|------------------------------|--------------------------------------------|
| `GOOGLE_EMAIL`               | Google account email for the bot           |
| `GOOGLE_PASSWORD`            | Google account app password                |
| `TRANSCRIPTION_BACKEND`      | `openai` (default) or `whisper` (local ASR + HF diarization) |
| `HF_TOKEN`                   | Hugging Face token — required only for `whisper` backend |
| `OPENAI_API_KEY`             | OpenAI — STT when backend is `openai`, plus summarisation |
| `OPENAI_TRANSCRIPTION_LANGUAGE` | Optional `hi` or `en` to bias the **whole** recording; leave empty for **mixed Hindi + English** (default) |
| `WHISPER_LANGUAGE`           | Local whisper only — language hint for faster-whisper |
| `PM_API_BASE_URL`            | External PM API base URL                   |
| `PM_API_KEY`                 | External PM API authentication key         |
| `POSTGRES_URL`               | PostgreSQL connection string               |
| `WHISPER_MODEL_SIZE`         | faster-whisper model (default: `large-v3`) |
| `WHISPER_DEVICE`             | `cpu` or `cuda` (default: `cpu`)           |
| `WHISPER_COMPUTE_TYPE`       | `int8`, `float16`, etc. (default: `int8`)  |

## API Endpoints

### Meetings

- **POST** `/api/v1/meetings/trigger` — Schedule bot to join a meeting
- **GET** `/api/v1/meetings` — List all meetings
- **GET** `/api/v1/meetings/{id}` — Get full meeting details
- **GET** `/api/v1/meetings/{id}/transcript` — Get raw + translated transcript
- **POST** `/api/v1/meetings/{id}/push-to-pm` — Push notes to PM integration

### System

- **GET** `/health` — Service health check

### Internal Webhooks

- **POST** `/api/v1/webhook/recording-complete` — Bot → API callback
- **POST** `/api/v1/webhook/pipeline-complete` — Pipeline → API callback

## LLM Cost (Estimate)

- `gpt-4o-mini` summarization is approximately **~$0.0015 per meeting** for typical internal sync calls.

## GPU Support

To enable GPU acceleration for faster-whisper and whisperx, update `docker-compose.yml`:

```yaml
pipeline:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    WHISPER_DEVICE: cuda
    WHISPER_COMPUTE_TYPE: float16
```

And use the CUDA PyTorch base in `pipeline/Dockerfile`.

## License

MIT
