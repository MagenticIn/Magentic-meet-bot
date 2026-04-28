###############################################################################
# Magentic-meetbot  ·  API service (FastAPI + Uvicorn)
###############################################################################
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.115.12 \
        "uvicorn[standard]==0.34.2" \
        pydantic==2.11.1 \
        pydantic-settings==2.9.1 \
        "sqlalchemy[asyncio]==2.0.40" \
        asyncpg==0.30.0 \
        alembic==1.15.2 \
        python-multipart==0.0.20 \
        httpx==0.28.1 \
        redis==5.3.0 \
        celery==5.5.2 \
        python-dotenv==1.1.0 \
        structlog==25.3.0 \
        tenacity==9.1.2

# Copy source
COPY api/ /app/api/
COPY integrations/ /app/integrations/
COPY pipeline/__init__.py /app/pipeline/__init__.py
COPY pipeline/worker.py /app/pipeline/worker.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
