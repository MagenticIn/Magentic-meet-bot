from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api.database import engine, init_db
from api.models import HealthResponse
from api.routes import meetings, webhook


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title="Magentic Meet Bot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meetings.router, prefix="/api/v1")
app.include_router(webhook.router, prefix="/api/v1")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/", include_in_schema=False)
async def landing_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
