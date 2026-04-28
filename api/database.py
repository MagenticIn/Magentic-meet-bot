"""
Magentic-meetbot  ·  database.py
────────────────────────────────
Async SQLAlchemy engine, session factory, and dependency for FastAPI.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.models import Base

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://meetbot:meetbot@postgres:5432/meetbot",
)

engine = create_async_engine(
    POSTGRES_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables (dev convenience — use Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
