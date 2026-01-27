# app/db/session_v2.py — FINAL SAFE VERSION

import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.utils.logger import log_info, log_warning

# -----------------------------------------------------------
# DATABASE CONFIGURATION
# -----------------------------------------------------------
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "marketing_advantage")

_env_database_url = os.getenv("DATABASE_URL")

if POSTGRES_PASSWORD:
    DATABASE_URL = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
elif _env_database_url:
    DATABASE_URL = _env_database_url
else:
    raise RuntimeError("Database credentials not provided.")

if "asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+asyncpg://"
    ).replace(
        "postgres://", "postgresql+asyncpg://"
    )

# -----------------------------------------------------------
# ENGINE
# -----------------------------------------------------------
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=25,
    max_overflow=50,
    pool_timeout=60,
    pool_recycle=1800,
)

# -----------------------------------------------------------
# SESSION FACTORY
# -----------------------------------------------------------
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# -----------------------------------------------------------
# ✅ FASTAPI-COMPATIBLE DEPENDENCY
# -----------------------------------------------------------
async def get_db() -> AsyncSession:
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        log_warning(f"[session_v2] Rolling back transaction: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()

# -----------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------
async def verify_connection():
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            log_info("[session_v2] ✅ Database connection verified.")
    except Exception as e:
        log_warning(f"[session_v2] ❌ Database connection failed: {e}")
        raise

# ---------------------------------------------------------
# Retrieval Adapter
# ---------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def get_async_session():
    """
    Thin adapter for retrieval CLI / runtime.

    Yields:
        Async SQLAlchemy session

    This adapter does NOT change existing session logic.
    """

    # 🔁 CASE 1: You already have async_session / AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as session:
            yield session
            return
    except NameError:
        pass

    # 🔁 CASE 2: You already have a get_session() async generator
    try:
        async for session in get_session():
            yield session
            return
    except NameError:
        pass

    raise RuntimeError(
        "No async session factory found. "
        "Please expose AsyncSessionLocal or get_session()."
    )
