from __future__ import annotations


from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ------------------------------------------------------------------
# Engine & session factory
# ------------------------------------------------------------------

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ANN001
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.close()


# ------------------------------------------------------------------
# Declarative base
# ------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables defined in metadata (safe to call on every startup)."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
