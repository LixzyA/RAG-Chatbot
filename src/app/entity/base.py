from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
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
# Initialisation & light-weight column migrations
# ------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables defined in metadata (safe to call on every startup).

    ``Base.metadata.create_all`` only creates missing tables — it does NOT
    add new columns to existing tables. ``_backfill_columns`` runs a
    SQLite-friendly ``ALTER TABLE`` for any columns added in newer model
    revisions so legacy dev databases pick them up without a manual wipe.
    """

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _backfill_columns(conn)


async def _backfill_columns(conn: AsyncConnection) -> None:
    """Add model columns missing from pre-existing tables.

    Each entry is ``(table, column, ddl_type)``. ``PRAGMA table_info`` is
    dialect-portable enough for SQLite (and harmless on others — the
    additions are skipped when the columns already exist). Keep entries
    additive and idempotent; destructive renames/drops belong in a real
    migration tool (e.g. Alembic).
    """
    additions: tuple[tuple[str, str, str], ...] = (
        ("rag_traces", "filters_applied", "JSON"),
    )
    for table, column, ddl_type in additions:
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {row[1] for row in result.fetchall()}
        if column in existing:
            continue
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
