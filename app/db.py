"""Database engine and session (SQLAlchemy 2.x).

Uses PostgreSQL via Docker. The ORM makes it swappable: point DATABASE_URL
elsewhere (e.g. SQLite) and the same code works.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import Settings


class Base(DeclarativeBase):
    pass


engine = create_engine(Settings.DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not exist (idempotent)."""
    import app.models  # noqa: F401 - ensures models are registered

    Base.metadata.create_all(engine)
