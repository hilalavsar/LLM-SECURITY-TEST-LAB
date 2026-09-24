"""Database engine and session (SQLAlchemy 2.x).

Uses PostgreSQL via Docker. The ORM makes it swappable: point DATABASE_URL
elsewhere (e.g. SQLite) and the same code works.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import Settings


class Base(DeclarativeBase):
    pass


engine = create_engine(Settings.DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

# Columns added after the first release. create_all() never alters an existing
# table, so a database created earlier gets them here. Forward-only, portable
# SQL (no IF NOT EXISTS, which SQLite lacks).
_ADDED_COLUMNS = {
    "test_runs": {
        "corpus": "VARCHAR(80) NOT NULL DEFAULT ''",
        "defense_snapshot": "TEXT NOT NULL DEFAULT ''",
    },
}


def _add_missing_columns() -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def init_db() -> None:
    """Create tables if they do not exist and add newer columns (idempotent)."""
    import app.models  # noqa: F401 - ensures models are registered

    Base.metadata.create_all(engine)
    _add_missing_columns()
