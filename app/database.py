"""SQLAlchemy engine / session setup (SQLite by default, any SQLAlchemy URL works)."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import DATA_DIR, DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")
_in_memory = _is_sqlite and (":memory:" in DATABASE_URL or DATABASE_URL in ("sqlite://", "sqlite:///"))

if _is_sqlite and not _in_memory:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

_kwargs: dict = {}
if _is_sqlite:
    _kwargs["connect_args"] = {"check_same_thread": False}
    if _in_memory:
        _kwargs["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, **_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from . import db_models  # noqa: F401  (register tables)
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
