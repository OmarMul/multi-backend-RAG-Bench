"""Database session factory and table creation helpers."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.db.models import Base


# ---------------------------------------------------------------------------
# Engine — created lazily; uses DATABASE_URL from .env
# ---------------------------------------------------------------------------
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,      # detect stale connections
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def create_tables() -> None:
    """Create all ORM tables if they don't already exist."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all ORM tables — use only in tests."""
    engine = _get_engine()
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session with automatic commit/rollback."""
    _get_engine()
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_db() -> bool:
    """Return True if the DB is reachable."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
