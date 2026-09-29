"""SQLAlchemy connection configuration for MySQL."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url() -> URL:
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL must be set to a MySQL connection URL.")
    url = make_url(value)
    if url.get_backend_name() != "mysql":
        raise RuntimeError("DATABASE_URL must use MySQL; no alternate database fallback is configured.")
    if url.database != "scribe":
        raise RuntimeError("DATABASE_URL must target the existing 'scribe' database.")
    return url


engine = create_engine(
    database_url(),
    pool_pre_ping=True,
    connect_args={"charset": "utf8mb4"},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
