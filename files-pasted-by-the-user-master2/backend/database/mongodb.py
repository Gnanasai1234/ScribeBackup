"""MongoDB connection for the Scribe backend.

One shared async client + database handle, and the small set of
indexes the app actually needs. No repository layer, no ORM-style
abstraction on top of Motor -- collections are used directly
(`get_database().documents`, `get_database().summaries`, etc.) from
the services in `backend/services/`.
"""
from __future__ import annotations

import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the shared MongoDB database handle, connecting on first use."""
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        name = os.getenv("MONGODB_DB", "scribe")
        _client = AsyncIOMotorClient(uri)
        _db = _client[name]
    return _db


async def create_indexes() -> None:
    """Create the indexes listed in docs/mongodb-schema.md. Safe to call repeatedly."""
    db = get_database()
    await db.users.create_index("email", unique=True)
    await db.documents.create_index("userId")
    await db.documents.create_index("createdAt")
    await db.chunks.create_index("documentId")
    await db.summaries.create_index("documentId")
    await db.validation.create_index("summaryId")
    await db.audio.create_index("documentId")
    await db.qa.create_index("documentId")
    await db.feedback.create_index("documentId")


async def close_connection() -> None:
    """Release the client. Call this on app shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None
