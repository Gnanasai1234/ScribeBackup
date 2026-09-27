"""CRUD helpers for the documents collection.

Deliberately thin -- a couple of functions is enough for a student
project. Copy this pattern (not a generic base class) for the other
collections: summaries, chunks, validation, audio, qa, feedback.
"""
from __future__ import annotations

from typing import Any

from bson import ObjectId

from ..database.mongodb import get_database
from ..models.document import Document, DocumentStatus


async def create_document(doc: Document) -> str:
    db = get_database()
    payload = doc.model_dump(by_alias=True, exclude={"id"})
    result = await db.documents.insert_one(payload)
    return str(result.inserted_id)


async def get_document(document_id: str) -> dict[str, Any] | None:
    db = get_database()
    return await db.documents.find_one({"_id": ObjectId(document_id)})


async def list_documents_for_user(user_id: str) -> list[dict[str, Any]]:
    db = get_database()
    cursor = db.documents.find({"userId": ObjectId(user_id)}).sort("createdAt", -1)
    return [doc async for doc in cursor]


async def update_document_status(document_id: str, status: DocumentStatus) -> None:
    db = get_database()
    await db.documents.update_one({"_id": ObjectId(document_id)}, {"$set": {"status": status}})
