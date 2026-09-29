"""Thin service facade over the SQLAlchemy document repository."""
from __future__ import annotations

from typing import Any

from ..database.repositories import get_document, list_documents, save_document


def create_document(doc: dict[str, Any]) -> str:
    save_document(doc)
    return str(doc["id"])


def get_document_by_id(document_id: str) -> dict[str, Any] | None:
    return get_document(document_id)


def list_documents_for_user(_user_id: str | None = None) -> list[dict[str, Any]]:
    return list_documents()


def update_document(document: dict[str, Any]) -> None:
    save_document(document)
