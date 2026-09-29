"""chunks table.

One row per document chunk, used for lexical/RAG retrieval. If you add
a vector database later, the embedding itself still does NOT belong
here -- only the text and its location in the source document.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .common import RecordId


class DocumentChunk(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    chunkNumber: int
    text: str
    pageNumber: int | None = None
    section: str | None = None
    embeddingReference: str | None = None

    model_config = {"populate_by_name": True}
