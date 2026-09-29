"""qa table.

One row per question asked about a document via the "Ask the Document"
feature. `sources` is a light pointer back to where the answer came
from -- just the page number(s), not a copy of the chunk text.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .common import RecordId


class QASource(BaseModel):
    pageNumber: int | None = None


class QuestionAnswer(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    question: str
    answer: str
    language: str = "en"
    sources: list[QASource] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
