"""feedback table.

User feedback on a specific piece of output (summary, audio, or a QA
answer) for a document.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from .common import RecordId

FeedbackType = Literal["summary", "audio", "qa"]


class Feedback(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    type: FeedbackType
    rating: int = Field(ge=1, le=5)
    feedback: str | None = None
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
