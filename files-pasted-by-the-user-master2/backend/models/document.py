"""documents table.

Metadata about an uploaded file. The file itself is never stored here --
only `fileUrl`, a path/URL to wherever it actually lives on disk or in
object storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from .common import RecordId

DocumentStatus = Literal["uploaded", "processing", "completed", "failed"]


class Document(BaseModel):
    id: RecordId | None = Field(default=None)
    userId: RecordId
    fileName: str
    fileType: str
    fileUrl: str
    language: str = "en"
    pageCount: int | None = None
    wordCount: int | None = None
    status: DocumentStatus = "uploaded"
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
