"""audio collection.

Metadata about a generated narration. Never stores the audio binary --
`audioUrl` points at the actual file. A summary can have multiple audio
rows (different languages and/or voices).
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .common import PyObjectId


class Audio(BaseModel):
    id: PyObjectId | None = Field(default=None, alias="_id")
    documentId: PyObjectId
    summaryId: PyObjectId
    language: str
    voiceId: str
    voiceName: str
    voiceStyle: str = "neutral"
    speed: float = 1.0
    audioUrl: str
    duration: int | None = None
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
