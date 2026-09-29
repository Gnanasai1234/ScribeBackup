"""audio table.

Metadata about a generated narration. Never stores the audio binary --
`audioUrl` points at the actual file. A summary can have multiple audio
rows (different languages and/or voices).
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .common import RecordId


class Audio(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    summaryId: RecordId
    language: str
    voiceId: str
    voiceName: str
    voiceStyle: str = "neutral"
    speed: float = 1.0
    audioUrl: str
    duration: int | None = None
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
