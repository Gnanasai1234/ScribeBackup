"""summaries table.

One row per generated summary. A single document can have several
summaries over time (different modes, languages, or re-runs).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from .common import RecordId

SummaryMode = Literal["quick", "standard", "deep", "auto", "custom"]


class Summary(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    summaryMode: SummaryMode = "standard"
    resolvedDepth: Literal["quick", "standard", "deep"] = "standard"
    summaryType: str = "executive"
    language: str = "en"
    purpose: str = "Executive briefing"
    audience: str = "Leadership"
    summary: str
    keyFindings: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    actionItems: list[str] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
