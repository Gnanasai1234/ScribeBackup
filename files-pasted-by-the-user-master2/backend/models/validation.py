"""validation table.

Stores the groundedness check run against a summary: an overall
status/score plus the per-claim breakdown. Claims are embedded
directly (not their own table) since they only ever make sense
in the context of one validation run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from .common import RecordId

ClaimStatus = Literal["supported", "partially_supported", "unsupported"]
ValidationStatus = Literal["passed", "failed", "partial"]


class Claim(BaseModel):
    claim: str
    status: ClaimStatus
    pageNumber: int | None = None
    chunkId: str | None = None
    evidence: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Validation(BaseModel):
    id: RecordId | None = Field(default=None)
    documentId: RecordId
    summaryId: RecordId
    status: ValidationStatus
    groundednessScore: float = Field(ge=0.0, le=1.0)
    claims: list[Claim] = Field(default_factory=list)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
