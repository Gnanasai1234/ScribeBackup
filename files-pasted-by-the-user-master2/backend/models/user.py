"""users collection.

Basic account info. Nothing about documents/summaries lives here --
those all reference a user via `userId`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, EmailStr, Field

from .common import PyObjectId


class User(BaseModel):
    id: PyObjectId | None = Field(default=None, alias="_id")
    name: str
    email: EmailStr
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}
