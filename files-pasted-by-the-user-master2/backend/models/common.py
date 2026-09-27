"""Shared helpers for Mongo-backed Pydantic models.

Kept in one tiny file on purpose -- every collection needs the same
"Mongo ObjectId <-> string" conversion, so we do it once here instead
of repeating it in every model.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator

# Lets a model field accept either a real ObjectId (as read back from
# Mongo) or a plain string (as passed in from request bodies), and
# always stores/serializes it as a string.
PyObjectId = Annotated[str, BeforeValidator(str)]
