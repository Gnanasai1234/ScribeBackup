"""Canonical language choices carried through a document analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LanguageContext:
    source_language: str
    requested_output_language: str
    target_language: str
    translation_required: bool

    @classmethod
    def resolve(cls, source_language: str, requested_output_language: str | None) -> "LanguageContext":
        requested = (requested_output_language or "auto").strip().lower()
        target = source_language if requested in {"", "auto", "original", "source"} else requested
        return cls(
            source_language=source_language,
            requested_output_language=requested,
            target_language=target,
            translation_required=target != source_language,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
