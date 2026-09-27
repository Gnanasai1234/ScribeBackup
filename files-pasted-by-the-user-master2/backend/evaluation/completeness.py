"""Reference coverage proxy, explicitly reported as lexical ROUGE-L recall."""
from __future__ import annotations

from .rouge import rouge_l


def reference_coverage(generated: str, reference: str) -> dict[str, object]:
    value = rouge_l(generated, reference)["recall"]
    return {"value": value, "available": True, "method": "ROUGE-L recall against the human reference; lexical proxy, not a factual completeness judgment."}
