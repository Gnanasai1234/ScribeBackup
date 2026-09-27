"""GovReport-specific adapter; human reference text is evaluation-only."""
from __future__ import annotations

from typing import Any

from evaluation.evaluator import evaluate_summary


def evaluate_record(result: dict[str, Any], reference_summary: str) -> dict[str, Any]:
    return evaluate_summary(result, reference_summary)
