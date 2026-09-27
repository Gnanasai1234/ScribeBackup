"""Reference-based GovReport evaluator; references never enter source validation."""
from __future__ import annotations

from typing import Any

from .completeness import reference_coverage
from .groundedness import citation_coverage, groundedness
from .rouge import calculate_rouge
from .semantic_similarity import semantic_similarity


def evaluate_summary(result: dict[str, Any], reference_summary: str) -> dict[str, Any]:
    generated = result.get("executive_summary", "")
    claims = result.get("claims", [])
    return {
        "metrics": {
            **calculate_rouge(generated, reference_summary),
            "semanticSimilarity": semantic_similarity(generated, reference_summary),
            "completeness": reference_coverage(generated, reference_summary),
            "groundedness": groundedness(claims),
            "citationCoverage": citation_coverage(claims),
            "relevance": {"value": None, "available": False, "reason": "No relevance labels or semantic evaluator are configured."},
            "factualConsistency": {"value": None, "available": False, "reason": "Automatic factual evaluation against a gold claim set is not available."},
            "validationAccuracy": {"value": None, "available": False, "reason": "GovReport does not provide claim-level validation labels."},
        },
        "summary": {"characters": len(generated), "words": len(generated.split()), "depth": result.get("summary_depth_resolved", result.get("depth")), "validated_claims": result.get("quality", {}).get("claims_checked", len(claims))},
        "reference_role": "evaluation_only",
        "source_of_truth": "original report chunks",
    }
