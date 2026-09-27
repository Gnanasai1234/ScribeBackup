"""Scores based on source validation results, never on the reference summary."""
from __future__ import annotations
from typing import Any


def groundedness(claims: list[dict[str, Any]]) -> dict[str, Any]:
    if not claims:
        return {"value": None, "available": False, "reason": "No claims were validated."}
    supported = sum(claim.get("status") == "SUPPORTED" for claim in claims)
    return {"value": round(supported / len(claims), 4), "available": True, "supported_claims": supported, "claims": len(claims)}


def citation_coverage(claims: list[dict[str, Any]]) -> dict[str, Any]:
    if not claims:
        return {"value": None, "available": False, "reason": "No claims were validated."}
    cited = sum(bool(claim.get("sources")) for claim in claims)
    return {"value": round(cited / len(claims), 4), "available": True, "cited_claims": cited, "claims": len(claims)}
