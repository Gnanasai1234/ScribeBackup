"""State-based sequential workflow for analysis, synthesis, validation, and quality."""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict


class AnalysisState(TypedDict):
    document: dict[str, Any]
    options: Any
    result: dict[str, Any] | None
    node_history: list[str]
    started_at: float


def document_analysis_agent(state: AnalysisState) -> AnalysisState:
    document = state["document"]
    if not document.get("chunks"):
        raise ValueError("Document analysis needs at least one extracted source chunk.")
    document["analysis_profile"] = {"word_count": len(document.get("text", "").split()), "chunk_count": len(document["chunks"]), "source_language": document.get("language", "en")}
    state["node_history"].append("document_analysis")
    return state


def summarization_agent(state: AnalysisState, summarize: Callable[[dict[str, Any], Any], dict[str, Any]]) -> AnalysisState:
    state["result"] = summarize(state["document"], state["options"])
    state["node_history"].append("summarization")
    return state


def summary_validation_agent(state: AnalysisState) -> AnalysisState:
    result = state["result"] or {}
    claims = result.get("claims", [])
    state["result"] = result | {
        "validation": {
            "status": result.get("validation_status", result.get("status", "unavailable")),
            "source_of_truth": "original_document_chunks",
            "language_context": state["document"].get("language_context"),
            "provider": result.get("validation_provider", "configured_provider"),
            "claims_checked": len(claims),
            "supported_claims": sum(c.get("status") == "SUPPORTED" for c in claims),
            "unsupported_claims": sum(c.get("status") == "UNSUPPORTED" for c in claims),
            "attempts": result.get("stats", {}).get("validation_attempts", 0),
            "max_attempts": result.get("stats", {}).get("max_validation_attempts", 3),
            "max_attempts_reached": result.get("stats", {}).get("validation_attempts", 0) >= result.get("stats", {}).get("max_validation_attempts", 3) and result.get("validation_status") not in {"passed", "no_claims"},
        }
    }
    state["node_history"].append("summary_validation")
    return state


def quality_evaluation_agent(state: AnalysisState) -> AnalysisState:
    result = state["result"] or {}
    result["stats"] = result.get("stats", {}) | {"processing_time_ms": round((time.perf_counter() - state["started_at"]) * 1000, 1)}
    result["workflow"] = state["node_history"] + ["quality_evaluation"]
    result.setdefault("evaluation", {"metrics_available": ["claim support ratio", "citation coverage", "semantic similarity"], "metrics_unavailable": ["ROUGE without a reference summary"]})
    state["node_history"].append("quality_evaluation")
    state["result"] = result
    return state


def run_analysis_workflow(document: dict[str, Any], options: Any, summarize: Callable[[dict[str, Any], Any], dict[str, Any]]) -> dict[str, Any]:
    """Execute a readable linear graph with shared structured state."""
    state: AnalysisState = {"document": document, "options": options, "result": None, "node_history": [], "started_at": time.perf_counter()}
    state = document_analysis_agent(state)
    state = summarization_agent(state, summarize)
    state = summary_validation_agent(state)
    state = quality_evaluation_agent(state)
    return state["result"] or {}
