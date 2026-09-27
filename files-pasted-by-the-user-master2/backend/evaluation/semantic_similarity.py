"""Semantic similarity evaluation using configured embedding provider (Groq API with local FastEmbed fallback)."""
from __future__ import annotations

import logging
from typing import Any

from services.embedding_service import cosine_similarity, get_embedding_service

LOGGER = logging.getLogger("scribe.evaluation")


def semantic_similarity(generated: str, reference: str) -> dict[str, Any]:
    if not generated or not reference:
        return {"value": None, "available": False, "reason": "Both generated text and reference text are required."}

    try:
        service = get_embedding_service()
        # Truncate texts safely to prevent token limit overflow on massive reference reports
        gen_trunc = generated[:2000]
        ref_trunc = reference[:2000]
        embeddings = service.get_embeddings([gen_trunc, ref_trunc])
        if len(embeddings) == 2:
            status = service.last_status or {}
            if not embeddings[0] or not embeddings[1] or status.get("effective_provider") == "lexical":
                return {"value": None, "available": False, "provider": "lexical", "model": None, "fallback_used": True, "reason": "No local embedding model is available; semantic similarity was not computed."}
            sim = cosine_similarity(embeddings[0], embeddings[1])
            provider_label = status.get("effective_provider", status.get("provider", "groq/local"))
            return {
                "value": round(float(sim), 4),
                "available": True,
                "provider": provider_label,
                "model": status.get("model", "bge-small-en-v1.5"),
                "fallback_used": status.get("fallback_used", False),
            }
    except Exception as exc:
        LOGGER.warning("Semantic similarity computation failed: %s", exc)
        return {"value": None, "available": False, "reason": f"Embedding computation error: {exc}"}

    return {"value": None, "available": False, "reason": "Could not compute embeddings."}
