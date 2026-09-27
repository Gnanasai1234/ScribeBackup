"""Embedding service with Groq API as primary and local FastEmbed as automatic fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from services.local_fallbacks import local_ollama_embeddings

LOGGER = logging.getLogger("scribe.embeddings")

_LOCAL_MODEL = None
_LOCAL_MODEL_NAME: str | None = None


def _get_local_model():
    """Load a cached multilingual FastEmbed model; never download on a request."""
    global _LOCAL_MODEL, _LOCAL_MODEL_NAME
    if _LOCAL_MODEL is None:
        try:
            from fastembed import TextEmbedding
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            cache_root = Path(os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")))
            model_cache = cache_root / "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
            cached = any(model_cache.rglob("model_optimized.onnx")) if model_cache.exists() else False
            if not cached:
                return None
            _LOCAL_MODEL = TextEmbedding(model_name=model_name, cache_dir=str(cache_root))
            _LOCAL_MODEL_NAME = model_name
            LOGGER.info("Loaded cached multilingual FastEmbed model=%s", model_name)
        except Exception as exc:
            LOGGER.info("Cached multilingual FastEmbed model is unavailable (%s)", type(exc).__name__)
            _LOCAL_MODEL = None
    return _LOCAL_MODEL


def cosine_similarity(vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingService:
    def __init__(self):
        # Default provider is groq if configured or specified, fallback is local
        self.provider = os.getenv("EMBEDDING_PROVIDER", "groq").strip().lower()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_base_url = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").strip()
        self.groq_model = os.getenv("GROQ_EMBEDDING_MODEL", "bge-small-en-v1.5").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.last_status: dict[str, Any] = {}

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Tries the configured provider, then installed local Ollama / cached
        multilingual FastEmbed, and finally empty vectors for lexical retrieval.
        """
        if not texts:
            return []

        # If primary provider is groq
        if self.provider == "groq" or (self.provider != "local" and self.groq_api_key):
            try:
                vectors = self._embed_groq(texts)
                self.last_status = {
                    "provider": "groq",
                    "effective_provider": "groq",
                    "model": self.groq_model,
                    "fallback_used": False,
                }
                return vectors
            except Exception as exc:
                LOGGER.warning("Groq embeddings failed (%s); trying local multilingual fallback", type(exc).__name__)
                return self._embed_local_fallback(texts, failure=exc)

        # If openai provider is specified
        if self.provider == "openai" and self.openai_api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key, timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "8")), max_retries=0)
                resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
                self.last_status = {
                    "provider": "openai",
                    "effective_provider": "openai",
                    "model": "text-embedding-3-small",
                    "fallback_used": False,
                }
                return [item.embedding for item in resp.data]
            except Exception as exc:
                LOGGER.warning("OpenAI embeddings failed (%s); trying local multilingual fallback", type(exc).__name__)
                return self._embed_local_fallback(texts, failure=exc)

        return self._embed_local_fallback(texts)

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single string."""
        results = self.get_embeddings([text])
        return results[0] if results else []

    def _embed_groq(self, texts: list[str]) -> list[list[float]]:
        """Attempt to call Groq OpenAI-compatible embedding endpoint."""
        from openai import OpenAI

        client = OpenAI(
            base_url=self.groq_base_url,
            api_key=self.groq_api_key or "gsk-placeholder",
            timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "8")),
            max_retries=0,
        )
        resp = client.embeddings.create(input=texts, model=self.groq_model)
        return [item.embedding for item in resp.data]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using an already-cached multilingual FastEmbed model."""
        model = _get_local_model()
        if model is None:
            return [[] for _ in texts]
        embeddings = list(model.embed(texts))
        return [vec.tolist() for vec in embeddings]

    def _embed_local_fallback(self, texts: list[str], failure: Exception | None = None) -> list[list[float]]:
        ollama_vectors = local_ollama_embeddings(texts)
        if ollama_vectors:
            self.last_status = {"provider": self.provider or "configured", "effective_provider": "local-ollama", "model": os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text:latest"), "fallback_used": True, "reason": type(failure).__name__ if failure else "configured local embedding"}
            return ollama_vectors
        fastembed_vectors = self._embed_local(texts)
        if fastembed_vectors and all(fastembed_vectors):
            self.last_status = {"provider": self.provider or "configured", "effective_provider": "local-fastembed-multilingual", "model": _LOCAL_MODEL_NAME, "fallback_used": True, "reason": type(failure).__name__ if failure else "local embedding"}
            return fastembed_vectors
        self.last_status = {"provider": self.provider or "configured", "effective_provider": "lexical", "model": None, "fallback_used": True, "reason": type(failure).__name__ if failure else "no cached local embedding model"}
        LOGGER.info("No local embedding model is available; retrieval will use lexical/source matching")
        return [[] for _ in texts]

    def get_info(self) -> dict[str, Any]:
        """Status information for health checks and UI display."""
        return {
            "configured_provider": self.provider,
            "groq_configured": bool(self.groq_api_key),
            "fallback_available": True,
            "fallback_engine": "installed Ollama embedding, cached multilingual FastEmbed, then lexical retrieval",
            "last_status": self.last_status,
        }


# Global singleton instance
_EMBEDDING_SERVICE = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    return _EMBEDDING_SERVICE
