"""Lazy, no-download local provider fallbacks shared by backend services."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

LOGGER = logging.getLogger("scribe.local_fallbacks")
_OLLAMA_MODELS: set[str] | None = None
_LOCAL_LLM_SERVICE: "LocalOllamaService | None" = None
_OLLAMA_READY: bool | None = None
_OLLAMA_FAILED_UNTIL = 0.0


def _ollama_ready() -> bool:
    """Probe the local Ollama HTTP API briefly before waiting on inference."""
    global _OLLAMA_READY
    if time.monotonic() < _OLLAMA_FAILED_UNTIL:
        return False
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            _OLLAMA_READY = response.status == 200
    except Exception:
        _OLLAMA_READY = False
    return bool(_OLLAMA_READY)


def installed_ollama_models(*, refresh: bool = False) -> set[str]:
    """Return models already installed in Ollama. Listing never pulls a model."""
    global _OLLAMA_MODELS
    if _OLLAMA_MODELS is not None and not refresh:
        return set(_OLLAMA_MODELS)
    binary = shutil.which("ollama")
    if not binary:
        _OLLAMA_MODELS = set()
        return set()
    try:
        result = subprocess.run([binary, "list"], capture_output=True, text=True, timeout=4, check=True)
        models = set()
        for line in result.stdout.splitlines()[1:]:
            fields = line.split()
            if fields:
                models.add(fields[0])
        _OLLAMA_MODELS = models
    except Exception as exc:
        LOGGER.info("Local Ollama model discovery unavailable (%s)", type(exc).__name__)
        _OLLAMA_MODELS = set()
    return set(_OLLAMA_MODELS)


def ollama_model_installed(model: str) -> bool:
    return any(candidate == model or candidate.split(":", 1)[0] == model.split(":", 1)[0] for candidate in installed_ollama_models())


class LocalOllamaService:
    """OpenAI-like JSON/text adapter for a model already installed in Ollama."""

    provider_name = "ollama-local"

    def __init__(self, model: str):
        self.model = model
        self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = max(2.0, min(90.0, float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "12"))))

    def complete(self, system: str, user: str, *, max_tokens: int = 1800) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            global _OLLAMA_READY, _OLLAMA_FAILED_UNTIL
            _OLLAMA_READY = False
            _OLLAMA_FAILED_UNTIL = time.monotonic() + 30
            raise RuntimeError(f"Local Ollama request failed ({type(exc).__name__})") from exc
        content = body.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Local Ollama returned an empty response")
        return content.strip()

    def complete_json(self, system: str, user: str, *, max_tokens: int = 1800) -> dict[str, Any]:
        content = self.complete(system, user, max_tokens=max_tokens)
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(content, index)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        global _OLLAMA_READY, _OLLAMA_FAILED_UNTIL
        _OLLAMA_READY = False
        _OLLAMA_FAILED_UNTIL = time.monotonic() + 30
        raise ValueError("Local Ollama response did not contain a JSON object")


def get_local_llm_service() -> LocalOllamaService | None:
    """Get the configured or first installed local chat model without downloading it."""
    global _LOCAL_LLM_SERVICE
    if _LOCAL_LLM_SERVICE is not None:
        return None if time.monotonic() < _OLLAMA_FAILED_UNTIL else _LOCAL_LLM_SERVICE
    available = installed_ollama_models()
    configured = os.getenv("LOCAL_LLM_MODEL", "").strip()
    if configured:
        match = next((model for model in available if model == configured), None)
        if match is None:
            LOGGER.info("Configured local model '%s' is not installed; skipping local LLM fallback", configured)
            return None
        selected = match
    else:
        chat_models = sorted(model for model in available if "embed" not in model.lower())
        selected = chat_models[0] if chat_models else None
    if not selected:
        return None
    if not _ollama_ready():
        LOGGER.info("Installed Ollama model is present but its local service is unavailable; skipping local LLM fallback")
        return None
    _LOCAL_LLM_SERVICE = LocalOllamaService(selected)
    LOGGER.info("Local Ollama fallback available using already-installed model=%s", selected)
    return _LOCAL_LLM_SERVICE


def local_ollama_embeddings(texts: list[str], model: str | None = None) -> list[list[float]] | None:
    """Use a locally installed Ollama embedding model; never trigger a pull."""
    selected = (model or os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text:latest")).strip()
    if not texts or not ollama_model_installed(selected) or not _ollama_ready():
        return None
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    payload = {"model": selected, "input": texts}
    request = urllib.request.Request(
        f"{base_url}/api/embed",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = body.get("embeddings")
        if isinstance(vectors, list) and len(vectors) == len(texts) and all(isinstance(v, list) and v for v in vectors):
            return [[float(value) for value in vector] for vector in vectors]
    except Exception as exc:
        global _OLLAMA_READY, _OLLAMA_FAILED_UNTIL
        _OLLAMA_READY = False
        _OLLAMA_FAILED_UNTIL = time.monotonic() + 30
        LOGGER.info("Local Ollama embeddings unavailable for model=%s (%s)", selected, type(exc).__name__)
    return None


def basic_local_claim_check(
    claim: str,
    evidence: list[str],
    *,
    claim_language: str,
    source_language: str,
    cited_ids_valid: bool,
) -> dict[str, Any]:
    """Conservative local grounding heuristic, explicitly weaker than LLM validation."""
    if not cited_ids_valid or not evidence:
        return {"status": "UNSUPPORTED", "confidence": 0.2, "reason": "No valid cited source chunk was available for the local check."}
    claim_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", claim))
    evidence_text = " ".join(evidence)
    evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", evidence_text))
    if claim_numbers - evidence_numbers:
        return {"status": "UNSUPPORTED", "confidence": 0.65, "reason": "A number in this claim was not found in its cited source text (local check)."}

    normalize = lambda value: re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
    claim_tokens = set(normalize(claim))
    source_tokens = set(normalize(evidence_text))
    if not claim_tokens:
        return {"status": "UNAVAILABLE", "confidence": None, "reason": "The local checker could not identify comparable claim text."}
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    same_language = claim_language == source_language
    exact = " ".join(normalize(claim)) in " ".join(normalize(evidence_text))
    if exact:
        return {"status": "SUPPORTED", "confidence": 0.85, "reason": "Claim text is present in cited source text (local exact-match check)."}
    if same_language and overlap >= 0.65:
        return {"status": "SUPPORTED", "confidence": 0.7, "reason": "Most claim terms appear in cited source text (basic local check)."}
    if same_language and overlap >= 0.35:
        return {"status": "PARTIALLY_SUPPORTED", "confidence": 0.5, "reason": "Some claim terms appear in cited source text (basic local check)."}
    if claim_numbers and claim_numbers <= evidence_numbers:
        return {"status": "PARTIALLY_SUPPORTED", "confidence": 0.45, "reason": "Cited source contains the claim's numbers, but local checking cannot verify its full meaning."}
    if not same_language:
        return {"status": "UNAVAILABLE", "confidence": None, "reason": "Local checking cannot verify cross-language meaning without a multilingual model."}
    return {"status": "UNAVAILABLE", "confidence": None, "reason": "Local text overlap was inconclusive; no semantic validation is available."}
