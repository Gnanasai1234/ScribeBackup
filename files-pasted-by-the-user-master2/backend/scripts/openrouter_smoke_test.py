"""Minimal OpenRouter API smoke test. Uses a tiny prompt and no document data."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import main as backend_main  # noqa: E402 - uses the backend's normal .env loading
from app.llm_service import OpenRouterProvider, _log_provider_error  # noqa: E402
from services.llm_logging import log_success  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provider = OpenRouterProvider()
    print(f"provider=openrouter model={provider.model} api_key_present={bool(provider.api_key)} endpoint={provider.base_url}")
    if not provider.api_key:
        print("request_started=false error_type=MissingAPIKey")
        return 2
    try:
        started = time.monotonic()
        response = provider.complete("Reply with the requested token only.", "Reply with exactly: OPENROUTER_TEST_OK", max_tokens=12)
        log_success(provider.provider_name, provider.model, 200, time.monotonic() - started, response)
        valid = response.strip() == "OPENROUTER_TEST_OK"
        print(f"response_received=true response_non_empty={bool(response.strip())} token_matched={valid}")
        return 0 if valid else 1
    except Exception as exc:
        _log_provider_error("openrouter", provider, exc)
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None) or "unknown"
        print(f"response_received=false error_type={type(exc).__name__} status={status}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
