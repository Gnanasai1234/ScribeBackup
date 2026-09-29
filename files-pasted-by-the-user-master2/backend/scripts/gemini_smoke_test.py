"""Minimal Gemini API smoke test. Uses no document data or other providers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(PROJECT / ".env")

from app.llm_service import GeminiProvider, _log_provider_error  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provider = GeminiProvider()
    print(f"provider=gemini model={provider.model} api_key_present={bool(provider.api_key)}")
    if not provider.api_key:
        print("request_started=false error_type=MissingAPIKey")
        return 2
    try:
        print("request_started=true")
        response = provider.complete(
            "Return exactly the word OK.", "Return the word OK", max_tokens=12
        )
        print(f"response_received=true response_non_empty={bool(response.strip())}")
        return 0 if response.strip() else 1
    except Exception as exc:
        _log_provider_error("gemini", provider, exc)
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None) or "unknown"
        category = getattr(exc, "category", type(exc).__name__)
        print(f"response_received=false error_type={category} status={status}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
