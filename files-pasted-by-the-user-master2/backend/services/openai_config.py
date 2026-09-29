"""Shared, secret-safe configuration checks for native OpenAI requests."""
from __future__ import annotations

DIRECT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def openai_key_problem(api_key: str) -> str | None:
    """Reject missing or obvious placeholder credentials before any request."""
    if not api_key:
        return "Direct OpenAI is enabled but OPENAI_API_KEY is missing or invalid. Add your OpenAI API key to the existing .env file."
    lowered = api_key.strip().lower()
    placeholders = {"test", "test-key", "placeholder", "changeme", "your-openai-api-key", "sk-test"}
    if lowered in placeholders or any(token in lowered for token in ("placeholder", "example", "dummy", "changeme")):
        return "Direct OpenAI is enabled but OPENAI_API_KEY appears to be a placeholder/test value. Add your real OpenAI API key to the existing .env file."
    if len(api_key.strip()) < 20 or not lowered.startswith("sk-"):
        return "Direct OpenAI is enabled but OPENAI_API_KEY appears invalid. Add your real OpenAI API key to the existing .env file."
    return None
