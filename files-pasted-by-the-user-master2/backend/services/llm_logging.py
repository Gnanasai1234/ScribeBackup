"""Safe, size-only request diagnostics shared by the LLM providers."""
from __future__ import annotations

import json
import logging
import os
import re
import traceback
from urllib.parse import urlsplit, urlunsplit
from typing import Any

LOGGER = logging.getLogger("scribe.llm")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|access[_-]?token|password)(\s*[:=]\s*)['\"]?[^\s,;}'\"]+"),
    re.compile(r"(?i)([?&](?:key|token|api_key|authorization)=)[^&\s]+"),
)


def safe_endpoint(url: str | None) -> str:
    """Drop URL credentials, query strings, and fragments before logging."""
    if not url:
        return "unknown"
    try:
        parsed = urlsplit(str(url))
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", "")) or "unknown"
    except Exception:
        return "configured-endpoint"


def safe_error_message(exc: BaseException) -> str:
    """Extract a concise provider message and redact common credential forms."""
    value: Any = getattr(exc, "body", None)
    if value is None and callable(getattr(exc, "read", None)):
        try:
            value = exc.read(8192)
        except Exception:
            value = None
    if isinstance(value, bytes):
        try:
            value = json.loads(value.decode("utf-8", errors="replace"))
        except Exception:
            value = None
    if isinstance(value, dict):
        error = value.get("error", value)
        if isinstance(error, dict):
            value = error.get("message") or error.get("detail") or error.get("type")
        elif isinstance(error, str):
            value = error
    message = str(value or exc).strip()
    message = redact_sensitive_text(message)
    return (message[:800] if message else type(exc).__name__)


def redact_sensitive_text(value: str) -> str:
    """Redact credential forms from messages and debug tracebacks."""
    message = str(value)
    for env_key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "SARVAM_API_KEY", "GEMINI_API_KEY"):
        secret = os.getenv(env_key, "").strip()
        if len(secret) >= 6:
            message = message.replace(secret, "[REDACTED]")
    message = _SECRET_VALUE_PATTERNS[0].sub("Bearer [REDACTED]", message)
    message = _SECRET_VALUE_PATTERNS[1].sub("[REDACTED]", message)
    message = _SECRET_VALUE_PATTERNS[2].sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", message)
    message = _SECRET_VALUE_PATTERNS[3].sub(lambda match: f"{match.group(1)}[REDACTED]", message)
    message = re.sub(r"https?://[^\s]+", lambda match: safe_endpoint(match.group(0)), message)
    return message


def status_from(exc: BaseException | None, default: Any = "unknown") -> Any:
    if exc is None:
        return default
    return getattr(exc, "status_code", None) or getattr(exc, "code", None) or default


def log_request(provider: str, model: str, url: str, timeout: float, system: str, user: str) -> None:
    LOGGER.info(
        "[LLM REQUEST] provider=%s model=%s url=%s timeout=%ss input_characters=%d prompt_characters=%d estimated_tokens=%d",
        provider, model, safe_endpoint(url), timeout, len(user), len(system), (len(system) + len(user) + 3) // 4,
    )


def log_response(provider: str, status: Any, response: Any, text: str | None, elapsed: float) -> None:
    usable = bool(isinstance(text, str) and text.strip())
    LOGGER.info(
        "[LLM RESPONSE] provider=%s status=%s response_type=%s response_characters=%d usable=%s elapsed=%.2fs",
        provider, status, type(text).__name__ if text is not None else "NoneType", len(text or "") if isinstance(text, str) else 0,
        str(usable).lower(), elapsed,
    )


def log_success(provider: str, model: str, status: Any, elapsed: float, text: str) -> None:
    LOGGER.info(
        "[LLM SUCCESS] provider=%s model=%s status=%s elapsed=%.2fs response_characters=%d",
        provider, model, status, elapsed, len(text),
    )


def log_error(provider: str, model: str, url: str, timeout: float, exc: BaseException,
              elapsed: float, *, status: Any = None, category: str | None = None) -> None:
    actual_status = status if status is not None else status_from(exc)
    safe_message = safe_error_message(exc)
    error_type = category or type(exc).__name__
    LOGGER.error(
        "[LLM ERROR] provider=%s model=%s url=%s error_type=%s exception_class=%s status=%s elapsed=%.2fs timeout=%ss message=%s",
        provider, model, safe_endpoint(url), error_type, type(exc).__name__, actual_status, elapsed, timeout, safe_message,
    )
    try:
        numeric_status = int(actual_status)
    except (TypeError, ValueError):
        numeric_status = None
    if numeric_status is not None and 400 <= numeric_status <= 599:
        LOGGER.error(
            "[LLM HTTP ERROR] provider=%s status=%s model=%s url=%s elapsed=%.2fs message=%s",
            provider, numeric_status, model, safe_endpoint(url), elapsed, safe_message,
        )
    if LOGGER.isEnabledFor(logging.DEBUG) or os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        LOGGER.error("[LLM DEBUG TRACE] provider=%s model=%s\n%s", provider, model, redact_sensitive_text(trace))
    try:
        setattr(exc, "_llm_logged", True)
    except Exception:
        pass


def log_empty(provider: str, model: str, url: str, timeout: float, elapsed: float,
              *, status: Any = "unknown") -> EmptyResponseError:
    exc = EmptyResponseError("Provider returned an empty response")
    log_error(provider, model, url, timeout, exc, elapsed, status=status, category="EmptyResponseError")
    return exc


class EmptyResponseError(ValueError):
    """Provider returned no usable text despite completing the request."""


class ResponseValidationError(ValueError):
    """Provider output was non-empty but failed application validation."""


def error_category(exc: BaseException, status: Any = None) -> str:
    """Classify common HTTP, network, timeout, and parsing failures safely."""
    status = status if status is not None else status_from(exc, default=None)
    text = safe_error_message(exc).lower()
    kind = type(exc).__name__.lower()
    if "api key is not configured" in text: return "MissingAPIKey"
    if "sdk unavailable" in text or "sdk compatibility" in text: return "SDKCompatibilityError"
    if "timeout" in kind or isinstance(exc, TimeoutError): return "TimeoutError"
    if "connection refused" in text or "actively refused" in text: return "ConnectionRefused"
    if "model" in text and any(word in text for word in ("loading", "starting", "not ready")): return "ModelLoading"
    if isinstance(exc, EmptyResponseError): return "EmptyResponseError"
    if isinstance(exc, ResponseValidationError): return "ResponseValidationError"
    if "jsondecode" in kind or "parse" in kind or isinstance(exc, json.JSONDecodeError): return "ResponseParsingError"
    if "invalid api key" in text or "api key not valid" in text or "unauthenticated" in text: return "AuthenticationError"
    if "model" in text and any(word in text for word in ("not found", "does not exist", "invalid model")): return "ModelNotFound"
    if "rate limit" in text or "too many requests" in text: return "RateLimitExceeded"
    if "quota" in text: return "QuotaExceeded"
    if "invalid request" in text or "validation error" in text: return "InvalidRequest"
    try: numeric_status = int(status) if status is not None else None
    except (TypeError, ValueError): numeric_status = None
    if numeric_status == 401: return "AuthenticationError"
    if numeric_status == 403: return "PermissionError"
    if numeric_status == 404: return "ModelNotFound" if "ollama" in text or "model" in text else "NotFound"
    if numeric_status == 429: return "RateLimitExceeded"
    if numeric_status == 400: return "InvalidRequest"
    if numeric_status == 503: return "ServiceUnavailable"
    if numeric_status is not None and numeric_status >= 500: return "ProviderServerError"
    if "unavailable" in text or "overloaded" in text: return "ServiceUnavailable"
    if "connect" in kind or "connection" in text or "network" in kind: return "NetworkError"
    return type(exc).__name__
