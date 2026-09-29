"""Shared backend-only client for the configured OpenAI Gateway."""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

LOGGER = logging.getLogger("scribe.openai_gateway")


class GatewayError(RuntimeError):
    """A gateway error that intentionally excludes request headers and bodies."""


class OpenAIGateway:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None) -> None:
        self.base_url = (base_url if base_url is not None else os.getenv("OPENAI_GATEWAY_URL", "https://keygateway1.arshnivlabs.com/v1")).strip().rstrip("/")
        self.api_key = (api_key if api_key is not None else (os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENAI_GATEWAY_API_KEY", ""))).strip()
        timeout_value = timeout if timeout is not None else os.getenv("OPENAI_GATEWAY_TIMEOUT", "120")
        try:
            self.timeout = float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("OPENAI_GATEWAY_TIMEOUT must be a positive number of seconds.") from exc
        if self.timeout <= 0:
            raise RuntimeError("OPENAI_GATEWAY_TIMEOUT must be a positive number of seconds.")
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            follow_redirects=False,
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _url(self, endpoint: str) -> str:
        if not self.configured:
            raise GatewayError("OpenAI Gateway URL or API key is not configured in the backend environment.")
        prefix = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return f"{prefix}/{endpoint.lstrip('/')}"

    def _request_json(self, method: str, endpoint: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(endpoint)
        try:
            response = self._client.request(method, url, json=payload)
        except httpx.HTTPError as exc:
            LOGGER.warning("Gateway %s request failed type=%s", endpoint, type(exc).__name__)
            raise GatewayError(f"OpenAI Gateway {endpoint} request failed ({type(exc).__name__}).") from exc
        if response.is_error:
            LOGGER.warning("Gateway %s request returned HTTP %s", endpoint, response.status_code)
            raise GatewayError(f"OpenAI Gateway {endpoint} returned HTTP {response.status_code}.")
        try:
            data = response.json()
        except ValueError as exc:
            raise GatewayError(f"OpenAI Gateway {endpoint} returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise GatewayError(f"OpenAI Gateway {endpoint} returned an unexpected response.")
        return data

    def health(self) -> dict[str, Any]:
        """Call GET /v1/health using the configured Bearer token."""
        return self._request_json("GET", "health")

    def embedding(self, text: str) -> list[float]:
        """Call embeddings with the gateway's required single-string body."""
        data = self._request_json("POST", "embeddings", payload={"input": text})
        vector: Any = None
        try:
            vector = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError):
            vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise GatewayError("OpenAI Gateway embeddings response did not contain a vector.")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise GatewayError("OpenAI Gateway embeddings response contained an invalid vector.") from exc


_GATEWAY: OpenAIGateway | None = None


def get_openai_gateway() -> OpenAIGateway:
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = OpenAIGateway()
    return _GATEWAY
