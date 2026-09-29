from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main


class FakeProvider:
    def __init__(self, name: str, model: str, result: str = "", error: Exception | None = None):
        self.provider_name = name
        self.model = model
        self.result = result
        self.error = error

    def complete(self, *_args, **_kwargs):
        if self.error:
            raise self.error
        return self.result


class LLMTestEndpointTests(unittest.TestCase):
    def test_endpoint_returns_actual_provider_and_execution_trace(self):
        service = main.LLMService(
            FakeProvider("openai-gateway", "gateway-test", "Gateway response"),
            FakeProvider("gemini", "gemini-test", "should not run"),
            FakeProvider("ollama-local", "ollama-test", "should not run"),
        )
        with patch.object(main, "LLMService", return_value=service):
            result = main.local_llm_test(main.LocalLLMTestRequest(prompt="Explain RAG."), {"id": 3})
        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "openai-gateway")
        self.assertEqual(result["provider_trace"], [{"provider": "openai-gateway", "status": "success"}])

    def test_endpoint_returns_safe_failure_and_complete_trace(self):
        service = main.LLMService(
            FakeProvider("openai-gateway", "gateway-test", error=RuntimeError("private error")),
            FakeProvider("gemini", "gemini-test", error=RuntimeError("private error")),
            FakeProvider("ollama-local", "ollama-test", error=RuntimeError("private error")),
        )
        with patch.object(main, "LLMService", return_value=service):
            result = main.local_llm_test(main.LocalLLMTestRequest(prompt="Explain RAG."), {"id": 3})
        self.assertEqual(result.status_code, 502)
        body = __import__("json").loads(result.body)
        self.assertEqual(body["error"], "All configured AI providers are currently unavailable.")
        self.assertEqual([row["status"] for row in body["provider_trace"]], ["failed", "failed", "failed"])
        self.assertNotIn("private error", result.body.decode())


if __name__ == "__main__":
    unittest.main()
