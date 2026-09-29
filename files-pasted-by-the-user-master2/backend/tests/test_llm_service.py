from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm_service import LLMGenerationError, LLMService, OpenAIProvider, OpenRouterProvider, _log_provider_error
from services.local_fallbacks import LocalOllamaService


class FakeProvider:
    def __init__(self, name, model, result=None, error=None):
        self.provider_name, self.model = name, model
        self.result, self.error = result, error
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    def complete_json(self, *args, **kwargs):
        return self.complete(*args, **kwargs)


class FakeLocal(FakeProvider):
    timeout = 10


class LLMServiceTests(unittest.TestCase):
    def test_openrouter_success_skips_openai_and_ollama(self):
        router = FakeProvider("openrouter", "or-model", "answer")
        openai_fallback = FakeProvider("openai-direct", "openai-model", "wrong")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(router, openai_fallback, local)
        self.assertEqual(service.complete("system", "question"), "answer")
        self.assertEqual([router.calls, openai_fallback.calls, local.calls], [1, 0, 0])
        self.assertEqual(service.provider_metadata["name"], "openrouter")
        self.assertEqual(service.provider_trace, [{"provider": "openrouter", "status": "success"}])

    def test_openrouter_failure_uses_openai_and_skips_ollama(self):
        router = FakeProvider("openrouter", "or-model", error=TimeoutError("timeout"))
        openai_fallback = FakeProvider("openai-direct", "openai-model", "OpenAI answer")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(router, openai_fallback, local)
        self.assertEqual(service.complete("system", "question"), "OpenAI answer")
        self.assertEqual([router.calls, openai_fallback.calls, local.calls], [1, 1, 0])
        self.assertEqual(service.provider_metadata["name"], "openai-direct")
        self.assertTrue(service.provider_metadata["fallback_used"])
        self.assertEqual(service.provider_metadata["fallback_reason"], "openrouter_failed")
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "success"])

    def test_openrouter_and_openai_failure_use_ollama(self):
        router = FakeProvider("openrouter", "or-model", error=RuntimeError("unavailable"))
        openai_fallback = FakeProvider("openai-direct", "openai-model", error=RuntimeError("unavailable"))
        local = FakeLocal("ollama-local", "llama-test", "local answer")
        service = LLMService(router, openai_fallback, local)
        self.assertEqual(service.complete("system", "question"), "local answer")
        self.assertEqual([router.calls, openai_fallback.calls, local.calls], [1, 1, 1])
        self.assertEqual(service.provider_metadata["name"], "ollama-local")
        self.assertEqual(service.provider_metadata["fallback_reason"], "openrouter_and_openai_failed")
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "failed", "success"])

    def test_empty_output_is_failure_and_advances_chain(self):
        router = FakeProvider("openrouter", "or-model", "  ")
        openai_fallback = FakeProvider("openai-direct", "openai-model", "usable")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(router, openai_fallback, local)
        self.assertEqual(service.complete("system", "question"), "usable")
        self.assertEqual([router.calls, openai_fallback.calls, local.calls], [1, 1, 0])

    def test_all_providers_fail_without_claiming_success(self):
        router = FakeProvider("openrouter", "or-model", error=RuntimeError())
        openai_fallback = FakeProvider("openai-direct", "openai-model", error=RuntimeError())
        local = FakeLocal("ollama-local", "llama-test", error=RuntimeError())
        service = LLMService(router, openai_fallback, local)
        with self.assertRaises(LLMGenerationError):
            service.complete("system", "question")
        self.assertEqual([router.calls, openai_fallback.calls, local.calls], [1, 1, 1])
        self.assertIsNone(service.provider_metadata["name"])
        self.assertIsNone(service.provider_metadata["model"])
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "failed", "failed"])

    def test_openai_provider_uses_native_sdk_endpoint_and_configured_model(self):
        test_key = "sk-" + "a" * 40
        with patch.dict("os.environ", {"OPENAI_API_KEY": test_key, "OPENAI_MODEL": "gpt-test"}):
            with patch("openai.OpenAI") as sdk:
                provider = OpenAIProvider()
        sdk.assert_called_once_with(
            api_key=test_key,
            base_url="https://api.openai.com/v1",
            timeout=30.0,
            max_retries=0,
        )
        self.assertEqual(provider.base_url, "https://api.openai.com/v1")
        self.assertEqual(provider.model, "gpt-test")
        self.assertEqual(provider.provider_name, "openai-direct")

    def test_placeholder_key_is_rejected_without_constructing_client(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "placeholder-key", "OPENAI_BASE_URL": "https://proxy.invalid/v1"}):
            with patch("openai.OpenAI") as sdk:
                provider = OpenAIProvider()
        sdk.assert_not_called()
        self.assertIsNone(provider.client)
        self.assertEqual(provider.base_url, "https://api.openai.com/v1")
        self.assertIn("placeholder/test", provider.config_error)

    def test_openrouter_uses_openrouter_endpoint_and_model(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"}):
            with patch("openai.OpenAI") as sdk:
                provider = OpenRouterProvider()
        sdk.assert_called_once()
        self.assertEqual(sdk.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(provider.model, "test/model")
        self.assertEqual(provider.provider_name, "openrouter")

    def test_real_provider_boundary_logs_request_sizes_response_and_success_without_prompt(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="TEST_OK"))],
            _response=SimpleNamespace(status_code=200),
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch("openai.OpenAI") as sdk:
            sdk.return_value = fake_client
            provider = OpenRouterProvider()
        openai_fallback = FakeProvider("openai-direct", "openai-model", "unused")
        service = LLMService(provider, openai_fallback, FakeLocal("ollama-local", "llama-test", "unused"))
        with self.assertLogs("scribe.llm", level="INFO") as captured:
            self.assertEqual(service.complete("safe instructions", "CONFIDENTIAL_DOCUMENT_CONTENT", max_tokens=8), "TEST_OK")
        output = "\n".join(captured.output)
        self.assertIn("[LLM REQUEST] provider=openrouter", output)
        self.assertIn("input_characters=", output)
        self.assertIn("prompt_characters=", output)
        self.assertIn("estimated_tokens=", output)
        self.assertIn("[LLM RESPONSE] provider=openrouter status=200", output)
        self.assertIn("usable=true", output)
        self.assertIn("[LLM SUCCESS] provider=openrouter", output)
        self.assertIn("elapsed=", output)
        self.assertNotIn("CONFIDENTIAL_DOCUMENT_CONTENT", output)

    def test_empty_provider_response_is_logged_and_falls_back(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            _response=SimpleNamespace(status_code=200),
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), patch("openai.OpenAI") as sdk:
            sdk.return_value = fake_client
            provider = OpenRouterProvider()
        openai_fallback = FakeProvider("openai-direct", "openai-model", "OpenAI answer")
        service = LLMService(provider, openai_fallback, FakeLocal("ollama-local", "llama-test", "unused"))
        with self.assertLogs("scribe.llm", level="INFO") as captured:
            self.assertEqual(service.complete("system", "question"), "OpenAI answer")
        output = "\n".join(captured.output)
        self.assertIn("error_type=EmptyResponseError", output)
        self.assertIn("[LLM FALLBACK] from=openrouter to=openai-direct", output)
        self.assertEqual(service.provider_metadata["name"], "openai-direct")

    def test_error_log_is_categorized_and_sanitized(self):
        error = RuntimeError("provider rejected credential sk-test-secret-value-12345")
        error.code = 503
        provider = FakeProvider("openrouter", "test-model")
        provider.timeout = 30
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-secret-value-12345"}):
            with self.assertLogs("scribe.llm", level="ERROR") as captured:
                _log_provider_error("openrouter", provider, error)
        output = "\n".join(captured.output)
        self.assertIn("error_type=ServiceUnavailable", output)
        self.assertIn("status=503", output)
        self.assertNotIn("sk-test-secret-value-12345", output)
        self.assertIn("[REDACTED]", output)

    def test_ollama_timeout_is_bounded_and_configurable(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "120"}):
            self.assertEqual(LocalOllamaService("model").timeout, 90.0)
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "12"}):
            self.assertEqual(LocalOllamaService("model").timeout, 12.0)


if __name__ == "__main__":
    unittest.main()

