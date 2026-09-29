from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm_service import GeminiProvider, GeminiProviderError, LLMGenerationError, LLMService, OpenAIGatewayProvider, _log_provider_error
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
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class FakeLocal(FakeProvider):
    timeout = 10


class LLMServiceTests(unittest.TestCase):
    def test_gateway_success_skips_gemini_and_ollama(self):
        gateway = FakeProvider("openai-gateway", "gateway-model", "answer")
        gemini = FakeProvider("gemini", "gemini-model", "wrong")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(gateway, gemini, local)
        self.assertEqual(service.complete("system", "question"), "answer")
        self.assertEqual([gateway.calls, gemini.calls, local.calls], [1, 0, 0])
        self.assertEqual(service.provider_metadata["name"], "openai-gateway")
        self.assertEqual(service.provider_trace, [{"provider": "openai-gateway", "status": "success"}])

    def test_gateway_failure_uses_gemini_and_skips_ollama(self):
        gateway = FakeProvider("openai-gateway", "gateway-model", error=TimeoutError("timeout"))
        gemini = FakeProvider("gemini", "gemini-model", "Gemini answer")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(gateway, gemini, local)
        self.assertEqual(service.complete("system", "question"), "Gemini answer")
        self.assertEqual([gateway.calls, gemini.calls, local.calls], [1, 1, 0])
        self.assertEqual(service.provider_metadata["fallback_reason"], "openai_gateway_failed")
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "success"])

    def test_gateway_and_gemini_failure_use_ollama(self):
        gateway = FakeProvider("openai-gateway", "gateway-model", error=RuntimeError("Gateway unavailable"))
        gemini = FakeProvider("gemini", "gemini-model", error=RuntimeError("Gemini unavailable"))
        local = FakeLocal("ollama-local", "llama-test", "local answer")
        service = LLMService(gateway, gemini, local)
        self.assertEqual(service.complete("system", "question"), "local answer")
        self.assertEqual([gateway.calls, gemini.calls, local.calls], [1, 1, 1])
        self.assertEqual(service.provider_metadata["fallback_reason"], "openai_gateway_and_gemini_failed")
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "failed", "success"])

    def test_empty_output_is_failure_and_advances_chain(self):
        gateway = FakeProvider("openai-gateway", "gateway-model", "  ")
        gemini = FakeProvider("gemini", "gemini-model", "usable")
        local = FakeLocal("ollama-local", "llama-test", "wrong")
        service = LLMService(gateway, gemini, local)
        self.assertEqual(service.complete("system", "question"), "usable")
        self.assertEqual([gateway.calls, gemini.calls, local.calls], [1, 1, 0])

    def test_all_providers_fail_without_claiming_any_provider_succeeded(self):
        gateway = FakeProvider("openai-gateway", "gateway-model", error=RuntimeError())
        gemini = FakeProvider("gemini", "gemini-model", error=RuntimeError())
        local = FakeLocal("ollama-local", "llama-test", error=RuntimeError())
        service = LLMService(gateway, gemini, local)
        with self.assertRaises(LLMGenerationError):
            service.complete("system", "question")
        self.assertEqual([gateway.calls, gemini.calls, local.calls], [1, 1, 1])
        self.assertIsNone(service.provider_metadata["name"])
        self.assertIsNone(service.provider_metadata["model"])
        self.assertEqual([row["status"] for row in service.provider_trace], ["failed", "failed", "failed"])

    def test_gateway_sdk_uses_configured_compatible_base_url(self):
        with patch.dict("os.environ", {"OPENAI_GATEWAY_URL": "https://keygateway1.arshnivlabs.com/v1", "OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as sdk:
                provider = OpenAIGatewayProvider()
        sdk.assert_called_once()
        self.assertEqual(sdk.call_args.kwargs["base_url"], "https://keygateway1.arshnivlabs.com/v1")
        self.assertEqual(provider.provider_name, "openai-gateway")

    def test_gemini_sdk_uses_configured_model_key_and_timeout(self):
        fake_client = MagicMock()
        fake_types = SimpleNamespace(HttpOptions=MagicMock(return_value="http-options"))
        fake_genai = SimpleNamespace(Client=MagicMock(return_value=fake_client))
        fake_google = MagicMock(genai=fake_genai)
        fake_genai_module = MagicMock(types=fake_types)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test", "GEMINI_TIMEOUT_SECONDS": "12"}):
            with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai_module}):
                provider = GeminiProvider()
        fake_genai.Client.assert_called_once_with(api_key="test-key", http_options="http-options")
        self.assertEqual(provider.provider_name, "gemini")
        self.assertEqual(provider.model, "gemini-test")
        self.assertEqual(provider.timeout, 12.0)

    def test_gemini_complete_uses_model_and_rejects_empty_sdk_responses(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = SimpleNamespace(text=" GEMINI_TEST_OK ")
        fake_types = SimpleNamespace(
            HttpOptions=MagicMock(return_value="http-options"),
            GenerateContentConfig=MagicMock(return_value="generation-config"),
        )
        fake_genai = SimpleNamespace(Client=MagicMock(return_value=fake_client))
        fake_google = MagicMock(genai=fake_genai)
        fake_genai_module = MagicMock(types=fake_types)
        env = {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"}
        with patch.dict("os.environ", env):
            with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai_module}):
                provider = GeminiProvider()
                result = provider.complete("Return the token", "Reply with exactly: GEMINI_TEST_OK")
                self.assertEqual(result, "GEMINI_TEST_OK")
                self.assertEqual(fake_client.models.generate_content.call_args.kwargs["model"], "gemini-test")
                self.assertIn("GEMINI_TEST_OK", fake_client.models.generate_content.call_args.kwargs["contents"])
                fake_client.models.generate_content.return_value = SimpleNamespace(text="")
                with self.assertRaises(GeminiProviderError) as raised:
                    provider.complete("Return a token", "test")
                self.assertEqual(raised.exception.category, "MalformedResponse")

    def test_gemini_response_parts_and_safety_errors(self):
        from app.llm_service import _gemini_response_text
        candidate = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[SimpleNamespace(text="Part response")]))
        self.assertEqual(_gemini_response_text(SimpleNamespace(text=None, candidates=[candidate])), "Part response")
        blocked = SimpleNamespace(text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[])
        with self.assertRaises(GeminiProviderError) as raised:
            _gemini_response_text(blocked)
        self.assertEqual(raised.exception.category, "SafetyBlock")

    def test_gemini_service_unavailable_log_is_specific_and_sanitized(self):
        error = RuntimeError("transient provider response contained private detail")
        error.code = 503
        provider = FakeProvider("gemini", "gemini-model")
        provider.timeout = 45
        with self.assertLogs("scribe.llm", level="ERROR") as captured:
            _log_provider_error("gemini", provider, error)
        output = "\n".join(captured.output)
        self.assertIn("error_type=ServiceUnavailable", output)
        self.assertIn("status=503", output)
        self.assertNotIn("private detail", output)

    def test_ollama_timeout_is_bounded_and_configurable(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "120"}):
            self.assertEqual(LocalOllamaService("model").timeout, 90.0)
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "12"}):
            self.assertEqual(LocalOllamaService("model").timeout, 12.0)


if __name__ == "__main__":
    unittest.main()
