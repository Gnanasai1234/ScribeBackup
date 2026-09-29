from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main


class FakeProvider:
    def __init__(self, name: str, model: str, result: str = "", error: Exception | None = None):
        self.provider_name = name
        self.model = model
        self.result = result
        self.error = error
        self.calls = []

    def complete(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result


class LLMTestEndpointTests(unittest.TestCase):
    def test_endpoint_returns_actual_provider_and_execution_trace(self):
        service = main.LLMService(
            FakeProvider("openrouter", "openrouter-test", "OpenRouter response"),
            FakeProvider("openai-direct", "openai-test", "should not run"),
            FakeProvider("ollama-local", "ollama-test", "should not run"),
        )
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=42) as history_mock:
            result = main.local_llm_test(main.LocalLLMTestRequest(prompt="Explain RAG."), {"id": 3})
        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["provider_trace"], [{"provider": "openrouter", "status": "success"}])
        self.assertEqual(result["history_id"], 42)
        self.assertTrue(result["history_persisted"])
        saved = history_mock.call_args.kwargs
        self.assertEqual(saved["user_id"], 3)
        self.assertEqual(saved["activity_type"], "llm_test")
        self.assertEqual(saved["input_text"], "Explain RAG.")
        self.assertEqual(saved["output_text"], "OpenRouter response")

    def test_endpoint_returns_safe_failure_and_complete_trace(self):
        service = main.LLMService(
            FakeProvider("openrouter", "openrouter-test", error=RuntimeError("private error")),
            FakeProvider("openai-direct", "openai-test", error=RuntimeError("private error")),
            FakeProvider("ollama-local", "ollama-test", error=RuntimeError("private error")),
        )
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=42) as history_mock:
            result = main.local_llm_test(main.LocalLLMTestRequest(prompt="Explain RAG."), {"id": 3})
        self.assertEqual(result.status_code, 502)
        body = __import__("json").loads(result.body)
        self.assertEqual(body["error"], "All configured AI providers are currently unavailable.")
        self.assertEqual([row["status"] for row in body["provider_trace"]], ["failed", "failed", "failed"])
        self.assertNotIn("private error", result.body.decode())

    def test_generation_settings_change_backend_prompt_and_are_returned(self):
        openrouter_provider = FakeProvider("openrouter", "openrouter-test", "Study response")
        service = main.LLMService(openrouter_provider, FakeProvider("openai-direct", "openai-test"), FakeProvider("ollama-local", "ollama-test"))
        payload = main.LocalLLMTestRequest(prompt="Explain photosynthesis.", purpose="study", audience="student", depth="deep")
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=42) as history_mock:
            result = main.local_llm_test(payload, {"id": 3})

        system_prompt, user_prompt = openrouter_provider.calls[0][0][:2]
        self.assertIn("concepts, definitions, important points", system_prompt)
        self.assertIn("Audience (student)", system_prompt)
        self.assertIn("avoid unnecessary jargon", system_prompt)
        self.assertIn("detailed response covering important context", system_prompt)
        self.assertEqual(user_prompt, "Explain photosynthesis.")
        self.assertEqual(result["generation_settings"], {"purpose": "study", "audience": "student", "depth": "deep"})

    def test_invalid_generation_settings_are_rejected(self):
        for field, value in (("purpose", "unknown"), ("audience", "unknown"), ("depth", "long")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                main.LocalLLMTestRequest(prompt="Test", **{field: value})

    def test_same_generation_settings_reach_openai_fallback_after_openrouter_failure(self):
        openrouter_provider = FakeProvider("openrouter", "openrouter-test", error=RuntimeError("unavailable"))
        openai_fallback = FakeProvider("openai-direct", "openai-test", "Research response")
        service = main.LLMService(openrouter_provider, openai_fallback, FakeProvider("ollama-local", "ollama-test"))
        payload = main.LocalLLMTestRequest(prompt="Review the study.", purpose="research", audience="researcher", depth="standard")
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=42) as history_mock:
            result = main.local_llm_test(payload, {"id": 3})

        self.assertEqual(result["provider"], "openai-direct")
        self.assertEqual(openrouter_provider.calls[0][0], openai_fallback.calls[0][0])
        self.assertEqual(openrouter_provider.calls[0][1], openai_fallback.calls[0][1])
        self.assertEqual(openai_fallback.calls[0][0][1], "Review the study.")

    def test_quick_standard_and_deep_settings_change_prompt_guidance(self):
        cases = [
            main.LocalLLMTestRequest(prompt="Summarize.", purpose="general", audience="general", depth="quick"),
            main.LocalLLMTestRequest(prompt="Summarize.", purpose="study", audience="student", depth="standard"),
            main.LocalLLMTestRequest(prompt="Summarize.", purpose="technical", audience="developer_technical", depth="deep"),
        ]
        prompts = [main._build_llm_test_system_prompt(payload) for payload in cases]
        self.assertIn("short overview", prompts[0])
        self.assertIn("concepts, definitions", prompts[1])
        self.assertIn("avoid unnecessary jargon", prompts[1])
        self.assertIn("architecture, implementation details", prompts[2])
        self.assertIn("detailed response covering important context", prompts[2])
        self.assertEqual(len(set(prompts)), 3)

    def test_ollama_fallback_receives_the_same_logical_request(self):
        openrouter_provider = FakeProvider("openrouter", "openrouter-test", error=RuntimeError("unavailable"))
        openai_fallback = FakeProvider("openai-direct", "openai-test", error=RuntimeError("unavailable"))
        ollama = FakeProvider("ollama-local", "ollama-test", "Local response")
        service = main.LLMService(openrouter_provider, openai_fallback, ollama)
        payload = main.LocalLLMTestRequest(prompt="Explain the architecture.", purpose="technical", audience="expert", depth="deep")
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=42) as history_mock:
            result = main.local_llm_test(payload, {"id": 3})

        self.assertEqual(result["provider"], "ollama-local")
        self.assertEqual(openrouter_provider.calls[0][0], openai_fallback.calls[0][0])
        self.assertEqual(openai_fallback.calls[0][0], ollama.calls[0][0])
        self.assertEqual(openrouter_provider.calls[0][1], openai_fallback.calls[0][1])
        self.assertEqual(openai_fallback.calls[0][1], ollama.calls[0][1])

    def test_translation_uses_existing_provider_chain_and_records_language(self):
        translated_text = "\u0c07\u0c26\u0c3f \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c35\u0c3e\u0c15\u0c4d\u0c2f\u0c02."
        openrouter_provider = FakeProvider("openrouter", "openrouter-test", translated_text)
        service = main.LLMService(openrouter_provider, FakeProvider("openai-direct", "openai-test"), FakeProvider("ollama-local", "ollama-test"))
        payload = main.LocalLLMTestRequest(prompt="This is a Telugu sentence.", purpose="study", audience="student", depth="deep", translate_to="te")
        with patch.object(main, "LLMService", return_value=service), patch.object(main, "record_history", return_value=43) as history_mock:
            result = main.local_llm_test(payload, {"id": 3})

        self.assertEqual(result["response"], translated_text)
        self.assertEqual(result["translation_language"], "te")
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(history_mock.call_args.kwargs["extra_data"]["translation_language"], "te")


if __name__ == "__main__":
    unittest.main()


