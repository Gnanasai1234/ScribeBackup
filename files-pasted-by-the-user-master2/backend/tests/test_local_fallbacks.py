import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import services.local_fallbacks as local_fallbacks
from services.local_fallbacks import LocalOllamaService, basic_local_claim_check, get_local_llm_service, installed_ollama_models
from services.embedding_service import EmbeddingService


class LocalFallbackTests(unittest.TestCase):
    def setUp(self):
        local_fallbacks._OLLAMA_MODELS = None
        local_fallbacks._OLLAMA_READY = None
        local_fallbacks._OLLAMA_FAILED_UNTIL = 0.0
        local_fallbacks._LOCAL_LLM_SERVICE = None

    class HttpResponse:
        status = 200
        def __init__(self, body):
            self.body = json.dumps(body).encode()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return self.body

    def test_basic_check_supports_exact_source_claim(self):
        result = basic_local_claim_check(
            "Revenue increased by 12%.", ["Revenue increased by 12%."],
            claim_language="en", source_language="en", cited_ids_valid=True,
        )
        self.assertEqual(result["status"], "SUPPORTED")

    def test_basic_check_rejects_unsupported_number(self):
        result = basic_local_claim_check(
            "Revenue increased by 18%.", ["Revenue increased by 12%."],
            claim_language="en", source_language="en", cited_ids_valid=True,
        )
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_basic_check_does_not_claim_cross_language_semantics(self):
        result = basic_local_claim_check(
            "राजस्व में वृद्धि हुई।", ["Revenue increased."],
            claim_language="hi", source_language="en", cited_ids_valid=True,
        )
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_embeddings_degrade_to_empty_vectors_for_lexical_search(self):
        service = EmbeddingService()
        service.provider = "local"
        with patch("services.embedding_service.local_ollama_embeddings", return_value=None), \
             patch("services.embedding_service._get_local_model", return_value=None):
            vectors = service.get_embeddings(["sample source text"])
        self.assertEqual(vectors, [[]])
        self.assertEqual(service.last_status["effective_provider"], "lexical")

    def test_ollama_discovery_only_lists_installed_models(self):
        with patch("services.local_fallbacks._OLLAMA_FAILED_UNTIL", 0), \
             patch("services.local_fallbacks.urllib.request.urlopen", return_value=self.HttpResponse({"models": [{"name": "llama3.2:1b"}, {"name": "nomic-embed-text:latest"}]})) as urlopen:
            models = installed_ollama_models(refresh=True)
        self.assertEqual(models, {"llama3.2:1b", "nomic-embed-text:latest"})
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://127.0.0.1:11434/api/tags")

    def test_generation_uses_configured_generate_endpoint_and_timeout(self):
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT_SECONDS": "120", "OLLAMA_BASE_URL": "http://127.0.0.1:11434"}), \
             patch("services.local_fallbacks.urllib.request.urlopen", return_value=self.HttpResponse({"response": "OK", "done": True})) as urlopen:
            service = LocalOllamaService("llama3.2:1b")
            self.assertEqual(service.complete("system", "Say OK"), "OK")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(payload["model"], "llama3.2:1b")
        self.assertIn("Say OK", payload["prompt"])
        self.assertIs(payload["stream"], False)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 90.0)

    def test_generation_bounds_large_pipeline_output_limit(self):
        with patch.dict("os.environ", {"OLLAMA_MAX_OUTPUT_TOKENS": "300"}), \
             patch("services.local_fallbacks.urllib.request.urlopen", return_value=self.HttpResponse({"response": "OK", "done": True})) as urlopen:
            LocalOllamaService("llama3.2:1b").complete("system", "prompt", max_tokens=3000)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["options"]["num_predict"], 300)

    def test_complete_json_sends_inferred_required_schema(self):
        service = LocalOllamaService("llama3.2:1b")
        schema = {"executive_summary": "one paragraph", "findings": [{"text": "fact", "source_chunk_ids": ["c1"]}]}
        response = self.HttpResponse({"response": '{"executive_summary":"Summary.","findings":[]}', "done": True})
        with patch("services.local_fallbacks.urllib.request.urlopen", return_value=response) as urlopen:
            service.complete_json("Return the requested shape", json.dumps({"required_shape": schema}))
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["format"]["required"], ["executive_summary", "findings"])
        self.assertEqual(payload["format"]["properties"]["findings"]["items"]["required"], ["text", "source_chunk_ids"])

    def test_configured_model_must_be_listed_by_health_endpoint(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "llama3.2:1b", "OLLAMA_BASE_URL": "http://127.0.0.1:11434"}), \
             patch("services.local_fallbacks.urllib.request.urlopen", return_value=self.HttpResponse({"models": [{"name": "llama3.2:1b"}, {"name": "nomic-embed-text:latest"}]})):
            service = get_local_llm_service()
        self.assertEqual(service.model, "llama3.2:1b")

    def test_json_response_repairs_once_and_extracts_fenced_json(self):
        service = LocalOllamaService("llama3.2:1b")
        service.complete = Mock(side_effect=["not json", '```json\n{"notes": [{"text":"Fact.","source_chunk_ids":["c1"]}]}\n```'])
        self.assertEqual(service.complete_json('Return JSON as {"notes":[{"text":"...","source_chunk_ids":[]}]}', "{}"), {"notes": [{"text": "Fact.", "source_chunk_ids": ["c1"]}]})
        self.assertEqual(service.complete.call_count, 2)
        self.assertTrue(service.complete.call_args.kwargs["json_mode"])

    def test_empty_notes_are_retried_and_not_accepted_as_complete(self):
        service = LocalOllamaService("llama3.2:1b")
        service.complete = Mock(side_effect=['{"notes": []}', '{"notes": [{"text":"Fact.","source_chunk_ids":["c1"]}]}'])
        result = service.complete_json('Return JSON as {"notes":[{"text":"...","source_chunk_ids":[]}]}', "{}")
        self.assertEqual(len(result["notes"]), 1)
        self.assertEqual(service.complete.call_count, 2)

    def test_empty_generation_response_fails_clearly(self):
        service = LocalOllamaService("llama3.2:1b")
        with patch("services.local_fallbacks.urllib.request.urlopen", return_value=self.HttpResponse({"response": "", "done": True})):
            with self.assertRaisesRegex(ValueError, "empty response"):
                service.complete("system", "prompt")

    def test_generation_timeout_is_reported_as_provider_error(self):
        service = LocalOllamaService("llama3.2:1b")
        with patch("services.local_fallbacks.urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                service.complete("system", "prompt")

    def test_unavailable_ollama_returns_no_local_service(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "llama3.2:1b"}), \
             patch("services.local_fallbacks._OLLAMA_FAILED_UNTIL", 0), \
             patch("services.local_fallbacks.urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertIsNone(get_local_llm_service())


if __name__ == "__main__":
    unittest.main()
