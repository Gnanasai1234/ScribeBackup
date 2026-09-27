import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.local_fallbacks import basic_local_claim_check, installed_ollama_models
from services.embedding_service import EmbeddingService


class LocalFallbackTests(unittest.TestCase):
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
        with patch("services.local_fallbacks.shutil.which", return_value="ollama.exe"), \
             patch("services.local_fallbacks.subprocess.run") as run:
            run.return_value.stdout = "NAME ID SIZE\nllama3.2:1b abc 1GB\nnomic-embed-text:latest def 300MB\n"
            run.return_value.check_returncode = lambda: None
            models = installed_ollama_models(refresh=True)
        self.assertEqual(models, {"llama3.2:1b", "nomic-embed-text:latest"})
        self.assertEqual(run.call_args.args[0], ["ollama.exe", "list"])


if __name__ == "__main__":
    unittest.main()
