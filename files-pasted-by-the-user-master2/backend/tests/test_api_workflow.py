from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import main
from app.auth import get_current_user
from app.main import DOCS, app


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 765432, "name": "Test User", "email": "test@example.com"}
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.patches = [
            patch.object(main, "save_document"), patch.object(main, "save_question"),
            patch.object(main, "find_duplicate_document", return_value=False),
            patch.object(main, "record_history"), patch.object(main, "save_combined_summary", return_value=1),
            patch.object(main, "get_local_llm_service", return_value=None),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        app.dependency_overrides.clear()
        for item in self.patches:
            item.stop()

    def test_process_persists_actual_openai_gateway_provider_metadata(self):
        text = "The board approved a two-year safety pilot. Quarterly inspections will track progress."
        doc = {"id": "openai-summary", "user_id": self.user["id"], "name": "summary.txt", "text": text,
               "pages": [], "chunks": main.make_chunks(text, []), "language": "en", "status": "UPLOADED", "result": None}
        service = type("Service", (), {
            "provider_metadata": {"name": "openai-gateway", "model": "gpt-4o-mini", "fallback_used": False},
            "provider_name": "openai-gateway", "model": "gpt-4o-mini",
        })()
        saved = lambda item: DOCS.__setitem__(item["id"], item)
        with patch.object(main, "LLMService", return_value=service), \
             patch.object(main, "summarize_document", return_value={"executive_summary": text, "claims": [], "status": "partial"}), \
             patch.object(main, "save_doc", side_effect=saved):
            result = main.process_document(doc, main.ProcessOptions(summary_depth="quick", language="en"))
        self.assertEqual(result["provider"], {"name": "openai-gateway", "model": "gpt-4o-mini", "fallback_used": False})
        DOCS.pop(doc["id"], None)

    def test_all_generation_providers_failed_uses_labeled_extractive_summary(self):
        text = "The board approved a two-year safety pilot. Quarterly inspections will track progress."
        doc = {"id": "all-providers-failed", "user_id": self.user["id"], "name": "summary.txt", "text": text,
               "pages": [], "chunks": main.make_chunks(text, []), "language": "en", "status": "UPLOADED", "result": None}
        service = type("Service", (), {
            "provider_metadata": {"name": None, "model": None, "fallback_used": True, "fallback_reason": "all providers failed"},
            "provider_name": "openai-gateway", "model": "gpt-4o-mini",
        })()
        saved = lambda item: DOCS.__setitem__(item["id"], item)
        with patch.object(main, "LLMService", return_value=service), \
             patch.object(main, "summarize_document", side_effect=main.LLMGenerationError("unavailable")), \
             patch.object(main, "save_doc", side_effect=saved):
            result = main.process_document(doc, main.ProcessOptions(summary_depth="quick", language="en"))
        self.assertTrue(result["executive_summary"])
        self.assertEqual(result["provider"]["name"], "deterministic-extractive")
        self.assertTrue(result["provider"]["fallback_used"])
        self.assertTrue(result["provider"]["fallback_used"])
        self.assertEqual(result["generation_status"], "partial_success")
        DOCS.pop(doc["id"], None)

    def test_rag_returns_controlled_failure_when_both_generation_providers_fail(self):
        doc_id = "rag-generation-failed"
        text = "The committee approved quarterly safety inspections at every site."
        DOCS[doc_id] = {"id": doc_id, "user_id": self.user["id"], "name": "safety.txt", "text": text,
                        "pages": [], "chunks": main.make_chunks(text, []), "language": "en"}
        embedding = type("Embedding", (), {"last_status": {"effective_provider": "lexical"},
            "get_embeddings": lambda self, texts: [], "get_embedding": lambda self, text: None})()
        with (
            patch.object(main, "get_embedding_service", return_value=embedding),
            patch.object(main, "answer_from_evidence", side_effect=main.LLMGenerationError("unavailable")),
        ):
            response = TestClient(app).post(f"/api/documents/{doc_id}/ask", json={"question": "What safety inspections were approved?"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("committee approved", response.text.lower())
        DOCS.pop(doc_id, None)


if __name__ == "__main__":
    unittest.main()
