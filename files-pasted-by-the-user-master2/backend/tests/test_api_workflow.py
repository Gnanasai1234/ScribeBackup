from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import DOCS, app
from app.auth import get_current_user
from app import main


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 765432, "name": "Test User", "email": "test@example.com"}
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.repo_patches = [
            patch.object(main, "save_document"), patch.object(main, "save_question"),
            patch.object(main, "find_duplicate_document", return_value=False),
            patch.object(main, "record_history"), patch.object(main, "save_combined_summary", return_value=1),
            patch.object(main, "get_local_llm_service", return_value=None),
            patch.object(main, "_get_fallback_llm_service", return_value=None),
        ]
        for item in self.repo_patches: item.start()

    def tearDown(self):
        app.dependency_overrides.clear()
        for item in self.repo_patches: item.stop()

    def test_primary_empty_response_uses_deterministic_summary_without_crashing(self):
        client = TestClient(app)
        text = "Quarterly inspections are required. The agency approved a two-year safety pilot."
        keep_in_memory = lambda doc: DOCS.__setitem__(doc["id"], doc)
        primary = type("Primary", (), {"provider_name": "sarvam", "model": "configured-model"})()
        with patch("app.main.OpenRouterService", return_value=primary), \
             patch("app.main.summarize_with_openrouter", side_effect=ValueError("sarvam returned an empty message")), \
             patch("app.main.get_local_llm_service", return_value=None), \
             patch("app.main._get_fallback_llm_service", return_value=None), \
             patch("app.main.save_doc", side_effect=keep_in_memory):
            uploaded = client.post("/api/documents/upload", files={"file": ("fallback.txt", text, "text/plain")})
            self.assertEqual(uploaded.status_code, 200)
            document_id = uploaded.json()["id"]
            processed = client.post(f"/api/documents/{document_id}/process", json={"summary_depth": "quick", "language": "en"})
        self.assertEqual(processed.status_code, 200)
        result = processed.json()
        self.assertTrue(result["executive_summary"])
        self.assertEqual(result["provider"]["name"], "local extractive fallback")
        self.assertEqual(result["provider_status"]["primary"], "FAILED (ValueError)")
        self.assertEqual(result["provider_status"]["final"], "DETERMINISTIC EXTRACTIVE SUMMARY")
        self.assertIn("The agency approved a two-year safety pilot.", result["executive_summary"])
        DOCS.pop(document_id, None)

    def test_auto_and_failed_translation_preserve_detected_telugu(self):
        client = TestClient(app)
        telugu_text = "తెలుగు భాషలో ఈ పత్రం వివరమైన విశ్లేషణను అందిస్తుంది. సంస్థ కొత్త ప్రణాళికను ఆమోదించి అమలు ప్రారంభించింది."
        keep_in_memory = lambda doc: DOCS.__setitem__(doc["id"], doc)
        with patch("app.main.OpenRouterService", side_effect=RuntimeError("No provider configured")), patch("app.main.save_doc", side_effect=keep_in_memory):
            uploaded = client.post("/api/documents/upload", files={"file": ("telugu.txt", telugu_text.encode("utf-8"), "text/plain")})
            self.assertEqual(uploaded.status_code, 200)
            document_id = uploaded.json()["id"]
            self.assertEqual(uploaded.json()["language"], "te")

            auto = client.post(
                f"/api/documents/{document_id}/process",
                json={"summary_depth": "quick", "source_language": "te", "output_language": "auto"},
            )
            self.assertEqual(auto.status_code, 200)
            self.assertEqual(auto.json()["source_language"], "te")
            self.assertEqual(auto.json()["language"], "te")
            self.assertTrue(auto.json()["executive_summary"])
            self.assertNotIn("translation is currently unavailable", auto.json()["executive_summary"].lower())

            english = client.post(
                f"/api/documents/{document_id}/process",
                json={"summary_depth": "quick", "source_language": "te", "output_language": "en"},
            )
            self.assertEqual(english.status_code, 200)
            self.assertEqual(english.json()["source_language"], "te")
            self.assertEqual(english.json()["requested_language"], "en")
            self.assertEqual(english.json()["language"], "te")
            self.assertTrue(english.json()["executive_summary"])
            DOCS.pop(document_id, None)

    def test_process_rag_validation_tts_and_reference_evaluation(self):
        client = TestClient(app)
        text = "The report recommends quarterly safety inspections at every site. The agency approved a two-year inspection pilot."
        with patch("app.main.OpenRouterService", side_effect=RuntimeError("No provider configured")):
            uploaded = client.post("/api/documents/upload", files={"file": ("report.txt", text, "text/plain")})
            self.assertEqual(uploaded.status_code, 200)
            document_id = uploaded.json()["id"]
            analyzed = client.post(f"/api/documents/{document_id}/process", json={"summary_depth": "deep", "purpose": "Risk analysis", "audience": "Leadership", "language": "en"})
            self.assertEqual(analyzed.status_code, 200)
            result = analyzed.json()
            self.assertEqual(result["summary_mode"], "deep")
            self.assertEqual(result["purpose"], "Risk analysis")
            self.assertEqual(result["audience"], "Leadership")
            self.assertEqual(result["validation"]["source_of_truth"], "original_document_chunks")
            self.assertIn("quality_evaluation", result["workflow"])

            DOCS[document_id]["reference_summary"] = "Quarterly inspections are recommended at all sites."
            evaluation = client.post("/api/evaluation/run", json={"document_id": document_id})
            self.assertEqual(evaluation.status_code, 200)
            self.assertEqual(evaluation.json()["reference_role"], "evaluation_only")
            self.assertIsNotNone(evaluation.json()["metrics"]["rougeL"]["f1"])

            answer = client.post(f"/api/documents/{document_id}/ask", json={"question": "What inspections are recommended?"})
            self.assertEqual(answer.status_code, 200)
            self.assertTrue(answer.json()["sources"])
            self.assertIn("chunk_id", answer.json()["sources"][0])

            audio = client.post(f"/api/documents/{document_id}/audio", json={"language": "en"})
            self.assertEqual(audio.status_code, 200)
            self.assertEqual(audio.json()["source"], "validated_summary")
            audio_auto = client.post(f"/api/documents/{document_id}/audio", json={})
            self.assertEqual(audio_auto.status_code, 200)
            self.assertEqual(audio_auto.json()["language"], "en")
            blocked_audio = client.post(f"/api/documents/{document_id}/audio", json={"language": "te"})
            self.assertIn(blocked_audio.status_code, [422, 503])


if __name__ == "__main__":
    unittest.main()
