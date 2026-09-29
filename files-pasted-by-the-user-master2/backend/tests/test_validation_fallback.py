from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
from docx import Document
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import ProcessOptions, make_chunks
from app.llm_service import LLMService, summarize_document
from services.summary_export import make_export_payload, render_summary


class ValidationFallbackTests(unittest.TestCase):
    def test_english_source_telugu_summary_survives_exhausted_validation(self):
        source = "The agency approved a two-year inspection pilot."
        telugu_summary = "\u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02 \u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c30\u0c4d"
        document = {
            "id": "doc_cross_language_retry_limit",
            "text": source,
            "chunks": make_chunks(source, []),
            "pages": [],
            "language": "en",
        }
        service = MagicMock(spec=LLMService)
        service.provider_name = "openai"
        service.model = "mock"
        validation_calls = 0
        regeneration_count = 0

        def complete_json(system, user, max_tokens=1800, **kwargs):
            nonlocal validation_calls, regeneration_count
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": source, "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                validation_calls += 1
                payload = json.loads(user)
                self.assertEqual(payload["source_language"], "en")
                self.assertEqual(payload["claim_language"], "te")
                return {"results": [{"id": "claim_01", "status": "UNSUPPORTED", "source_chunk_ids": [], "reason": "Evidence did not establish this translation.", "confidence": 0.0}]}
            if "enterprise document intelligence platform" in system:
                if "validation_feedback" in user:
                    regeneration_count += 1
                return {
                    "executive_summary": f"{telugu_summary} version {regeneration_count + 1}",
                    "findings": [{"text": f"{telugu_summary} version {regeneration_count + 1}", "source_chunk_ids": ["chunk_001"]}],
                }
            self.fail(f"Unexpected provider stage: {system[:90]}")

        service.complete_json.side_effect = complete_json
        env = {"SUMMARY_MAX_REGENERATION_ATTEMPTS": "2", "MAX_VALIDATION_RETRIES": "2"}
        with patch.dict("os.environ", env):
            result = summarize_document(document, ProcessOptions(language="te"), service)

        self.assertEqual(validation_calls, 3)
        self.assertEqual(regeneration_count, 2)
        self.assertEqual(result["language"], "te")
        self.assertIn("version 3", result["executive_summary"])
        self.assertEqual(result["validation_state"], "VALIDATION_FAILED_AFTER_RETRIES")
        self.assertTrue(result["validation_warning"])
        self.assertIn("after 3 of 3 validation attempts", result["warning_message"])
        self.assertEqual(result["supported_claims_count"], 0)
        self.assertEqual(result["unsupported_claims_count"], 1)
        self.assertEqual(result["claims"][0]["status"], "UNSUPPORTED")

        exported = make_export_payload({"name": "cross-language.txt"}, result)
        self.assertIn("version 3", exported["summary"])
        self.assertIn("Validation incomplete", exported["validation_warning"])
        markdown, _, _ = render_summary(exported, "md")
        self.assertIn(b"VALIDATION_FAILED_AFTER_RETRIES", markdown)
        self.assertIn(b"Validation incomplete", markdown)
        txt, _, _ = render_summary(exported, "txt")
        self.assertIn("VALIDATION WARNING", txt.decode("utf-8"))
        docx, _, _ = render_summary(exported, "docx")
        self.assertIn("VALIDATION WARNING", "\n".join(p.text for p in Document(BytesIO(docx)).paragraphs))
        pdf, _, _ = render_summary(exported, "pdf")
        pdf_text = "\n".join(page.get_text() for page in fitz.open(stream=pdf, filetype="pdf"))
        self.assertIn("VALIDATION WARNING", pdf_text)


if __name__ == "__main__":
    unittest.main()
