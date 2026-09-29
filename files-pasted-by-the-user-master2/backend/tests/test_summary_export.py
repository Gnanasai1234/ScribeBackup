from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz
from docx import Document
from fastapi.testclient import TestClient

from app import main
from app.auth import get_current_user
from services.summary_export import make_export_payload, render_summary


class SummaryExportTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 41701, "name": "Export User", "email": "export@example.test"}
        main.app.dependency_overrides[get_current_user] = lambda: self.user
        self.history = patch.object(main, "record_history")
        self.history_mock = self.history.start()
        self.addCleanup(self.history.stop)
        self.addCleanup(main.app.dependency_overrides.clear)

    def payload(self):
        return make_export_payload(
            {"name": "quarterly-report.txt", "language": "en"},
            {
                "executive_summary": "Validated export marker. The report confirms steady progress.",
                "language": "en", "depth": "standard", "findings": ["Finding one."],
                "facts": ["Fact one."], "risks": [], "decisions": ["Decision one."],
                "generated_at": "2026-09-29T12:00:00+00:00",
            },
        )

    def test_each_format_contains_final_summary_and_only_present_sections(self):
        payload = self.payload()
        pdf, _, _ = render_summary(payload, "pdf")
        pdf_doc = fitz.open(stream=pdf, filetype="pdf")
        pdf_text = "\n".join(page.get_text() for page in pdf_doc)
        self.assertIn("Validated export marker.", pdf_text)
        self.assertIn("Key Findings", pdf_text)
        self.assertNotIn("Risks and Issues", pdf_text)

        docx, _, _ = render_summary(payload, "docx")
        docx_text = "\n".join(p.text for p in Document(BytesIO(docx)).paragraphs)
        self.assertIn("Validated export marker.", docx_text)
        self.assertIn("Important Facts", docx_text)

        txt, _, _ = render_summary(payload, "txt")
        self.assertIn("Validated export marker.", txt.decode("utf-8"))
        markdown, _, _ = render_summary(payload, "md")
        self.assertIn("## Key Findings", markdown.decode("utf-8"))
        data, _, _ = render_summary(payload, "json")
        structured = json.loads(data)
        self.assertEqual(structured["key_findings"], ["Finding one."])
        self.assertEqual(structured["important_facts"], ["Fact one."])
        self.assertNotIn("risks", structured)

    def test_pdf_paginates_long_summary(self):
        payload = make_export_payload(
            {"name": "long.txt"},
            {"executive_summary": "START_MARKER " + ("Long validated sentence. " * 1200) + "END_MARKER", "language": "en"},
        )
        content, _, _ = render_summary(payload, "pdf")
        document = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)
        self.assertGreater(len(document), 1)
        self.assertIn("START_MARKER", text)
        self.assertIn("END_MARKER", text)

    def test_export_endpoint_uses_owner_scoped_stored_result_and_records_history(self):
        stored = {"id": "owned-document", "name": "quarterly-report.txt", "language": "en",
                  "result": self.payload() | {"executive_summary": "Validated export marker. The report confirms steady progress."}}
        client = TestClient(main.app)
        with patch.object(main, "load_document_from_db", return_value=stored) as load:
            response = client.get("/api/documents/owned-document/export?format=txt")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Validated export marker.", response.text)
        load.assert_called_once_with("owned-document", self.user["id"])
        self.history_mock.assert_called_once()
        self.assertEqual(self.history_mock.call_args.kwargs["activity_type"], "export")
        self.assertEqual(self.history_mock.call_args.kwargs["extra_data"]["export_format"], "txt")

    def test_export_endpoint_hides_foreign_documents_and_rejects_bad_format(self):
        client = TestClient(main.app)
        with patch.object(main, "load_document_from_db", return_value=None):
            self.assertEqual(client.get("/api/documents/foreign/export?format=pdf").status_code, 404)
        with patch.object(main, "load_document_from_db", return_value={"name": "x", "result": self.payload()}):
            self.assertEqual(client.get("/api/documents/owned/export?format=exe").status_code, 422)


if __name__ == "__main__":
    unittest.main()
