import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import main
from app.auth import get_current_user


class GuardrailApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.user = {"id": 765433, "name": "Test User", "email": "guardrails@example.com"}
        main.app.dependency_overrides[get_current_user] = lambda: self.user
        self.repo_patches = [
            patch.object(main, "save_document"), patch.object(main, "save_question"),
            patch.object(main, "find_duplicate_document", side_effect=lambda user_id, content_hash, text_hash: any(
                d.get("user_id") == user_id and (d.get("content_sha256") == content_hash or d.get("extracted_text_sha256") == text_hash)
                for d in main.DOCS.values())),
            patch.object(main, "create_upload_batch"), patch.object(main, "update_upload_batch"),
            patch.object(main, "record_history"), patch.object(main, "save_combined_summary", return_value=1),
        ]
        for item in self.repo_patches: item.start()
        self.addCleanup(main.DOCS.clear)
        self.addCleanup(main.app.dependency_overrides.clear)
        for item in self.repo_patches: self.addCleanup(item.stop)

    def remember(self, doc):
        main.DOCS[doc["id"]] = doc

    def test_upload_rejects_traversal_unsupported_and_duplicate_documents(self):
        with patch.object(main, "save_doc", side_effect=self.remember):
            traversal = self.client.post("/api/documents/upload", files={"file": ("../note.txt", b"A readable document with meaningful source text.", "text/plain")})
            unsupported = self.client.post("/api/documents/upload", files={"file": ("note.pdf", b"A readable document with meaningful source text.", "application/pdf")})
            valid = self.client.post("/api/documents/upload", files={"file": ("note.txt", b"A readable document with meaningful source text.", "text/plain")})
            duplicate = self.client.post("/api/documents/upload", files={"file": ("again.txt", b"A readable document with meaningful source text.", "text/plain")})
        self.assertEqual(traversal.status_code, 400)
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)

    def test_batch_upload_keeps_one_bad_document_isolated(self):
        files = [
            ("files", ("good.txt", b"This document contains enough meaningful source text.", "text/plain")),
            ("files", ("bad.pdf", b"not a pdf document at all", "application/pdf")),
            ("files", ("table.csv", b"year,value\n2025,42\n", "text/csv")),
        ]
        with patch.object(main, "save_doc", side_effect=self.remember):
            response = self.client.post("/api/documents/upload-batch", files=files)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["succeeded"], 2)
        self.assertEqual(body["failed"], 1)
        self.assertEqual([item["status"] for item in body["documents"]], ["uploaded", "failed", "uploaded"])
        self.assertEqual(body["documents"][1]["error"], "Document processing failed.")

    def test_batch_processing_is_bounded_and_marks_combined_results(self):
        for document_id in ("doc-a", "doc-b", "doc-c"):
            main.DOCS[document_id] = {"id": document_id, "user_id": self.user["id"], "name": document_id, "text": "Enough meaningful source text for a valid document.", "chunks": [], "pages": [], "language": "en"}
        lock = threading.Lock()
        active = 0
        peak = 0

        def fake_process(doc, options):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return {"executive_summary": f"Summary for {doc['id']}", "claims": [{"claim": doc["id"], "sources": [{"document_id": doc["id"], "chunk_id": "chunk_001"}]}], "validation_status": "passed", "translation_validation_status": "validated"}

        with patch.dict("os.environ", {"MAX_CONCURRENT_DOCUMENTS": "2"}), patch.object(main, "process_document", side_effect=fake_process):
            response = self.client.post("/api/documents/process-batch", json={"document_ids": ["doc-a", "doc-b", "doc-c"]})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["run_type"], "COMBINED_SUMMARY")
        self.assertEqual(body["status"], "completed")
        self.assertLessEqual(peak, 2)
        self.assertEqual(len(body["claims"]), 3)
        self.assertIn("doc-a", body["combined_summary"])


if __name__ == "__main__":
    unittest.main()
