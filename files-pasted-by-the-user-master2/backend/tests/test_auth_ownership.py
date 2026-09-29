from __future__ import annotations

import sys
import unittest
import fitz
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import main
from app.main import app


class AuthOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.users = {}
        self.next_id = 1
        def create_user(name, email, password_hash):
            user = SimpleNamespace(id=self.next_id, name=name, email=email, password_hash=password_hash)
            self.next_id += 1
            self.users[email] = user
            return user
        self.patches = [
            patch("app.auth.create_user", side_effect=create_user),
            patch("app.auth.get_user_by_email", side_effect=lambda email: self.users.get(email)),
            patch("app.auth.get_user_by_id", side_effect=lambda user_id: next((u for u in self.users.values() if u.id == user_id), None)),
            patch.object(main, "load_document_from_db", return_value=None),
            patch.object(main, "save_document"),
            patch.object(main, "find_duplicate_document", side_effect=lambda user_id, content_hash, text_hash: any(
                d.get("user_id") == user_id and (d.get("content_sha256") == content_hash or d.get("extracted_text_sha256") == text_hash)
                for d in main.DOCS.values())),
            patch.object(main, "list_documents_from_db", side_effect=lambda user_id: [d for d in main.DOCS.values() if d.get("user_id") == user_id]),
            patch.object(main, "save_question"),
            patch.object(main, "delete_document_from_db", return_value=True),
            patch.object(main, "record_history"),
            patch.object(main, "list_history_from_db", return_value=[]),
            patch.object(main, "list_history_documents_from_db", return_value={"items": [], "page": 1, "limit": 20, "total": 0, "pages": 0}),
            patch.object(main, "create_upload_batch"),
            patch.object(main, "update_upload_batch"),
            patch.object(main, "save_combined_summary"),
        ]
        for item in self.patches: item.start()
        self.client = TestClient(app)
        self.previous_docs = dict(main.DOCS)

    def tearDown(self):
        main.DOCS.clear()
        main.DOCS.update(self.previous_docs)
        for item in reversed(self.patches): item.stop()

    def register_and_login(self, name: str, email: str):
        registered = self.client.post("/api/auth/register", json={"name": name, "email": email, "password": "secret6"})
        self.assertEqual(registered.status_code, 200)
        self.assertNotIn("password", registered.json())
        self.assertNotIn("password_hash", registered.json())
        login = self.client.post("/api/auth/login", json={"email": email, "password": "secret6"})
        self.assertEqual(login.status_code, 200)
        self.assertIn("scribe_session", self.client.cookies)
        return login.json()

    def test_register_login_me_logout_and_friendly_failures(self):
        anonymous = self.client.get("/api/auth/me")
        self.assertEqual(anonymous.status_code, 401)
        too_short = self.client.post("/api/auth/register", json={"name": "A", "email": "a@example.com", "password": "12345"})
        self.assertEqual(too_short.status_code, 422)
        user = self.register_and_login("Alice", "  ALICE@Example.com ")
        self.assertEqual(user["email"], "alice@example.com")
        self.assertEqual(self.client.get("/api/auth/me").json(), user)
        invalid = self.client.post("/api/auth/login", json={"email": "alice@example.com", "password": "incorrect"})
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(invalid.json()["detail"], "Invalid email or password.")
        duplicate = self.client.post("/api/auth/register", json={"name": "Other", "email": "alice@example.com", "password": "secret6"})
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["detail"], "Email already registered.")
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_documents_are_private_and_owner_is_taken_from_session(self):
        alice = self.register_and_login("Alice", "alice@example.com")
        owner_doc = {"id": "alice-doc", "user_id": alice["id"], "name": "reportA.txt", "text": "Alice document", "chunks": [], "pages": [], "language": "en", "status": "UPLOADED", "created_at": "2026-01-01T00:00:00+00:00", "result": None}
        main.DOCS[owner_doc["id"]] = owner_doc
        uploaded = self.client.post("/api/documents/upload", files={"file": ("alice-upload.txt", b"A personal report with enough words to process.", "text/plain")}, data={"user_id": "999999"})
        self.assertEqual(uploaded.status_code, 200)
        uploaded_id = uploaded.json()["id"]
        self.assertEqual(main.DOCS[uploaded_id]["user_id"], alice["id"])
        self.assertEqual({d["id"] for d in self.client.get("/api/documents").json()}, {"alice-doc", uploaded_id})
        bob = self.register_and_login("Bob", "bob@example.com")
        self.assertEqual(bob["id"], alice["id"] + 1)
        self.assertEqual(self.client.get("/api/documents").json(), [])
        self.assertEqual(self.client.get("/api/history").json()["items"], [])
        self.assertEqual(self.client.get("/api/documents/alice-doc").status_code, 404)
        self.assertEqual(self.client.post("/api/documents/process-batch", json={"document_ids": ["alice-doc"]}).status_code, 404)
        self.assertEqual(self.client.post("/api/documents/alice-doc/retry", json={"summary_depth": "quick"}).status_code, 404)
        self.assertEqual(self.client.delete("/api/documents/alice-doc").status_code, 404)
        for path in ("/api/documents/alice-doc/summary", "/api/documents/alice-doc/sources",
                     "/api/documents/alice-doc/validation", "/api/documents/alice-doc/audio/timings",
                     "/api/documents/alice-doc/export?format=txt"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(self.client.post("/api/documents/alice-doc/process", json={"summary_depth": "quick"}).status_code, 404)
        self.assertEqual(self.client.post("/api/documents/alice-doc/ask", json={"question": "private?"}).status_code, 404)
        self.assertEqual(self.client.post("/api/documents/alice-doc/translate", json={"language": "hi"}).status_code, 404)
        self.assertEqual(self.client.post("/api/documents/alice-doc/audio", json={"language": "en"}).status_code, 404)
        self.assertEqual(self.client.get("/api/history?page=1&limit=20").json()["items"], [])
        main.list_history_documents_from_db.assert_called_with(  # history query uses the account from the session
            bob["id"], page=1, limit=20, search=None, file_type=None, status=None,
            language=None, date_from=None, date_to=None, sort="newest")
        unauthenticated = TestClient(app)
        self.assertEqual(unauthenticated.get("/api/documents").status_code, 401)
        self.assertEqual(unauthenticated.get("/api/documents/alice-doc").status_code, 401)
        self.assertEqual(unauthenticated.get("/api/dataset/govreport").status_code, 401)

    def test_same_content_is_private_duplicate_per_user(self):
        self.register_and_login("Alice", "alice@example.com")
        pdf = fitz.open()
        pdf.new_page().insert_text((72, 72), "A shared PDF report with enough meaningful source text for duplicate checks.")
        pdf_bytes = pdf.tobytes()
        pdf.close()
        payload = {"file": ("report.pdf", pdf_bytes, "application/pdf")}
        alice_first = self.client.post("/api/documents/upload", files=payload)
        alice_duplicate = self.client.post("/api/documents/upload", files=payload)
        self.assertEqual(alice_first.status_code, 200)
        self.assertEqual(alice_duplicate.status_code, 409)
        self.client.post("/api/auth/logout")
        self.register_and_login("Bob", "bob@example.com")
        bob_first = self.client.post("/api/documents/upload", files=payload)
        bob_duplicate = self.client.post("/api/documents/upload", files=payload)
        self.assertEqual(bob_first.status_code, 200)
        self.assertNotEqual(alice_first.json()["id"], bob_first.json()["id"])
        self.assertEqual(bob_duplicate.status_code, 409)

    def test_owner_can_retry_and_delete_their_document(self):
        user = self.register_and_login("Casey", "casey@example.com")
        main.DOCS["casey-doc"] = {"id": "casey-doc", "user_id": user["id"], "name": "casey.txt", "text": "Private text", "chunks": [], "pages": [], "language": "en", "status": "UPLOADED", "created_at": "2026-01-01T00:00:00+00:00", "result": None}
        with patch.object(main, "process_document", return_value={"status": "retried"}):
            retried = self.client.post("/api/documents/casey-doc/retry", json={"summary_depth": "quick"})
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json(), {"status": "retried"})
        self.assertEqual(self.client.delete("/api/documents/casey-doc").status_code, 204)
        self.assertNotIn("casey-doc", main.DOCS)


if __name__ == "__main__":
    unittest.main()
