from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from sqlalchemy.dialects import mysql

from app import main
from app.auth import get_current_user
from database.repositories import search_documents


class DocumentSearchTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 712, "name": "Search User", "email": "search@example.test"}
        main.app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(main.app)
        self.addCleanup(main.app.dependency_overrides.clear)

    def test_search_endpoint_trims_query_and_scopes_to_authenticated_user(self):
        result = [{"id": "owned", "name": "Quarterly Report.pdf", "language": "en", "status": "COMPLETED",
                   "created_at": "2026-01-01T00:00:00+00:00", "text": "report", "chunks": [], "pages": []}]
        with patch.object(main, "search_documents_from_db", return_value=result) as search:
            response = self.client.get("/api/documents?search=%20%20GOV%20%20")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["name"] for item in response.json()], ["Quarterly Report.pdf"])
        search.assert_called_once_with(self.user["id"], "GOV")

    def test_empty_search_uses_existing_list_route_and_search_failure_is_clear(self):
        with patch.object(main, "list_documents_from_db", return_value=[]) as list_docs, \
             patch.object(main, "search_documents_from_db") as search:
            response = self.client.get("/api/documents?search=%20%20")
        self.assertEqual(response.status_code, 200)
        list_docs.assert_called_once_with(self.user["id"])
        search.assert_not_called()

        with patch.object(main, "search_documents_from_db", side_effect=RuntimeError("database details")):
            response = self.client.get("/api/documents?search=report")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Unable to search documents. Please try again.")
        self.assertNotIn("database details", response.text)

    def test_search_query_is_parameterized_and_owner_scoped(self):
        session = MagicMock()
        session.__enter__.return_value.execute.return_value.all.return_value = []
        with patch("database.repositories.SessionLocal", return_value=session):
            self.assertEqual(search_documents(712, "Report"), [])
        statement = session.__enter__.return_value.execute.call_args.args[0]
        compiled = statement.compile(dialect=mysql.dialect())
        self.assertIn("documents.user_id", str(compiled))
        self.assertIn("lower(documents.name)", str(compiled))
        self.assertIn("lower(documents.text)", str(compiled))
        self.assertTrue(any(value == 712 for value in compiled.params.values()))
        self.assertTrue(any(value == "report" for value in compiled.params.values()))

    def test_search_query_length_is_bounded(self):
        with patch.object(main, "search_documents_from_db") as search:
            response = self.client.get("/api/documents", params={"search": "r" * 257})
        self.assertEqual(response.status_code, 422)
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
