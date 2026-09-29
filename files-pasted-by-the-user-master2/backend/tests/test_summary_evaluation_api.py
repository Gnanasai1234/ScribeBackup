from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app import main
from app.auth import get_current_user


class SummaryEvaluationApiTests(unittest.TestCase):
    def setUp(self):
        self.user = {"id": 50111, "name": "Evaluation User", "email": "evaluation@example.test"}
        main.app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(main.app)
        self.addCleanup(main.app.dependency_overrides.clear)

    def test_refresh_reads_persisted_report_and_returns_all_metrics(self):
        report = {"evaluation_status": "completed", "overall_score": 81.5,
                  "metrics": {key: {"score": 80, "status": "completed", "explanation": "source-based"}
                              for key in ("relevance", "completeness", "factual_consistency", "groundedness", "summary_quality", "validation_accuracy")}}
        owned = {"id": "mine", "user_id": self.user["id"], "result": {"evaluation": report}}
        with patch.object(main, "get_doc", return_value=owned), patch.object(main, "get_latest_summary_evaluation", return_value=report) as load:
            response = self.client.get("/api/documents/mine/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["metrics"]), set(report["metrics"]))
        load.assert_not_called()

    def test_persisted_lookup_is_used_when_result_snapshot_has_no_report(self):
        report = {"evaluation_status": "completed", "overall_score": 75,
                  "metrics": {"relevance": {"score": 75}}}
        owned = {"id": "mine", "user_id": self.user["id"], "result": {"executive_summary": "Summary"}}
        with patch.object(main, "get_doc", return_value=owned), patch.object(main, "get_latest_summary_evaluation", return_value=report) as load:
            response = self.client.get("/api/evaluation/mine")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), report)
        load.assert_called_once_with("mine", self.user["id"])

    def test_foreign_document_is_not_retrievable(self):
        foreign = {"id": "theirs", "user_id": 90221, "result": {"evaluation": {"overall_score": 88}}}
        with patch.dict(main.DOCS, {"theirs": foreign}), patch.object(main, "load_document_from_db", return_value=None):
            response = self.client.get("/api/documents/theirs/evaluation")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
