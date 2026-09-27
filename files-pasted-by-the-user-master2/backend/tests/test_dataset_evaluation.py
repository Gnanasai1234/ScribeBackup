from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dataset.govreport_loader import get_record, stream_records
from evaluation.completeness import reference_coverage
from evaluation.evaluator import evaluate_summary
from evaluation.rouge import calculate_rouge


class FakeStream:
    def __init__(self, rows):
        self.rows = rows

    def take(self, limit):
        return iter(self.rows[:limit])


class DatasetEvaluationTests(unittest.TestCase):
    def test_stream_loader_uses_split_streaming_and_limit(self):
        rows = [{"id": str(i), "report": f"Report {i}", "summary": f"Summary {i}"} for i in range(4)]
        load_dataset = unittest.mock.Mock(return_value=FakeStream(rows))
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load_dataset)}):
            selected = stream_records(split="validation", limit=2)
            found = get_record("2", split="test", limit=3)
        self.assertEqual([row["id"] for row in selected], ["0", "1"])
        self.assertEqual(found["report"], "Report 2")
        self.assertEqual(load_dataset.call_args_list[0].kwargs, {"split": "validation", "streaming": True})

    def test_stream_loader_derives_stable_id_when_hub_export_omits_id(self):
        row = {"report": "Stable original report text.", "summary": "Reference only."}
        with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=lambda *_args, **_kwargs: FakeStream([row]))}):
            selected = stream_records(split="train", limit=1)
        self.assertTrue(selected[0]["id"].startswith("content-"))
        self.assertEqual(selected[0]["id_source"], "content_hash_fallback")

    def test_rouge_and_reference_evaluator_report_individual_metrics(self):
        metrics = calculate_rouge("the report found a risk", "the report found a risk")
        self.assertEqual(metrics["rouge1"]["f1"], 1.0)
        self.assertEqual(metrics["rouge2"]["f1"], 1.0)
        self.assertEqual(metrics["rougeL"]["f1"], 1.0)
        result = {"executive_summary": "the report found a risk", "claims": [{"status": "SUPPORTED", "sources": [{"chunk_id": "c1"}]}], "quality": {"claims_checked": 1}, "depth": "standard"}
        evaluation = evaluate_summary(result, "the report found a risk")
        self.assertEqual(evaluation["metrics"]["groundedness"]["value"], 1.0)
        self.assertEqual(evaluation["metrics"]["citationCoverage"]["value"], 1.0)
        self.assertIn(evaluation["metrics"]["semanticSimilarity"]["value"], [1.0, None])
        self.assertEqual(reference_coverage(result["executive_summary"], "the report found a risk")["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
