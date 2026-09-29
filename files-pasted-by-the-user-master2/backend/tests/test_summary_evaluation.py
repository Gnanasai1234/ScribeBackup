from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.summary_evaluation import evaluate_summary
from services.summary_export import make_export_payload, render_summary


class SummaryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            "Revenue increased by 25 percent in 2025 after international expansion. "
            "The company approved a two-year factory investment to reduce delivery delays. "
            "Executives identified supply costs as the main risk and recommended supplier diversification. "
            "The report expects hiring to continue through 2026."
        )
        self.chunks = [{"id": "chunk-1", "documentId": "doc-1", "page": 12,
                        "section": "Performance", "text": self.source}]

    def test_returns_distinct_calculated_metrics_and_normalized_overall(self):
        claims = [{"claim": "Revenue increased by 25 percent in 2025.", "status": "SUPPORTED",
                   "confidence": 0.9, "evidence": "Revenue increased by 25 percent in 2025 after international expansion.",
                   "sources": [{"document_id": "doc-1", "page": 12, "section": "Performance", "chunk_id": "chunk-1"}]},
                  {"claim": "Revenue doubled in 2025.", "status": "UNSUPPORTED", "confidence": 0.2,
                   "evidence": "Revenue increased by 25 percent in 2025 after international expansion.",
                   "sources": [{"document_id": "doc-1", "page": 12, "section": "Performance", "chunk_id": "chunk-1"}]}]
        result = evaluate_summary(self.source,
            "Revenue increased by 25 percent in 2025. The company approved a factory investment and identified supply costs as a risk.",
            claims=claims, chunks=self.chunks, source_language="en", summary_language="en", depth="standard")
        self.assertEqual(set(result["metrics"]), {"relevance", "completeness", "factual_consistency", "groundedness", "summary_quality", "validation_accuracy"})
        scores = [metric["score"] for metric in result["metrics"].values() if metric["score"] is not None]
        self.assertTrue(all(0 <= score <= 100 for score in scores))
        self.assertNotEqual(len(set(scores)), 1)
        weighted = sum(result["metrics"][name]["score"] * weight for name, weight in result["overall_formula"]["weights"].items()
                       if result["metrics"][name]["score"] is not None)
        denominator = sum(weight for name, weight in result["overall_formula"]["weights"].items()
                          if result["metrics"][name]["score"] is not None)
        self.assertAlmostEqual(result["overall_score"], weighted / denominator, places=2)
        self.assertEqual(result["claims"][0]["sources"][0]["page"], 12)
        self.assertIn("Revenue doubled", result["unsupported_claims"][0]["claim"])

    def test_no_claims_marks_validation_not_applicable_and_excludes_weight(self):
        result = evaluate_summary(self.source, "Revenue increased by 25 percent in 2025.",
                                  chunks=self.chunks, source_language="en", summary_language="en")
        validation = result["metrics"]["validation_accuracy"]
        self.assertIsNone(validation["score"])
        self.assertEqual(validation["status"], "not_applicable")
        self.assertEqual(validation["explanation"], "No factual claims requiring validation were detected.")
        self.assertEqual(result["evaluation_status"], "not_applicable")
        self.assertEqual(result["applicable_metric_count"], 3)

    def test_unusable_summary_has_no_scores(self):
        result = evaluate_summary(self.source, None, source_language="en")
        self.assertEqual(result["evaluation_status"], "unavailable")
        self.assertIsNone(result["overall_score"])
        self.assertTrue(all(item["score"] is None for item in result["metrics"].values()))

    def test_cross_language_summary_does_not_get_lexical_similarity_scores(self):
        result = evaluate_summary(self.source, "Los ingresos aumentaron durante el año.",
                                  source_language="en", summary_language="es")
        self.assertIsNone(result["metrics"]["relevance"]["score"])
        self.assertIsNone(result["metrics"]["completeness"]["score"])
        self.assertIn("differs from the source", result["warnings"][0])

    def test_short_and_large_source_documents_are_scored_without_placeholders(self):
        short = evaluate_summary("Sales rose 10 percent.", "Sales rose 10 percent.",
                                 source_language="en", summary_language="en")
        self.assertIsNotNone(short["metrics"]["relevance"]["score"])
        long_source = " ".join(f"The annual report records initiative {i} and a measurable milestone for operations." for i in range(800))
        long_summary = "The annual report records initiative 12 and a measurable milestone for operations."
        large = evaluate_summary(long_source, long_summary, source_language="en", summary_language="en", depth="deep")
        self.assertGreater(large["metrics"]["completeness"]["score"], 0)
        self.assertTrue(all(metric["score"] is None or 0 <= metric["score"] <= 100 for metric in large["metrics"].values()))

    def test_regenerated_summary_gets_a_different_fingerprint(self):
        first = evaluate_summary(self.source, "Revenue increased by 25 percent in 2025.", source_language="en", summary_language="en")
        second = evaluate_summary(self.source, "Factory investment will reduce delivery delays.", source_language="en", summary_language="en")
        self.assertNotEqual(first["summary_fingerprint"], second["summary_fingerprint"])

    def test_export_includes_evaluation_report(self):
        evaluation = evaluate_summary(self.source, "Revenue increased by 25 percent in 2025.", source_language="en", summary_language="en")
        payload = make_export_payload({"name": "annual.txt", "language": "en"}, {
            "executive_summary": "Revenue increased by 25 percent in 2025.", "language": "en", "evaluation": evaluation,
        })
        text_export, _, _ = render_summary(payload, "txt")
        self.assertIn("Summary Quality Evaluation", text_export.decode("utf-8"))
        self.assertIn("Validation Accuracy: N/A", text_export.decode("utf-8"))
        self.assertIn("summary_quality_evaluation", make_export_payload({"name": "annual.txt"}, {
            "executive_summary": "Revenue increased by 25 percent in 2025.", "evaluation": evaluation,
        }))


if __name__ == "__main__":
    unittest.main()
