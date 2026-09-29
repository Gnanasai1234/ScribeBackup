import unittest
from services.document_comparison import (
    clean_text_for_comparison,
    extract_numbers_and_quantities,
    extract_dates,
    detect_entity_differences,
    segment_document_into_sections,
    perform_document_comparison,
)


class TestDocumentComparison(unittest.TestCase):

    def test_clean_text(self):
        text = "Hello   world!\r\n\r\n\n\nTest   line."
        cleaned = clean_text_for_comparison(text)
        self.assertEqual(cleaned, "Hello world!\n\nTest line.")

    def test_numeric_change_detection(self):
        text_a = "The platform supports up to 100 users with 2.7 seconds latency."
        text_b = "The platform supports up to 150 users with 1.7 seconds latency."
        diffs = detect_entity_differences(text_a, text_b)
        self.assertTrue(any("100" in d["old_value"] and "150" in d["new_value"] for d in diffs))

    def test_date_change_detection(self):
        text_a = "The project deadline is June 30, 2026."
        text_b = "The project deadline is July 15, 2026."
        diffs = detect_entity_differences(text_a, text_b)
        self.assertTrue(any(d["type"] == "DATE_CHANGE" for d in diffs))

    def test_identical_documents(self):
        text = "1. Introduction\nThe system was deployed across five regions.\n\n2. Architecture\nCentralized database stores user records."
        res = perform_document_comparison(
            doc_a_info={"name": "spec_v1.txt"},
            doc_b_info={"name": "spec_v1_copy.txt"},
            text_a=text,
            text_b=text,
        )
        self.assertEqual(res["statistics"]["added"], 0)
        self.assertEqual(res["statistics"]["removed"], 0)
        self.assertEqual(res["statistics"]["modified"], 0)
        self.assertGreater(res["statistics"]["unchanged"], 0)
        self.assertFalse(res["is_unrelated"])

    def test_added_content(self):
        text_a = "1. Introduction\nThe system was deployed across five regions."
        text_b = "1. Introduction\nThe system was deployed across five regions.\n\n2. Monitoring\nThe platform now includes real-time Prometheus monitoring."
        res = perform_document_comparison(
            doc_a_info={"name": "doc_a.txt"},
            doc_b_info={"name": "doc_b.txt"},
            text_a=text_a,
            text_b=text_b,
        )
        self.assertGreater(res["statistics"]["added"], 0)
        self.assertTrue(any(item["status"] == "ADDED" for item in res["comparison_items"]))

    def test_removed_content(self):
        text_a = "1. Introduction\nThe system was deployed across five regions.\n\n2. Legacy\nOld legacy script is retained for backups."
        text_b = "1. Introduction\nThe system was deployed across five regions."
        res = perform_document_comparison(
            doc_a_info={"name": "doc_a.txt"},
            doc_b_info={"name": "doc_b.txt"},
            text_a=text_a,
            text_b=text_b,
        )
        self.assertGreater(res["statistics"]["removed"], 0)
        self.assertTrue(any(item["status"] == "REMOVED" for item in res["comparison_items"]))

    def test_numeric_value_change(self):
        text_a = "1. Capacity\nThe cluster maintains 100 users."
        text_b = "1. Capacity\nThe cluster maintains 150 users."
        res = perform_document_comparison(
            doc_a_info={"name": "v1.txt"},
            doc_b_info={"name": "v2.txt"},
            text_a=text_a,
            text_b=text_b,
        )
        self.assertGreater(res["statistics"]["modified"], 0)
        modified_items = [it for it in res["comparison_items"] if it["status"] == "MODIFIED"]
        self.assertTrue(len(modified_items) > 0)
        self.assertTrue(any("100" in ed.get("old_value", "") and "150" in ed.get("new_value", "") for it in modified_items for ed in it.get("specific_changes", [])))

    def test_unrelated_documents(self):
        text_a = "The evolution of modern computational architecture spans several decades of microprocessors, paging mechanisms, and memory management units."
        text_b = "అద్భుతమైన విజువల్ ఎఫెక్ట్స్, భారీ యాక్షన్ సన్నివేశాలు మరియు హీరోల మధ్య సమన్వయంతో ఈ సినిమా అభిమానులకు ఒక గొప్ప అనుభూతిని పంచుతుంది."
        res = perform_document_comparison(
            doc_a_info={"name": "Computational_Systems.pdf"},
            doc_b_info={"name": "Justice_League_Telugu.docx"},
            text_a=text_a,
            text_b=text_b,
            lang_a="en",
            lang_b="te",
        )
        self.assertTrue(res["is_unrelated"])
        self.assertEqual(res["topic_overlap"], "LOW")
        self.assertIn("Low semantic overlap", res["executive_summary"])


if __name__ == "__main__":
    unittest.main()
