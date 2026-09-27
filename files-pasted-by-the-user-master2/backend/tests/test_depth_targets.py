import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import ProcessOptions, make_chunks, process_extractively, resolve_summary_depth
from app.openrouter import summarize_with_openrouter


class DepthTargetTests(unittest.TestCase):
    def setUp(self):
        # Create a document with 80+ distinct sentences for depth testing
        self.sentences = [
            f"Finding {i}: The system recorded measurable operational performance metric {i} indicating stable throughput and low latency across the regional cluster."
            for i in range(1, 85)
        ]
        self.long_text = " ".join(self.sentences)
        self.long_doc = {
            "text": self.long_text,
            "chunks": make_chunks(self.long_text, []),
            "pages": [],
            "language": "en",
        }

    def test_extractive_same_document_quick_standard_deep_progression(self):
        """Test the SAME document across Quick, Standard, and Deep to verify progressive detail."""
        quick_opts = ProcessOptions(summary_depth="quick")
        standard_opts = ProcessOptions(summary_depth="standard")
        deep_opts = ProcessOptions(summary_depth="deep")

        quick_res = process_extractively(self.long_doc, quick_opts)
        standard_res = process_extractively(self.long_doc, standard_opts)
        deep_res = process_extractively(self.long_doc, deep_opts)

        quick_lines = len(quick_res["claims"])
        standard_lines = len(standard_res["claims"])
        deep_lines = len(deep_res["claims"])

        # Quick targets 10-15 lines
        self.assertGreaterEqual(quick_lines, 10, f"Quick lines ({quick_lines}) below target 10-15")
        self.assertLessEqual(quick_lines, 15, f"Quick lines ({quick_lines}) above target 10-15")

        # Standard targets 20-30 lines
        self.assertGreaterEqual(standard_lines, 20, f"Standard lines ({standard_lines}) below target 20-30")
        self.assertLessEqual(standard_lines, 30, f"Standard lines ({standard_lines}) above target 20-30")

        # Deep targets 40-60+ lines
        self.assertGreaterEqual(deep_lines, 40, f"Deep lines ({deep_lines}) below target 40-60+")

        # Progression check: Quick < Standard < Deep
        self.assertLess(quick_lines, standard_lines, "Quick must be strictly less than Standard")
        self.assertLess(standard_lines, deep_lines, "Standard must be strictly less than Deep")

        # Content progression check: executive summary length
        self.assertLess(len(quick_res["executive_summary"]), len(standard_res["executive_summary"]))
        self.assertLess(len(standard_res["executive_summary"]), len(deep_res["executive_summary"]))

    def test_extractive_auto_sizing_by_document_length(self):
        """Test Auto mode dynamically scaling across short, medium, and long documents."""
        # Short document (e.g. 20 sentences)
        short_text = " ".join(self.sentences[:20])
        short_doc = {"text": short_text, "chunks": make_chunks(short_text, []), "pages": [], "language": "en"}
        auto_short = process_extractively(short_doc, ProcessOptions(summary_depth="auto", summary_mode="auto"))
        short_lines = len(auto_short["claims"])
        self.assertGreaterEqual(short_lines, 15, f"Auto short ({short_lines}) below target 15-25")
        self.assertLessEqual(short_lines, 25, f"Auto short ({short_lines}) above target 15-25")

        # Medium document (e.g. 45 sentences)
        med_text = " ".join(self.sentences[:45])
        med_doc = {"text": med_text, "chunks": make_chunks(med_text, []), "pages": [], "language": "en"}
        auto_med = process_extractively(med_doc, ProcessOptions(summary_depth="auto", summary_mode="auto"))
        med_lines = len(auto_med["claims"])
        self.assertGreaterEqual(med_lines, 25, f"Auto medium ({med_lines}) below target 25-45")
        self.assertLessEqual(med_lines, 45, f"Auto medium ({med_lines}) above target 25-45")

        # Long document (80+ sentences)
        auto_long = process_extractively(self.long_doc, ProcessOptions(summary_depth="auto", summary_mode="auto"))
        long_lines = len(auto_long["claims"])
        self.assertGreaterEqual(long_lines, 45, f"Auto long ({long_lines}) below target 45-70+")

        # Progression check for Auto
        self.assertLess(short_lines, med_lines)
        self.assertLess(med_lines, long_lines)

    def test_extractive_custom_detail_levels(self):
        """Test Custom mode respects detail levels and deep does not cap to standard."""
        custom_quick = process_extractively(self.long_doc, ProcessOptions(summary_depth="custom", custom_detail="quick"))
        custom_standard = process_extractively(self.long_doc, ProcessOptions(summary_depth="custom", custom_detail="standard"))
        custom_deep = process_extractively(self.long_doc, ProcessOptions(summary_depth="custom", custom_detail="deep"))

        self.assertGreaterEqual(len(custom_quick["claims"]), 10)
        self.assertGreaterEqual(len(custom_standard["claims"]), 20)
        self.assertGreaterEqual(len(custom_deep["claims"]), 40)
        self.assertGreater(len(custom_deep["claims"]), len(custom_standard["claims"]))

    def test_extractive_custom_sections_match_selected_categories_including_clear_all(self):
        text = "The report identifies a serious risk in the regional process. Revenue reached 24 percent growth in 2025. The board decided to delay approval until June."
        doc = {"text": text, "chunks": make_chunks(text, []), "pages": [], "language": "en"}
        selected = process_extractively(doc, ProcessOptions(summary_depth="custom", custom_detail="standard", custom_sections=["risks", "decisions"]))
        self.assertTrue(selected["claims"])
        self.assertEqual({claim["category"] for claim in selected["claims"]}, {"risks", "decisions"})
        self.assertEqual(selected["facts"], [])

        cleared = process_extractively(doc, ProcessOptions(summary_depth="custom", custom_sections=[]))
        self.assertEqual(cleared["claims"], [])
        self.assertTrue(all(not cleared[section] for section in ("findings", "facts", "risks", "decisions", "recommendations", "action_items")))
        self.assertEqual(cleared["validation_status"], "no_claims")
        self.assertTrue(cleared["executive_summary"])

    def test_openrouter_prompt_and_tokens_for_all_modes(self):
        """Test OpenRouter prompt mode and max_tokens allowances across quick, standard, deep, and auto."""
        import json

        class CaptureService:
            model = "test/llm"
            def __init__(self):
                self.calls = []

            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                self.calls.append((system, payload, kwargs.get("max_tokens", 0)))
                if "passages" in payload:
                    return {"notes": [{"text": p["text"], "source_chunk_ids": [p["chunk_id"]]} for p in payload["passages"]]}
                if "source_notes" in payload:
                    notes = payload["source_notes"]
                    # Generate points reflecting prompt request
                    return {
                        "findings": [{"text": f"Finding note {i}", "source_chunk_ids": [notes[0]["source_chunk_ids"][0]]} for i in range(len(notes))]
                    }
                return {"results": [{"id": c["id"], "status": "SUPPORTED", "source_chunk_ids": c["cited_chunk_ids"], "reason": "Entailed."} for c in payload.get("claims", [])]}

        service = CaptureService()
        doc = {"text": self.long_text, "chunks": make_chunks(self.long_text, []), "pages": [], "language": "en"}

        # Quick
        summarize_with_openrouter(doc, ProcessOptions(summary_depth="quick"), service)
        quick_call = [c for c in service.calls if "source_notes" in c[1]][-1]
        self.assertIn("10 to 15", quick_call[1]["depth_requirements"])
        self.assertGreaterEqual(quick_call[2], 2800)

        # Standard
        summarize_with_openrouter(doc, ProcessOptions(summary_depth="standard"), service)
        std_call = [c for c in service.calls if "source_notes" in c[1]][-1]
        self.assertIn("20 to 30", std_call[1]["depth_requirements"])
        self.assertGreaterEqual(std_call[2], 4500)

        # Deep
        summarize_with_openrouter(doc, ProcessOptions(summary_depth="deep"), service)
        deep_call = [c for c in service.calls if "source_notes" in c[1]][-1]
        self.assertIn("40 to 60+", deep_call[1]["depth_requirements"])
        self.assertGreaterEqual(deep_call[2], 7500)

        # Token allowance progression: Quick < Standard < Deep
        self.assertLess(quick_call[2], std_call[2])
        self.assertLess(std_call[2], deep_call[2])


if __name__ == "__main__":
    unittest.main()
