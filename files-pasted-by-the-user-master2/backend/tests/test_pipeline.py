import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import ProcessOptions, detect_language, extract_text, make_chunks, normalize, process_document, process_extractively, resolve_summary_depth
from app.openrouter import summarize_with_openrouter


class PipelineTests(unittest.TestCase):
    def test_summary_depth_request_and_auto_resolution(self):
        self.assertEqual(ProcessOptions.model_validate({"summary_depth": "deep"}).summary_depth, "deep")
        self.assertEqual(ProcessOptions.model_validate({"depth": "quick"}).summary_depth, "quick")
        self.assertEqual(resolve_summary_depth("auto", {"chunks": [{}] * 8}), "quick")
        self.assertEqual(resolve_summary_depth("auto", {"chunks": [{}] * 9}), "standard")
        self.assertEqual(resolve_summary_depth("auto", {"chunks": [{}] * 31}), "deep")
        self.assertEqual(ProcessOptions(summary_depth="custom", custom_detail="deep").summary_depth, "custom")
        self.assertEqual(resolve_summary_depth("custom", {"chunks": [{}]}, ProcessOptions(summary_depth="custom", custom_detail="deep")), "deep")

    def test_normalize_and_chunk_preserve_source_structure(self):
        text = normalize("Heading\r\n\r\nA meaningful sentence about transit service and passenger safety.\r\n\r\nAnother detailed sentence about the annual budget and staffing levels.")
        chunks = make_chunks(text, [{"page": 4, "text": text}])
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["section"], "Document text")
        self.assertEqual(chunks[0]["page"], 4)
        self.assertEqual(chunks[0]["language"], "en")

    def test_process_claims_are_source_linked_and_validated(self):
        text = "The board approved a two-year service pilot. The pilot will be reviewed every quarter."
        doc = {"text": text, "chunks": make_chunks(text, []), "pages": [], "language": "en"}
        with patch("app.main.OpenRouterService", side_effect=RuntimeError("Deterministic unit test")):
            result = process_document(doc, ProcessOptions(depth="quick"))
        self.assertEqual(result["status"], "validated")
        self.assertTrue(result["claims"])
        self.assertTrue(all(c["status"] == "SUPPORTED" and c["sources"] for c in result["claims"]))

    def test_extractively_generated_summary_length_tracks_depth(self):
        sentences = [f"The committee recorded distinct operational evidence point number {i} for the annual review." for i in range(1, 16)]
        text = " ".join(sentences)
        doc = {"text": text, "chunks": make_chunks(text, []), "pages": [], "language": "en"}
        quick = process_extractively(doc, ProcessOptions(summary_depth="quick"))
        deep = process_extractively(doc, ProcessOptions(summary_depth="deep"))
        self.assertLess(len(quick["executive_summary"]), len(deep["executive_summary"]))
        self.assertEqual(quick["stats"]["context_chunks_used"], len({c["sources"][0]["chunk_id"] for c in quick["claims"] if c["sources"]}))

    def test_backend_state_keeps_depth_purpose_audience_and_requested_language(self):
        text = "The agency identified a substantial cybersecurity risk requiring mitigation before deployment."
        doc = {"text": text, "chunks": make_chunks(text, []), "pages": [], "language": "en"}
        with unittest.mock.patch("app.main.OpenRouterService", side_effect=RuntimeError("No configured provider")):
            result = process_document(doc, ProcessOptions(summary_depth="deep", purpose="Risk analysis", audience="Technical audience", language="te"))
        self.assertEqual(result["summary_mode"], "deep")
        self.assertEqual(result["summary_depth_resolved"], "deep")
        self.assertEqual(result["purpose"], "Risk analysis")
        self.assertEqual(result["audience"], "Technical audience")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["requested_language"], "te")
        self.assertIn("summary_validation", result["workflow"])


    def test_json_parser_reports_readable_text(self):
        text, pages = extract_text("record.json", b'{"title":"Transit plan","year":2025}')
        self.assertIn('"Transit plan"', text)
        self.assertEqual(pages, [])

    def test_language_detection_handles_supported_scripts(self):
        self.assertEqual(detect_language("తెలుగు భాష మరియు పత్రం"), "te")
        self.assertEqual(detect_language("தமிழ் மொழி மற்றும் ஆவணம்"), "ta")
        self.assertEqual(detect_language("हिंदी भाषा और दस्तावेज़"), "hi")
        self.assertEqual(detect_language("ಕನ್ನಡ ಭಾಷೆ ಮತ್ತು ದಾಖಲೆ"), "kn")
        self.assertEqual(detect_language("తెలుగు భాష మరియు పత్రం English language"), "te")
        self.assertEqual(detect_language("12345 !!!"), "und")

    def test_openrouter_summary_requires_validated_source_claims(self):
        class FakeOpenRouter:
            model = "test/fake"

            def complete_json(self, system, user, **kwargs):
                import json
                payload = json.loads(user)
                if "passages" in payload:
                    return {"notes": [{"text": payload["passages"][0]["text"], "source_chunk_ids": [payload["passages"][0]["chunk_id"]]}]}
                if "source_notes" in payload:
                    cid = payload["source_notes"][0]["source_chunk_ids"][0]
                    sentence = payload["source_notes"][0]["text"]
                    return {"executive_summary": "UNTRUSTED SUMMARY MUST NOT BE USED", "findings": [{"text": sentence, "source_chunk_ids": [cid]}]}
                return {"results": [{"id": claim["id"], "status": "SUPPORTED", "source_chunk_ids": claim["cited_chunk_ids"], "reason": "Directly supported."} for claim in payload["claims"]]}

        text = "The board approved a two-year service pilot."
        doc = {"text": text, "chunks": [{"id": "chunk_001", "text": text, "page": 1, "section": "Decision", "language": "en"}], "pages": [{"page": 1}], "language": "en"}
        result = summarize_with_openrouter(doc, ProcessOptions(depth="quick"), FakeOpenRouter())
        self.assertEqual(result["status"], "validated")
        self.assertNotIn("UNTRUSTED", result["executive_summary"])
        self.assertEqual(result["claims"][0]["sources"][0]["chunk_id"], "chunk_001")

    def test_openrouter_depth_changes_context_prompt_and_generation_capacity(self):
        import json

        class CapturingOpenRouter:
            model = "test/fake"

            def __init__(self):
                self.calls = []

            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                self.calls.append((system, payload, kwargs.get("max_tokens")))
                if "passages" in payload:
                    return {"notes": [{"text": p["text"], "source_chunk_ids": [p["chunk_id"]]} for p in payload["passages"]]}
                if "source_notes" in payload:
                    note = payload["source_notes"][0]
                    return {"findings": [{"text": note["text"], "source_chunk_ids": note["source_chunk_ids"]}]}
                return {"results": [{"id": c["id"], "status": "SUPPORTED", "source_chunk_ids": c["cited_chunk_ids"], "reason": "Supported."} for c in payload["claims"]]}

        chunks = [{"id": f"chunk_{i:03d}", "text": f"Section {i} contains substantive evidence and a distinct operational finding." , "page": i, "section": f"Section {i}", "language": "en"} for i in range(1, 10)]
        doc = {"text": " ".join(c["text"] for c in chunks), "chunks": chunks, "pages": [], "language": "en"}
        quick_service, deep_service = CapturingOpenRouter(), CapturingOpenRouter()
        quick = summarize_with_openrouter(doc, ProcessOptions(summary_depth="quick"), quick_service)
        deep = summarize_with_openrouter(doc, ProcessOptions(summary_depth="deep"), deep_service)
        quick_generation = next(c for c in quick_service.calls if "source_notes" in c[1])
        deep_generation = next(c for c in deep_service.calls if "source_notes" in c[1])
        self.assertLess(len(quick_generation[1]["source_notes"]), len(deep_generation[1]["source_notes"]))
        self.assertLess(quick_generation[2], deep_generation[2])
        self.assertIn("Concise brief", quick_generation[0])
        self.assertIn("Detailed analysis", deep_generation[0])
        self.assertEqual(quick["summary_depth_resolved"], "quick")
        self.assertEqual(deep["summary_depth_resolved"], "deep")

    def test_purpose_and_audience_reach_synthesis_prompt(self):
        import json

        class Capture:
            model = "test/fake"
            def __init__(self): self.synthesis = None
            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                if "passages" in payload:
                    return {"notes": [{"text": "Source supports a cyber risk finding.", "source_chunk_ids": ["chunk_001"]}]}
                if "source_notes" in payload:
                    self.synthesis = (system, payload)
                    return {"findings": [{"text": "A cyber risk requires mitigation.", "source_chunk_ids": ["chunk_001"]}]}
                return {"results": [{"id": item["id"], "status": "SUPPORTED", "source_chunk_ids": item["cited_chunk_ids"], "reason": "Supported", "confidence": .91} for item in payload["claims"]]}

        text = "The agency identified a substantial cybersecurity risk requiring mitigation before deployment."
        doc = {"text": text, "chunks": [{"id": "chunk_001", "text": text, "page": 3, "section": "Risk", "language": "en"}], "pages": [], "language": "en"}
        service = Capture()
        result = summarize_with_openrouter(doc, ProcessOptions(summary_depth="standard", purpose="Risk analysis", audience="Technical audience"), service)
        system, payload = service.synthesis
        self.assertIn("threats", payload["purpose_instructions"])
        self.assertIn("technical terminology", payload["audience_instructions"])
        self.assertEqual(result["claims"][0]["confidence"], .91)
        self.assertEqual(result["claims"][0]["sources"][0]["page"], 3)

    def test_telugu_translation_is_checked_against_original_report(self):
        import json

        class Translator:
            model = "test/fake"
            def __init__(self): self.validation_languages = []
            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                if "passages" in payload:
                    p = payload["passages"][0]
                    return {"notes": [{"text": p["text"], "source_chunk_ids": [p["chunk_id"]]}]}
                if "source_notes" in payload:
                    note = payload["source_notes"][0]
                    return {"findings": [{"text": note["text"], "source_chunk_ids": note["source_chunk_ids"]}]}
                if "target_language" in payload:
                    return {"findings": ["సైబర్ ప్రమాదాన్ని తగ్గించాలి."]}
                claim = payload["claims"][0]
                self.validation_languages.append(claim["claim"])
                return {"results": [{"id": claim["id"], "status": "SUPPORTED", "source_chunk_ids": claim["cited_chunk_ids"], "reason": "Meaning matches source.", "confidence": .88}]}

        text = "The agency identified a substantial cybersecurity risk requiring mitigation before deployment."
        doc = {"text": text, "chunks": [{"id": "chunk_001", "text": text, "page": 3, "section": "Risk", "language": "en"}], "pages": [], "language": "en"}
        service = Translator()
        result = summarize_with_openrouter(doc, ProcessOptions(summary_depth="standard", language="te"), service)
        self.assertEqual(result["language"], "te")
        self.assertEqual(result["translation_validation_status"], "validated_against_original_report")
        self.assertTrue(any("సైబర్" in claim for claim in service.validation_languages))
        self.assertEqual(result["claims"][0]["sources"][0]["chunk_id"], "chunk_001")


if __name__ == "__main__":
    unittest.main()
