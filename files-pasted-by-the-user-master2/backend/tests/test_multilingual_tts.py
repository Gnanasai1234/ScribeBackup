from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import DOCS, app
from services.tts_service import (
    estimate_timings,
    split_into_sentences,
    voices_for,
)


class MultilingualTtsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.doc_id = "test_doc_tts"
        DOCS[self.doc_id] = {
            "id": self.doc_id,
            "name": "test.txt",
            "language": "en",
            "status": "ready",
            "chunks": [{"id": "c1", "section": "intro", "text": "Safety inspections are mandatory quarterly.", "language": "en"}],
            "stats": {"words": 5, "chunks": 1, "pages": None},
            "result": {
                "status": "validated",
                "depth": "standard",
                "language": "en",
                "purpose": "Executive briefing",
                "executive_summary": "Quarterly safety inspections are mandatory at every facility. Management approved the safety protocol.",
                "findings": ["Quarterly inspections required."],
                "claims": [{"id": "claim_01", "claim": "Inspections are required.", "status": "SUPPORTED", "sources": [{"chunk_id": "c1"}]}],
                "quality": {"groundedness": 1.0, "claims_checked": 1},
                "stats": {"words": 15, "chunks": 1, "context_chunks_used": 1, "pages": None, "language": "en"},
                "limitations": [],
            },
        }

    def tearDown(self):
        DOCS.pop(self.doc_id, None)

    def test_multilingual_sentence_splitting(self):
        # English
        en_text = "First sentence. Second sentence! Third one?"
        en_sents = split_into_sentences(en_text)
        self.assertEqual(len(en_sents), 3)

        # Telugu
        te_text = "ఈ నివేదిక భద్రతా ప్రమాణాలను వివరిస్తుంది. ఇది చాలా ముఖ్యమైన సమాచారం."
        te_sents = split_into_sentences(te_text)
        self.assertEqual(len(te_sents), 2)
        self.assertIn("ఈ నివేదిక", te_sents[0])

        # Hindi with Devanagari danda
        hi_text = "यह पहली पंक्ति है। यह दूसरी पंक्ति है।"
        hi_sents = split_into_sentences(hi_text)
        self.assertEqual(len(hi_sents), 2)

        # Multi-paragraph with newlines
        para_text = "Heading text\n\nParagraph one sentence. Another sentence.\n\nParagraph two sentence."
        para_sents = split_into_sentences(para_text)
        self.assertGreaterEqual(len(para_sents), 3)

    def test_estimate_timings_structure(self):
        text = "Sentence one. Sentence two. Sentence three."
        timings = estimate_timings(text, total_duration=9.0)
        self.assertEqual(len(timings), 3)
        self.assertEqual(timings[0]["start"], 0.0)
        self.assertAlmostEqual(timings[-1]["end"], 9.0, places=1)
        # Monotonicity check
        for i in range(len(timings) - 1):
            self.assertLessEqual(timings[i]["start"], timings[i]["end"])
            self.assertAlmostEqual(timings[i]["end"], timings[i + 1]["start"], delta=0.01)

    def test_voices_supported_for_all_multilingual_languages(self):
        languages = ["en", "te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "ur"]
        for lang in languages:
            voices = voices_for(lang)
            self.assertTrue(len(voices) > 0, f"No voices configured for language: {lang}")
            for v in voices:
                self.assertIn("voice_id", v)
                self.assertIn("name", v)

        # Unsupported language
        self.assertEqual(voices_for("unsupported_xyz"), [])

    def test_tts_voices_api_endpoint(self):
        res = self.client.get("/api/tts/voices?language=te&provider=edge")
        self.assertEqual(res.status_code, 200)
        voices = res.json()
        self.assertTrue(len(voices) >= 2)
        voice_ids = [v["voice_id"] for v in voices]
        self.assertIn("te-IN-ShrutiNeural", voice_ids)

        # 422 for unsupported language
        bad_res = self.client.get("/api/tts/voices?language=xx")
        self.assertEqual(bad_res.status_code, 422)

    def test_tts_audio_edge_success_and_headers(self):
        mock_audio = b"\xff\xf3mockaudiobytes"
        mock_timings = [
            {"index": 0, "start": 0.1, "end": 2.5, "text": "Quarterly safety inspections are mandatory at every facility."},
            {"index": 1, "start": 2.5, "end": 4.8, "text": "Management approved the safety protocol."},
        ]
        with patch("app.main.synthesize_with_timings", return_value=(mock_audio, mock_timings)):
            res = self.client.post(
                f"/api/documents/{self.doc_id}/audio",
                json={"language": "en", "provider": "edge", "voice_id": "en-US-AriaNeural"},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers.get("x-tts-provider"), "edge-neural")
            self.assertEqual(res.headers.get("x-tts-voice"), "en-US-AriaNeural")
            self.assertEqual(res.headers.get("x-tts-fallback-used"), "false")
            b64_timings = res.headers.get("x-tts-timings-base64")
            self.assertIsNotNone(b64_timings)
            decoded_timings = json.loads(base64.b64decode(b64_timings).decode("utf-8"))
            self.assertEqual(len(decoded_timings), 2)
            self.assertEqual(decoded_timings[0]["start"], 0.1)

    def test_tts_edge_failure_falls_back_to_browser_without_silent_failure(self):
        with patch("app.main.synthesize_with_timings", side_effect=RuntimeError("Edge connection reset")):
            res = self.client.post(
                f"/api/documents/{self.doc_id}/audio",
                json={"language": "en", "provider": "edge"},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["provider"], "browser")
            self.assertTrue(data["fallback_used"])
            self.assertIn("fallback_reason", data)
            self.assertEqual(data["language"], "en")
            self.assertTrue(len(data["timings"]) >= 2)

    def test_tts_audio_preserves_multilingual_language(self):
        # When language matches document language, narration text is preserved
        DOCS[self.doc_id]["result"]["language"] = "te"
        DOCS[self.doc_id]["result"]["executive_summary"] = "ఈ నివేదిక భద్రతా ప్రమాణాలను వివరిస్తుంది."
        res = self.client.post(
            f"/api/documents/{self.doc_id}/audio",
            json={"language": "te", "provider": "browser"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["language"], "te")
        self.assertEqual(data["text"], "ఈ నివేదిక భద్రతా ప్రమాణాలను వివరిస్తుంది.")

    def test_audio_timings_endpoint(self):
        res = self.client.get(f"/api/documents/{self.doc_id}/audio/timings")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("timings", data)
        self.assertIn("text", data)
        self.assertTrue(len(data["timings"]) > 0)


if __name__ == "__main__":
    unittest.main()
