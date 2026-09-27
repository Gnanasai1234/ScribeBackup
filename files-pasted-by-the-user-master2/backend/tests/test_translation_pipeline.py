import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import ProcessOptions, make_chunks, process_document, process_extractively
from app.openrouter import (
    LANGUAGE_SCRIPTS,
    OpenRouterService,
    summarize_with_openrouter,
    translate_for_audio,
    verify_script,
)


class TranslationPipelineTests(unittest.TestCase):
    def setUp(self):
        # Sample Telugu source text from the user's document
        self.telugu_text = "పాండురాజు మరణానంతరం ధృతరాష్ట్రుడు రాజ్యపాలన చేపట్టాడు. పాండవులు మరియు కౌరవులు హస్తినాపురంలో కలసి పెరిగారు. కురుక్షేత్ర యుద్ధంలో పాండవులు విజయం సాధించారు."
        self.kannada_sentence = "ಪಾಂಡುರಾಜನ ಮರಣದ ನಂತರ ಧೃತರಾಷ್ಟ್ರನು ರಾಜ್ಯಭಾರವನ್ನು ವಹಿಸಿಕೊಂಡನು."
        self.telugu_sentence = "పాండురాజు మరణానంతరం ధృతరాష్ట్రుడు రాజ్యపాలన చేపట్టాడు."
        self.tamil_sentence = "பாண்டுராஜுவின் மரணத்திற்குப் பிறகு திருதராஷ்டிரன் ஆட்சிப் பொறுப்பேற்றார்."
        self.hindi_sentence = "पांडुराज की मृत्यु के बाद धृतराष्ट्र ने राज्य का शासन संभाला।"
        self.english_sentence = "After the death of Panduraju, Dhritarashtra took over the kingdom governance."

    def test_script_verification_logic(self):
        """Test verify_script accurately accepts correct scripts and strictly rejects foreign scripts."""
        # Kannada verification
        self.assertTrue(verify_script(self.kannada_sentence, "kn"))
        self.assertFalse(verify_script(self.telugu_sentence, "kn"), "Telugu must not pass Kannada verification")
        self.assertFalse(verify_script(self.tamil_sentence, "kn"))

        # Telugu verification
        self.assertTrue(verify_script(self.telugu_sentence, "te"))
        self.assertFalse(verify_script(self.kannada_sentence, "te"), "Kannada must not pass Telugu verification")

        # Tamil verification
        self.assertTrue(verify_script(self.tamil_sentence, "ta"))
        self.assertFalse(verify_script(self.kannada_sentence, "ta"))

        # Hindi verification
        self.assertTrue(verify_script(self.hindi_sentence, "hi"))
        self.assertFalse(verify_script(self.telugu_sentence, "hi"))

        # English verification
        self.assertTrue(verify_script(self.english_sentence, "en"))
        self.assertFalse(verify_script(self.telugu_sentence, "en"), "Telugu must not pass English verification")

    def _has_script(self, text: str, lang_code: str) -> bool:
        start, end = LANGUAGE_SCRIPTS[lang_code][2], LANGUAGE_SCRIPTS[lang_code][3]
        return any(start <= ord(c) <= end for c in text)

    def test_kannada_translation_produces_kannada_and_no_telugu(self):
        """Selecting Kannada must produce Kannada script (0x0C80-0x0CFF) and no Telugu script."""
        doc = {
            "id": "doc_te_01",
            "text": self.telugu_text,
            "chunks": make_chunks(self.telugu_text, []),
            "pages": [],
            "language": "te",
        }

        mock_service = MagicMock(spec=OpenRouterService)
        mock_service.provider_name = "openrouter"
        mock_service.model = "openai/gpt-4o-mini"

        # Mock complete_json responses:
        # 1. Notes extraction
        # 2. Synthesis in Telugu (source language)
        # 3. Validation
        # 4. Translation into Kannada
        # 5. Cross-lingual validation
        def mock_complete_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            elif "enterprise document intelligence platform" in system:
                return {
                    "findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}],
                    "facts": [],
                    "risks": [],
                    "decisions": [],
                    "recommendations": [],
                    "action_items": [],
                }
            elif "Validate each claim strictly" in system:
                return {
                    "results": [
                        {
                            "id": "claim_01",
                            "status": "SUPPORTED",
                            "source_chunk_ids": ["chunk_001"],
                            "reason": "Direct evidence match.",
                            "confidence": 0.95,
                        }
                    ]
                }
            elif "Translate the following content into Kannada" in system:
                # LLM correctly responds with Kannada claims
                return {
                    "target_language": "Kannada",
                    "validated_summary_fields": {
                        "findings": [self.kannada_sentence],
                    },
                }
            return {}

        mock_service.complete_json.side_effect = mock_complete_json

        result = summarize_with_openrouter(
            doc,
            ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="kn"),
            mock_service,
        )

        self.assertEqual(result["language"], "kn")
        self.assertEqual(result["source_language"], "te")
        self.assertEqual(result["status"], "validated")

        # Executive summary must contain Kannada script
        self.assertTrue(self._has_script(result["executive_summary"], "kn"), "Executive summary must contain Kannada script")
        # Executive summary must NOT contain Telugu script
        self.assertFalse(self._has_script(result["executive_summary"], "te"), "Executive summary must NOT contain Telugu script")
        self.assertIn(self.kannada_sentence, result["executive_summary"])

    def test_switching_languages_telugu_to_kannada_and_back(self):
        """Ensure switching between Telugu and Kannada on the same document does not reuse stale cached state."""
        doc = {
            "id": "doc_switch_01",
            "text": self.telugu_text,
            "chunks": make_chunks(self.telugu_text, []),
            "pages": [],
            "language": "te",
        }

        mock_service = MagicMock(spec=OpenRouterService)
        mock_service.provider_name = "openrouter"
        mock_service.model = "openai/gpt-4o-mini"

        def mock_complete_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            elif "enterprise document intelligence platform" in system:
                return {
                    "findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}],
                    "facts": [], "risks": [], "decisions": [], "recommendations": [], "action_items": [],
                }
            elif "Validate each claim strictly" in system:
                return {
                    "results": [
                        {
                            "id": "claim_01",
                            "status": "SUPPORTED",
                            "source_chunk_ids": ["chunk_001"],
                            "reason": "Direct evidence match.",
                            "confidence": 0.95,
                        }
                    ]
                }
            elif "Translate the following content into Kannada" in system:
                return {
                    "validated_summary_fields": {
                        "findings": [self.kannada_sentence],
                    }
                }
            return {}

        mock_service.complete_json.side_effect = mock_complete_json

        # 1. First run with Telugu (same as doc source language)
        res_te = summarize_with_openrouter(
            doc,
            ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="te"),
            mock_service,
        )
        self.assertEqual(res_te["language"], "te")
        self.assertTrue(self._has_script(res_te["executive_summary"], "te"))
        self.assertFalse(self._has_script(res_te["executive_summary"], "kn"))

        # 2. Switch directly to Kannada
        res_kn = summarize_with_openrouter(
            doc,
            ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="kn"),
            mock_service,
        )
        self.assertEqual(res_kn["language"], "kn")
        self.assertTrue(self._has_script(res_kn["executive_summary"], "kn"))
        self.assertFalse(self._has_script(res_kn["executive_summary"], "te"))

        # 3. Switch back to Telugu
        res_te_again = summarize_with_openrouter(
            doc,
            ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="te"),
            mock_service,
        )
        self.assertEqual(res_te_again["language"], "te")
        self.assertTrue(self._has_script(res_te_again["executive_summary"], "te"))
        self.assertFalse(self._has_script(res_te_again["executive_summary"], "kn"))

    def test_direct_target_generation_uses_target_language_before_translation(self):
        doc = {"id": "doc_direct", "text": self.english_sentence, "chunks": make_chunks(self.english_sentence, []), "pages": [], "language": "en"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"
        target_claim = self.telugu_sentence

        def complete_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.english_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                self.assertIn("requested target language", system)
                return {"findings": [{"text": target_claim, "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                self.assertIn("Telugu", system)
                return {"results": [{"id": "claim_01", "status": "SUPPORTED", "source_chunk_ids": ["chunk_001"], "reason": "à°†à°§à°¾à°°à°‚à°¤à±‹ à°¸à°°à°¿à°ªà±‹à°¤à±à°‚à°¦à°¿", "confidence": 0.9}]}
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        result = summarize_with_openrouter(doc, ProcessOptions(language="te"), service)
        self.assertEqual(result["language"], "te")
        self.assertEqual(result["translation_validation_status"], "generated_directly_in_target_language")
        self.assertEqual(result["language_context"], {"source_language": "en", "requested_output_language": "te", "target_language": "te", "translation_required": True})
        self.assertEqual(result["claims"][0]["status"], "SUPPORTED")
        self.assertFalse(any("Translate the following content" in call.args[0] for call in service.complete_json.call_args_list))

    def test_regeneration_keeps_direct_target_language(self):
        doc = {"id": "doc_direct_revision", "text": self.english_sentence, "chunks": make_chunks(self.english_sentence, []), "pages": [], "language": "en"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"
        validation_calls = 0

        def complete_json(system, user, max_tokens=1800):
            nonlocal validation_calls
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.english_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "validation_feedback" in user:
                self.assertIn("Telugu", user)
                self.assertIn("requested target language", system)
                return {"findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                return {"findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                validation_calls += 1
                return {"results": [{"id": "claim_01", "status": "UNSUPPORTED" if validation_calls == 1 else "SUPPORTED", "source_chunk_ids": ["chunk_001"], "reason": "à°¸à°°à°¿à°ªà±‹à°¤à±à°‚à°¦à°¿", "confidence": 0.9}]}
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        result = summarize_with_openrouter(doc, ProcessOptions(language="te"), service)
        self.assertEqual(result["language"], "te")
        self.assertEqual(result["stats"]["regeneration_count"], 1)
        self.assertEqual(result["claims"][0]["status"], "SUPPORTED")
        self.assertEqual(validation_calls, 2)

    def test_direct_generation_language_matrix_covers_all_summary_sections(self):
        cases = [
            ("te", "te", self.telugu_text, self.telugu_sentence),
            ("te", "hi", self.telugu_text, self.hindi_sentence),
            ("te", "en", self.telugu_text, self.english_sentence),
            ("en", "te", self.english_sentence, self.telugu_sentence),
            ("en", "hi", self.english_sentence, self.hindi_sentence),
            ("kn", "hi", self.kannada_sentence, self.hindi_sentence),
        ]
        categories = ("findings", "facts", "risks", "decisions", "recommendations", "action_items")
        for source_lang, target_lang, source_text, target_sentence in cases:
            with self.subTest(source=source_lang, target=target_lang):
                doc = {"id": f"doc_{source_lang}_{target_lang}", "text": source_text, "chunks": make_chunks(source_text, []), "pages": [], "language": source_lang}
                service = MagicMock(spec=OpenRouterService)
                service.provider_name = "openrouter"
                service.model = "mock"

                def complete_json(system, user, max_tokens=1800):
                    if "Extract source-grounded notes" in system:
                        return {"notes": [{"text": source_text, "source_chunk_ids": ["chunk_001"]}]}
                    if "enterprise document intelligence platform" in system:
                        return {category: [{"text": target_sentence, "source_chunk_ids": ["chunk_001"]}] for category in categories}
                    if "Validate each claim strictly" in system:
                        claims = json.loads(user)["claims"]
                        return {"results": [{"id": claim["id"], "status": "SUPPORTED", "source_chunk_ids": claim["cited_chunk_ids"], "reason": target_sentence, "confidence": 0.95} for claim in claims]}
                    self.fail(f"Unexpected model stage: {system[:100]}")

                service.complete_json.side_effect = complete_json
                result = summarize_with_openrouter(doc, ProcessOptions(language=target_lang), service)
                self.assertEqual(result["language"], target_lang)
                self.assertEqual(result["source_language"], source_lang)
                self.assertTrue(verify_script(result["executive_summary"], target_lang))
                for category in categories:
                    self.assertTrue(result[category], category)
                    self.assertTrue(all(verify_script(value, target_lang) for value in result[category]), category)
                self.assertTrue(all(verify_script(claim["claim"], target_lang) for claim in result["claims"]))

    def test_translation_validation_failure_falls_back_without_regeneration(self):
        doc = {"id": "doc_mismatch", "text": self.telugu_text, "chunks": make_chunks(self.telugu_text, []), "pages": [], "language": "te"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"
        validation_calls = 0

        def complete_json(system, user, max_tokens=1800):
            nonlocal validation_calls
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                return {"findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "Translate the following content into Kannada" in system:
                return {"findings": [self.kannada_sentence]}
            if "Validate each claim strictly" in system:
                validation_calls += 1
                status = "SUPPORTED" if validation_calls == 1 else "UNSUPPORTED"
                refs = ["chunk_001"] if status == "SUPPORTED" else []
                return {"results": [{"id": "claim_01", "status": status, "source_chunk_ids": refs, "reason": "Mismatch", "confidence": 0.2}]}
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        with patch("app.openrouter._get_fallback_llm_service", return_value=None):
            result = summarize_with_openrouter(doc, ProcessOptions(language="kn"), service)
        self.assertEqual(result["language"], "te")
        self.assertEqual(result["translation_validation_status"], "translation_validation_failed_source_fallback")
        self.assertEqual(result["claims"][0]["status"], "SUPPORTED")
        self.assertTrue(self._has_script(result["claims"][0]["claim"], "te"))
        self.assertEqual(result["stats"]["regeneration_count"], 0)
        self.assertEqual(validation_calls, 2)

    def test_validation_provider_failure_retries_three_times_and_keeps_summary(self):
        doc = {"id": "doc_validation_unavailable", "text": self.english_sentence, "chunks": make_chunks(self.english_sentence, []), "pages": [], "language": "en"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"

        def complete_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.english_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                return {"findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                raise RuntimeError("validator provider unavailable")
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        result = summarize_with_openrouter(doc, ProcessOptions(language="en"), service)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["validation_status"], "unavailable")
        self.assertEqual(result["claims"][0]["status"], "UNAVAILABLE")
        self.assertTrue(result["executive_summary"])
        self.assertEqual(result["stats"]["regeneration_count"], 0)
        self.assertEqual(result["stats"]["validation_attempts"], 3)
        self.assertTrue(any("after 3 attempts" in message for message in result["limitations"]))

    def test_validation_stops_after_three_attempts_and_returns_latest_summary(self):
        doc = {"id": "doc_validation_partial", "text": self.english_sentence, "chunks": make_chunks(self.english_sentence, []), "pages": [], "language": "en"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"
        validation_calls = 0
        regenerations = 0

        def complete_json(system, user, max_tokens=1800):
            nonlocal validation_calls, regenerations
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.english_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                payload = json.loads(user)
                if "current_summary" in payload:
                    regenerations += 1
                    self.assertIn("Do not invent information", payload["instruction"])
                    self.assertIn("chunk_001", json.dumps(payload["source_notes"]))
                return {"findings": [{"text": f"Draft version {regenerations + 1} says revenue doubled.", "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                validation_calls += 1
                payload = json.loads(user)
                self.assertIn("chunk_001", payload["claims"][0]["cited_chunk_ids"])
                return {"results": [{"id": "claim_01", "status": "UNSUPPORTED", "source_chunk_ids": [], "reason": "Not in source", "confidence": 0.1}]}
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        result = summarize_with_openrouter(doc, ProcessOptions(language="en"), service)
        self.assertEqual(validation_calls, 3)
        self.assertEqual(regenerations, 2)
        self.assertEqual(result["stats"]["validation_attempts"], 3)
        self.assertEqual(result["stats"]["regeneration_count"], 2)
        self.assertIn("Draft version 3", result["executive_summary"])
        self.assertEqual(result["validation_status"], "failed")
        self.assertTrue(any("latest usable summary is shown" in message for message in result["limitations"]))

    def test_zero_claims_is_distinct_from_validation_unavailable(self):
        doc = {"id": "doc_no_claims", "text": self.english_sentence, "chunks": make_chunks(self.english_sentence, []), "pages": [], "language": "en"}
        service = MagicMock(spec=OpenRouterService)
        service.provider_name = "openrouter"
        service.model = "mock"
        validation_calls = 0

        def complete_json(system, user, max_tokens=1800):
            nonlocal validation_calls
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.english_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                return {"executive_summary": "The report describes governance after a leadership transition."}
            if "Validate each claim strictly" in system:
                validation_calls += 1
            self.fail(f"Unexpected model stage: {system[:100]}")

        service.complete_json.side_effect = complete_json
        result = summarize_with_openrouter(doc, ProcessOptions(language="en"), service)
        self.assertEqual(result["validation_status"], "no_claims")
        self.assertEqual(result["source_validation_status"], "no_claims")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["stats"]["validation_attempts"], 0)
        self.assertTrue(result["executive_summary"])
        self.assertEqual(validation_calls, 0)

    def test_fallback_rejects_telugu_when_kannada_requested(self):
        """If the translation model fails or erroneously returns Telugu, never silently substitute Telugu."""
        doc = {
            "id": "doc_fb_01",
            "text": self.telugu_text,
            "chunks": make_chunks(self.telugu_text, []),
            "pages": [],
            "language": "te",
        }

        mock_service = MagicMock(spec=OpenRouterService)
        mock_service.provider_name = "openrouter"
        mock_service.model = "openai/gpt-4o-mini"

        # Model erroneously returns Telugu text for a Kannada translation request
        def mock_complete_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            elif "enterprise document intelligence platform" in system:
                return {
                    "findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}],
                    "facts": [], "risks": [], "decisions": [], "recommendations": [], "action_items": [],
                }
            elif "Validate each claim strictly" in system:
                return {
                    "results": [
                        {
                            "id": "claim_01",
                            "status": "SUPPORTED",
                            "source_chunk_ids": ["chunk_001"],
                            "reason": "Direct evidence match.",
                            "confidence": 0.95,
                        }
                    ]
                }
            elif "Translate the following content into Kannada" in system:
                # Returns Telugu text erroneously instead of Kannada
                return {
                    "validated_summary_fields": {
                        "findings": [self.telugu_sentence],
                    }
                }
            return {}

        mock_service.complete_json.side_effect = mock_complete_json

        # Ensure fallback LLM service is not providing Kannada here
        with patch("app.openrouter._get_fallback_llm_service", return_value=None):
            result = summarize_with_openrouter(
                doc,
                ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="kn"),
                mock_service,
            )

        self.assertEqual(result["language"], "te")
        self.assertEqual(result["requested_language"], "kn")
        self.assertEqual(result["translation_validation_status"], "translation_unavailable_source_fallback")
        self.assertTrue(result["executive_summary"])
        self.assertTrue(self._has_script(result["executive_summary"], "te"), "A failed translation should preserve the usable source-language summary")

    def test_extractive_fallback_preserves_source_summary_when_translation_unavailable(self):
        """When local mode cannot translate, return useful Telugu with its actual language marked."""
        doc = {
            "id": "doc_extractive_te",
            "text": self.telugu_text,
            "chunks": make_chunks(self.telugu_text, []),
            "pages": [],
            "language": "te",
        }

        result = process_extractively(
            doc,
            ProcessOptions(summary_depth="standard", purpose="Research", audience="Research team", language="kn"),
        )

        self.assertIn(result["status"], {"validated", "partial"})
        self.assertTrue(result["executive_summary"])
        self.assertEqual(result["requested_language"], "kn")
        self.assertEqual(result["source_language"], "te")
        self.assertEqual(result["language"], "te")
        self.assertEqual(result["translation_validation_status"], "translation_unavailable_source_fallback")
        self.assertTrue(self._has_script(result["executive_summary"], "te"))

    def test_translate_for_audio_strictly_checks_script(self):
        """translate_for_audio must strictly verify the output script and reject wrong language output."""
        mock_service = MagicMock(spec=OpenRouterService)
        mock_service.provider_name = "openrouter"
        mock_service.complete.return_value = self.kannada_sentence

        # Kannada translation returns Kannada script
        kannada_out = translate_for_audio("Some English text", "kn", mock_service)
        self.assertEqual(kannada_out, self.kannada_sentence)

        # If LLM returns Telugu for a Kannada audio translation request, it must raise ValueError
        mock_service.complete.return_value = self.telugu_sentence
        with patch("app.openrouter._get_fallback_llm_service", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                translate_for_audio("Some English text", "kn", mock_service)
            self.assertIn("Kannada translation is currently unavailable", str(ctx.exception))


    def test_local_translation_fallback_preserves_telugu_to_hindi_direction(self):
        doc = {"id": "doc_local_te_hi", "text": self.telugu_text, "chunks": make_chunks(self.telugu_text, []), "pages": [], "language": "te"}
        primary = MagicMock(spec=OpenRouterService)
        primary.provider_name = "openrouter"
        primary.model = "mock"

        def primary_json(system, user, max_tokens=1800):
            if "Extract source-grounded notes" in system:
                return {"notes": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "enterprise document intelligence platform" in system:
                # Force target-script retry to source generation, then exercise the translation path.
                return {"findings": [{"text": self.telugu_sentence, "source_chunk_ids": ["chunk_001"]}]}
            if "Validate each claim strictly" in system:
                claims = json.loads(user)["claims"]
                return {"results": [{"id": c["id"], "status": "SUPPORTED", "source_chunk_ids": c["cited_chunk_ids"], "reason": "validated", "confidence": 0.9} for c in claims]}
            if "Translate the following content into Hindi" in system:
                raise RuntimeError("primary translation unavailable")
            self.fail(f"Unexpected primary stage: {system[:100]}")

        primary.complete_json.side_effect = primary_json
        local = MagicMock()
        local.provider_name = "ollama-local"
        local.model = "llama-local"
        hindi = self.hindi_sentence
        def local_json(system, user, max_tokens=1800):
            payload = json.loads(user)
            self.assertEqual(payload["source_language"], "te")
            self.assertEqual(payload["target_language"], "hi")
            return {"findings": [hindi]}
        local.complete_json.side_effect = local_json

        with patch("app.openrouter.get_local_llm_service", return_value=local), patch("app.openrouter._get_fallback_llm_service", return_value=None):
            result = summarize_with_openrouter(doc, ProcessOptions(language="hi"), primary)

        self.assertEqual(result["language"], "hi")
        self.assertEqual(result["translation_provider"], "ollama-local")
        self.assertTrue(self._has_script(result["executive_summary"], "hi"))

if __name__ == "__main__":
    unittest.main()

