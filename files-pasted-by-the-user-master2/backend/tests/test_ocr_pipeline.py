import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class OcrPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.created_ids = []

    def tearDown(self):
        for document_id in self.created_ids:
            main.DOCS.pop(document_id, None)

    def remember_doc(self, doc):
        self.created_ids.append(doc["id"])
        main.DOCS[doc["id"]] = doc

    def test_image_ocr_text_is_added_to_upload_document_and_chunks(self):
        image = Image.new("RGB", (120, 60), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        mock_ocr = {"text": "Annual revenue increased by 24 percent.", "confidence": 91.4,
                    "records": [{"page": None, "source": "image", "text": "Annual revenue increased by 24 percent.", "ocr_confidence": 91.4}],
                    "warning": None}
        with patch.object(main, "extract_image_ocr", return_value=mock_ocr), patch.object(main, "save_doc", side_effect=self.remember_doc):
            response = self.client.post("/api/documents/upload", files={"file": ("chart.png", output.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200)
        doc = main.get_doc(response.json()["id"])
        self.assertIn("Annual revenue increased by 24 percent.", doc["text"])
        self.assertEqual(doc["ocr_records"][0]["ocr_confidence"], 91.4)
        self.assertIn("Annual revenue increased by 24 percent.", " ".join(c["text"] for c in doc["chunks"]))
        self.assertTrue(doc["chunks"][0]["contains_ocr"])
        self.assertEqual(doc["chunks"][0]["ocr_confidence"], 91.4)

    def test_pdf_page_ocr_is_merged_with_page_text(self):
        page = {"page": 1, "text": ""}
        with patch.object(main, "extract_text", return_value=("", [page])), \
             patch.object(main, "extract_pdf_ocr", return_value=([{"page": 1, "text": "[Image 1 OCR] Revenue rose 24 percent."}], [{"page": 1, "source": "image", "text": "Revenue rose 24 percent.", "ocr_confidence": 88.0}], [])), \
             patch.object(main, "save_doc", side_effect=self.remember_doc):
            response = self.client.post("/api/documents/upload", files={"file": ("scan.pdf", b"mock-pdf", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        doc = main.get_doc(response.json()["id"])
        self.assertIn("Revenue rose 24 percent.", doc["text"])
        self.assertEqual(doc["chunks"][0]["page"], 1)
        self.assertEqual(doc["ocr_records"][0]["ocr_confidence"], 88.0)

    def test_explicit_missing_language_is_reported_without_english_fallback(self):
        with patch("services.ocr_service._tesseract", return_value=((object(), {"eng"}), None)):
            from services.ocr_service import _language_config
            lang, warning = _language_config(None, {"eng"}, "te")
        self.assertEqual(lang, "")
        self.assertIn("tel", warning)

    def test_unavailable_local_ocr_returns_a_warning(self):
        from services.ocr_service import extract_image_ocr
        image = Image.new("RGB", (80, 40), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        with patch("services.ocr_service._tesseract", return_value=(None, {"status": "executable-not-found", "message": "Tesseract OCR executable was not found."})):
            result = extract_image_ocr(output.getvalue(), "en")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["records"], [])
        self.assertEqual(result["status"], "executable-not-found")
        self.assertIn("Tesseract OCR executable was not found", result["warning"])

    def test_ocr_failure_preserves_normally_extracted_pdf_text(self):
        normal_text = "The native PDF paragraph remains usable even if image OCR fails."
        pages = [{"page": 1, "text": normal_text}]
        warning = "Tesseract executable exists but OCR processing failed."
        with patch.object(main, "extract_text", return_value=(normal_text, pages)), \
             patch.object(main, "extract_pdf_ocr", return_value=(pages, [], [warning])), \
             patch.object(main, "save_doc", side_effect=self.remember_doc):
            response = self.client.post("/api/documents/upload", files={"file": ("mixed.pdf", b"mock-pdf", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        doc = main.get_doc(response.json()["id"])
        self.assertIn("native PDF paragraph", doc["text"])
        self.assertEqual(doc["ocr_warnings"], [warning])


if __name__ == "__main__":
    unittest.main()
