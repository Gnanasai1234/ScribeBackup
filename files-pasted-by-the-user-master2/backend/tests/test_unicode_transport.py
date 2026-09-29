import sys
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import extract_text


class UnicodeEcho(BaseModel):
    text: str


class UnicodeTransportTests(unittest.TestCase):
    def test_utf8_text_extraction_preserves_multilingual_content(self):
        original = "తెలుగు నివేదిక — हिंदी सारांश — ಕನ್ನಡ — தமிழ் — മലയാളം — 😀"
        extracted, pages = extract_text("unicode.txt", original.encode("utf-8"))
        self.assertEqual(extracted, original)
        self.assertEqual(pages, [])

    def test_invalid_utf8_text_is_rejected_instead_of_replaced(self):
        with self.assertRaises(HTTPException) as raised:
            extract_text("broken.txt", b"hello \xff world")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("UTF-8", raised.exception.detail)

    def test_fastapi_json_round_trip_preserves_unicode(self):
        app = FastAPI()

        @app.post("/echo")
        def echo(body: UnicodeEcho):
            return {"text": body.text}

        original = "Résumé · తెలుగు · 日本語 · हिन्दी · 😀 — “quoted text”"
        response = TestClient(app).post("/echo", json={"text": original})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertEqual(response.json()["text"], original)
        self.assertEqual(response.content.decode("utf-8"), response.text)


if __name__ == "__main__":
    unittest.main()
