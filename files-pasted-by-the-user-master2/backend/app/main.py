from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import AliasChoices, BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load the existing project .env before any service (including OCR) is configured.
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR.parent.parent / ".env")

from database.repositories import (database_schema_status, get_document as load_document_from_db,
    list_documents as list_documents_from_db, save_document, save_question, delete_document as delete_document_from_db,
    find_duplicate_document, search_documents as search_documents_from_db,
    save_document_chunk_embeddings,
    create_upload_batch, update_upload_batch, get_upload_batch, record_history, list_history as list_history_from_db,
    list_history_documents as list_history_documents_from_db,
    save_combined_summary, get_combined_summary)
from .auth import router as auth_router, get_current_user

from .openrouter import OpenRouterService, _get_fallback_llm_service, answer_with_openrouter, summarize_with_openrouter, translate_for_audio
from models.language_context import LanguageContext
from .textmatch import term_hits, term_present
from workflow.analysis import run_analysis_workflow
from dataset.govreport_evaluator import evaluate_record
from dataset.govreport_loader import SPLITS as GOVREPORT_SPLITS, get_record as get_govreport_record, stream_records as stream_govreport_records
from dataset.govreport_service import to_document as govreport_to_document
from services.tts_service import (
    synthesize as synthesize_audio,
    synthesize_with_timings,
    voices_for as tts_voices_for,
    get_default_voice_for,
    estimate_timings,
)
from services.embedding_service import cosine_similarity, get_embedding_service
from services.openai_gateway import get_openai_gateway
from services.local_fallbacks import get_local_llm_service
from services.speech_to_text import (
    AudioRecordingEmpty,
    InvalidAudioRecording,
    NoTranscription,
    SpeechModelUnavailable,
    transcribe_local_audio,
)
from services.ocr_service import configure_tesseract, extract_image_ocr, extract_pdf_ocr, ocr_status
from services.guardrails import config_int, validate_upload, validate_document_text
from services.summary_export import (
    ExportTooLarge,
    SUPPORTED_EXPORT_FORMATS,
    SummaryNotAvailable,
    make_combined_export_payload,
    make_export_payload,
    render_summary,
)

MAX_BYTES = config_int("MAX_UPLOAD_MB", 25) * 1024 * 1024
ALLOWED = {".pdf", ".docx", ".txt", ".json", ".csv", ".jpg", ".jpeg", ".png"}
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}
AUTO_LANGUAGE_VALUES = {"", "auto", "original", "source"}
DOCS: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger("scribe.openrouter")
TTS_LOGGER = logging.getLogger("scribe.tts")


def save_doc(doc: dict[str, Any]) -> None:
    doc_id = doc.get("id")
    if not doc_id:
        return
    save_document(doc)
    DOCS[doc_id] = doc

async def verify_tesseract_at_startup() -> None:
    """Check executable availability/version only; OCR stays lazy until an upload."""
    diagnostic = configure_tesseract(force=True)
    if diagnostic["status"] == "ready":
        LOGGER.info("Local Tesseract OCR is ready (version %s).", diagnostic.get("version", "unknown"))
    else:
        LOGGER.warning("Local Tesseract OCR unavailable status=%s", diagnostic["status"])


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    database_status = database_schema_status()
    if not database_status["connected"]:
        raise RuntimeError(
            "Could not connect to the configured MySQL database 'scribe' "
            f"(error type: {database_status.get('error_type', 'unknown')}). Check DATABASE_URL and MySQL availability."
        )
    LOGGER.info(
        "Database connection established host=%s port=%s database=%s",
        database_status.get("host"), database_status.get("port"), database_status.get("database"),
    )
    if not database_status["schema_ready"]:
        raise RuntimeError(
            "MySQL is reachable, but the required Scribe schema is not ready: "
            f"missing_tables={database_status['missing_tables']} "
            f"missing_columns={database_status['missing_columns']} "
            f"configured_database={database_status['configured_database']!r}. "
            "Run `..\\venv\\Scripts\\alembic.exe upgrade head` from the backend directory."
        )
    LOGGER.info("Database schema ready (%d mapped tables).", database_status["required_table_count"])
    await verify_tesseract_at_startup()
    yield


app = FastAPI(
    title="Scribe Intelligence API",
    version="0.1.0",
    description="Local-first, source-grounded document intelligence MVP",
    lifespan=app_lifespan,
)
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()
if len(SESSION_SECRET) < 32:
    raise RuntimeError("Set SESSION_SECRET to a random value of at least 32 characters in the environment.")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="scribe_session",
                   same_site="lax", https_only=os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=[
        "Content-Disposition",
        "X-TTS-Provider",
        "X-TTS-Voice",
        "X-TTS-Language",
        "X-TTS-Fallback-Used",
        "X-TTS-Timings",
        "X-TTS-Timings-Base64",
        "X-TTS-Narration-Text-Base64",
    ],
)
app.include_router(auth_router)


@app.exception_handler(Exception)
async def safe_unhandled_error(_: Request, exc: Exception):
    # Do not expose exception text, prompts, credentials, or filesystem paths to clients.
    LOGGER.error("Unhandled request failure type=%s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Document processing failed."})


@app.exception_handler(RequestValidationError)
async def safe_validation_error(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid request data."})


class ProcessOptions(BaseModel):
    summary_depth: str = Field(default="standard", validation_alias=AliasChoices("summary_depth", "depth"))
    summary_depth_requested: str | None = Field(default=None, exclude=True)
    summary_mode: str | None = Field(default=None, exclude=True)
    output_language_requested: str | None = Field(default=None, exclude=True)
    source_language: str | None = Field(default=None, exclude=True)
    language_context: LanguageContext | None = Field(default=None, exclude=True)
    purpose: str = Field(default="Executive briefing", max_length=160)
    language: str = Field(default="auto", validation_alias=AliasChoices("output_language", "language"))
    audience: str = Field(default="Leadership", max_length=160)
    focus: list[str] = Field(default_factory=list, max_length=20)
    custom_sections: list[str] | None = Field(default=None, max_length=12)
    custom_detail: str = "standard"

    model_config = {"populate_by_name": True}

    @field_validator("language", mode="before")
    @classmethod
    def normalize_output_language(cls, value: Any) -> str:
        language = str(value or "auto").strip().lower()
        if language in AUTO_LANGUAGE_VALUES:
            return "auto"
        if language not in LANGUAGES:
            raise ValueError(f"output_language must be auto or one of: {', '.join(sorted(LANGUAGES))}")
        return language

    @field_validator("summary_depth")
    @classmethod
    def validate_depth(cls, value: str) -> str:
        if value not in {"quick", "standard", "deep", "auto", "custom"}:
            raise ValueError("summary_depth must be quick, standard, deep, auto, or custom")
        return value

    @field_validator("custom_detail")
    @classmethod
    def validate_custom_detail(cls, value: str) -> str:
        if value not in {"quick", "standard", "deep"}:
            raise ValueError("custom_detail must be quick, standard, or deep")
        return value


class LocalLLMTestRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: Any) -> str:
        return str(value or "").strip()


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    output_language: str = "auto"  # defaults to the source language unless explicitly selected

    @field_validator("output_language", mode="before")
    @classmethod
    def validate_output_language(cls, value: Any) -> str:
        language = str(value or "auto").strip().lower()
        if language in AUTO_LANGUAGE_VALUES:
            return "auto"
        if language not in LANGUAGES:
            raise ValueError(f"output_language must be auto or one of: {', '.join(sorted(LANGUAGES))}")
        return language


class AudioRequest(BaseModel):
    language: str = "auto"
    voice_id: str = "browser-default"
    provider: str | None = None
    rate: float = Field(default=1.0, ge=0.7, le=1.3)


class EvaluationRunRequest(BaseModel):
    document_id: str


class DepthComparisonRequest(BaseModel):
    purpose: str = "Executive briefing"
    audience: str = "Leadership"
    language: str = "auto"


class BatchProcessRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    options: ProcessOptions = Field(default_factory=ProcessOptions)


def extract_text(filename: str, data: bytes) -> tuple[str, list[dict[str, Any]]]:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8-sig", errors="replace"), []
    if ext == ".csv":
        import csv
        import io
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
        return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows if any(cell.strip() for cell in row)), []
    if ext == ".json":
        try:
            value = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, "The JSON file is malformed or unreadable.") from exc
        return json.dumps(value, ensure_ascii=False, indent=2), []
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise HTTPException(400, "This PDF is password protected.")
            if len(reader.pages) > config_int("MAX_PDF_PAGES", 2000):
                raise HTTPException(413, "The document exceeds the configured page limit.")
            pages = [{"page": i, "text": (page.extract_text() or "").strip()} for i, page in enumerate(reader.pages, 1)]
            return "\n\n".join(p["text"] for p in pages), pages
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "This PDF could not be read. Try exporting it as text or DOCX.") from exc
    if ext == ".docx":
        try:
            from docx import Document
            import io
            document = Document(io.BytesIO(data))
            return "\n\n".join(p.text for p in document.paragraphs if p.text.strip()), []
        except Exception as exc:
            raise HTTPException(400, "This DOCX could not be read. Check that the file is not corrupted.") from exc
    raise HTTPException(415, "Supported formats are PDF, DOCX, TXT, CSV, JSON, JPG, JPEG, and PNG.")


def normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    # Pick the dominant supported writing system. Latin text is English by
    # default; supported non-Latin scripts are never silently relabeled English.
    script_ranges = (
        ("hi", 0x0900, 0x097F),  # Devanagari (Hindi/Marathi)
        ("bn", 0x0980, 0x09FF),  # Bengali
        ("pa", 0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
        ("gu", 0x0A80, 0x0AFF),  # Gujarati
        ("ta", 0x0B80, 0x0BFF),  # Tamil
        ("te", 0x0C00, 0x0C7F),  # Telugu
        ("kn", 0x0C80, 0x0CFF),  # Kannada
        ("ml", 0x0D00, 0x0D7F),  # Malayalam
        ("ur", 0x0600, 0x06FF),  # Arabic script (Urdu)
    )
    letters = [ord(c) for c in text if c.isalpha()]
    if letters:
        text_points = [ord(c) for c in text if not c.isspace()]
        counts = [(sum(low <= n <= high for n in text_points), code) for code, low, high in script_ranges]
        count, code = max(counts, default=(0, "und"))
        latin_count = sum(n < 0x0250 for n in letters)
        if count and count >= latin_count:
            return code
        if latin_count:
            return "en"
        return "und"
    return "und"


def resolve_output_language(requested: str | None, source_language: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in AUTO_LANGUAGE_VALUES:
        return source_language
    return normalized


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+|\n+", text) if len(s.strip()) > 15]


def make_chunks(text: str, pages: list[dict[str, Any]], chunk_size: int = 900, overlap: int = 120, document_id: str | None = None) -> list[dict[str, Any]]:
    chunk_size = min(chunk_size, config_int("MAX_CHUNK_CHARS", 4000))
    overlap = min(overlap, max(0, chunk_size // 5))
    chunks: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", cursor + chunk_size // 2, end)
            if boundary > cursor:
                end = boundary
        piece = text[cursor:end].strip()
        if piece:
            chunks.append({"id": f"chunk_{len(chunks)+1:03d}", "text": piece})
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    headings = [(match.start(), (match.group(1) or match.group(2)).strip().lstrip("# ")) for match in re.finditer(r"(?m)^(?:#{1,3}\s+(.{2,120})|([A-Z][A-Z 0-9'-]{5,100}))$", text)]
    for number, chunk in enumerate(chunks, start=1):
        chunk_prefix = re.sub(r"\s+", " ", chunk["text"][:45]).strip()
        matching_page = next((p for p in pages if chunk_prefix in re.sub(r"\s+", " ", p.get("text", "")).strip()), None)
        chunk["page"] = matching_page.get("page") if matching_page else None
        ocr_record = matching_page.get("ocr") if matching_page else None
        if ocr_record and ocr_record.get("text") and ocr_record["text"][:min(40, len(ocr_record["text"]))] in chunk["text"]:
            chunk["contains_ocr"] = True
            chunk["ocr_confidence"] = ocr_record.get("ocr_confidence")
            chunk["source"] = "image"
        chunk_start = text.find(chunk["text"][:45])
        prior_headings = [entry for entry in headings if entry[0] <= chunk_start] if chunk_start >= 0 else []
        chunk["section"] = prior_headings[-1][1] if prior_headings else "Document text"
        chunk["chunk_number"] = number
        if document_id:
            chunk["documentId"] = document_id
            chunk["document_id"] = document_id
        chunk["chunk_id"] = chunk["id"]
        chunk["content"] = chunk["text"]
        chunk["language"] = detect_language(chunk["text"])
    return chunks


def keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text.lower())
    stop = set("this that with from have were will their there about would could should into more than been also they them then when what which while document report summary section findings these those".split())
    return [w for w, _ in Counter(w for w in words if w not in stop).most_common(12)]


def resolve_summary_depth(requested: str, doc: dict[str, Any], options: ProcessOptions | None = None) -> str:
    """Choose depth from document length plus section/numeric complexity clues."""
    if requested != "auto":
        if requested == "custom":
            return (options or ProcessOptions()).custom_detail
        return requested
    chunk_count = len(doc.get("chunks", []))
    chunk_sections = {chunk.get("section") for chunk in doc.get("chunks", []) if chunk.get("section") and chunk.get("section") != "Document text"}
    text = doc.get("text", "")
    numeric_ratio = sum(char.isdigit() for char in text) / max(1, len(text))
    if chunk_count <= 8:
        return "standard" if len(chunk_sections) >= 6 or numeric_ratio >= .035 else "quick"
    if chunk_count <= 30:
        return "deep" if len(chunk_sections) >= 12 or numeric_ratio >= .05 else "standard"
    return "deep"


def process_extractively(doc: dict[str, Any], options: ProcessOptions) -> dict[str, Any]:
    text = doc["text"]
    chunks = doc["chunks"]
    requested_output_language = options.output_language_requested or options.language
    output_language = resolve_output_language(options.language, doc.get("language", "und"))
    language_context = options.language_context or LanguageContext.resolve(doc.get("language", "und"), requested_output_language)
    if options.language != output_language:
        options = options.model_copy(update={"language": output_language})
        options.output_language_requested = requested_output_language
    sentences = split_sentences(text)
    terms = set(keywords(text))
    scored = sorted(sentences, key=lambda s: (term_hits(s, terms), min(len(s), 300)), reverse=True)
    
    requested_mode = getattr(options, "summary_mode", None) or getattr(options, "summary_depth_requested", None) or options.summary_depth
    words = len(re.findall(r"\b\w+\b", text, re.UNICODE))
    chunk_count = len(chunks)
    sent_count = len(sentences)
    is_long_doc = words > 1400 or chunk_count > 16 or sent_count >= 50
    is_short_doc = (words < 500 or chunk_count <= 5 or sent_count <= 22) and not is_long_doc

    if requested_mode == "auto":
        # Dynamic Auto targets based on document length:
        # Short document (< 600 words or <= 8 chunks): 15-25 lines
        # Medium document (600-2500 words or 9-30 chunks): 25-45 lines
        # Long document (> 2500 words or > 30 chunks): 45-70+ lines
        if is_short_doc:
            target_count = min(len(sentences), 20)
        elif is_long_doc:
            target_count = min(len(sentences), 65)
        else:
            target_count = min(len(sentences), 35)
    elif requested_mode == "custom":
        detail = getattr(options, "custom_detail", "standard")
        if detail == "quick":
            target_count = min(len(sentences), 15)
        elif detail == "deep":
            target_count = min(len(sentences), 60)
        else:
            target_count = min(len(sentences), 28)
    else:
        depth = options.summary_depth
        if depth == "quick":
            target_count = min(len(sentences), 12)
            if len(sentences) > 1 and target_count >= len(sentences):
                target_count = max(1, len(sentences) - 2)
        elif depth == "standard":
            target_count = min(len(sentences), 25)
        elif depth == "deep":
            target_count = min(len(sentences), 65 if is_long_doc else 50)
        else:
            target_count = min(len(sentences), 25)

    selected = []
    seen = set()
    for sentence in scored:
        norm = re.sub(r"\W", "", sentence.lower())
        if norm not in seen:
            selected.append(sentence)
            seen.add(norm)
        if len(selected) >= target_count:
            break
    if not selected and text:
        selected = [text[:500]]

    claims = []
    categorized: dict[str, list[str]] = {
        "findings": [], "facts": [], "risks": [], "decisions": [], "recommendations": [], "action_items": []
    }
    for i, sentence in enumerate(selected, 1):
        source = next((c for c in chunks if sentence.lower() in c["text"].lower()), None)
        source_refs = ([{"document_id": doc.get("id"), "page": source.get("page") or "unavailable", "section": source.get("section", "Document text"), "chunk_id": source["id"]}] if source else [])
        lower_s = sentence.lower()
        if any(w in lower_s for w in ("risk", "threat", "vulnerab", "hazard", "concern", "expos")):
            cat = "risks"
        elif any(w in lower_s for w in ("recommend", "should", "suggest", "advis", "propos")):
            cat = "recommendations"
        elif any(w in lower_s for w in ("decid", "approv", "adopt", "determin", "resolv")):
            cat = "decisions"
        elif any(w in lower_s for w in ("action", "next step", "implement", "schedul", "deadlin", "mileston")):
            cat = "action_items"
        elif any(char.isdigit() for char in sentence) or any(w in lower_s for w in ("percent", "%", "total", "increas", "decreas", "million", "billion")):
            cat = "facts"
        else:
            cat = "findings"
        categorized[cat].append(sentence)
        claims.append({
            "id": f"claim_{i:02d}", "category": cat, "claim": sentence,
            "status": "SUPPORTED" if source else "UNAVAILABLE",
            "sources": source_refs,
            "evidence": source["text"] if source else "",
            "confidence": 1.0 if source else None,
            "reason": "Exact sentence retained from extracted source text." if source else "No source passage was available to validate this claim."
        })
    if requested_mode == "custom" and options.custom_sections is not None:
        selected_categories = {name.strip().lower().replace(" ", "_") for name in options.custom_sections}
        claims = [claim for claim in claims if claim["category"] in selected_categories]
        for category in categorized:
            if category not in selected_categories:
                categorized[category] = []
        for index, claim in enumerate(claims, 1):
            claim["id"] = f"claim_{index:02d}"
    if not categorized["findings"] and selected and (requested_mode != "custom" or options.custom_sections is None):
        categorized["findings"] = selected[:min(len(selected), 15)]
    local_groundedness = (
        None if not claims or any(c["status"] == "UNAVAILABLE" for c in claims)
        else round(sum(c["status"] == "SUPPORTED" for c in claims) / len(claims), 2)
    )

    if options.language != doc["language"]:
        target_name = LANGUAGES.get(options.language, options.language)
        source_name = LANGUAGES.get(doc["language"], doc["language"])
        _translation_fallback_msg = (
            f"\u26a0\ufe0f Translation into {target_name} was unavailable. "
            f"Showing the usable source-language summary ({source_name}) instead."
        )
        result = {
            "depth": options.summary_depth, "summary_mode": options.summary_mode or options.summary_depth,
            "summary_depth_requested": options.summary_depth_requested,
            "summary_depth_resolved": options.summary_depth, "purpose": options.purpose, "audience": options.audience,
            "language": doc["language"], "output_language": doc["language"],
            "actual_output_language": doc["language"],
            "requested_language": requested_output_language, "requested_output_language": requested_output_language,
            "source_language": doc["language"], "language_context": language_context.as_dict(),
            "source_validation_status": "passed" if claims and all(c["status"] == "SUPPORTED" for c in claims) else ("partial" if claims else "no_claims"),
            "validation_status": "local_basic" if claims else "no_claims",
            "validation_state": "PARTIALLY_VALIDATED" if claims else "NO_CLAIMS",
            "validation_warning": bool(claims),
            "warning_message": "Only deterministic local source matching was available; review the summary and claims against the original document." if claims else None,
            "validation_attempts": 0,
            "supported_claims_count": sum(c["status"] == "SUPPORTED" for c in claims),
            "unsupported_claims_count": sum(c["status"] != "SUPPORTED" for c in claims),
            "validation_provider": "local_basic",
            "translation_validation_status": "translation_unavailable_source_fallback",
            "translation_warning": True,
            "translation_warning_message": _translation_fallback_msg,
            "translation_failed_fallback": True,
            "status": "translation_unavailable",
            "executive_summary": " ".join(selected),
            **categorized,
            "claims": claims, "quality": {"groundedness": local_groundedness, "claims_checked": len(claims), "evaluation": "Deterministic source-overlap check; translation unavailable, showing the source-language summary."},
            "stats": {"pages": len(doc.get("pages", [])) or None, "words": words, "estimated_tokens": round(words * 1.35), "sections": len(re.findall(r"(?m)^(?:#{1,3} |[A-Z][A-Z 0-9'-]{5,100})$", text)), "chunks": len(chunks), "context_chunks_used": len({s.get("sources", [{}])[0].get("chunk_id") for s in claims if s.get("sources")}), "language": doc["language"], "scanned": bool(doc.get("pages") and not text.strip())},
            "limitations": [_translation_fallback_msg],
            "provider": {"name": "local fallback", "model": None, "fallback_used": True},
        }
        doc["result"] = result
        doc["options"] = options.model_dump() | {"summary_depth": options.summary_depth, "summary_mode": options.summary_mode or options.summary_depth, "output_language": options.language}
        doc["status"] = "COMPLETED"
        return result


    result = {
        "depth": options.summary_depth, "summary_mode": options.summary_mode or options.summary_depth,
        "summary_depth_requested": options.summary_depth_requested,
        "summary_depth_resolved": options.summary_depth, "purpose": options.purpose, "audience": options.audience, "language": doc["language"], "output_language": doc["language"], "requested_language": requested_output_language, "requested_output_language": requested_output_language,
        "source_language": doc["language"], "language_context": language_context.as_dict(),
        "source_validation_status": "passed" if claims and all(c["status"] == "SUPPORTED" for c in claims) else ("partial" if claims else "no_claims"),
        "validation_status": "local_basic" if claims else "no_claims",
        "validation_state": "PARTIALLY_VALIDATED" if claims else "NO_CLAIMS",
        "validation_warning": bool(claims),
        "warning_message": "Only deterministic local source matching was available; review the summary and claims against the original document." if claims else None,
        "validation_attempts": 0,
        "supported_claims_count": sum(c["status"] == "SUPPORTED" for c in claims),
        "unsupported_claims_count": sum(c["status"] != "SUPPORTED" for c in claims),
        "validation_provider": "local_basic",
        "translation_validation_status": "validated_in_source_language",
        "status": "validated" if claims and all(c["status"] == "SUPPORTED" for c in claims) else "partial",
        "executive_summary": " ".join(selected),
        **categorized,
        "claims": claims, "quality": {"groundedness": local_groundedness, "claims_checked": len(claims), "evaluation": "Deterministic source-overlap check; no learned evaluator configured."},
        "stats": {"pages": len(doc.get("pages", [])) or None, "words": words, "estimated_tokens": round(words * 1.35), "sections": len(re.findall(r"(?m)^(?:#{1,3} |[A-Z][A-Z 0-9-]{5,}$)", text)), "chunks": len(chunks), "context_chunks_used": len({s.get("sources", [{}])[0].get("chunk_id") for s in claims if s.get("sources")}), "language": doc["language"], "scanned": bool(doc.get("pages") and not text.strip())},
        "limitations": ["Summary is extractive and uses local heuristics; it does not call an LLM.", "Purpose and audience style instructions require a configured semantic model.", "Risk, decision, and action extraction require a configured semantic model and are left empty rather than guessed."],
        "provider": {"name": "local fallback", "model": None, "fallback_used": False},
    }
    doc["result"] = result
    doc["options"] = options.model_dump() | {"summary_depth": options.summary_depth, "summary_mode": options.summary_mode or options.summary_depth, "output_language": options.language}
    doc["status"] = "COMPLETED"
    return result


def process_document(doc: dict[str, Any], options: ProcessOptions) -> dict[str, Any]:
    """Use OpenRouter when configured; retain a clearly labeled local fallback."""
    doc["text"] = validate_document_text(doc.get("text", ""))
    if not doc.get("chunks"):
        doc["chunks"] = make_chunks(doc["text"], doc.get("pages", []), document_id=doc.get("id"))
    doc["chunks"] = [c for c in doc["chunks"] if isinstance(c.get("text"), str) and c["text"].strip()]
    if not doc["chunks"]:
        raise HTTPException(422, "Unable to extract meaningful text from this document.")
    source_language = detect_language(doc.get("text", "")) or doc.get("language", "und")
    doc["language"] = source_language
    requested_output_language = options.language
    language_context = LanguageContext.resolve(source_language, requested_output_language)
    target_language = resolve_output_language(requested_output_language, source_language)
    requested_depth = options.summary_depth
    resolved_depth = resolve_summary_depth(requested_depth, doc, options)
    resolved_options = options.model_copy(update={"summary_depth": resolved_depth, "language": target_language, "language_context": language_context})
    resolved_options.summary_depth_requested = requested_depth
    resolved_options.summary_mode = requested_depth
    resolved_options.output_language_requested = requested_output_language
    doc["language_context"] = language_context.as_dict()
    LOGGER.info(
        "Source language: %s | Requested output language: %s | Final output language: %s | Translation required: %s",
        source_language, requested_output_language, target_language, source_language != target_language,
    )
    LOGGER.info("[ANALYSIS] document_id=%s requested_depth=%s resolved_depth=%s output_language=%s", doc.get("id", "unknown"), requested_depth, resolved_depth, target_language)
    try:
        service = OpenRouterService()
    except RuntimeError as exc:
        service = None
        LOGGER.warning("Configured summary provider is unavailable (%s)", type(exc).__name__)

    def summarize(source: dict[str, Any], current_options: ProcessOptions) -> dict[str, Any]:
        primary_error: Exception | None = None
        local_error: Exception | None = None
        if service is not None:
            try:
                primary_result = summarize_with_openrouter(source, current_options, service)
                LOGGER.info("[LLM PROVIDER] selected=%s model=%s", service.provider_name, service.model)
                return primary_result
            except Exception as exc:
                primary_error = exc
                LOGGER.warning("Configured summary provider=%s failed type=%s; trying Ollama", service.provider_name, type(exc).__name__)
        else:
            LOGGER.warning("Configured summary provider is unavailable; trying Ollama")

        local_service = get_local_llm_service()
        if local_service is not None:
            try:
                local_result = summarize_with_openrouter(source, current_options, local_service)
                local_result["provider"] = {"name": local_service.provider_name, "model": local_service.model, "fallback_used": True}
                local_result["provider_status"] = {
                    "primary": (f"FAILED ({type(primary_error).__name__})" if primary_error else "UNAVAILABLE"),
                    "ollama": "AVAILABLE - generated summary; source checked locally",
                    "final": "OLLAMA SUMMARY",
                }
                local_result["limitations"] = ["External summary provider was unavailable; an already-installed local LLM generated this result.", *local_result.get("limitations", [])]
                LOGGER.info("[LLM PROVIDER] selected=ollama-local model=%s reason=external_unavailable_or_failed", local_service.model)
                return local_result
            except Exception as exc:
                local_error = exc
                # Include the validation reason (never prompt/source text) so a
                # structured-output rejection isn't mistaken for a network timeout.
                LOGGER.warning(
                    "Ollama summary failed model=%s timeout=%ss type=%s reason=%s; using deterministic extractive summary",
                    local_service.model,
                    local_service.timeout,
                    type(exc).__name__,
                    str(exc)[:240],
                )

        if service is not None:
            secondary_service = _get_fallback_llm_service(service)
            if secondary_service is not None:
                try:
                    secondary_result = summarize_with_openrouter(source, current_options, secondary_service)
                    secondary_result["provider"] = {"name": secondary_service.provider_name, "model": secondary_service.model, "fallback_used": True}
                    secondary_result["limitations"] = [f"Primary provider {service.provider_name} failed; configured secondary provider {secondary_service.provider_name} generated this result.", *secondary_result.get("limitations", [])]
                    LOGGER.info("[SUMMARY FALLBACK] selected=%s model=%s", secondary_service.provider_name, secondary_service.model)
                    return secondary_result
                except Exception as exc:
                    LOGGER.warning("Configured secondary LLM failed; using deterministic extractive summary (%s)", type(exc).__name__)

        fallback = process_extractively(source, current_options)
        target_name = LANGUAGES.get(current_options.language, current_options.language)
        limitations = [
            "External providers were unavailable; deterministic sentence ranking produced this source-grounded extractive summary.",
            "The local extractive fallback does not infer risks, decisions, or actions and does not translate text.",
        ]
        if current_options.language != source["language"]:
            limitations.insert(0, f"Translation into {target_name} was unavailable; the usable source-language summary is shown.")
        elif primary_error is not None:
            limitations.insert(0, f"The configured summary provider failed ({type(primary_error).__name__}); a local extractive fallback was used.")
        fallback["limitations"] = limitations
        fallback["provider"] = {"name": "local extractive fallback", "model": None, "fallback_used": True}
        fallback["provider_status"] = {
            "primary": (f"FAILED ({type(primary_error).__name__})" if primary_error else "UNAVAILABLE"),
            "ollama": (f"AVAILABLE - generation failed ({type(local_error).__name__})" if local_error else "UNAVAILABLE - model or API check failed"),
            "final": "DETERMINISTIC EXTRACTIVE SUMMARY",
        }
        LOGGER.info("[SUMMARY FALLBACK] selected=extractive source=%s final=%s", source["language"], fallback.get("language"))
        return fallback

    result = run_analysis_workflow(doc, resolved_options, summarize)
    if doc.get("ocr_detected"):
        result.setdefault("limitations", []).append("OCR text was included in the source. Check low-confidence image text against the original page.")
    result.setdefault("limitations", []).extend(w for w in doc.get("ocr_warnings", []) if w not in result.get("limitations", []))
    LOGGER.info(
        "Final output language: %s (source=%s, requested=%s, translation_status=%s)",
        result.get("language"), source_language, requested_output_language,
        result.get("translation_validation_status"),
    )
    LOGGER.info(
        "[ANALYSIS RESULT] source=%s target=%s summary_generated=%s translation_status=%s validation_status=%s regeneration_count=%s final_output=%s",
        source_language, target_language, bool(result.get("executive_summary")),
        result.get("translation_validation_status"), result.get("validation_status"),
        result.get("stats", {}).get("regeneration_count", 0), result.get("language"),
    )
    doc["result"] = result
    doc["options"] = options.model_dump() | {"summary_depth": resolved_depth, "summary_mode": requested_depth, "custom_sections": options.custom_sections, "custom_detail": options.custom_detail, "source_language": source_language, "requested_output_language": requested_output_language, "output_language": result.get("language"), "resolved_output_language": result.get("language")}
    doc["status"] = "COMPLETED"
    save_doc(doc)
    if doc.get("user_id") is not None:
        provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
        record_history(user_id=int(doc["user_id"]), document_id=str(doc["id"]),
            activity_type="summarization", title=f"Summarized {doc.get('name', 'document')}",
            status=str(result.get("status", "completed")), output_text=result.get("executive_summary"),
            language=result.get("language"), provider=provider.get("name"), model=provider.get("model"),
            extra_data={"summary_type": result.get("summary_mode", result.get("depth")),
                        "validation_status": result.get("validation_status"),
                        "fallback_used": provider.get("fallback_used", False)})
        if result.get("validation_status"):
            record_history(user_id=int(doc["user_id"]), document_id=str(doc["id"]),
                activity_type="validation", title=f"Validated {doc.get('name', 'document')}",
                status=str(result.get("validation_status")), language=result.get("language"),
                provider=result.get("validation_provider"), extra_data={
                    "claims_checked": (result.get("quality") or {}).get("claims_checked"),
                })
        requested_language = options.language if options.language not in AUTO_LANGUAGE_VALUES else source_language
        if requested_language != source_language:
            translated = result.get("language") == requested_language
            record_history(user_id=int(doc["user_id"]), document_id=str(doc["id"]),
                activity_type="translation", title=f"Translated {doc.get('name', 'document')}",
                status="completed" if translated else "fallback", language=result.get("language"),
                provider=result.get("translation_provider") or provider.get("name"),
                model=provider.get("model"), extra_data={
                    "requested_language": requested_language,
                    "validation_status": result.get("translation_validation_status"),
                })
    return result


@app.get("/api/health")
def health():
    database_status = database_schema_status()
    requested_llm_provider = os.getenv("LLM_PROVIDER", "sarvam").strip().lower() or "sarvam"
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if requested_llm_provider == "sarvam":
        active_llm_provider = "sarvam" if sarvam_key else ("openrouter" if openrouter_key else ("groq" if groq_key else "local-demo"))
    elif requested_llm_provider == "groq":
        active_llm_provider = "groq" if groq_key else ("openrouter" if openrouter_key else "local-demo")
    else:
        active_llm_provider = "openrouter" if openrouter_key else ("groq" if groq_key else "local-demo")
    local_llm_service = get_local_llm_service() if active_llm_provider == "local-demo" else None
    if local_llm_service is not None:
        active_llm_provider = "ollama-local"
    gateway_configured = get_openai_gateway().configured
    if gateway_configured:
        active_llm_provider = "openai-gateway"
        local_llm_service = None
    llm_configured = active_llm_provider != "local-demo"
    embedding_svc = get_embedding_service()
    emb_info = embedding_svc.get_info()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "groq").strip()
    vector_db_url = os.getenv("VECTOR_DB_URL", "").strip()
    openai_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "status": "ok" if database_status["schema_ready"] else "degraded",
        "mode": active_llm_provider,
        "requested_llm_provider": "openai-gateway" if gateway_configured else requested_llm_provider,
        "llm": "configured" if llm_configured else "not-configured",
        "model": (
            "gateway-managed" if active_llm_provider == "openai-gateway" else
            os.getenv("SARVAM_MODEL", "sarvam-105b") if active_llm_provider == "sarvam" else
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if active_llm_provider == "groq" else
            os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") if active_llm_provider == "openrouter" else
            local_llm_service.model if local_llm_service is not None else None
        ),
        "ocr": ocr_status(),
        "embeddings": "configured",
        "embedding_provider": "openai-gateway" if emb_info.get("gateway_configured") else embedding_provider,
        "embedding_details": emb_info,
        "vector_database": "configured" if vector_db_url else "mysql-document-chunk-vectors",
        "retrieval": "hybrid-semantic",
        "database": "connected" if database_status["connected"] else "unavailable",
        "database_name": database_status["database"],
        "database_schema": "ready" if database_status["schema_ready"] else "missing",
        "missing_tables": database_status["missing_tables"],
        "missing_columns": database_status["missing_columns"],
        "tts": {"default": os.getenv("TTS_PROVIDER", "browser").strip().lower(), "available": ["browser", "edge-neural"]},
        "stt": {
            "provider": "openai-whisper-with-local-fallback" if openai_configured else "faster-whisper-local",
            "fallback": "faster-whisper-local",
            "status": "cloud-primary-local-fallback" if openai_configured else "local-offline",
            "languages": list(LANGUAGES.keys()),
        },
    }


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Use cloud Whisper when configured, otherwise fall back to local Faster Whisper."""
    if language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of: {', '.join(sorted(LANGUAGES))}")

    max_audio_bytes = 25 * 1024 * 1024
    audio_bytes = await audio.read(max_audio_bytes + 1)
    if len(audio_bytes) > max_audio_bytes:
        raise HTTPException(413, "Audio recording exceeds 25 MiB.")
    if not audio_bytes:
        raise HTTPException(422, "Audio recording was empty.")

    async def use_local_stt(reason: str) -> dict[str, str]:
        LOGGER.info("[STT PROVIDER] selected=faster-whisper-local language=%s reason=%s", language, reason)
        try:
            local_result = await run_in_threadpool(
                transcribe_local_audio,
                audio_bytes,
                content_type=audio.content_type,
                language=language,
            )
        except AudioRecordingEmpty as exc:
            raise HTTPException(422, str(exc)) from exc
        except InvalidAudioRecording as exc:
            raise HTTPException(415, str(exc)) from exc
        except NoTranscription as exc:
            raise HTTPException(422, str(exc)) from exc
        except SpeechModelUnavailable as exc:
            LOGGER.error("Local speech-to-text unavailable model=%s type=%s", os.getenv("LOCAL_STT_MODEL", "base"), type(exc).__name__)
            raise HTTPException(503, "Local speech-to-text service is unavailable; install/cache the configured Faster Whisper model.") from exc
        except Exception as exc:
            LOGGER.warning("Local speech-to-text failed type=%s", type(exc).__name__)
            raise HTTPException(502, "Local speech-to-text could not process this audio recording.") from exc
        transcript = str(local_result.get("transcript", "")).strip()
        if not transcript:
            raise HTTPException(422, "Speech-to-text returned no transcription.")
        LOGGER.info("[STT PROVIDER] used=faster-whisper-local language=%s characters=%d", local_result.get("language", language), len(transcript))
        return {"text": transcript, "language": str(local_result.get("language") or language), "provider": "faster-whisper-local"}

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        return await use_local_stt("external_credential_missing")

    # A configured but invalid/unreachable paid provider must not prevent local STT.
    try:
        import io
        from openai import OpenAI as OpenAIClient

        client = OpenAIClient(api_key=openai_key)
        filename = audio.filename or f"recording.{_audio_extension(audio.content_type or '')}"
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
        )
        text = (transcription.text or "").strip()
    except Exception as exc:
        LOGGER.warning("OpenAI Whisper unavailable type=%s; falling back to local STT", type(exc).__name__)
        return await use_local_stt("external_provider_failed")
    if text:
        LOGGER.info("[STT PROVIDER] used=openai-whisper language=%s characters=%d", language, len(text))
        return {"text": text, "language": language, "provider": "openai-whisper"}
    LOGGER.warning("OpenAI Whisper returned no transcription; falling back to local STT")
    return await use_local_stt("external_empty_transcription")


def _audio_extension(content_type: str) -> str:
    """Guess a file extension from a MIME content type."""
    ct = content_type.lower()
    if "webm" in ct:
        return "webm"
    if "ogg" in ct:
        return "ogg"
    if "mp4" in ct or "mpeg" in ct:
        return "mp4"
    if "wav" in ct:
        return "wav"
    return "webm"




@app.post("/api/demo")
def create_demo(current_user: dict[str, Any] = Depends(get_current_user)):
    text = (APP_DIR.parent.parent / "sample_data" / "northstar_annual_review.txt").read_text(encoding="utf-8")
    doc_id = str(uuid.uuid4())
    pages = []
    doc = {"id": doc_id, "user_id": int(current_user["id"]), "name": "Northstar Annual Review · Demo.txt", "text": normalize(text), "pages": pages, "chunks": make_chunks(text, pages, document_id=doc_id), "language": "en", "status": "UPLOADED", "created_at": datetime.now(timezone.utc).isoformat(), "result": None}
    save_doc(doc)
    result = process_extractively(doc, ProcessOptions(summary_depth="standard", purpose="Executive briefing", language="en", audience="Leadership"))
    return public_doc(doc) | {"result": result}


@app.get("/api/dataset/govreport")
def govreport_dataset(split: str = "train", limit: int | None = None, evaluation_mode: bool = False, current_user: dict[str, Any] = Depends(get_current_user)):
    if split not in GOVREPORT_SPLITS:
        raise HTTPException(422, "split must be train, validation, or test")
    try:
        records = stream_govreport_records(split=split, limit=limit, evaluation=evaluation_mode)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"dataset": "ccdv/govreport-summarization", "split": split, "count": len(records), "records": [{"id": r["id"], "id_source": r.get("id_source", "dataset"), "report_characters": len(r["report"]), "summary_characters": len(r["summary"]), "preview": r["report"][:280]} for r in records]}


@app.post("/api/dataset/govreport/{record_id}/load")
def load_govreport_document(record_id: str, split: str = "train", limit: int | None = None, evaluation_mode: bool = False, current_user: dict[str, Any] = Depends(get_current_user)):
    if split not in GOVREPORT_SPLITS:
        raise HTTPException(422, "split must be train, validation, or test")
    try:
        record = get_govreport_record(record_id, split=split, limit=limit, evaluation=evaluation_mode)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    if not record:
        raise HTTPException(404, "GovReport record not found in the configured bounded subset.")
    doc = govreport_to_document(record, split)
    doc["user_id"] = int(current_user["id"])
    save_doc(doc)
    return public_doc(doc) | {"dataset": doc["dataset"]}


async def _upload_file(file: UploadFile, ocr_language: str, current_user: dict[str, Any], batch_id: str | None = None):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file exceeds the configured upload limit.")
    try:
        upload_info = validate_upload(file.filename or "", data)
    except OverflowError as exc:
        raise HTTPException(413, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    name, ext = upload_info["name"], upload_info["extension"]
    selected_ocr_language = None if ocr_language.strip().lower() in AUTO_LANGUAGE_VALUES else ocr_language.strip().lower()
    if selected_ocr_language and selected_ocr_language not in {"en", "hi", "te", "ta", "kn"}:
        raise HTTPException(422, "OCR input language must be auto, English, Hindi, Telugu, Tamil, or Kannada.")
    text, pages = ("", []) if ext in {".jpg", ".jpeg", ".png"} else extract_text(name, data)
    ocr_records: list[dict[str, Any]] = []
    ocr_warnings: list[str] = []
    if ext == ".pdf":
        pages, ocr_records, ocr_warnings = extract_pdf_ocr(data, pages, selected_ocr_language)
        text = "\n\n".join(page.get("text", "") for page in pages)
    elif ext in {".jpg", ".jpeg", ".png"}:
        result = extract_image_ocr(data, selected_ocr_language)
        text = result["text"]
        ocr_records = result.get("records", [])
        if result.get("warning"):
            ocr_warnings.append(result["warning"])
        if text:
            text = f"[Image 1 OCR | confidence {result.get('confidence') if result.get('confidence') is not None else 'unavailable'}%]\n{text}"
            pages = [{"page": 1, "text": text, "ocr": ocr_records[0] if ocr_records else None}]
    text = normalize(text)
    try:
        text = validate_document_text(text)
    except OverflowError as exc:
        raise HTTPException(413, str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        if ocr_warnings:
            detail += " " + " ".join(ocr_warnings)
        raise HTTPException(422, detail) from exc
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    owner_id = int(current_user["id"])
    if find_duplicate_document(owner_id, upload_info["sha256"], text_hash):
        raise HTTPException(409, "This document or its extracted text was already uploaded. Open the existing document instead of uploading the same content again.")
    doc_id = str(uuid.uuid4())
    chunks = make_chunks(text, pages, document_id=doc_id)
    embedding_svc = get_embedding_service()
    try:
        vectors = embedding_svc.get_embeddings([chunk["text"][:1200] for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            if vector:
                chunk["embedding"] = vector
        LOGGER.info("Document chunk embeddings generated provider=%s chunks=%d", embedding_svc.last_status.get("effective_provider", "lexical"), sum(bool(chunk.get("embedding")) for chunk in chunks))
    except Exception as exc:
        # Upload and text retrieval remain available if all embedding providers fail.
        LOGGER.warning("Document embeddings unavailable type=%s; RAG will use retrieval fallbacks", type(exc).__name__)
    doc = {"id": doc_id, "user_id": owner_id, "batch_id": batch_id, "name": name, "content_sha256": upload_info["sha256"], "extracted_text_sha256": text_hash, "text": text, "pages": pages, "ocr_records": ocr_records, "ocr_warnings": ocr_warnings, "ocr_detected": bool(ocr_records), "chunks": chunks, "language": detect_language(text), "status": "UPLOADED", "created_at": datetime.now(timezone.utc).isoformat(), "result": None}
    try:
        save_doc(doc)
    except Exception as exc:
        from sqlalchemy.exc import IntegrityError
        if isinstance(exc, IntegrityError) and find_duplicate_document(owner_id, upload_info["sha256"], text_hash):
            raise HTTPException(409, "This document or its extracted text was already uploaded. Open the existing document instead of uploading the same content again.") from exc
        raise
    record_history(user_id=int(current_user["id"]), document_id=doc_id, activity_type="upload",
        title=f"Uploaded {name}", status="completed", input_text=name, language=doc["language"],
        related_document_ids=[doc_id], extra_data={"batch_id": batch_id} if batch_id else None)
    record_history(user_id=int(current_user["id"]), document_id=doc_id, activity_type="extraction",
        title=f"Extracted {name}", status="completed", language=doc["language"],
        extra_data={"ocr_used": bool(ocr_records), "page_count": len(pages) or None})
    return public_doc(doc)


@app.post("/api/documents/upload")
async def upload(file: UploadFile = File(...), ocr_language: str = Form(default="auto"), current_user: dict[str, Any] = Depends(get_current_user)):
    return await _upload_file(file, ocr_language, current_user)


@app.post("/api/documents/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...), ocr_language: str = Form(default="auto"), current_user: dict[str, Any] = Depends(get_current_user)):
    limit = config_int("MAX_DOCUMENTS_PER_BATCH", 8)
    total_limit = config_int("MAX_TOTAL_UPLOAD_MB", 80) * 1024 * 1024
    if not files or len(files) > limit:
        raise HTTPException(413, "The document batch exceeds the configured document limit.")
    batch_id = str(uuid.uuid4())
    owner_id = int(current_user["id"])
    create_upload_batch(owner_id, len(files), batch_id)
    records = []
    total = 0
    seen: set[str] = set()
    for item in files:
        data = await item.read(MAX_BYTES + 1)
        total += len(data)
        if len(data) > MAX_BYTES or total > total_limit:
            records.append({"name": re.sub(r"[\x00-\x1f\x7f]", "_", Path(item.filename or "Document").name)[:180], "status": "failed", "error": "Document processing failed."})
            continue
        try:
            info = validate_upload(item.filename or "", data)
            if info["sha256"] in seen:
                raise ValueError("Duplicate document in batch.")
            seen.add(info["sha256"])
            # Reuse the same extraction, OCR, storage and validation path as a single upload.
            item.file.seek(0)
            records.append({"status": "uploaded", "document": await _upload_file(item, ocr_language, current_user, batch_id)})
        except HTTPException as exc:
            records.append({"name": re.sub(r"[\x00-\x1f\x7f]", "_", Path(item.filename or "Document").name)[:180], "status": "failed", "error": "Document processing failed."})
        except (ValueError, OverflowError):
            records.append({"name": re.sub(r"[\x00-\x1f\x7f]", "_", Path(item.filename or "Document").name)[:180], "status": "failed", "error": "Document processing failed."})
    succeeded = sum(x["status"] == "uploaded" for x in records)
    failed = sum(x["status"] == "failed" for x in records)
    batch_status = "completed" if failed == 0 else ("failed" if succeeded == 0 else "partially_failed")
    update_upload_batch(batch_id, owner_id, succeeded, failed, batch_status)
    record_history(user_id=owner_id, activity_type="upload_batch", title=f"Uploaded {succeeded} of {len(files)} documents",
        status=batch_status, related_document_ids=[x["document"]["id"] for x in records if x["status"] == "uploaded"],
        extra_data={"batch_id": batch_id, "total_files": len(files), "completed_files": succeeded, "failed_files": failed})
    return {"batch_id": batch_id, "status": batch_status, "total_files": len(files),
            "completed_files": succeeded, "failed_files": failed, "documents": records,
            "succeeded": succeeded, "failed": failed}


@app.get("/api/upload-batches/{batch_id}")
def get_upload_batch_status(batch_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    batch = get_upload_batch(batch_id, int(current_user["id"]))
    if batch is None:
        raise HTTPException(404, "Upload batch not found.")
    return batch


def public_doc(doc: dict[str, Any]):
    return {k: doc[k] for k in ("id", "name", "language", "status", "created_at")} | {"batch_id": doc.get("batch_id"), "ocr_detected": bool(doc.get("ocr_detected")), "ocr_warnings": doc.get("ocr_warnings", []), "stats": {"words": len(re.findall(r"\b\w+\b", doc["text"])), "chunks": len(doc["chunks"]), "pages": doc.get("pages_count", len(doc["pages"])) or None}}


def get_doc(doc_id: str, current_user: dict[str, Any]):
    doc = DOCS.get(doc_id)
    if doc and doc.get("user_id") != current_user["id"]:
        doc = None
    if not doc:
        doc = load_document_from_db(doc_id, int(current_user["id"]))
        if doc:
            DOCS[doc_id] = doc
    if not doc:
        raise HTTPException(404, "Document not found. Please upload your document again.")
    return doc


@app.get("/api/documents")
def list_documents(search: str = "", current_user: dict[str, Any] = Depends(get_current_user)):
    user_id = int(current_user["id"])
    term = search.strip()
    if len(term) > 256:
        raise HTTPException(422, "Search terms must be 256 characters or fewer.")
    try:
        documents = search_documents_from_db(user_id, term) if term else list_documents_from_db(user_id)
    except Exception as exc:
        LOGGER.error("Document search failed user_id=%s error_type=%s", user_id, type(exc).__name__)
        raise HTTPException(500, "Unable to search documents. Please try again.") from exc
    return [public_doc(d) for d in documents]


@app.post("/api/local-llm/test")
def local_llm_test(payload: LocalLLMTestRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    """Run a prompt through the configured, already-installed local Ollama model."""
    service = get_local_llm_service()
    if service is None:
        raise HTTPException(503, "LLM stage failed: Ollama or the configured local model is unavailable.")
    try:
        response_text = service.complete(
            "Answer the user's prompt clearly and concisely. If asked to summarize, summarize only the supplied text and do not invent facts.",
            payload.prompt,
            max_tokens=800,
        )
    except Exception as exc:
        LOGGER.warning("Local LLM test failed model=%s type=%s", service.model, type(exc).__name__)
        raise HTTPException(502, f"LLM stage failed: local Ollama request failed ({type(exc).__name__}).") from exc
    return {"response": response_text, "provider": service.provider_name, "model": service.model}


@app.post("/api/speech/transcribe")
async def transcribe_speech(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Transcribe microphone audio with the local, offline speech model."""
    if file.content_type and not file.content_type.lower().startswith("audio/"):
        raise HTTPException(415, "Audio recording was invalid: upload an audio recording.")
    max_audio_bytes = 15 * 1024 * 1024
    audio = await file.read(max_audio_bytes + 1)
    if len(audio) > max_audio_bytes:
        raise HTTPException(413, "Audio recording was invalid: recording exceeds 15 MiB.")
    LOGGER.info("STT upload received bytes=%d mime=%s language=%s", len(audio), file.content_type or "unknown", language)
    try:
        result = await run_in_threadpool(transcribe_local_audio, audio, content_type=file.content_type, language=language)
    except AudioRecordingEmpty as exc:
        raise HTTPException(422, str(exc)) from exc
    except InvalidAudioRecording as exc:
        raise HTTPException(415, str(exc)) from exc
    except NoTranscription as exc:
        raise HTTPException(422, str(exc)) from exc
    except SpeechModelUnavailable as exc:
        raise HTTPException(503, "Local speech-to-text service is unavailable.") from exc
    except Exception as exc:
        LOGGER.warning("Speech-to-Text request failed type=%s", type(exc).__name__)
        raise HTTPException(502, f"Local speech-to-text could not process the recording ({type(exc).__name__}).") from exc
    return result


@app.get("/api/documents/{doc_id}")
def document(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    return public_doc(doc) | {"preview": doc["text"][:1600], "extracted_text": doc["text"],
        "result": doc["result"], "options": doc.get("options"),
        "audio": doc.get("audio_timings"), "ocr_detected": bool(doc.get("ocr_detected"))}


def _export_filename(title: str, extension: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(title).stem).strip(".-_")[:80]
    return f"{base or 'scribe-summary'}.{extension}"


def _export_response(payload: dict[str, Any], export_format: str, *, title: str,
                     user_id: int, document_id: str | None,
                     related_document_ids: list[str] | None = None) -> Response:
    try:
        content, media_type, extension = render_summary(payload, export_format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        LOGGER.warning("Summary export generation failed format=%s type=%s", export_format, type(exc).__name__)
        raise HTTPException(500, "Export generation failed. Please try another format.") from exc
    record_history(user_id=user_id, document_id=document_id, activity_type="export",
        title=f"Exported summary as {extension.upper()}", status="completed", language=payload.get("language"),
        related_document_ids=related_document_ids,
        extra_data={"export_format": extension, "generated_at": payload.get("generated_at")})
    return Response(content=content, media_type=media_type, headers={
        "Content-Disposition": f'attachment; filename="{_export_filename(title, extension)}"',
        "X-Content-Type-Options": "nosniff",
    })


def record_audio_history(doc: dict[str, Any], *, language: str, provider: str,
                        voice_id: str, fallback_used: bool) -> None:
    owner_id = doc.get("user_id")
    if owner_id is None:
        return
    record_history(user_id=int(owner_id), document_id=str(doc["id"]), activity_type="audio",
        title=f"Generated audio for {doc.get('name', 'document')}", status="completed",
        language=language, provider=provider,
        extra_data={"voice_id": voice_id, "fallback_used": fallback_used})


def record_processing_failure(doc: dict[str, Any], user_id: int) -> None:
    doc["status"] = "FAILED"
    try:
        save_doc(doc)
    except Exception as exc:
        LOGGER.warning("Could not persist failed document status type=%s", type(exc).__name__)
    try:
        record_history(user_id=user_id, document_id=str(doc["id"]), activity_type="processing",
            title=f"Processing failed for {doc.get('name', 'document')}", status="failed",
            language=doc.get("language"), error_message="Document processing failed.")
    except Exception as exc:
        LOGGER.warning("Could not record processing failure event type=%s", type(exc).__name__)


@app.get("/api/documents/{doc_id}/export")
def export_document_summary(doc_id: str, format: str,
                            current_user: dict[str, Any] = Depends(get_current_user)):
    if format.strip().lower() not in SUPPORTED_EXPORT_FORMATS:
        raise HTTPException(422, "Unsupported export format. Choose PDF, DOCX, TXT, Markdown, or JSON.")
    user_id = int(current_user["id"])
    doc = load_document_from_db(doc_id, user_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    try:
        payload = make_export_payload(doc, doc.get("result") or {})
    except SummaryNotAvailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except ExportTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    return _export_response(payload, format, title=doc["name"], user_id=user_id, document_id=doc_id)


@app.get("/api/combined-summaries/{summary_id}/export")
def export_combined_summary(summary_id: int, format: str,
                            current_user: dict[str, Any] = Depends(get_current_user)):
    if format.strip().lower() not in SUPPORTED_EXPORT_FORMATS:
        raise HTTPException(422, "Unsupported export format. Choose PDF, DOCX, TXT, Markdown, or JSON.")
    user_id = int(current_user["id"])
    combined = get_combined_summary(summary_id, user_id)
    if combined is None:
        raise HTTPException(404, "Combined summary not found.")
    try:
        payload = make_combined_export_payload(combined)
    except SummaryNotAvailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except ExportTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    return _export_response(payload, format, title="scribe-combined-summary", user_id=user_id,
        document_id=None, related_document_ids=combined.get("document_ids", []))


@app.post("/api/documents/{doc_id}/process")
def process(doc_id: str, options: ProcessOptions, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    try:
        result = process_document(doc, options)
    except Exception:
        record_processing_failure(doc, int(current_user["id"]))
        raise
    if doc.get("reference_summary"):
        doc["evaluation"] = evaluate_record(result, doc["reference_summary"])
        result["evaluation"] = doc["evaluation"]
    return result


@app.post("/api/documents/process-batch")
def process_batch(payload: BatchProcessRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    limit = config_int("MAX_DOCUMENTS_PER_BATCH", 8)
    if len(payload.document_ids) > limit or len(set(payload.document_ids)) != len(payload.document_ids):
        raise HTTPException(422, "The document batch is invalid or exceeds the configured limit.")
    # Validate ownership before starting any work so a mixed/foreign batch is
    # rejected as one request and can never produce a partial cross-user result.
    for document_id in payload.document_ids:
        get_doc(document_id, current_user)
    concurrency = min(config_int("MAX_CONCURRENT_DOCUMENTS", 2), len(payload.document_ids))
    results: dict[str, dict[str, Any]] = {}

    def run_one(document_id: str) -> tuple[str, dict[str, Any]]:
        item = None
        try:
            item = get_doc(document_id, current_user)
            summary_result = process_document(item, payload.options.model_copy(deep=True))
            return document_id, {"document_id": document_id, "status": "completed", "progress": 100, "summary": summary_result, "validation_result": summary_result.get("validation"), "translation_result": summary_result.get("translation_validation_status"), "error": None}
        except Exception:
            if item is not None:
                record_processing_failure(item, int(current_user["id"]))
            return document_id, {"document_id": document_id, "status": "failed", "progress": 0, "summary": None, "validation_result": None, "translation_result": None, "error": "Document processing failed."}

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="scribe-document") as pool:
        futures = [pool.submit(run_one, document_id) for document_id in payload.document_ids]
        for future in as_completed(futures):
            document_id, item_result = future.result()
            results[document_id] = item_result
    ordered = [results[document_id] for document_id in payload.document_ids]
    combined_claims = []
    combined_sections = []
    for item in ordered:
        if item["summary"]:
            doc_name = get_doc(item["document_id"], current_user).get("name", item["document_id"])
            combined_sections.append({"document_id": item["document_id"], "document_name": doc_name,
                "summary": item["summary"].get("executive_summary", ""),
                "findings": item["summary"].get("findings", [])})
            combined_claims.extend(item["summary"].get("claims", []))
    combined_summary = "\n\n".join(f"{section['document_name']}: {section['summary']}" for section in combined_sections)
    batch_status = "completed" if all(item["status"] == "completed" for item in ordered) else "partial"
    output_language = next((item["summary"].get("language") for item in ordered if item.get("summary")), payload.options.language)
    owner_id = int(current_user["id"])
    combined_summary_id = save_combined_summary(user_id=owner_id, document_ids=list(payload.document_ids),
        summary_text=combined_summary, language=output_language or "und", status=batch_status,
        details={"sections": combined_sections, "claims": combined_claims})
    record_history(user_id=owner_id, activity_type="combined_summary", title=f"Combined summary of {len(ordered)} documents",
        status=batch_status, output_text=combined_summary, language=output_language,
        related_document_ids=list(payload.document_ids), extra_data={"document_count": len(ordered), "combined_summary_id": combined_summary_id})
    return {
        "combined_summary_id": combined_summary_id,
        "run_type": "COMBINED_SUMMARY",
        "documents": ordered,
        "combined_summary": combined_summary,
        "combined_sections": combined_sections,
        "claims": combined_claims,
        "status": batch_status,
    }


@app.get("/api/history")
@app.get("/api/history/documents", include_in_schema=False)
def history_documents(page: int = 1, limit: int = 20, activity_type: str | None = None,
                      search: str | None = None,
                      file_type: str | None = None, status: str | None = None,
                      language: str | None = None, date_from: datetime | None = None,
                      date_to: datetime | None = None, sort: str = "newest",
                      current_user: dict[str, Any] = Depends(get_current_user)):
    if page < 1 or limit < 1 or limit > 100:
        raise HTTPException(422, "page must be positive and limit must be between 1 and 100.")
    if activity_type:
        return list_history_from_db(int(current_user["id"]), page=page, limit=limit, activity_type=activity_type)
    if sort not in {"newest", "oldest"}:
        raise HTTPException(422, "sort must be newest or oldest.")
    if date_from and date_to and date_from >= date_to:
        raise HTTPException(422, "date_from must be earlier than date_to.")
    return list_history_documents_from_db(int(current_user["id"]), page=page, limit=limit,
        search=search, file_type=file_type, status=status, language=language,
        date_from=date_from, date_to=date_to, sort=sort)


@app.delete("/api/documents/{doc_id}")
def delete_user_document(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    get_doc(doc_id, current_user)
    if not delete_document_from_db(doc_id, int(current_user["id"])):
        raise HTTPException(404, "Document not found.")
    DOCS.pop(doc_id, None)
    return Response(status_code=204)


@app.post("/api/documents/{doc_id}/retry")
def retry_document(doc_id: str, options: ProcessOptions, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    try:
        return process_document(doc, options)
    except Exception:
        record_processing_failure(doc, int(current_user["id"]))
        raise


@app.get("/api/documents/{doc_id}/summary")
def summary(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    result = get_doc(doc_id, current_user).get("result")
    if not result:
        raise HTTPException(409, "Process the document before requesting its summary.")
    return result


@app.get("/api/documents/{doc_id}/sources")
def sources(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    return [{"id": c["id"], "documentId": c.get("documentId", doc_id), "chunkId": c["id"], "chunkNumber": c.get("chunk_number"), "page": c.get("page"), "section": c["section"], "language": c["language"], "text": c["text"]} for c in doc["chunks"]]


@app.get("/api/documents/{doc_id}/validation")
def validation(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    result = get_doc(doc_id, current_user).get("result")
    if not result:
        raise HTTPException(409, "Process the document before requesting validation.")
    return {
        "status": result["status"],
        "validation_status": result.get("validation_status", result["status"]),
        "validation_state": result.get("validation_state"),
        "validation_warning": result.get("validation_warning", False),
        "warning_message": result.get("warning_message"),
        "validation_attempts": result.get("validation_attempts", (result.get("stats") or {}).get("validation_attempts", 0)),
        "supported_claims_count": result.get("supported_claims_count", 0),
        "unsupported_claims_count": result.get("unsupported_claims_count", 0),
        "validation_provider": result.get("validation_provider"),
        "claims": result["claims"],
        "quality": result["quality"],
    }


@app.post("/api/documents/{doc_id}/ask")
def ask(doc_id: str, payload: AskRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    if len(payload.question) > config_int("MAX_QUESTION_CHARS", 4000) or not payload.question.strip():
        raise HTTPException(422, "Enter a question within the configured length limit.")
    doc["text"] = validate_document_text(doc.get("text", ""))
    output_lang = resolve_output_language(payload.output_language, doc.get("language", "und"))
    qterms = set(keywords(payload.question))
    if not qterms:
        qterms = set(re.findall(r"[^\W_]{2,}", payload.question.casefold(), flags=re.UNICODE))

    chunks = doc["chunks"]
    emb_svc = get_embedding_service()
    ranked_chunks = chunks
    semantic_used = False

    try:
        # Pre-compute or fetch embeddings for chunks (cached directly on chunk dicts)
        uncached = [c for c in chunks if not c.get("embedding")]
        if uncached:
            vectors = emb_svc.get_embeddings([c["text"][:1200] for c in uncached])
            persisted_vectors: dict[str, list[float]] = {}
            for c, v in zip(uncached, vectors):
                if v:
                    c["embedding"] = v
                    persisted_vectors[str(c["id"])] = v
            if persisted_vectors:
                save_document_chunk_embeddings(doc_id, int(current_user["id"]), persisted_vectors)

        q_vec = emb_svc.get_embedding(payload.question)
        semantic_used = bool(q_vec) and all(bool(c.get("embedding")) for c in chunks)

        def hybrid_score(c: dict[str, Any]) -> float:
            c_vec = c.get("embedding")
            sim = cosine_similarity(q_vec, c_vec) if (c_vec and q_vec) else 0.0
            hits = term_hits(c["text"], qterms)
            norm_hits = hits / max(1, len(qterms))
            return 0.65 * max(0.0, sim) + 0.35 * norm_hits

        if semantic_used:
            ranked_chunks = sorted(chunks, key=hybrid_score, reverse=True)
            evidence = [c for c in ranked_chunks[:3] if term_present(c["text"], qterms) or cosine_similarity(q_vec, c["embedding"]) >= 0.28]
            if not evidence and ranked_chunks and cosine_similarity(q_vec, ranked_chunks[0]["embedding"]) > 0.22:
                evidence = [ranked_chunks[0]]
        else:
            ranked_chunks = sorted(chunks, key=lambda c: term_hits(c["text"], qterms), reverse=True)
            evidence = [c for c in ranked_chunks if term_hits(c["text"], qterms) > 0][:3]
            LOGGER.info("[RETRIEVAL FALLBACK] provider=%s; using in-memory lexical matching", emb_svc.last_status.get("effective_provider", "lexical"))
    except Exception as exc:
        LOGGER.warning("Semantic ranking encountered error type=%s; using lexical ranking", type(exc).__name__)
        ranked = sorted(chunks, key=lambda c: term_hits(c["text"], qterms), reverse=True)
        evidence = [c for c in ranked[:3] if term_present(c["text"], qterms)]

    if not evidence:
        no_evidence_msgs = {
            "en": "I couldn't find sufficient evidence in the extracted document to answer that.",
            "hi": "इस दस्तावेज़ में पर्याप्त साक्ष्य नहीं मिला।",
            "te": "ఈ పత్రంలో తగినంత ఆధారాలు కనుగొనలేదు.",
            "ta": "இந்த ஆவணத்தில் போதுமான ஆதாரம் கிடைக்கவில்லை.",
            "kn": "ಈ ದಾಖಲೆಯಲ್ಲಿ ಸಾಕಷ್ಟು ಪುರಾವೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
            "ml": "ഈ രേഖയിൽ മതിയായ തെളിവുകൾ കണ്ടെത്താനായില്ല.",
            "bn": "এই নথিতে পর্যাপ্ত প্রমাণ পাওয়া যায়নি।",
            "mr": "या दस्तऐवजात पुरेसा पुरावा सापडला नाही.",
            "gu": "આ દસ્તાવેજમાં પૂરતો પુરાવો મળ્યો નથી.",
            "ur": "اس دستاویز میں کافی ثبوت نہیں ملے۔",
        }
        response = {"answer": no_evidence_msgs.get(output_lang, no_evidence_msgs["en"]), "sources": [], "grounded": False, "output_language": output_lang, "retrieval_provider": emb_svc.last_status.get("effective_provider", "lexical")}
        save_question(doc_id, payload.question, response, output_lang)
        return response

    source_list = [{"document_id": doc["id"], "chunk_id": c["id"], "page": c.get("page") or "unavailable", "section": c.get("section", "Document text"), "text": c["text"][:600]} for c in evidence]
    answer = None
    answer_provider = None
    try:
        service = OpenRouterService()
    except Exception as exc:
        service = None
        LOGGER.info("[LLM PROVIDER] hosted provider unavailable for Q&A type=%s; trying local Ollama", type(exc).__name__)
    if service is not None:
        try:
            answer = answer_with_openrouter(payload.question, evidence, service, output_language=output_lang)
            answer_provider = service.provider_name
            LOGGER.info("[LLM PROVIDER] selected=%s model=%s operation=question_answering", service.provider_name, service.model)
        except Exception as exc:
            LOGGER.warning("Hosted LLM Q&A failed provider=%s type=%s; trying local Ollama", service.provider_name, type(exc).__name__)
    if answer is None:
        local_service = get_local_llm_service()
        if local_service is not None:
            try:
                answer = answer_with_openrouter(payload.question, evidence, local_service, output_language=output_lang)
                answer_provider = local_service.provider_name
                LOGGER.info("[LLM PROVIDER] selected=%s model=%s operation=question_answering", answer_provider, local_service.model)
            except Exception as exc:
                LOGGER.warning("Local Ollama Q&A failed model=%s type=%s; returning source excerpts", local_service.model, type(exc).__name__)
    if answer is not None:
        mode_desc = "Answer grounded in hybrid semantic/lexical evidence" if semantic_used else "Answer grounded in lexical evidence"
        response = {"answer": answer, "sources": source_list, "grounded": True, "output_language": output_lang, "retrieval_provider": emb_svc.last_status.get("effective_provider", "lexical"), "llm_provider": answer_provider, "mode": mode_desc}
        save_question(doc_id, payload.question, response, output_lang)
        return response
    LOGGER.warning("No LLM provider produced a Q&A response; returning source excerpts")
    response = {"answer": "\n\n".join(c["text"][:600] for c in evidence), "sources": source_list, "grounded": True, "output_language": doc.get("language", "und"), "requested_language": output_lang, "retrieval_provider": emb_svc.last_status.get("effective_provider", "lexical"), "mode": "source-language retrieval fallback; no generated answer"}
    save_question(doc_id, payload.question, response, output_lang)
    return response


class TranslateRequest(BaseModel):
    language: str = Field(default="auto", validation_alias=AliasChoices("output_language", "language"))
    source_language: str | None = Field(default=None, exclude=True)
    summary_depth: str = "standard"
    custom_detail: str = "standard"
    custom_sections: list[str] | None = None
    purpose: str = "Executive briefing"
    audience: str = "Leadership"

    model_config = {"populate_by_name": True}

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, value: Any) -> str:
        language = str(value or "auto").strip().lower()
        if language in AUTO_LANGUAGE_VALUES:
            return "auto"
        if language not in LANGUAGES:
            raise ValueError(f"output_language must be auto or one of: {', '.join(sorted(LANGUAGES))}")
        return language


@app.post("/api/documents/{doc_id}/translate")
def translate_summary(doc_id: str, payload: TranslateRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    """
    Generate a summary in a new output language from an already-processed document.

    This reuses the stored document chunks — no re-upload required.  The same
    summarize → validate → translate → validate-translation pipeline runs as
    during normal processing, but the raw document bytes are not needed again.

    Input  : doc_id path param + { language, summary_depth, purpose, audience }
    Output : same shape as POST /process
    """
    doc = get_doc(doc_id, current_user)
    options = ProcessOptions(
        summary_depth=payload.summary_depth,
        custom_detail=payload.custom_detail,
        custom_sections=payload.custom_sections,
        language=payload.language,
        purpose=payload.purpose,
        audience=payload.audience,
    )
    try:
        result = process_document(doc, options)
    except Exception:
        record_processing_failure(doc, int(current_user["id"]))
        raise
    if doc.get("reference_summary"):
        doc["evaluation"] = evaluate_record(result, doc["reference_summary"])
        result["evaluation"] = doc["evaluation"]
    return result



@app.get("/api/tts/languages")
def tts_languages():
    return [{"code": k, "name": v} for k, v in LANGUAGES.items()]


@app.get("/api/tts/voices")
def voices(language: str = "en", provider: str = "edge"):
    if language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of: {', '.join(sorted(LANGUAGES.keys()))}")
    if provider == "edge":
        return [{**voice, "language": language, "provider": "online neural speech"} for voice in tts_voices_for(language)]
    return [{"voice_id": "browser-default", "name": f"System voice ({LANGUAGES[language]})", "language": language, "provider": "browser speech synthesis"}]


@app.get("/api/documents/{doc_id}/audio/timings")
def get_audio_timings(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    if "audio_timings" in doc:
        return doc["audio_timings"]
    result = doc.get("result")
    if not result:
        raise HTTPException(404, "No summary available for this document.")
    text = result.get("executive_summary", "")
    return {
        "timings": estimate_timings(text),
        "text": text,
        "language": result.get("language", "en"),
        "provider": "estimated",
        "voice_id": "estimated",
        "fallback_used": False,
    }


@app.post("/api/documents/{doc_id}/audio")
def audio(doc_id: str, payload: AudioRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    import base64
    doc = get_doc(doc_id, current_user)
    result = doc.get("result")
    # TEMPORARILY DISABLED: Exact-status audio gate blocked usable partial summaries.
    # Keep validation statuses and claim checks intact; allow narration only when
    # there is real summary text to speak.
    # if not result or result.get("status") != "validated":
    #     raise HTTPException(409, "Audio is available only after summary validation completes.")
    if not result or not str(result.get("executive_summary", "")).strip():
        raise HTTPException(409, "Process the document and generate a summary before requesting audio.")
    payload.language = resolve_output_language(payload.language, result.get("language") or doc.get("language", "und"))
    if payload.language not in LANGUAGES:
        raise HTTPException(422, f"Audio language '{payload.language}' is not supported. Supported languages: {', '.join(sorted(LANGUAGES.keys()))}.")

    # Preserve user's requested language throughout
    narration_text = result["executive_summary"]
    if payload.language != result.get("language"):
        translated_text = None
        try:
            hosted_service = OpenRouterService()
        except RuntimeError as exc:
            hosted_service = None
            LOGGER.info("Hosted LLM unavailable for audio translation type=%s; trying local Ollama", type(exc).__name__)
        if hosted_service is not None:
            try:
                translated_text = translate_for_audio(narration_text, payload.language, hosted_service)
                LOGGER.info("[LLM PROVIDER] selected=%s model=%s operation=audio_translation", hosted_service.provider_name, hosted_service.model)
            except Exception as exc:
                LOGGER.warning("Hosted audio translation failed provider=%s type=%s; trying local Ollama", hosted_service.provider_name, type(exc).__name__)
        if translated_text is None:
            local_service = get_local_llm_service()
            if local_service is not None:
                try:
                    translated_text = translate_for_audio(narration_text, payload.language, local_service)
                    LOGGER.info("[LLM PROVIDER] selected=%s model=%s operation=audio_translation", local_service.provider_name, local_service.model)
                except Exception as exc:
                    LOGGER.warning("Local audio translation failed model=%s type=%s", local_service.model, type(exc).__name__)
        if translated_text is None:
            raise HTTPException(503, "Audio translation is unavailable: configure a local Ollama model or a hosted LLM provider.")
        narration_text = translated_text

    # Determine requested provider
    if payload.provider:
        primary_provider = payload.provider.strip().lower()
    else:
        primary_provider = "browser" if os.getenv("PYTEST_CURRENT_TEST") else os.getenv("TTS_PROVIDER", "edge").strip().lower()

    if primary_provider not in {"edge", "browser"}:
        raise HTTPException(422, "provider must be browser or edge")

    TTS_LOGGER.info(
        "TTS request received: doc_id=%s, language=%s (%s), requested_provider=%s, requested_voice=%s, rate=%.2f",
        doc_id, payload.language, LANGUAGES[payload.language], primary_provider, payload.voice_id, payload.rate
    )

    edge_voices = tts_voices_for(payload.language)
    edge_available = len(edge_voices) > 0

    # Primary provider is Edge (online neural TTS)
    if primary_provider == "edge":
        fallback_reason = ""
        if edge_available:
            allowed_voice_ids = {v["voice_id"] for v in edge_voices}
            voice_id = payload.voice_id
            if voice_id == "browser-default" or voice_id not in allowed_voice_ids:
                voice_id = edge_voices[0]["voice_id"]

            TTS_LOGGER.info("Attempting primary TTS provider=edge with voice=%s for language=%s", voice_id, payload.language)
            try:
                audio_bytes, timings = synthesize_with_timings(narration_text, voice_id, payload.rate)
                TTS_LOGGER.info("TTS succeeded using provider=edge, voice=%s, language=%s, segments=%d, bytes=%d",
                                voice_id, payload.language, len(timings), len(audio_bytes))

                doc["audio_timings"] = {
                    "timings": timings,
                    "text": narration_text,
                    "language": payload.language,
                    "provider": "edge",
                    "voice_id": voice_id,
                    "fallback_used": False,
                }
                save_doc(doc)

                timings_b64 = base64.b64encode(json.dumps(timings, ensure_ascii=False).encode("utf-8")).decode("ascii")
                narration_b64 = base64.b64encode(narration_text.encode("utf-8")).decode("ascii")
                filename = f"scribe-briefing-{payload.language}.mp3"
                headers = {
                    "Content-Disposition": f'inline; filename="{filename}"',
                    "X-TTS-Provider": "edge-neural",
                    "X-TTS-Voice": voice_id,
                    "X-TTS-Language": payload.language,
                    "X-TTS-Fallback-Used": "false",
                    "X-TTS-Timings-Base64": timings_b64,
                    "X-TTS-Narration-Text-Base64": narration_b64,
                }
                record_audio_history(doc, language=payload.language, provider="edge-neural",
                    voice_id=voice_id, fallback_used=False)
                return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)
            except Exception as exc:
                TTS_LOGGER.warning(
                    "Primary TTS provider 'edge' failed for language=%s, voice=%s: %s. Falling back to browser TTS.",
                    payload.language, voice_id, exc
                )
                fallback_reason = f"Primary provider 'edge' failed ({exc}); fallen back to browser TTS."
        else:
            TTS_LOGGER.info(
                "Language '%s' is not supported by edge-tts neural voices. Falling back to browser TTS.",
                payload.language
            )
            fallback_reason = f"Language '{payload.language}' is not available in edge-tts; fallen back to browser."

        # Browser fallback for edge failure / unavailability
        timings = estimate_timings(narration_text)
        doc["audio_timings"] = {
            "timings": timings,
            "text": narration_text,
            "language": payload.language,
            "provider": "browser",
            "voice_id": "browser-default",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
        }
        save_doc(doc)
        record_audio_history(doc, language=payload.language, provider="browser",
            voice_id="browser-default", fallback_used=True)
        TTS_LOGGER.info("Serving browser fallback TTS for language=%s", payload.language)
        return {
            "status": "ready_for_browser_playback",
            "source": "validated_summary" if result.get("status") == "validated" else "summary",
            "validation_status": result.get("status", "unavailable"),
            "language": payload.language,
            "text": narration_text,
            "provider": "browser",
            "voice_id": "browser-default",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "timings": timings,
        }

    # Primary provider is Browser
    TTS_LOGGER.info("Using provider=browser for language=%s, voice=%s", payload.language, payload.voice_id)
    timings = estimate_timings(narration_text)
    doc["audio_timings"] = {
        "timings": timings,
        "text": narration_text,
        "language": payload.language,
        "provider": "browser",
        "voice_id": payload.voice_id or "browser-default",
        "fallback_used": False,
    }
    save_doc(doc)
    record_audio_history(doc, language=payload.language, provider="browser",
        voice_id=payload.voice_id or "browser-default", fallback_used=False)
    return {
        "status": "ready_for_browser_playback",
        "source": "validated_summary" if result.get("status") == "validated" else "summary",
        "validation_status": result.get("status", "unavailable"),
        "language": payload.language,
        "text": narration_text,
        "provider": "browser",
        "voice_id": payload.voice_id or "browser-default",
        "fallback_used": False,
        "timings": timings,
    }


@app.post("/api/documents/compare")
async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...), current_user: dict[str, Any] = Depends(get_current_user)):
    a, b = await file_a.read(MAX_BYTES + 1), await file_b.read(MAX_BYTES + 1)
    total_limit = config_int("MAX_TOTAL_UPLOAD_MB", 50) * 1024 * 1024
    if max(len(a), len(b)) > MAX_BYTES or len(a) + len(b) > total_limit:
        raise HTTPException(413, "The upload batch exceeds the configured size limit.")
    try:
        info_a = validate_upload(file_a.filename or "", a)
        info_b = validate_upload(file_b.filename or "", b)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(400, "One of the comparison files is invalid or unsupported.") from exc
    if info_a["sha256"] == info_b["sha256"]:
        raise HTTPException(409, "The same document was selected twice.")
    ta, _ = extract_text(info_a["name"], a)
    tb, _ = extract_text(info_b["name"], b)
    ta, tb = validate_document_text(normalize(ta)), validate_document_text(normalize(tb))
    sa, sb = set(re.findall(r"[^.!?\n]+[.!?]?", normalize(ta))), set(re.findall(r"[^.!?\n]+[.!?]?", normalize(tb)))
    return {"added": sorted(s.strip() for s in sb-sa if s.strip()), "removed": sorted(s.strip() for s in sa-sb if s.strip()), "method": "exact sentence comparison", "limitations": "No semantic change classification is configured."}


@app.get("/api/evaluation/{doc_id}")
def evaluation(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    result = doc.get("result")
    if not result:
        raise HTTPException(409, "Process the document before requesting evaluation.")
    if doc.get("reference_summary"):
        return doc.get("evaluation") or evaluate_record(result, doc["reference_summary"])
    return {"metrics": {"groundedness": {"value": result["quality"].get("groundedness"), "available": True}}, "metrics_unavailable": ["ROUGE", "semantic similarity", "reference completeness"]}


@app.post("/api/evaluation/run")
def run_evaluation(payload: EvaluationRunRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(payload.document_id, current_user)
    if not doc.get("reference_summary"):
        raise HTTPException(422, "This document has no GovReport reference summary for evaluation.")
    if not doc.get("result"):
        raise HTTPException(409, "Process the document before running evaluation.")
    doc["evaluation"] = evaluate_record(doc["result"], doc["reference_summary"])
    return doc["evaluation"]


@app.post("/api/evaluation/{doc_id}/compare-depths")
def compare_summary_depths(doc_id: str, payload: DepthComparisonRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    if not doc.get("reference_summary"):
        raise HTTPException(422, "Depth comparison requires a loaded GovReport record.")
    rows = []
    for depth in ("quick", "standard", "deep"):
        options = ProcessOptions(summary_depth=depth, purpose=payload.purpose, audience=payload.audience, language=payload.language)
        start = time.perf_counter()
        result = process_document(doc, options)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        rows.append({"depth": depth, "processing_time_ms": elapsed_ms, **evaluate_record(result, doc["reference_summary"])})
    doc["depth_comparison"] = rows
    return {"document_id": doc_id, "reference_role": "evaluation_only", "comparisons": rows}
