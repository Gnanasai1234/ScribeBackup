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
from typing import Any, Literal

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
_log_level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(
    level=getattr(logging, _log_level_name, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from database.repositories import (database_schema_status, get_document as load_document_from_db,
    list_documents as list_documents_from_db, save_document, save_question, delete_document as delete_document_from_db,
    find_duplicate_document, search_documents as search_documents_from_db,
    save_document_chunk_embeddings,
    create_upload_batch, update_upload_batch, get_upload_batch, record_history, list_history as list_history_from_db,
    list_history_documents as list_history_documents_from_db,
    save_combined_summary, get_combined_summary, get_latest_summary_evaluation)
from .auth import router as auth_router, get_current_user

from .llm_service import LLMGenerationError, LLMService, OpenAIProvider, OpenRouterProvider, answer_from_evidence, summarize_document, translate_for_audio
from services.openai_config import DIRECT_OPENAI_BASE_URL, openai_key_problem
from models.language_context import LanguageContext
from .textmatch import term_hits, term_present
from workflow.analysis import run_analysis_workflow
from services.tts_service import (
    synthesize as synthesize_audio,
    synthesize_with_timings,
    voices_for as tts_voices_for,
    get_default_voice_for,
    estimate_timings,
)
from services.embedding_service import cosine_similarity, get_embedding_service
from services.document_comparison import perform_document_comparison
from services.local_fallbacks import get_local_llm_service
from services.llm_logging import log_error as log_llm_error, log_success as log_llm_success
from services.summary_evaluation import evaluate_summary
from services.speech_to_text import (
    AudioRecordingEmpty,
    InvalidAudioRecording,
    NoTranscription,
    SpeechModelUnavailable,
    transcribe_local_audio,
)
from services.ocr_service import configure_tesseract, extract_image_ocr, extract_pdf_ocr, ocr_status
from services.guardrails import config_int, validate_upload, validate_document_text, is_usable_summary, summary_text
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
LOGGER = logging.getLogger("scribe.llm")
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
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_problem = openai_key_problem(openai_key)
    configured_openai_url = os.getenv("OPENAI_BASE_URL", DIRECT_OPENAI_BASE_URL).strip().rstrip("/")
    LOGGER.info("[CONFIG] OPENROUTER_API_KEY configured=%s", str(bool(os.getenv("OPENROUTER_API_KEY", "").strip())).lower())
    LOGGER.info("[CONFIG] OPENROUTER_MODEL=%s", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    LOGGER.info("[CONFIG] OPENAI_API_KEY configured=%s", str(bool(openai_key) and not openai_problem).lower())
    LOGGER.info("[CONFIG] OPENAI_MODEL=%s", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    LOGGER.info("[CONFIG] OPENAI_BASE_URL=%s", configured_openai_url if configured_openai_url == DIRECT_OPENAI_BASE_URL else "invalid (must be official direct endpoint)")
    LOGGER.info("[CONFIG] OLLAMA_BASE_URL=%s", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    LOGGER.info("[CONFIG] OLLAMA_MODEL=%s", os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
    if openai_problem:
        LOGGER.error("[CONFIG ERROR] %s", openai_problem)
    if configured_openai_url != DIRECT_OPENAI_BASE_URL:
        LOGGER.error("[CONFIG ERROR] OPENAI_BASE_URL must be %s; custom proxies are disabled.", DIRECT_OPENAI_BASE_URL)
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
            "Run `.\\venv\\Scripts\\alembic.exe upgrade head` from the backend directory."
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
    prompt: str = Field(min_length=1, max_length=30000)
    document_id: str | None = Field(default=None, max_length=80)
    purpose: Literal["general", "study", "business", "technical", "research", "presentation", "key_findings", "custom"] = "general"
    audience: Literal["general", "student", "developer_technical", "business_professional", "executive", "researcher", "expert"] = "general"
    depth: Literal["quick", "standard", "deep"] = "standard"
    translate_to: str | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def normalize_prompt(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("document_id", mode="before")
    @classmethod
    def normalize_document_id(cls, value: Any) -> str | None:
        document_id = str(value or "").strip()
        return document_id or None

    @field_validator("translate_to", mode="before")
    @classmethod
    def validate_translation_target(cls, value: Any) -> str | None:
        target = str(value or "").strip().lower()
        if not target:
            return None
        if target not in LANGUAGES:
            raise ValueError(f"translate_to must be one of: {', '.join(sorted(LANGUAGES))}")
        return target


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
    narration_text: str | None = Field(default=None, max_length=30000)
    narration_language: str | None = None

    @field_validator("narration_text", mode="before")
    @classmethod
    def normalize_narration_text(cls, value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    @field_validator("narration_language", mode="before")
    @classmethod
    def validate_narration_language(cls, value: Any) -> str | None:
        language = str(value or "").strip().lower()
        if not language:
            return None
        if language not in LANGUAGES:
            raise ValueError(f"narration_language must be one of: {', '.join(sorted(LANGUAGES))}")
        return language


class BatchProcessRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    options: ProcessOptions = Field(default_factory=ProcessOptions)


def extract_text(filename: str, data: bytes) -> tuple[str, list[dict[str, Any]]]:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        try:
            return data.decode("utf-8-sig"), []
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "The text document must be UTF-8 encoded.") from exc
    if ext == ".csv":
        import csv
        import io
        try:
            csv_text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "The CSV document must be UTF-8 encoded.") from exc
        rows = csv.reader(io.StringIO(csv_text))
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
            "provider": {"name": "deterministic-extractive", "model": None, "fallback_used": True,
                         "fallback_reason": "translation_unavailable"},
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
        "provider": {"name": "deterministic-extractive", "model": None, "fallback_used": True,
                     "fallback_reason": "no_llm_generation_requested"},
    }
    doc["result"] = result
    doc["options"] = options.model_dump() | {"summary_depth": options.summary_depth, "summary_mode": options.summary_mode or options.summary_depth, "output_language": options.language}
    doc["status"] = "COMPLETED"
    return result


def process_document(doc: dict[str, Any], options: ProcessOptions) -> dict[str, Any]:
    """Generate through the configured provider chain, then use a deterministic summary."""
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
    LOGGER.info("[SUMMARY] document_id=%s extracted_text_length=%d chunk_count=%d generation_started", doc.get("id", "unknown"), len(doc["text"]), len(doc["chunks"]))
    service = LLMService()

    def summarize(source: dict[str, Any], current_options: ProcessOptions) -> dict[str, Any]:
        try:
            summary = summarize_document(source, current_options, service)
            text = summary_text(summary.get("executive_summary")) or summary_text(summary.get("summary"))
            if not is_usable_summary(text):
                raise LLMGenerationError("Configured providers returned no usable summary.")
            summary["executive_summary"] = text
            summary["provider"] = service.provider_metadata
            summary["generation_status"] = "success"
            return summary
        except Exception as exc:
            metadata = service.provider_metadata
            LOGGER.warning("[SUMMARY] generation unavailable error_type=%s provider_trace=%s", type(exc).__name__, json.dumps(getattr(service, "provider_trace", []), separators=(",", ":")))
            LOGGER.warning("[SUMMARY] Using deterministic extractive fallback")
            fallback = process_extractively(source, current_options)
            target_name = LANGUAGES.get(current_options.language, current_options.language)
            limitations = ["Configured providers could not generate a usable summary; deterministic sentence ranking produced this source-grounded extractive summary.",
                           "The local extractive fallback does not infer risks, decisions, or actions and does not translate text."]
            if current_options.language != source["language"]:
                limitations.insert(0, f"Translation into {target_name} was unavailable; the usable source-language summary is shown.")
            fallback["limitations"] = limitations
            fallback["provider"] = {
                "name": "deterministic-extractive", "model": None, "fallback_used": True,
                "fallback_reason": metadata.get("fallback_reason", type(exc).__name__),
                "providers_attempted": ["openrouter", "openai-direct", "ollama-local"],
            }
            fallback["generation_status"] = "partial_success" if is_usable_summary(fallback.get("executive_summary")) else "failed"
            fallback["provider_status"] = {"primary": "OpenRouter failed or returned unusable output", "fallback": "OpenAI and Ollama Local failed or were unavailable", "final": "deterministic extractive summary"}
            return fallback

    result = run_analysis_workflow(doc, resolved_options, summarize)
    # A valid JSON response can still omit the requested summary fields. Treat
    # that as a generation failure and use the same source-backed fallback as
    # provider exceptions instead of persisting a blank successful result.
    generated_summary = summary_text(result.get("executive_summary")) or summary_text(result.get("summary"))
    if not is_usable_summary(generated_summary):
        LOGGER.warning("[SUMMARY] document_id=%s generated_text_length=%d unusable_response; fallback_started", doc.get("id", "unknown"), len(generated_summary))
        fallback = process_extractively(doc, resolved_options)
        fallback_summary = summary_text(fallback.get("executive_summary"))
        if is_usable_summary(fallback_summary):
            primary_info = result.get("provider") if isinstance(result.get("provider"), dict) else {}
            primary_name = primary_info.get("name") or service.provider_name
            primary_model = primary_info.get("model") or (service.model if service is not None else None)
            result = fallback
            result["limitations"] = [
                "The configured summary provider returned no usable summary; a deterministic extractive fallback was used.",
                *result.get("limitations", []),
            ]
            result["provider"] = {"name": "deterministic-extractive", "model": None, "fallback_used": True,
                                   "fallback_reason": "all_configured_providers_failed",
                                   "providers_attempted": ["openrouter", "openai-direct", "ollama-local"],
                                   "primary_provider": primary_name, "primary_model": primary_model}
            result["provider_status"] = {"primary": f"{primary_name} RETURNED NO USABLE SUMMARY", "final": "DETERMINISTIC EXTRACTIVE SUMMARY"}
            result["generation_status"] = "partial_success"
            result["warning"] = "The AI provider returned no usable summary; a source-grounded extractive summary was used."
            LOGGER.info("[SUMMARY] document_id=%s fallback_completed text_length=%d provider=deterministic_extractive_fallback", doc.get("id", "unknown"), len(fallback_summary))
        else:
            result["status"] = "failed"
            result["generation_status"] = "failed"
            result["executive_summary"] = None
            result["summary"] = None
            result["error"] = "No usable summary could be generated from the extracted document text."
            result["provider"] = {"name": "none", "model": None, "fallback_used": True}
            LOGGER.error("[SUMMARY] no deterministic fallback was usable document_id=%s provider_trace=%s", doc.get("id", "unknown"), json.dumps(getattr(service, "provider_trace", []), separators=(",", ":")))
    else:
        result["executive_summary"] = generated_summary
        provider_info = result.get("provider") if isinstance(result.get("provider"), dict) else {}
        fallback_used = bool(provider_info.get("fallback_used")) or provider_info.get("name") == "deterministic-extractive"
        result["provider"] = provider_info | {"fallback_used": fallback_used}
        result["generation_status"] = "partial_success" if fallback_used or result.get("validation_warning") or result.get("translation_failed_fallback") else "success"
    if result.get("generation_status") != "failed":
        result["summary"] = result.get("executive_summary")
    LOGGER.info("[SUMMARY] document_id=%s final_summary_length=%d generation_status=%s provider=%s model=%s", doc.get("id", "unknown"), len(result.get("executive_summary") or ""), result.get("generation_status"), (result.get("provider") or {}).get("name"), (result.get("provider") or {}).get("model"))
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
        source_language, target_language, is_usable_summary(result.get("executive_summary")),
        result.get("translation_validation_status"), result.get("validation_status"),
        result.get("stats", {}).get("regeneration_count", 0), result.get("language"),
    )
    evaluation_summary = result.get("_evaluation_input_summary")
    evaluation_claims = result.get("_evaluation_input_claims")
    if not evaluation_summary:
        evaluation_summary = result.get("executive_summary")
        evaluation_claims = result.get("claims", [])
    try:
        result["evaluation"] = evaluate_summary(
            doc.get("text", ""), evaluation_summary,
            claims=evaluation_claims or [], chunks=doc.get("chunks", []),
            source_language=source_language,
            summary_language=source_language if result.get("_evaluation_input_summary") else result.get("language", source_language),
            purpose=options.purpose, depth=resolved_depth, provider=result.get("provider"),
        )
    except Exception as exc:
        # A scoring implementation error must never be replaced with invented
        # values. Preserve the analysis response and mark every score unavailable.
        LOGGER.exception("Summary evaluation failed type=%s", type(exc).__name__)
        result["evaluation"] = evaluate_summary(doc.get("text", ""), None)
        result["evaluation"]["evaluation_status"] = "failed"
        result["evaluation"]["reason"] = "Evaluation could not be completed; no scores were produced."
    result.pop("_evaluation_input_summary", None)
    result.pop("_evaluation_input_claims", None)
    doc["result"] = result
    doc["options"] = options.model_dump() | {"summary_depth": resolved_depth, "summary_mode": requested_depth, "custom_sections": options.custom_sections, "custom_detail": options.custom_detail, "source_language": source_language, "requested_output_language": requested_output_language, "output_language": result.get("language"), "resolved_output_language": result.get("language")}
    doc["status"] = "FAILED" if result.get("generation_status") == "failed" else "COMPLETED"
    save_doc(doc)
    LOGGER.info("[SUMMARY] document_id=%s persistence=success user_id=%s", doc.get("id", "unknown"), doc.get("user_id", "unknown"))
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
    local_llm = get_local_llm_service()
    embedding_svc = get_embedding_service()
    emb_info = embedding_svc.get_info()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", DIRECT_OPENAI_BASE_URL).strip().rstrip("/")
    direct_openai_configured = bool(openai_key) and openai_key_problem(openai_key) is None and openai_base_url == DIRECT_OPENAI_BASE_URL
    return {
        "status": "ok" if database_status["schema_ready"] else "degraded",
        "mode": "openrouter" if os.getenv("OPENROUTER_API_KEY", "").strip() else ("openai-direct" if direct_openai_configured else ("ollama-local" if local_llm else "unavailable")),
        "llm": {
            "primary_provider": "openrouter",
            "primary_url": "https://openrouter.ai/api/v1",
            "primary_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "primary_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "secondary_provider": "openai-direct",
            "secondary_url": DIRECT_OPENAI_BASE_URL,
            "secondary_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "secondary_configured": direct_openai_configured,
            "secondary_configuration_error": (openai_key_problem(openai_key) or ("OPENAI_BASE_URL must be https://api.openai.com/v1. Custom OpenAI proxies are not supported." if openai_base_url != DIRECT_OPENAI_BASE_URL else None)),
            "fallback_provider": "ollama-local",
            "fallback_model": local_llm.model if local_llm else os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
            "fallback_available": local_llm is not None,
        "priority": ["openrouter", "openai-direct", "ollama-local"],
        },
        "database": database_status,
        "embedding": emb_info,
        "embeddings": {
            "provider": os.getenv("EMBEDDING_PROVIDER", "groq").strip(),
            "model": emb_info.get("model"),
            "configured": emb_info.get("configured", False),
        },
        "vector_db": {"configured": bool(os.getenv("VECTOR_DB_URL", "").strip()), "connected": False},
        "speech": {"provider": os.getenv("TTS_PROVIDER", "edge").strip().lower(), "configured": True},
        "ocr": ocr_status(),
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
    key_problem = openai_key_problem(openai_key)
    if key_problem:
        return await use_local_stt("external_credential_missing_or_invalid")

    # A configured but invalid/unreachable paid provider must not prevent local STT.
    try:
        import io
        from openai import OpenAI as OpenAIClient

        client = OpenAIClient(
            api_key=openai_key,
            base_url=DIRECT_OPENAI_BASE_URL,
            timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=0,
        )
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
    LOGGER.info("[DEMO] created document_id=%s user_id=%s extracted_text_length=%d chunk_count=%d", doc_id, current_user["id"], len(doc["text"]), len(doc["chunks"]))
    result = process_document(doc, ProcessOptions(summary_depth="standard", purpose="Executive briefing", language="en", audience="Leadership"))
    return public_doc(doc) | {"result": result}


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
    limit = config_int("MAX_DOCUMENTS_PER_BATCH", 10)
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


def _build_llm_test_system_prompt(payload: LocalLLMTestRequest) -> str:
    purpose_guidance = {
        "general": "Provide a clear, balanced general summary or answer.",
        "study": "Focus on concepts, definitions, important points, examples, and learning-oriented explanations.",
        "business": "Focus on decisions, outcomes, risks, metrics, actions, and business impact.",
        "technical": "Focus on architecture, implementation details, technologies, processes, dependencies, and technical findings.",
        "research": "Focus on methodology, findings, evidence, limitations, and conclusions.",
        "presentation": "Organize the key message and supporting points for a clear presentation.",
        "key_findings": "Prioritize the most important findings and their supporting evidence.",
        "custom": "Follow the user's stated purpose and requested output closely.",
    }
    audience_guidance = {
        "general": "Use clear language suitable for a broad audience.",
        "student": "Explain concepts clearly and avoid unnecessary jargon.",
        "developer_technical": "Use appropriate technical terminology and implementation details.",
        "business_professional": "Emphasize practical implications, trade-offs, and actions.",
        "executive": "Be concise and focus on decisions, impact, risks, and recommendations.",
        "researcher": "Preserve methodological nuance, evidence, and limitations.",
        "expert": "Allow technical and domain-specific detail without oversimplifying.",
    }
    depth_guidance = {
        "quick": "Give a short overview with only the most important points.",
        "standard": "Give a balanced response covering the main information.",
        "deep": "Give a detailed response covering important context, evidence, relationships, and supporting details.",
    }
    return "\n".join((
        "Answer the user's prompt clearly. If asked to summarize, summarize only supplied text and do not invent facts.",
        f"Purpose ({payload.purpose}): {purpose_guidance[payload.purpose]}",
        f"Audience ({payload.audience}): {audience_guidance[payload.audience]}",
        f"Depth ({payload.depth}): {depth_guidance[payload.depth]}",
    ))


@app.post("/api/llm/test")
@app.post("/api/local-llm/test", include_in_schema=False)
def local_llm_test(payload: LocalLLMTestRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    """Run a real prompt through the same authenticated centralized provider chain."""
    LOGGER.info("[LLM TEST] request_started user_id=%s", current_user.get("id", "authenticated"))
    if payload.document_id:
        get_doc(payload.document_id, current_user)
    service = LLMService()
    system_prompt = _build_llm_test_system_prompt(payload)
    try:
        if payload.translate_to:
            response_text = translate_for_audio(payload.prompt, payload.translate_to, service)
        else:
            response_text = service.complete(
                system_prompt,
                payload.prompt,
                max_tokens={"quick": 400, "standard": 800, "deep": 1400}[payload.depth],
            )
    except Exception as exc:
        LOGGER.warning("[LLM TEST] request_failed error_type=%s provider_trace=%s", type(exc).__name__, json.dumps(getattr(service, "provider_trace", []), separators=(",", ":")))
        return JSONResponse(status_code=502, content={
            "success": False,
            "error": "All configured AI providers are currently unavailable.",
            "provider_trace": service.provider_trace,
        })
    metadata = service.provider_metadata
    try:
        history_id = record_history(
            user_id=int(current_user["id"]),
            document_id=payload.document_id,
            activity_type="llm_test",
            title=(f"Translated {payload.purpose.replace('_', ' ').title()} prompt response"
                   if payload.translate_to else f"{payload.purpose.replace('_', ' ').title()} prompt response"),
            status="completed",
            input_text=payload.prompt,
            output_text=response_text,
            provider=metadata.get("name"),
            model=metadata.get("model"),
            extra_data={
                "purpose": payload.purpose,
                "audience": payload.audience,
                "depth": payload.depth,
                "translation_language": payload.translate_to,
                "fallback_used": bool(metadata.get("fallback_used")),
                "fallback_reason": metadata.get("fallback_reason"),
            },
        )
    except Exception as exc:
        history_id = None
        LOGGER.warning("[LLM TEST] history_save_failed error_type=%s", type(exc).__name__)
    return {
        "success": True,
        "response": response_text,
        "generation_settings": {
            "purpose": payload.purpose,
            "audience": payload.audience,
            "depth": payload.depth,
        },
        "history_id": history_id,
        "history_persisted": history_id is not None,
        "translation_language": payload.translate_to,
        "provider": metadata.get("name"),
        "model": metadata.get("model"),
        "fallback_used": bool(metadata.get("fallback_used")),
        "fallback_reason": metadata.get("fallback_reason"),
        "provider_metadata": metadata,
        "provider_trace": service.provider_trace,
    }


@app.post("/api/llm/diagnostics")
def llm_provider_diagnostics(current_user: dict[str, Any] = Depends(get_current_user)):
    """Explicitly probe each configured provider separately; startup generates no test traffic."""
    probe_system = "Return exactly the requested diagnostic token and nothing else."
    results: list[dict[str, Any]] = []
    for provider in (OpenRouterProvider(), OpenAIProvider()):
        name = provider.provider_name
        entry = {"provider": name, "model": provider.model,
                 "endpoint": provider.base_url, "base_url": provider.base_url,
                 "api_key_present": bool(provider.api_key), "request_started": False,
                 "configured": bool(provider.client), "reachable": False, "model_valid": False, "available": False,
                 "request_success": False, "response_non_empty": False}
        if name == "openai-direct":
            entry["credential_valid"] = not bool(provider.config_error)
            if provider.config_error:
                entry["configuration_error"] = provider.config_error
        if provider.client:
            probe_started = time.monotonic()
            try:
                entry["request_started"] = True
                token = "OPENROUTER_TEST_OK" if name == "openrouter" else "OPENAI_DIRECT_TEST_OK"
                response_text = provider.complete(probe_system, f"Reply with exactly: {token}", max_tokens=12)
                entry["response_non_empty"] = bool(response_text and response_text.strip())
                entry["response"] = response_text
                entry["token_matched"] = token in response_text
                entry["reachable"] = True
                entry["model_valid"] = entry["token_matched"]
                entry["request_success"] = entry["response_non_empty"] and entry["token_matched"]
                entry["available"] = entry["request_success"]
                if entry["request_success"]:
                    log_llm_success(name, provider.model, 200, time.monotonic() - probe_started, response_text)
                else:
                    error = ValueError("Diagnostic response did not match the expected token")
                    log_llm_error(name, provider.model, provider.base_url, provider.timeout, error,
                                  time.monotonic() - probe_started, status=200, category="ResponseValidationError")
            except Exception as exc:
                entry["error_type"] = _diagnostic_error_type(exc)
                entry["http_status"] = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                entry["error_message"] = _safe_diagnostic_message(exc)
        elif getattr(provider, "client_error", None):
            entry["error_type"] = "SDKCompatibilityError"
            entry["error_message"] = "Provider SDK could not be initialized."
        else:
            if name == "openai-direct" and getattr(provider, "config_error", None):
                entry["error_type"] = "InvalidConfiguration"
                entry["error_message"] = provider.config_error
            else:
                entry["error_type"] = "MissingAPIKey"
                key_name = "OPENROUTER_API_KEY" if name == "openrouter" else "OPENAI_API_KEY"
                entry["error_message"] = f"{key_name} is missing or empty; no request was sent."
        results.append(entry)
    local = get_local_llm_service()
    entry = {"provider": "ollama-local", "model": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
             "endpoint": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
             "configured": local is not None, "reachable": False, "model_installed": local is not None, "available": False}
    if local is not None:
        entry["model"] = local.model
        probe_started = time.monotonic()
        try:
            response_text = local.complete(probe_system, "Reply with exactly: OLLAMA_TEST_OK", max_tokens=12)
            entry["response_non_empty"] = bool(response_text and response_text.strip())
            entry["token_matched"] = "OLLAMA_TEST_OK" in response_text
            entry["reachable"] = True
            entry["request_success"] = entry["response_non_empty"] and entry["token_matched"]
            entry["available"] = entry["request_success"]
            if entry["request_success"]:
                log_llm_success("ollama-local", local.model, 200, time.monotonic() - probe_started, response_text)
            else:
                error = ValueError("Diagnostic response did not match the expected token")
                log_llm_error("ollama-local", local.model, entry["endpoint"], getattr(local, "timeout", 0),
                              error, time.monotonic() - probe_started, status=200, category="ResponseValidationError")
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
            entry["error_message"] = _safe_diagnostic_message(exc)
    else:
        entry["error_type"] = "ConfigurationError"
        entry["error_message"] = "Ollama endpoint is unavailable or the configured model is not installed."
    results.append(entry)
    return {"priority": ["openrouter", "openai-direct", "ollama-local"], "providers": results}


def _safe_diagnostic_message(exc: Exception) -> str:
    status = str(getattr(exc, "status_code", None) or getattr(exc, "code", None) or "")
    if "timeout" in type(exc).__name__.lower():
        return "Provider request timed out."
    if status == "401":
        return "Authentication failed. Check the configured API credential."
    if status == "403":
        return "Provider rejected the request because of permissions."
    if status == "429":
        return "Provider quota or rate limit was exceeded."
    if status == "404":
        return "Configured model or endpoint was not found."
    if status == "503":
        return "Provider is temporarily unavailable or overloaded."
    if status.startswith("5"):
        return "Provider service returned a server error."
    if status == "400":
        return "Provider rejected the request; check model and payload compatibility."
    if isinstance(exc, ValueError):
        return "Provider response was empty or invalid."
    return f"Provider request failed ({type(exc).__name__})."


def _diagnostic_error_type(exc: Exception) -> str:
    if getattr(exc, "category", None):
        return str(exc.category)
    status = str(getattr(exc, "status_code", None) or getattr(exc, "code", None) or "")
    if status == "401":
        return "AuthenticationError"
    if status == "403":
        return "PermissionError"
    if status == "404":
        return "ModelNotFound"
    if status == "429":
        return "RateLimitOrQuotaExceeded"
    if status == "503":
        return "ServiceUnavailable"
    if status == "400":
        return "InvalidRequest"
    if "timeout" in type(exc).__name__.lower():
        return "TimeoutError"
    if "connect" in type(exc).__name__.lower():
        return "NetworkError"
    return type(exc).__name__


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
    return result


@app.post("/api/documents/process-batch")
def process_batch(payload: BatchProcessRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    limit = config_int("MAX_DOCUMENTS_PER_BATCH", 10)
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
    service = LLMService()
    try:
        answer = answer_from_evidence(payload.question, evidence, service, output_language=output_lang)
    except LLMGenerationError as exc:
        LOGGER.warning("Grounded Q&A generation unavailable type=%s provider_trace=%s", type(exc).__name__, json.dumps(getattr(service, "provider_trace", []), separators=(",", ":")))
        raise HTTPException(503, "Configured providers could not generate a grounded answer.") from exc
    mode_desc = "Answer grounded in hybrid semantic/lexical evidence" if semantic_used else "Answer grounded in lexical evidence"
    provider = service.provider_metadata
    response = {
        "answer": answer, "sources": source_list, "grounded": True,
        "output_language": output_lang, "retrieval_provider": emb_svc.last_status.get("effective_provider", "lexical"),
        "llm_provider": provider.get("name"), "llm_model": provider.get("model"),
        "fallback_used": provider.get("fallback_used", False), "mode": mode_desc,
    }
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
    narration_text = payload.narration_text or result["executive_summary"]
    narration_language = payload.narration_language or result.get("language")
    if payload.language != narration_language:
        translation_service = LLMService()
        try:
            translated_text = translate_for_audio(narration_text, payload.language, translation_service)
            LOGGER.info("[LLM PROVIDER] selected=%s model=%s operation=audio_translation", translation_service.provider_name, translation_service.model)
        except (LLMGenerationError, ValueError) as exc:
            LOGGER.warning("Audio translation unavailable type=%s", type(exc).__name__)
            raise HTTPException(503, "Audio translation is unavailable from the configured providers.") from exc
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
async def compare(
    file_a: UploadFile | None = File(None),
    file_b: UploadFile | None = File(None),
    doc_id_a: str | None = Form(None),
    doc_id_b: str | None = Form(None),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    ta, tb = "", ""
    pages_a: list[dict[str, Any]] = []
    pages_b: list[dict[str, Any]] = []
    info_a: dict[str, Any] = {}
    info_b: dict[str, Any] = {}

    if doc_id_a and doc_id_b:
        doc_a = get_doc(doc_id_a, current_user)
        doc_b = get_doc(doc_id_b, current_user)
        ta, pages_a = doc_a.get("text", ""), doc_a.get("pages", [])
        tb, pages_b = doc_b.get("text", ""), doc_b.get("pages", [])
        info_a = {"name": doc_a["name"], "sha256": doc_a.get("content_sha256", doc_id_a)}
        info_b = {"name": doc_b["name"], "sha256": doc_b.get("content_sha256", doc_id_b)}
    elif file_a and file_b:
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
        ta, pages_a = extract_text(info_a["name"], a)
        tb, pages_b = extract_text(info_b["name"], b)
    else:
        raise HTTPException(422, "Provide either two files (file_a and file_b) or two document IDs (doc_id_a and doc_id_b).")

    ta, tb = validate_document_text(normalize(ta)), validate_document_text(normalize(tb))
    lang_a = detect_language(ta)
    lang_b = detect_language(tb)

    llm_service = LLMService()

    result = perform_document_comparison(
        doc_a_info=info_a,
        doc_b_info=info_b,
        text_a=ta,
        text_b=tb,
        pages_a=pages_a,
        pages_b=pages_b,
        lang_a=lang_a,
        lang_b=lang_b,
        llm_service=llm_service,
    )

    try:
        record_history(
            user_id=int(current_user["id"]),
            activity_type="document_comparison",
            title=f"Compare: {info_a['name']} vs {info_b['name']}",
            status="COMPLETED",
            input_text=f"A: {info_a['name']} ({result['document_a']['language']}) | B: {info_b['name']} ({result['document_b']['language']})",
            output_text=result["executive_summary"],
            language=result["document_a"]["language_code"],
            extra_data=result,
        )
    except Exception as exc:
        LOGGER.warning("Failed to record comparison history: %s", exc)

    return result


@app.get("/api/evaluation/{doc_id}")
@app.get("/api/documents/{doc_id}/evaluation")
def evaluation(doc_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    doc = get_doc(doc_id, current_user)
    result = doc.get("result")
    if result and isinstance(result.get("evaluation"), dict):
        return result["evaluation"]
    stored = get_latest_summary_evaluation(doc_id, int(current_user["id"]))
    if stored:
        return stored
    if not result:
        raise HTTPException(409, "Process the document before requesting evaluation.")
    raise HTTPException(409, "Evaluation is not available for this summary.")
