"""Request-path guardrails for untrusted documents and model responses."""
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
import csv
from io import BytesIO
from pathlib import PurePath
from typing import Any


MIN_USABLE_SUMMARY_CHARS = 12


def summary_text(value: Any) -> str:
    """Extract readable summary text from plain or structured model output."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(parsed, (dict, list)):
                return summary_text(parsed)
        return text
    if isinstance(value, list):
        parts = [summary_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        preferred = ("executive_summary", "summary", "overview", "text", "key_findings", "findings", "facts", "risks", "decisions", "recommendations", "action_items")
        parts = [summary_text(value[key]) for key in preferred if key in value]
        return " ".join(part for part in parts if part)
    return ""


def is_usable_summary(value: Any) -> bool:
    """Return whether a summary contains enough non-whitespace text to display."""
    return isinstance(value, str) and len(value.strip()) >= MIN_USABLE_SUMMARY_CHARS


def config_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def llm_retry_limit() -> int:
    raw = os.getenv("MAX_LLM_RETRIES", os.getenv("LLM_MAX_RETRIES", "1"))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1


def validate_upload(filename: str, data: bytes, *, existing_hashes: set[str] | None = None) -> dict[str, Any]:
    """Validate extension, path-like name, bytes, size, type, emptiness and duplicates."""
    if not filename or re.search(r"(^|[\\/])\.\.(?:[\\/]|$)", filename) or "\x00" in filename:
        raise ValueError("The filename is invalid.")
    base_name = re.split(r"[\\/]", filename)[-1]
    if base_name in {"", ".", ".."}:
        raise ValueError("The filename is invalid.")
    ext = PurePath(base_name).suffix.lower()
    allowed = {".pdf", ".docx", ".txt", ".json", ".csv", ".jpg", ".jpeg", ".png"}
    if ext not in allowed:
        raise ValueError("This file type is not supported.")
    max_bytes = config_int("MAX_UPLOAD_MB", 25) * 1024 * 1024
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > max_bytes:
        raise OverflowError("The file exceeds the configured upload limit.")

    valid = False
    if ext == ".pdf":
        valid = data.startswith(b"%PDF-")
    elif ext in {".jpg", ".jpeg"}:
        valid = data.startswith(b"\xff\xd8\xff")
    elif ext == ".png":
        valid = data.startswith(b"\x89PNG\r\n\x1a\n")
    elif ext == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                names = archive.namelist()
                unpacked_cap = config_int("MAX_ARCHIVE_UNPACKED_MB", 100) * 1024 * 1024
                valid = ("[Content_Types].xml" in names and "word/document.xml" in names
                         and sum(info.file_size for info in archive.infolist()) <= unpacked_cap)
        except (zipfile.BadZipFile, OSError):
            valid = False
    elif ext == ".json":
        try:
            json.loads(data.decode("utf-8-sig"))
            valid = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid = False
    elif ext == ".txt":
        try:
            data.decode("utf-8-sig")
            valid = True
        except UnicodeDecodeError:
            valid = False
    elif ext == ".csv":
        try:
            decoded = data.decode("utf-8-sig")
            list(csv.reader(decoded.splitlines()))
            valid = True
        except (UnicodeDecodeError, csv.Error):
            valid = False
    if not valid:
        raise ValueError("The file contents do not match a readable supported file.")
    digest = hashlib.sha256(data).hexdigest()
    if existing_hashes and digest in existing_hashes:
        raise ValueError("This document has already been uploaded.")
    return {"extension": ext, "sha256": digest, "size": len(data), "name": re.sub(r"[\x00-\x1f\x7f]", "_", base_name)[:180]}


def validate_document_text(text: str, *, max_chars: int | None = None) -> str:
    if not isinstance(text, str):
        raise ValueError("Unable to extract meaningful text from this document.")
    cleaned = text.replace("\x00", " ").replace("\ufffd", " ")
    cleaned = "".join(ch for ch in cleaned if ch in "\n\r\t" or ord(ch) >= 32)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    meaningful = sum(ch.isalnum() for ch in cleaned)
    min_chars = config_int("MIN_DOCUMENT_TEXT_CHARS", 20)
    if len(cleaned) < min_chars or meaningful < min(12, min_chars // 2):
        raise ValueError("Unable to extract meaningful text from this document.")
    limit = max_chars or config_int("MAX_DOCUMENT_CHARS", 1_000_000)
    if len(cleaned) > limit:
        raise OverflowError("Extracted document text exceeds the configured limit.")
    return cleaned


def validate_llm_input(system: str, user: str, *, operation: str, content: str = "") -> None:
    if operation not in {"map", "summary", "validation", "revision", "translation", "question", "audio_translation"}:
        raise ValueError("Invalid model operation.")
    if not system.strip() or not user.strip() or not content.strip():
        raise ValueError("Model input is empty.")
    if len(user) > config_int("MAX_CONTEXT_CHARS", 100_000):
        raise ValueError("Model context exceeds the configured limit.")


def operation_for_prompt(system: str) -> str:
    lowered = system.casefold()
    if "audio" in lowered and "translat" in lowered:
        return "audio_translation"
    if "translat" in lowered:
        return "translation"
    if "validate each claim" in lowered or "claim validation" in lowered:
        return "validation"
    if "revise" in lowered or "regenerat" in lowered:
        return "revision"
    if "question" in lowered or "answer" in lowered:
        return "question"
    if "passages" in lowered or "extract source-grounded notes" in lowered:
        return "map"
    return "summary"


def validate_provider_response(response: Any) -> str:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The model returned an empty response.")
    return response.strip()


def validate_llm_output(data: Any, *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise ValueError("The model returned malformed structured output.")
    return data


def validate_model_json_contract(data: dict[str, Any], system: str, user: str) -> dict[str, Any]:
    """Apply operation-specific shape checks before generated JSON enters the workflow."""
    try:
        payload = json.loads(user)
    except (TypeError, json.JSONDecodeError):
        payload = {}
    shape = payload.get("required_shape") if isinstance(payload, dict) else None
    if isinstance(shape, dict):
        for field, sample in shape.items():
            if field not in data:
                raise ValueError("The model response is missing a required field.")
            value = data[field]
            if isinstance(sample, str) and not isinstance(value, str):
                raise ValueError("The model response has an invalid field type.")
            if field == "executive_summary" and (not isinstance(value, str) or not value.strip()):
                raise ValueError("The model response has an empty executive summary.")
            if isinstance(sample, list) and not isinstance(value, list):
                raise ValueError("The model response has an invalid field type.")
            if isinstance(sample, list) and sample and isinstance(sample[0], dict):
                for item in value:
                    if (not isinstance(item, dict) or not isinstance(item.get("text"), str)
                            or not isinstance(item.get("source_chunk_ids"), list)
                            or not item["source_chunk_ids"]):
                        raise ValueError("The model response contains an invalid source-linked item.")
    if isinstance(payload, dict) and "passages" in payload:
        notes = data.get("notes")
        if not isinstance(notes, list) or not notes or any(
            not isinstance(item, dict) or not isinstance(item.get("text"), str)
            or not item["text"].strip() or not isinstance(item.get("source_chunk_ids"), list)
            or not item["source_chunk_ids"] for item in notes
        ):
            raise ValueError("The model response contains invalid source notes.")
    if isinstance(payload, dict) and "claims" in payload:
        results = data.get("results")
        allowed = {item.get("id") for item in payload["claims"] if isinstance(item, dict)}
        seen: set[str] = set()
        if not isinstance(results, list):
            raise ValueError("The model response is missing claim validation results.")
        for item in results:
            if (not isinstance(item, dict) or item.get("id") not in allowed
                    or item.get("status") not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
                    or not isinstance(item.get("source_chunk_ids"), list)
                    or not isinstance(item.get("reason"), str)):
                raise ValueError("The model response contains malformed claim validation results.")
            seen.add(item["id"])
        if seen != allowed:
            raise ValueError("The model response did not validate every claim.")
    if isinstance(payload, dict) and "validated_summary_fields" in payload:
        for category in payload["validated_summary_fields"]:
            if not isinstance(data.get(category), list) or any(not isinstance(item, str) or not item.strip() for item in data[category]):
                raise ValueError("The model response contains malformed translated fields.")
    return data


def validate_summary(summary: str, *, max_chars: int | None = None) -> str:
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("The summary is empty.")
    limit = max_chars or config_int("MAX_SUMMARY_CHARS", 40_000)
    return summary.strip()[:limit]


def validate_claims(claims: list[dict[str, Any]], chunk_ids: set[str]) -> list[dict[str, Any]]:
    valid = []
    # TRANSLATION_UNAVAILABLE is a sentinel added when translation completely failed;
    # it is distinct from claim-level validation statuses and must be preserved.
    statuses = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "UNAVAILABLE", "TRANSLATION_UNAVAILABLE"}
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
            continue
        sources = [s for s in claim.get("sources", []) if isinstance(s, dict) and s.get("chunk_id") in chunk_ids]
        claim["sources"] = sources
        if claim.get("status") not in statuses:
            claim["status"] = "UNCERTAIN"
        valid.append(claim)
    return valid


def unsupported_critical_values(claim: str, evidence: str) -> set[str]:
    """Find numbers, dates, currencies and identifier-like values absent from evidence."""
    pattern = re.compile(
        r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d+)?%?|\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b|"
        r"\b\d[\d,]*(?:\.\d+)?\s?%|\b\d[\d,]*(?:\.\d+)?\b|\b[A-Z]{2,}[\w-]*\d[\w-]*\b)",
        re.IGNORECASE,
    )
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    evidence_values = {normalize(value) for value in pattern.findall(evidence)}
    return {value for value in pattern.findall(claim) if normalize(value) not in evidence_values}


def validate_translation(translated: str, original: str) -> str:
    if not isinstance(translated, str) or not translated.strip():
        return original
    return translated.strip()
