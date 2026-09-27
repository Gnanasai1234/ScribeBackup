"""Convert one selected GovReport row into the app's in-memory document shape."""
from __future__ import annotations

import re
import uuid
from typing import Any


def to_document(record: dict[str, str], split: str) -> dict[str, Any]:
    # Imported lazily to avoid making the dataset helper the app entry point.
    from app.main import detect_language, make_chunks, normalize

    report = normalize(record["report"])
    pages: list[dict[str, Any]] = []
    document_id = str(uuid.uuid4())
    return {
        "id": document_id,
        "name": f"GovReport {record['id']}.txt",
        "text": report,
        "pages": pages,
        "chunks": make_chunks(report, pages, document_id=document_id),
        "language": detect_language(report),
        "status": "UPLOADED",
        "created_at": None,
        "result": None,
        "dataset": {"name": "ccdv/govreport-summarization", "split": split, "record_id": record["id"], "record_id_source": record.get("id_source", "dataset")},
        # The reference is evaluator-only and is never passed to summarization or validation.
        "reference_summary": re.sub(r"\s+", " ", record["summary"]).strip(),
    }
