"""Bounded streaming access to GovReport; source records remain dataset-backed."""
from __future__ import annotations

import os
from hashlib import sha256
from collections.abc import Iterable
from typing import Any

DATASET_ID = os.getenv("GOVREPORT_DATASET_ID", "ccdv/govreport-summarization")
SPLITS = {"train", "validation", "test"}


def configured_limit(*, evaluation: bool = False) -> int:
    name = "GOVREPORT_EVALUATION_LIMIT" if evaluation else "GOVREPORT_DEV_LIMIT"
    default = 500 if evaluation else 20
    maximum = 50_000 if evaluation else 100
    try:
        requested = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if requested < 1:
        raise ValueError(f"{name} must be a positive integer")
    return min(requested, maximum)


def stream_records(split: str | None = None, limit: int | None = None, *, evaluation: bool = False) -> list[dict[str, str]]:
    """Read only the first bounded number of records from the selected split."""
    selected_split = split or os.getenv("GOVREPORT_SPLIT", "train")
    if selected_split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    maximum = 50_000 if evaluation else 100
    row_limit = configured_limit(evaluation=evaluation) if limit is None else limit
    if row_limit < 1 or row_limit > maximum:
        raise ValueError(f"limit must be between 1 and {maximum} for this mode")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("GovReport loading needs the optional 'datasets' dependency. Install backend/requirements.txt.") from exc

    try:
        load_options: dict[str, Any] = {"split": selected_split, "streaming": True}
        if cache_dir := os.getenv("GOVREPORT_CACHE_DIR", "").strip():
            load_options["cache_dir"] = cache_dir
        rows: Iterable[dict[str, Any]] = load_dataset(DATASET_ID, **load_options)
        selected = []
        for row in rows.take(row_limit):
            report = str(row["report"])
            summary = str(row["summary"])
            # The Hub's current Parquet export omits the card's `id` column.
            # Use the original id when present, otherwise a stable content ID.
            original_id = row.get("id")
            record_id = original_id or f"content-{sha256(report.encode('utf-8')).hexdigest()[:20]}"
            selected.append({"id": str(record_id), "id_source": "dataset" if original_id else "content_hash_fallback", "report": report, "summary": summary})
        return selected
    except Exception as exc:
        raise RuntimeError(f"Could not stream the GovReport {selected_split} split: {exc}") from exc


def get_record(record_id: str, split: str | None = None, limit: int | None = None, *, evaluation: bool = False) -> dict[str, str] | None:
    """Find one ID inside the bounded development subset."""
    for record in stream_records(split=split, limit=limit, evaluation=evaluation):
        if record["id"] == record_id:
            return record
    return None
