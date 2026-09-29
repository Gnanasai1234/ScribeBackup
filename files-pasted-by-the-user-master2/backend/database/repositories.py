"""Transactional document repository used by API routes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import String, cast, func, inspect, or_, select, text
from sqlalchemy.orm import selectinload

from .connection import Base, SessionLocal, engine
from .models import (AudioRecord, DocumentChunkRecord, DocumentPageRecord, DocumentRecord,
                     CombinedSummaryRecord, HistoryRecord, QuestionAnswerRecord, SummaryClaimRecord,
                     SummaryRecord, TranslationRecord, UploadBatchRecord, ValidationResultRecord,
                     UserRecord)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def save_document(doc: dict[str, Any]) -> None:
    """Atomically upsert a document snapshot and its normalized child records."""
    doc_id = str(doc["id"])
    created = datetime.fromisoformat(str(doc.get("created_at") or datetime.now(timezone.utc).isoformat()))
    if created.tzinfo:
        created = created.astimezone(timezone.utc).replace(tzinfo=None)
    pages = doc.get("pages", []) or []
    chunks = doc.get("chunks", []) or []
    audio = doc.get("audio_timings") or {}
    result = doc.get("result")
    with SessionLocal.begin() as session:
        row = session.get(DocumentRecord, doc_id)
        owner_id = int(doc["user_id"]) if doc.get("user_id") is not None else None
        batch_id = doc.get("batch_id")
        if batch_id:
            batch = session.get(UploadBatchRecord, str(batch_id))
            if batch is None or owner_id is None or batch.user_id != owner_id:
                raise PermissionError("Upload batch does not belong to the document owner.")
        if row is None:
            row = DocumentRecord(id=doc_id, name=doc.get("name", "Document"), text=doc.get("text", ""),
                                 language=doc.get("language", "und"), status=doc.get("status", "UPLOADED"),
                                 created_at=created, user_id=owner_id, batch_id=str(batch_id) if batch_id else None,
                                 content_sha256=doc.get("content_sha256"),
                                 extracted_text_sha256=doc.get("extracted_text_sha256"))
            session.add(row)
        else:
            if owner_id is not None and row.user_id not in (None, owner_id):
                raise PermissionError("Document does not belong to the supplied owner.")
            if owner_id is not None and row.user_id is None:
                row.user_id = owner_id
            if batch_id:
                row.batch_id = str(batch_id)
            if doc.get("content_sha256"):
                row.content_sha256 = doc["content_sha256"]
            if doc.get("extracted_text_sha256"):
                row.extracted_text_sha256 = doc["extracted_text_sha256"]
        row.name = doc.get("name", "Document")
        row.text = doc.get("text", "")
        row.language = doc.get("language", "und")
        row.status = doc.get("status", "UPLOADED")
        row.ocr_records = _jsonable(doc.get("ocr_records", []))
        row.ocr_warnings = _jsonable(doc.get("ocr_warnings", []))
        row.ocr_detected = bool(doc.get("ocr_detected"))
        row.options = _jsonable(doc.get("options")) if doc.get("options") is not None else None
        row.evaluation = _jsonable(doc.get("evaluation")) if doc.get("evaluation") is not None else None
        row.reference_summary = doc.get("reference_summary")
        row.dataset = _jsonable(doc.get("dataset")) if doc.get("dataset") is not None else None
        row.depth_comparison = _jsonable(doc.get("depth_comparison")) if doc.get("depth_comparison") is not None else None
        row.audio_timings = _jsonable(audio) if audio else None
        row.pages.clear()
        row.chunks.clear()
        row.summaries.clear()
        row.validations.clear()
        row.translations.clear()
        row.audio_files.clear()
        # Flush orphan deletes before inserting replacement child rows. This
        # keeps the upsert safe when an uploaded PDF is processed and saved again
        # in the same document lifecycle (page numbers are unique per document).
        session.flush()
        for p in pages:
            row.pages.append(DocumentPageRecord(page_number=int(p.get("page", p.get("page_number", 0)) or 0),
                                                text=p.get("text", ""), ocr_data=_jsonable(p.get("ocr")) if p.get("ocr") else None))
        for index, chunk in enumerate(chunks, 1):
            row.chunks.append(DocumentChunkRecord(
                source_chunk_id=str(chunk.get("id") or f"chunk_{index:03d}"), chunk_number=int(chunk.get("chunk_number", index)),
                text=chunk.get("text", ""), page_number=chunk.get("page"), section=chunk.get("section"),
                language=chunk.get("language"), source=chunk.get("source"),
                contains_ocr=bool(chunk.get("contains_ocr")), ocr_confidence=chunk.get("ocr_confidence"),
                embedding=_jsonable(chunk.get("embedding")) if chunk.get("embedding") else None))
        if result:
            summary = SummaryRecord(summary_text=result.get("executive_summary", ""), language=result.get("language", doc.get("language", "und")),
                                    mode=result.get("summary_mode", result.get("depth", "standard")), created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                                    result_data=_jsonable(result))
            for index, claim in enumerate(result.get("claims", []) or [], 1):
                summary.claims.append(SummaryClaimRecord(ordinal=index, claim=claim.get("claim", ""), category=claim.get("category"),
                    status=claim.get("status"), evidence=claim.get("evidence"), confidence=claim.get("confidence"), sources=_jsonable(claim.get("sources", []))))
            row.summaries.append(summary)
            quality = result.get("quality", {}) or {}
            if result.get("validation") or result.get("claims"):
                row.validations.append(ValidationResultRecord(status=result.get("validation_status", result.get("status", "unavailable")),
                    groundedness_score=quality.get("groundedness"), created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            translations = result.get("translations", {})
            if isinstance(translations, dict):
                for lang, translated in translations.items():
                    if isinstance(translated, str):
                        row.translations.append(TranslationRecord(source_language=doc.get("language", "und"), target_language=lang,
                            translated_text=translated, created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
            translated_summary = result.get("executive_summary", "")
            result_language = result.get("language", doc.get("language", "und"))
            if translated_summary and result_language != doc.get("language", "und"):
                row.translations.append(TranslationRecord(source_language=doc.get("language", "und"), target_language=result_language,
                    translated_text=translated_summary, created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        if audio:
            row.audio_files.append(AudioRecord(language=audio.get("language", row.language), voice_id=audio.get("voice_id", "browser-default"),
                provider=audio.get("provider", "browser"), file_path=audio.get("file_path"), timings=_jsonable(audio.get("timings", [])),
                narration_text=audio.get("text"), fallback_used=bool(audio.get("fallback_used")),
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)))


def get_document(document_id: str, user_id: int | None = None) -> dict[str, Any] | None:
    with SessionLocal() as session:
        stmt = select(DocumentRecord).where(DocumentRecord.id == document_id)
        if user_id is not None:
            stmt = stmt.where(DocumentRecord.user_id == user_id)
        stmt = stmt.options(
            selectinload(DocumentRecord.pages), selectinload(DocumentRecord.chunks), selectinload(DocumentRecord.summaries),
            selectinload(DocumentRecord.audio_files))
        row = session.scalar(stmt)
        if row is None:
            return None
        summary = max(row.summaries, key=lambda s: s.id or 0, default=None)
        created = row.created_at.replace(tzinfo=timezone.utc).isoformat()
        return {
            "id": row.id, "name": row.name, "text": row.text, "language": row.language, "status": row.status, "user_id": row.user_id,
            "batch_id": row.batch_id,
            "created_at": created, "pages": [{"page": p.page_number, "text": p.text, **({"ocr": p.ocr_data} if p.ocr_data else {})} for p in row.pages],
            "chunks": [{"id": c.source_chunk_id, "documentId": row.id, "text": c.text, "page": c.page_number, "section": c.section,
                        "language": c.language, "chunk_number": c.chunk_number, "source": c.source, "contains_ocr": c.contains_ocr,
                        "ocr_confidence": c.ocr_confidence, "embedding": c.embedding} for c in row.chunks],
            "ocr_records": row.ocr_records or [], "ocr_warnings": row.ocr_warnings or [], "ocr_detected": row.ocr_detected,
            "options": row.options, "evaluation": row.evaluation, "reference_summary": row.reference_summary,
            "dataset": row.dataset, "depth_comparison": row.depth_comparison,
            "result": ({**summary.result_data, "generated_at": summary.created_at.replace(tzinfo=timezone.utc).isoformat()} if summary else None), "audio_timings": row.audio_timings,
        }


def find_duplicate_document(user_id: int, content_sha256: str, extracted_text_sha256: str) -> bool:
    """Match file or extracted-text hashes only within the authenticated owner's documents."""
    with SessionLocal() as session:
        direct_match = session.scalar(select(DocumentRecord.id).where(
            DocumentRecord.user_id == user_id,
            or_(DocumentRecord.content_sha256 == content_sha256,
                DocumentRecord.extracted_text_sha256 == extracted_text_sha256),
        ).limit(1))
        if direct_match is not None:
            return True
        # Hash older owner documents lazily in MySQL so records predating the
        # additive migration retain duplicate detection without loading text.
        legacy_match = session.scalar(select(DocumentRecord.id).where(
            DocumentRecord.user_id == user_id, DocumentRecord.extracted_text_sha256.is_(None),
            func.lower(func.sha2(DocumentRecord.text, 256)) == extracted_text_sha256).limit(1))
        return legacy_match is not None


def save_document_chunk_embeddings(document_id: str, user_id: int,
                                   embeddings: dict[str, list[float]]) -> int:
    """Persist vectors only for chunks on a document owned by the caller."""
    if not embeddings:
        return 0
    with SessionLocal.begin() as session:
        owned_document = session.scalar(select(DocumentRecord.id).where(
            DocumentRecord.id == document_id, DocumentRecord.user_id == user_id))
        if owned_document is None:
            return 0
        rows = session.scalars(select(DocumentChunkRecord).where(
            DocumentChunkRecord.document_id == document_id,
            DocumentChunkRecord.source_chunk_id.in_(list(embeddings)),
        )).all()
        for row in rows:
            vector = embeddings.get(row.source_chunk_id)
            row.embedding = _jsonable(vector) if vector else None
        return len(rows)


def list_documents(user_id: int | None = None) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(DocumentRecord)
        if user_id is not None:
            stmt = stmt.where(DocumentRecord.user_id == user_id)
        rows = session.scalars(stmt.options(selectinload(DocumentRecord.pages),
            selectinload(DocumentRecord.chunks)).order_by(DocumentRecord.created_at.desc())).all()
        return [{"id": row.id, "name": row.name, "text": row.text, "language": row.language, "status": row.status, "user_id": row.user_id,
                 "batch_id": row.batch_id,
                 "created_at": row.created_at.replace(tzinfo=timezone.utc).isoformat(),
                 "pages": [{"page": p.page_number, "text": p.text} for p in row.pages],
            "chunks": [{"id": c.source_chunk_id} for c in row.chunks],
                 "ocr_detected": row.ocr_detected, "ocr_warnings": row.ocr_warnings or [], "result": None}
                for row in rows]


def search_documents(user_id: int, query: str) -> list[dict[str, Any]]:
    """Search one owner's persisted documents by name, language, metadata, or extracted text."""
    term = query.strip().casefold()
    if not term:
        return list_documents(user_id)
    # contains() binds the pattern as a parameter and auto-escapes LIKE wildcards,
    # so a user's percent/underscore characters are literal search text.
    predicates = [
        func.lower(column).contains(term, autoescape=True)
        for column in (DocumentRecord.name, DocumentRecord.language, DocumentRecord.status, DocumentRecord.text)
    ]
    predicates.append(func.lower(cast(DocumentRecord.dataset, String)).contains(term, autoescape=True))
    with SessionLocal() as session:
        page_count = select(func.count(DocumentPageRecord.id)).where(
            DocumentPageRecord.document_id == DocumentRecord.id
        ).scalar_subquery()
        rows = session.execute(
            select(DocumentRecord, page_count.label("page_count"))
            .where(DocumentRecord.user_id == user_id, or_(*predicates))
            .options(selectinload(DocumentRecord.chunks))
            .order_by(DocumentRecord.created_at.desc())
        ).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "text": row.text,
                "language": row.language,
                "status": row.status,
                "user_id": row.user_id,
                "batch_id": row.batch_id,
                "created_at": row.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "pages": [],
                "pages_count": int(pages_count or 0),
                "chunks": [{"id": chunk.source_chunk_id} for chunk in row.chunks],
                "ocr_detected": row.ocr_detected,
                "ocr_warnings": row.ocr_warnings or [],
                "result": None,
            }
            for row, pages_count in rows
        ]


def create_user(name: str, email: str, password_hash: str) -> UserRecord:
    with SessionLocal.begin() as session:
        user = UserRecord(name=name, email=email, password_hash=password_hash,
                          created_at=datetime.now(timezone.utc).replace(tzinfo=None))
        session.add(user)
        session.flush()
        session.refresh(user)
        session.expunge(user)
        return user


def get_user_by_email(email: str) -> UserRecord | None:
    with SessionLocal() as session:
        return session.scalar(select(UserRecord).where(UserRecord.email == email))


def get_user_by_id(user_id: int) -> UserRecord | None:
    with SessionLocal() as session:
        return session.get(UserRecord, user_id)


def delete_document(document_id: str, user_id: int) -> bool:
    with SessionLocal.begin() as session:
        row = session.scalar(select(DocumentRecord).where(
            DocumentRecord.id == document_id, DocumentRecord.user_id == user_id))
        if row is None:
            return False
        session.delete(row)
        return True


def database_is_connected() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def database_schema_status() -> dict[str, Any]:
    """Check the configured MySQL database and required mapped tables/columns."""
    configured_database = engine.url.database
    try:
        with engine.connect() as connection:
            current_database = connection.execute(text("SELECT DATABASE()")).scalar_one_or_none()
            inspector = inspect(connection)
            existing_tables = set(inspector.get_table_names())
            expected_tables = set(Base.metadata.tables)
            missing_tables = sorted(expected_tables - existing_tables)
            missing_columns: dict[str, list[str]] = {}
            for table_name in sorted(expected_tables & existing_tables):
                existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
                missing = sorted({column.name for column in Base.metadata.tables[table_name].columns} - existing_columns)
                if missing:
                    missing_columns[table_name] = missing
            connected = current_database is not None
            schema_ready = (
                connected
                and configured_database == "scribe"
                and current_database == configured_database
                and not missing_tables
                and not missing_columns
            )
            return {
                "connected": connected,
                "schema_ready": schema_ready,
                "database": current_database,
                "configured_database": configured_database,
                "host": engine.url.host,
                "port": engine.url.port or 3306,
                "tables": sorted(existing_tables),
                "required_table_count": len(expected_tables),
                "missing_tables": missing_tables,
                "missing_columns": missing_columns,
            }
    except Exception as exc:
        return {
            "connected": False,
            "schema_ready": False,
            "database": configured_database,
            "configured_database": configured_database,
            "host": engine.url.host,
            "port": engine.url.port or 3306,
            "tables": [],
            "required_table_count": len(Base.metadata.tables),
            "missing_tables": [],
            "missing_columns": {},
            "error_type": type(exc).__name__,
        }


def save_question(document_id: str, question: str, answer: dict[str, Any], language: str) -> None:
    with SessionLocal.begin() as session:
        session.add(QuestionAnswerRecord(document_id=document_id, question=question,
            answer=answer.get("answer", ""), language=language, sources=_jsonable(answer.get("sources", [])),
            created_at=datetime.now(timezone.utc).replace(tzinfo=None)))


def create_upload_batch(user_id: int, total_files: int, batch_id: str) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with SessionLocal.begin() as session:
        session.add(UploadBatchRecord(id=batch_id, user_id=user_id, total_files=total_files,
            completed_files=0, failed_files=0, status="processing", created_at=now, updated_at=now))


def update_upload_batch(batch_id: str, user_id: int, completed_files: int, failed_files: int, status: str) -> bool:
    with SessionLocal.begin() as session:
        batch = session.scalar(select(UploadBatchRecord).where(
            UploadBatchRecord.id == batch_id, UploadBatchRecord.user_id == user_id))
        if batch is None:
            return False
        batch.completed_files = completed_files
        batch.failed_files = failed_files
        batch.status = status
        batch.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return True


def get_upload_batch(batch_id: str, user_id: int) -> dict[str, Any] | None:
    with SessionLocal() as session:
        batch = session.scalar(select(UploadBatchRecord).where(
            UploadBatchRecord.id == batch_id, UploadBatchRecord.user_id == user_id).options(
                selectinload(UploadBatchRecord.documents)))
        if batch is None:
            return None
        return {
            "id": batch.id, "total_files": batch.total_files, "completed_files": batch.completed_files,
            "failed_files": batch.failed_files, "status": batch.status,
            "created_at": batch.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "updated_at": batch.updated_at.replace(tzinfo=timezone.utc).isoformat(),
            "documents": [{"id": doc.id, "name": doc.name, "status": doc.status}
                          for doc in batch.documents if doc.user_id == user_id],
        }


def record_history(*, user_id: int, activity_type: str, title: str, status: str,
                   document_id: str | None = None, input_text: str | None = None,
                   output_text: str | None = None, language: str | None = None,
                   provider: str | None = None, model: str | None = None,
                   error_message: str | None = None, related_document_ids: list[str] | None = None,
                   extra_data: dict[str, Any] | None = None) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with SessionLocal.begin() as session:
        session.add(HistoryRecord(user_id=user_id, document_id=document_id, activity_type=activity_type,
            title=title, input_text=input_text, output_text=output_text, language=language,
            provider=provider, model=model, status=status, error_message=error_message,
            related_document_ids=_jsonable(related_document_ids) if related_document_ids is not None else None,
            extra_data=_jsonable(extra_data) if extra_data is not None else None, created_at=now, updated_at=now))


def list_history(user_id: int, *, page: int = 1, limit: int = 20, activity_type: str | None = None) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        stmt = select(HistoryRecord).where(HistoryRecord.user_id == user_id)
        if activity_type:
            stmt = stmt.where(HistoryRecord.activity_type == activity_type)
        rows = session.scalars(stmt.order_by(HistoryRecord.created_at.desc(), HistoryRecord.id.desc())
            .offset((page - 1) * limit).limit(limit)).all()
        return [{"id": row.id, "document_id": row.document_id, "activity_type": row.activity_type,
            "title": row.title, "input_text": row.input_text, "output_text": row.output_text,
            "language": row.language, "provider": row.provider, "model": row.model,
            "status": row.status, "error_message": row.error_message,
            "related_document_ids": row.related_document_ids, "extra_data": row.extra_data,
            "created_at": row.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "updated_at": row.updated_at.replace(tzinfo=timezone.utc).isoformat()} for row in rows]


def list_history_documents(user_id: int, *, page: int = 1, limit: int = 20,
                           search: str | None = None, file_type: str | None = None,
                           status: str | None = None, language: str | None = None,
                           date_from: datetime | None = None, date_to: datetime | None = None,
                           sort: str = "newest") -> dict[str, Any]:
    """Return one page of the user's documents with summary and processing metadata."""
    filters = [DocumentRecord.user_id == user_id]
    if search:
        filters.append(DocumentRecord.name.ilike(f"%{search.strip()}%"))
    if file_type:
        suffix = f"%.{file_type.strip().lower().lstrip('.')}"
        filters.append(func.lower(DocumentRecord.name).like(suffix))
    if status:
        filters.append(func.lower(DocumentRecord.status) == status.strip().lower())
    if language:
        filters.append(func.lower(DocumentRecord.language) == language.strip().lower())
    if date_from:
        filters.append(DocumentRecord.created_at >= date_from)
    if date_to:
        filters.append(DocumentRecord.created_at < date_to)
    with SessionLocal() as session:
        total = int(session.scalar(select(func.count()).select_from(DocumentRecord).where(*filters)) or 0)
        ordering = (DocumentRecord.created_at.asc(), DocumentRecord.id.asc()) if sort == "oldest" else (DocumentRecord.created_at.desc(), DocumentRecord.id.desc())
        rows = session.scalars(select(DocumentRecord).where(*filters)
            .options(selectinload(DocumentRecord.summaries), selectinload(DocumentRecord.audio_files),
                     selectinload(DocumentRecord.translations), selectinload(DocumentRecord.history))
            .order_by(*ordering)
            .offset((page - 1) * limit).limit(limit)).all()
        items = []
        for row in rows:
            summary = max(row.summaries, key=lambda item: item.id or 0, default=None)
            result = summary.result_data if summary else {}
            processing_event = max((event for event in row.history
                                    if event.activity_type in {"summarization", "processing"}),
                                   key=lambda event: event.created_at, default=None)
            provider = result.get("provider") if isinstance(result, dict) else None
            validation = result.get("validation_status", result.get("status")) if isinstance(result, dict) else None
            summary_status = result.get("status") if isinstance(result, dict) else None
            if summary_status is None and processing_event and processing_event.status == "failed":
                summary_status = "failed"
            generated_at = (summary.created_at if summary else processing_event.created_at if processing_event else None)
            generated_at = generated_at.replace(tzinfo=timezone.utc).isoformat() if generated_at else None
            items.append({
                "id": row.id, "name": row.name, "file_type": row.name.rsplit(".", 1)[-1].lower() if "." in row.name else "unknown",
                "language": row.language, "status": row.status,
                "created_at": row.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "processed_at": generated_at, "summary_status": summary_status,
                "summary_type": result.get("summary_mode", result.get("depth")) if isinstance(result, dict) else None,
                "validation_status": validation,
                "provider": provider.get("name") if isinstance(provider, dict) else None,
                "model": provider.get("model") if isinstance(provider, dict) else None,
                "fallback_used": provider.get("fallback_used") if isinstance(provider, dict) else None,
                "has_audio": bool(row.audio_files or row.audio_timings),
                "has_translation": bool(row.translations), "can_export": summary is not None,
                "preview": (result.get("executive_summary", "")[:360] if isinstance(result, dict) else ""),
            })
        return {"items": items, "page": page, "limit": limit, "total": total,
                "pages": (total + limit - 1) // limit}


def save_combined_summary(*, user_id: int, document_ids: list[str], summary_text: str,
                          language: str, status: str = "completed",
                          details: dict[str, Any] | None = None) -> int:
    with SessionLocal.begin() as session:
        # Store only document IDs that still belong to this authenticated user.
        owned_ids = set(session.scalars(select(DocumentRecord.id).where(
            DocumentRecord.user_id == user_id, DocumentRecord.id.in_(document_ids))).all())
        if owned_ids != set(document_ids):
            raise PermissionError("Combined summary contains a document not owned by the user.")
        ordered_ids = [document_id for document_id in document_ids if document_id in owned_ids]
        record = CombinedSummaryRecord(user_id=user_id, document_ids=ordered_ids,
            summary_text=summary_text, language=language, status=status,
            details=_jsonable(details) if details is not None else None,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None))
        session.add(record)
        session.flush()
        return int(record.id)


def get_combined_summary(summary_id: int, user_id: int) -> dict[str, Any] | None:
    with SessionLocal() as session:
        record = session.scalar(select(CombinedSummaryRecord).where(
            CombinedSummaryRecord.id == summary_id, CombinedSummaryRecord.user_id == user_id))
        if record is None:
            return None
        owned_documents = session.execute(select(DocumentRecord.id, DocumentRecord.name).where(
            DocumentRecord.user_id == user_id, DocumentRecord.id.in_(record.document_ids))).all()
        names = {document_id: name for document_id, name in owned_documents}
        details = record.details or {}
        sections = details.get("sections", []) if isinstance(details, dict) else []
        claims = details.get("claims", []) if isinstance(details, dict) else []
        return {
            "id": record.id, "summary": record.summary_text, "language": record.language,
            "status": record.status, "document_ids": record.document_ids,
            "document_names": [names[doc_id] for doc_id in record.document_ids if doc_id in names],
            "sections": sections, "claims": claims,
            "generated_at": record.created_at.replace(tzinfo=timezone.utc).isoformat(),
        }
