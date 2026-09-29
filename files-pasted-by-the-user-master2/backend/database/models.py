"""Relational persistence models. All text columns use MySQL utf8mb4."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .connection import Base


class UserRecord(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    history: Mapped[list["HistoryRecord"]] = relationship(cascade="all, delete-orphan", back_populates="user")
    upload_batches: Mapped[list["UploadBatchRecord"]] = relationship(cascade="all, delete-orphan", back_populates="user")
    combined_summaries: Mapped[list["CombinedSummaryRecord"]] = relationship(cascade="all, delete-orphan", back_populates="user")


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_content_sha256", "user_id", "content_sha256", unique=True),
        Index("ix_documents_user_text_sha256", "user_id", "extracted_text_sha256", unique=True),
        Index("ix_documents_user_created_at", "user_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("upload_batches.id", ondelete="SET NULL"), nullable=True, index=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    extracted_text_sha256: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False, default="und")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    ocr_records: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    ocr_warnings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    ocr_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    options: Mapped[dict | None] = mapped_column(JSON)
    audio_timings: Mapped[dict | None] = mapped_column(JSON)
    batch: Mapped["UploadBatchRecord | None"] = relationship(back_populates="documents", passive_deletes=True)
    history: Mapped[list["HistoryRecord"]] = relationship(back_populates="document", passive_deletes=True)
    pages: Mapped[list["DocumentPageRecord"]] = relationship(cascade="all, delete-orphan", order_by="DocumentPageRecord.page_number")
    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(cascade="all, delete-orphan", order_by="DocumentChunkRecord.chunk_number")
    summaries: Mapped[list["SummaryRecord"]] = relationship(cascade="all, delete-orphan")
    validations: Mapped[list["ValidationResultRecord"]] = relationship(cascade="all, delete-orphan")
    translations: Mapped[list["TranslationRecord"]] = relationship(cascade="all, delete-orphan")
    audio_files: Mapped[list["AudioRecord"]] = relationship(cascade="all, delete-orphan")
    questions: Mapped[list["QuestionAnswerRecord"]] = relationship(cascade="all, delete-orphan")


class DocumentPageRecord(Base):
    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_document_page_number"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(LONGTEXT, nullable=False, default="")
    ocr_data: Mapped[dict | None] = mapped_column(JSON)


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (Index("ix_document_chunks_document_id", "document_id"),)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    chunk_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_chunk_id: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(512))
    language: Mapped[str | None] = mapped_column(String(12))
    source: Mapped[str | None] = mapped_column(String(32))
    contains_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    embedding: Mapped[list[float] | None] = mapped_column(JSON)


class SummaryRecord(Base):
    __tablename__ = "summaries"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    result_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    claims: Mapped[list["SummaryClaimRecord"]] = relationship(cascade="all, delete-orphan")


class SummaryEvaluationRecord(Base):
    __tablename__ = "summary_evaluations"
    __table_args__ = (UniqueConstraint("summary_id", "evaluation_version", name="uq_summary_evaluation_version"),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    evaluation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Float)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    completeness_score: Mapped[float | None] = mapped_column(Float)
    factual_consistency_score: Mapped[float | None] = mapped_column(Float)
    groundedness_score: Mapped[float | None] = mapped_column(Float)
    summary_quality_score: Mapped[float | None] = mapped_column(Float)
    validation_accuracy_score: Mapped[float | None] = mapped_column(Float)
    evaluation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    evaluation_details: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ValidationResultRecord(Base):
    __tablename__ = "validation_results"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    groundedness_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SummaryClaimRecord(Base):
    __tablename__ = "summary_claims"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    summary_id: Mapped[int] = mapped_column(ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    claim: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(40))
    evidence: Mapped[str | None] = mapped_column(LONGTEXT)
    confidence: Mapped[float | None] = mapped_column(Float)
    sources: Mapped[list | None] = mapped_column(JSON)


class TranslationRecord(Base):
    __tablename__ = "translations"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    source_language: Mapped[str] = mapped_column(String(12), nullable=False)
    target_language: Mapped[str] = mapped_column(String(12), nullable=False)
    translated_text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AudioRecord(Base):
    __tablename__ = "audio_files"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    voice_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(2048))
    timings: Mapped[list | None] = mapped_column(JSON)
    narration_text: Mapped[str | None] = mapped_column(LONGTEXT)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class QuestionAnswerRecord(Base):
    __tablename__ = "question_answers"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    answer: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


Index("ix_audio_document_language", AudioRecord.document_id, AudioRecord.language)


class HistoryRecord(Base):
    __tablename__ = "history"
    __table_args__ = (
        Index("ix_history_user_created_at", "user_id", "created_at"),
        Index("ix_history_user_activity_type", "user_id", "activity_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    activity_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    input_text: Mapped[str | None] = mapped_column(LONGTEXT)
    output_text: Mapped[str | None] = mapped_column(LONGTEXT)
    language: Mapped[str | None] = mapped_column(String(12))
    provider: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    related_document_ids: Mapped[list | None] = mapped_column(JSON)
    extra_data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    user: Mapped["UserRecord"] = relationship(back_populates="history")
    document: Mapped["DocumentRecord | None"] = relationship(back_populates="history")


class UploadBatchRecord(Base):
    __tablename__ = "upload_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    total_files: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_files: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_files: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    user: Mapped["UserRecord"] = relationship(back_populates="upload_batches")
    documents: Mapped[list["DocumentRecord"]] = relationship(back_populates="batch", passive_deletes=True)


class CombinedSummaryRecord(Base):
    __tablename__ = "combined_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    summary_text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    language: Mapped[str] = mapped_column(String(12), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    user: Mapped["UserRecord"] = relationship(back_populates="combined_summaries")
