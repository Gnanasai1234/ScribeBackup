"""Store upload and extracted-text hashes for owner-scoped duplicate checks."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_user_scoped_document_hashes"
down_revision = "0004_combined_summary_details"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _has_column(name: str) -> bool:
    return any(item["name"] == name for item in _inspector().get_columns("documents"))


def _has_index(name: str) -> bool:
    return any(item["name"] == name for item in _inspector().get_indexes("documents"))


def upgrade() -> None:
    if not _has_column("content_sha256"):
        op.add_column("documents", sa.Column("content_sha256", sa.String(length=64), nullable=True))
    if not _has_column("extracted_text_sha256"):
        op.add_column("documents", sa.Column("extracted_text_sha256", sa.String(length=64), nullable=True))
    if not _has_index("ix_documents_user_content_sha256"):
        op.create_index("ix_documents_user_content_sha256", "documents", ["user_id", "content_sha256"], unique=True)
    if not _has_index("ix_documents_user_text_sha256"):
        op.create_index("ix_documents_user_text_sha256", "documents", ["user_id", "extracted_text_sha256"], unique=True)
    if not _has_index("ix_documents_user_created_at"):
        op.create_index("ix_documents_user_created_at", "documents", ["user_id", "created_at"])


def downgrade() -> None:
    # Keep hashes and indexes so rollback cannot disable owner-scoped duplicate protection.
    pass
