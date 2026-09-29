"""Persist ownership-scoped quality reports for individual summary versions."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0009_summary_evaluations"
down_revision = "0008_chunk_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("summary_evaluations"):
        return
    op.create_table(
        "summary_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary_id", sa.Integer(), sa.ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("evaluation_version", sa.String(length=64), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("completeness_score", sa.Float(), nullable=True),
        sa.Column("factual_consistency_score", sa.Float(), nullable=True),
        sa.Column("groundedness_score", sa.Float(), nullable=True),
        sa.Column("summary_quality_score", sa.Float(), nullable=True),
        sa.Column("validation_accuracy_score", sa.Float(), nullable=True),
        sa.Column("evaluation_status", sa.String(length=32), nullable=False),
        sa.Column("evaluation_details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("summary_id", "evaluation_version", name="uq_summary_evaluation_version"),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_summary_evaluations_document_id", "summary_evaluations", ["document_id"])
    op.create_index("ix_summary_evaluations_user_id", "summary_evaluations", ["user_id"])
    op.create_index("ix_summary_evaluations_summary_id", "summary_evaluations", ["summary_id"])
    op.create_index("ix_summary_evaluations_summary_fingerprint", "summary_evaluations", ["summary_fingerprint"])


def downgrade() -> None:
    # Keep evaluation history across schema downgrades.
    pass
