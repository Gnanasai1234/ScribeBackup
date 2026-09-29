"""Persist generated chunk vectors for semantic retrieval across restarts."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0008_chunk_embeddings"
down_revision = "0007_tighten_empty_owner_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("document_chunks"):
        return
    columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    if "embedding" not in columns:
        op.add_column("document_chunks", sa.Column("embedding", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Keep stored vectors so downgrades cannot silently remove RAG data.
    pass
