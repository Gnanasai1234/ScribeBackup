"""Persist validated per-document sections for combined-summary exports."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_combined_summary_details"
down_revision = "0003_upload_batches_and_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("combined_summaries") and not any(
        column["name"] == "details" for column in inspector.get_columns("combined_summaries")
    ):
        op.add_column("combined_summaries", sa.Column("details", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Keep previously generated combined results and their structured details.
    pass
