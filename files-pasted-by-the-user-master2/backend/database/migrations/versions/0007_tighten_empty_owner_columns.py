"""Enforce non-null user ownership where no legacy unowned rows exist.

The 0003 compatibility path permits NULL owner IDs only when needed to preserve
pre-existing rows whose owner cannot be inferred safely.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_tighten_empty_owner_columns"
down_revision = "0006_repair_incomplete_schema"
branch_labels = None
depends_on = None


def _columns(table: str) -> dict[str, dict]:
    return {column["name"]: column for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    for table in ("upload_batches", "history", "combined_summaries"):
        columns = _columns(table)
        owner_column = columns.get("user_id")
        if owner_column is None or not owner_column.get("nullable", True):
            continue
        has_unowned_rows = bind.execute(sa.text(f"SELECT 1 FROM `{table}` WHERE user_id IS NULL LIMIT 1")).first()
        if has_unowned_rows:
            # Keep existing records intact; the runtime scopes results to user_id,
            # so an unassigned legacy row cannot be exposed to an account.
            continue
        op.alter_column(
            table,
            "user_id",
            existing_type=sa.Integer(),
            existing_nullable=True,
            nullable=False,
        )


def downgrade() -> None:
    # Ownership constraints stay in place to avoid weakening account isolation.
    pass
