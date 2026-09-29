"""Repair ownership on evaluation rows created by the legacy initial revision."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0010_eval_user_owner"
down_revision = "0009_summary_evaluations"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def upgrade() -> None:
    inspector = _inspector()
    if not inspector.has_table("summary_evaluations"):
        return

    columns = {item["name"]: item for item in inspector.get_columns("summary_evaluations")}
    if "user_id" not in columns:
        op.add_column("summary_evaluations", sa.Column("user_id", sa.Integer(), nullable=True))

    # Existing evaluations inherit ownership only from their owning document.
    # Abort without changing rows when that mapping cannot establish a user.
    op.execute(sa.text(
        "UPDATE summary_evaluations AS e "
        "JOIN documents AS d ON d.id = e.document_id "
        "SET e.user_id = d.user_id WHERE e.user_id IS NULL"
    ))
    unresolved = op.get_bind().execute(sa.text(
        "SELECT COUNT(*) FROM summary_evaluations WHERE user_id IS NULL"
    )).scalar_one()
    if unresolved:
        raise RuntimeError(
            f"Cannot enforce evaluation ownership: {unresolved} existing rows have no document owner. "
            "Assign those documents to users, then rerun the migration."
        )

    inspector = _inspector()
    column = {item["name"]: item for item in inspector.get_columns("summary_evaluations")}['user_id']
    if column.get("nullable", True):
        op.alter_column("summary_evaluations", "user_id", existing_type=sa.Integer(), nullable=False)

    indexes = {item["name"] for item in _inspector().get_indexes("summary_evaluations")}
    if "ix_summary_evaluations_user_id" not in indexes:
        op.create_index("ix_summary_evaluations_user_id", "summary_evaluations", ["user_id"])

    foreign_keys = _inspector().get_foreign_keys("summary_evaluations")
    ownership_fk_exists = any(
        item.get("constrained_columns") == ["user_id"] and item.get("referred_table") == "users"
        for item in foreign_keys
    )
    if not ownership_fk_exists:
        op.create_foreign_key(
            "fk_summary_evaluations_user_id_users",
            "summary_evaluations",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    # Keep ownership data and constraints on downgrade.
    pass
