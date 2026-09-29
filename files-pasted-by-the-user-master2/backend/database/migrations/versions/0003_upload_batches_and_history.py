"""Add persistent upload batches, history, and combined summaries.

The documents.batch_id addition is nullable so all pre-existing documents are
preserved without guessing which upload batch they belonged to.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "0003_upload_batches_and_history"
down_revision = "0002_user_ownership"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _table_exists(name: str) -> bool:
    return _inspector().has_table(name)


def _column_exists(table: str, column: str) -> bool:
    return any(item["name"] == column for item in _inspector().get_columns(table))


def _index_exists(table: str, index: str) -> bool:
    return any(item["name"] == index for item in _inspector().get_indexes(table))


def _foreign_key_exists(table: str, name: str) -> bool:
    return any(item.get("name") == name for item in _inspector().get_foreign_keys(table))


def _table_has_rows(table: str) -> bool:
    return op.get_bind().execute(sa.text(f"SELECT 1 FROM `{table}` LIMIT 1")).first() is not None


def upgrade() -> None:
    if not _table_exists("upload_batches"):
        op.create_table(
            "upload_batches",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("total_files", sa.Integer(), nullable=False),
            sa.Column("completed_files", sa.Integer(), nullable=False),
            sa.Column("failed_files", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_upload_batches_user_id_users", ondelete="CASCADE"),
            mysql_charset="utf8mb4", mysql_collate="utf8mb4_unicode_ci",
        )
    elif not _column_exists("upload_batches", "user_id"):
        # Preserve existing batch records; do not invent an owner for old rows.
        op.add_column("upload_batches", sa.Column("user_id", sa.Integer(), nullable=_table_has_rows("upload_batches")))
    if not _foreign_key_exists("upload_batches", "fk_upload_batches_user_id_users"):
        op.create_foreign_key("fk_upload_batches_user_id_users", "upload_batches", "users",
            ["user_id"], ["id"], ondelete="CASCADE")
    for name, columns in (
        ("ix_upload_batches_user_id", ["user_id"]),
        ("ix_upload_batches_status", ["status"]),
        ("ix_upload_batches_created_at", ["created_at"]),
    ):
        if not _index_exists("upload_batches", name):
            op.create_index(name, "upload_batches", columns)

    if not _column_exists("documents", "batch_id"):
        op.add_column("documents", sa.Column("batch_id", sa.String(length=36), nullable=True))
    if not _index_exists("documents", "ix_documents_batch_id"):
        op.create_index("ix_documents_batch_id", "documents", ["batch_id"])
    if not _foreign_key_exists("documents", "fk_documents_batch_id_upload_batches"):
        op.create_foreign_key(
            "fk_documents_batch_id_upload_batches", "documents", "upload_batches",
            ["batch_id"], ["id"], ondelete="SET NULL",
        )

    if not _table_exists("history"):
        op.create_table(
            "history",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("document_id", sa.String(length=36), nullable=True),
            sa.Column("activity_type", sa.String(length=64), nullable=False),
            sa.Column("title", sa.String(length=512), nullable=False),
            sa.Column("input_text", LONGTEXT(), nullable=True),
            sa.Column("output_text", LONGTEXT(), nullable=True),
            sa.Column("language", sa.String(length=12), nullable=True),
            sa.Column("provider", sa.String(length=64), nullable=True),
            sa.Column("model", sa.String(length=255), nullable=True),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("related_document_ids", sa.JSON(), nullable=True),
            sa.Column("extra_data", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_history_user_id_users", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["document_id"], ["documents.id"], name="fk_history_document_id_documents", ondelete="SET NULL"),
            mysql_charset="utf8mb4", mysql_collate="utf8mb4_unicode_ci",
        )
    elif not _column_exists("history", "user_id"):
        # Preserve legacy history rows without guessing which account owns them.
        op.add_column("history", sa.Column("user_id", sa.Integer(), nullable=_table_has_rows("history")))
    if not _foreign_key_exists("history", "fk_history_user_id_users"):
        op.create_foreign_key("fk_history_user_id_users", "history", "users",
            ["user_id"], ["id"], ondelete="CASCADE")
    if not _foreign_key_exists("history", "fk_history_document_id_documents"):
        op.create_foreign_key("fk_history_document_id_documents", "history", "documents",
            ["document_id"], ["id"], ondelete="SET NULL")
    for name, columns in (
        ("ix_history_user_id", ["user_id"]),
        ("ix_history_document_id", ["document_id"]),
        ("ix_history_activity_type", ["activity_type"]),
        ("ix_history_status", ["status"]),
        ("ix_history_created_at", ["created_at"]),
        ("ix_history_user_created_at", ["user_id", "created_at"]),
        ("ix_history_user_activity_type", ["user_id", "activity_type"]),
    ):
        if not _index_exists("history", name):
            op.create_index(name, "history", columns)

    if not _table_exists("combined_summaries"):
        op.create_table(
            "combined_summaries",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("document_ids", sa.JSON(), nullable=False),
            sa.Column("summary_text", LONGTEXT(), nullable=False),
            sa.Column("language", sa.String(length=12), nullable=False),
            sa.Column("provider", sa.String(length=64), nullable=True),
            sa.Column("model", sa.String(length=255), nullable=True),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_combined_summaries_user_id_users", ondelete="CASCADE"),
            mysql_charset="utf8mb4", mysql_collate="utf8mb4_unicode_ci",
        )
    elif not _column_exists("combined_summaries", "user_id"):
        # Preserve legacy summaries; unowned rows remain inaccessible to users.
        op.add_column("combined_summaries", sa.Column("user_id", sa.Integer(), nullable=_table_has_rows("combined_summaries")))
    if not _foreign_key_exists("combined_summaries", "fk_combined_summaries_user_id_users"):
        op.create_foreign_key("fk_combined_summaries_user_id_users", "combined_summaries", "users",
            ["user_id"], ["id"], ondelete="CASCADE")
    for name, columns in (
        ("ix_combined_summaries_user_id", ["user_id"]),
        ("ix_combined_summaries_status", ["status"]),
        ("ix_combined_summaries_created_at", ["created_at"]),
    ):
        if not _index_exists("combined_summaries", name):
            op.create_index(name, "combined_summaries", columns)


def downgrade() -> None:
    # Keep user data and uploaded documents intact. This migration is additive;
    # removing its tables/column could destroy user history and batch links.
    pass
