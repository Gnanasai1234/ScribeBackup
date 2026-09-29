"""Add users and optional ownership for existing documents.

Existing document rows are kept with a NULL user_id until an administrator
assigns them explicitly.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_user_ownership"
down_revision = "0001_initial_mysql"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _table_exists(name: str) -> bool:
    return _inspector().has_table(name)


def _column_exists(table: str, column: str) -> bool:
    return _table_exists(table) and any(item["name"] == column for item in _inspector().get_columns(table))


def _index_exists(table: str, name: str) -> bool:
    return _table_exists(table) and any(item["name"] == name for item in _inspector().get_indexes(table))


def _foreign_key_exists(table: str, name: str) -> bool:
    return _table_exists(table) and any(item.get("name") == name for item in _inspector().get_foreign_keys(table))


def upgrade() -> None:
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=160), nullable=False),
            sa.Column("email", sa.String(length=320), nullable=False),
            sa.Column("password_hash", sa.String(length=255), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("email", name="uq_users_email"),
            mysql_charset="utf8mb4",
            mysql_collate="utf8mb4_unicode_ci",
        )
    if not _index_exists("users", "ix_users_email"):
        op.create_index("ix_users_email", "users", ["email"], unique=False)
    if _table_exists("documents") and not _column_exists("documents", "user_id"):
        op.add_column("documents", sa.Column("user_id", sa.Integer(), nullable=True))
    if _table_exists("documents") and not _index_exists("documents", "ix_documents_user_id"):
        op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    if _table_exists("documents") and not _foreign_key_exists("documents", "fk_documents_user_id_users"):
        op.create_foreign_key("fk_documents_user_id_users", "documents", "users", ["user_id"], ["id"])


def downgrade() -> None:
    # Intentionally retain users and ownership data; production downgrade must
    # never delete user records or their documents.
    pass
