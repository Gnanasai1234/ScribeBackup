"""Create missing mapped tables and indexes without replacing existing data.

This repairs databases where earlier initialization stopped partway through.
Only absent tables/indexes/foreign keys are created; existing rows and tables
are never dropped or truncated.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from database import models  # noqa: F401 - ensure every mapped table is registered
from database.connection import Base

revision = "0006_repair_incomplete_schema"
down_revision = "0005_user_scoped_document_hashes"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _create_table_from_model(table) -> None:
    columns = [column.copy() for column in table.columns]
    constraints = [constraint for constraint in table.constraints]
    op.create_table(
        table.name,
        *columns,
        *constraints,
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )


def upgrade() -> None:
    # Create only tables that are absent. Sorting respects mapped FK dependencies.
    for table in Base.metadata.sorted_tables:
        if not _inspector().has_table(table.name):
            _create_table_from_model(table)

    # Repair absent indexes when all indexed columns are present.
    for table in Base.metadata.sorted_tables:
        if not _inspector().has_table(table.name):
            continue
        columns = {item["name"] for item in _inspector().get_columns(table.name)}
        existing_indexes = {item["name"] for item in _inspector().get_indexes(table.name)}
        for index in sorted(table.indexes, key=lambda item: item.name or ""):
            if index.name in existing_indexes:
                continue
            index_columns = [column.name for column in index.columns]
            if all(column in columns for column in index_columns):
                op.create_index(index.name, table.name, index_columns, unique=index.unique)

    # Add missing foreign keys after any absent target tables have been created.
    for table in Base.metadata.sorted_tables:
        if not _inspector().has_table(table.name):
            continue
        inspector = _inspector()
        columns = {item["name"] for item in inspector.get_columns(table.name)}
        foreign_keys = inspector.get_foreign_keys(table.name)
        existing_pairs = {
            (tuple(item.get("constrained_columns") or ()), item.get("referred_table"),
             tuple(item.get("referred_columns") or ()))
            for item in foreign_keys
        }
        for constraint in table.foreign_key_constraints:
            local_columns = tuple(element.parent.name for element in constraint.elements)
            remote_columns = tuple(element.column.name for element in constraint.elements)
            remote_table = constraint.elements[0].column.table.name
            if not set(local_columns).issubset(columns) or not _inspector().has_table(remote_table):
                continue
            identity = (local_columns, remote_table, remote_columns)
            if identity in existing_pairs:
                continue
            op.create_foreign_key(
                constraint.name,
                table.name,
                remote_table,
                list(local_columns),
                list(remote_columns),
                ondelete=constraint.ondelete,
                onupdate=constraint.onupdate,
            )


def downgrade() -> None:
    # Repairs are intentionally retained on downgrade to avoid deleting user data.
    pass
