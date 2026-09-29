"""Initial normalized Scribe schema on MySQL."""
from __future__ import annotations

from pathlib import Path

from alembic import op
from dotenv import load_dotenv

# Alembic imports revision modules before loading env.py. Load the project
# settings here so importing Base does not fail when DATABASE_URL is in .env.
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from database.connection import Base
from database import models  # noqa: F401

revision = "0001_initial_mysql"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in Base.metadata.sorted_tables:
        # Keep this migration stable as later models are introduced. In
        # particular, 0002 creates users and adds documents.user_id.
        if table.name == "users":
            continue
        columns = [column.copy() for column in table.columns if column.name != "user_id"]
        for column in columns:
            column.unique = False
        constraints = [constraint for constraint in table.constraints
                       if not any(column.name == "user_id" for column in getattr(constraint, "columns", []))]
        op.create_table(table.name, *columns, *constraints,
                        mysql_charset="utf8mb4", mysql_collate="utf8mb4_unicode_ci")


def downgrade() -> None:
    for table in reversed(Base.metadata.sorted_tables):
        op.drop_table(table.name)
