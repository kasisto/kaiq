"""Fix document count sync in collections.

Revision ID: 3507fec61762
Revises: c45a9cf6a8a4
Create Date: 2025-01-08
"""

import os
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3507fec61762"
down_revision: Union[str, None] = "c45a9cf6a8a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

project_name = os.getenv("R2R_PROJECT_NAME")
if not project_name:
    raise ValueError(
        "Environment variable `R2R_PROJECT_NAME` must be provided to migrate, "
        "it should be set equal to the value of `project_name` in your `r2r.toml`."
    )


def upgrade() -> None:
    """Recalculate document_count for all collections based on actual documents."""
    op.execute(f"""
        UPDATE {project_name}.collections c
        SET document_count = COALESCE(
            (SELECT COUNT(*)
             FROM {project_name}.documents d
             WHERE c.id = ANY(d.collection_ids)),
            0
        )
    """)


def downgrade() -> None:
    """No downgrade needed - this is a data fix."""
    pass
