"""Add storage duration type

Revision ID: d0316acef56d
Revises: 37cf20593582
Create Date: 2025-04-28 12:22:51.475310

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0316acef56d'
down_revision: Union[str, None] = '37cf20593582'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
