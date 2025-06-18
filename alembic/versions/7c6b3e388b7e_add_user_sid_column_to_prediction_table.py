"""Add user_sid column to prediction table

Revision ID: 7c6b3e388b7e
Revises: 018d6436b6f8
Create Date: 2025-06-18 10:42:23.708308

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c6b3e388b7e'
down_revision: Union[str, None] = '018d6436b6f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Добавляем колонку user_sid в таблицу prediction
    op.add_column('prediction',
        sa.Column('user_sid', sa.String(length=255), nullable=True)
    )

    # Создаем индекс для производительности
    op.create_index('ix_prediction_user_sid', 'prediction', ['user_sid'])


def downgrade() -> None:
    """Downgrade schema."""
    # Удаляем индекс
    op.drop_index('ix_prediction_user_sid', table_name='prediction')

    # Удаляем колонку user_sid
    op.drop_column('prediction', 'user_sid')