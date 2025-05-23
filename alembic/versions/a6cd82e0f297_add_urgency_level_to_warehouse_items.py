"""Add urgency_level to warehouse items

Revision ID: a6cd82e0f297
Revises: c912b5318def
Create Date: 2025-04-28 12:38:17.137111

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a6cd82e0f297'
down_revision: Union[str, None] = 'c912b5318def'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Сначала создаем ENUM тип
    urgency_level_type = sa.Enum('NORMAL', 'URGENT', 'CRITICAL', name='urgencylevel')
    urgency_level_type.create(op.get_bind(), checkfirst=True)

    # Теперь добавляем колонку, используя созданный тип ENUM
    op.add_column('warehouseitem', sa.Column('urgency_level', urgency_level_type, nullable=False, server_default='NORMAL'))

    # Убираем server_default после добавления колонки
    op.alter_column('warehouseitem', 'urgency_level',
               existing_type=urgency_level_type,
               server_default=None,
               nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Удаляем колонку
    op.drop_column('warehouseitem', 'urgency_level')

    # Удаляем ENUM тип после удаления колонки
    sa.Enum(name='urgencylevel').drop(op.get_bind(), checkfirst=True)