"""Add currency and storage_duration

Revision ID: f8f729bfedef
Revises: 08359cc8a297
Create Date: 2025-04-21 19:10:58.960032

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8f729bfedef'
down_revision: Union[str, None] = '08359cc8a297'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Сначала создаем тип ENUM
    currency_enum = sa.Enum('KZT', 'USD', 'EUR', 'RUB', name='currency')
    currency_enum.create(op.get_bind(), checkfirst=True)

    # Затем добавляем колонки
    op.add_column('product', sa.Column('currency', currency_enum, server_default='KZT', nullable=False))
    op.add_column('product', sa.Column('storage_duration', sa.Integer(), server_default='30', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    # Удаляем колонки
    op.drop_column('product', 'storage_duration')
    op.drop_column('product', 'currency')

    # Удаляем ENUM тип после удаления колонки
    sa.Enum(name='currency').drop(op.get_bind(), checkfirst=True)