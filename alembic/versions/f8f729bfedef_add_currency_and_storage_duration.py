"""Add storage duration type

Revision ID: %(revision_id)s
Revises: f8f729bfedef
Create Date: %(create_date)s

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '%(revision_id)s'
down_revision = 'f8f729bfedef'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Создаем ENUM тип
    storage_duration_type = sa.Enum('DAY', 'MONTH', 'YEAR', name='storagedurationtype')
    storage_duration_type.create(op.get_bind(), checkfirst=True)

    # Затем добавляем столбец с использованием этого типа
    op.add_column('product',
                  sa.Column('storage_duration_type',
                            sa.Enum('DAY', 'MONTH', 'YEAR', name='storagedurationtype'),
                            server_default='DAY',
                            nullable=False))


def downgrade() -> None:
    # Удаляем столбец
    op.drop_column('product', 'storage_duration_type')

    # Удаляем ENUM тип
    op.execute('DROP TYPE storagedurationtype')