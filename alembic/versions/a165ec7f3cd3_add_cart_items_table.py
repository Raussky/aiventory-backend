"""add cart items table

Revision ID: a165ec7f3cd3
Revises: a6cd82e0f297
Create Date: 2025-05-24 21:38:15.672580

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a165ec7f3cd3'
down_revision: Union[str, None] = 'a6cd82e0f297'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Создаем таблицу cartitem
    op.create_table('cartitem',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('sid', sa.String(length=22), nullable=False),
        sa.Column('store_item_sid', sa.String(length=22), nullable=False),
        sa.Column('user_sid', sa.String(length=22), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('price_per_unit', sa.Float(), nullable=False),
        sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['store_item_sid'], ['storeitem.sid'], ),
        sa.ForeignKeyConstraint(['user_sid'], ['user.sid'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cartitem_sid'), 'cartitem', ['sid'], unique=True)
    op.create_index(op.f('ix_cartitem_user_sid'), 'cartitem', ['user_sid'], unique=False)
    op.create_index(op.f('ix_cartitem_store_item_sid'), 'cartitem', ['store_item_sid'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_cartitem_store_item_sid'), table_name='cartitem')
    op.drop_index(op.f('ix_cartitem_user_sid'), table_name='cartitem')
    op.drop_index(op.f('ix_cartitem_sid'), table_name='cartitem')
    op.drop_table('cartitem')