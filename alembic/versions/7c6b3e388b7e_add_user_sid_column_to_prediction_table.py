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

    connection = op.get_bind()
    result = connection.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'prediction' AND column_name = 'user_sid'"
    ))

    if result.fetchone() is None:
        op.add_column('prediction',
            sa.Column('user_sid', sa.String(length=22), nullable=True)
        )

        op.execute("""
            UPDATE prediction p
            SET user_sid = (
                SELECT DISTINCT u.user_sid
                FROM sale s
                JOIN storeitem si ON s.store_item_sid = si.sid
                JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
                JOIN upload u ON wi.upload_sid = u.sid
                WHERE wi.product_sid = p.product_sid
                LIMIT 1
            )
            WHERE p.user_sid IS NULL
        """)

        op.execute("""
            DELETE FROM prediction
            WHERE user_sid IS NULL
        """)

        op.alter_column('prediction', 'user_sid',
                       existing_type=sa.String(length=22),
                       nullable=False)

        op.create_foreign_key('fk_prediction_user', 'prediction', 'user', ['user_sid'], ['sid'])

        op.create_index('ix_prediction_user_sid', 'prediction', ['user_sid'])

        op.create_index('ix_prediction_product_user', 'prediction',
                       ['product_sid', 'user_sid'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_prediction_product_user', table_name='prediction')
    op.drop_index('ix_prediction_user_sid', table_name='prediction')
    op.drop_constraint('fk_prediction_user', 'prediction', type_='foreignkey')
    op.drop_column('prediction', 'user_sid')