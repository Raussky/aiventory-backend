"""Add user-specific predictions and forecast bounds

Revision ID: ae666d42ee9a
Revises: e332d21eb82d
Create Date: 2025-06-08 05:35:02.197840

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ae666d42ee9a'
down_revision: Union[str, None] = 'e332d21eb82d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()

    result = connection.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'prediction' AND column_name = 'user_sid'"
    ))
    has_user_sid = result.fetchone() is not None

    if not has_user_sid:
        op.add_column('prediction', sa.Column('user_sid', sa.String(length=22), nullable=True))

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

        op.create_foreign_key('fk_prediction_user_sid', 'prediction', 'user', ['user_sid'], ['sid'])
        op.create_index('ix_prediction_user_sid', 'prediction', ['user_sid'], unique=False)

    op.add_column('prediction', sa.Column('forecast_qty_lower', sa.Float(), nullable=True))
    op.add_column('prediction', sa.Column('forecast_qty_upper', sa.Float(), nullable=True))
    op.add_column('prediction', sa.Column('created_at', sa.DateTime(timezone=True), nullable=True))

    op.execute("""
        UPDATE prediction
        SET forecast_qty_lower = GREATEST(0, forecast_qty * 0.8),
            forecast_qty_upper = forecast_qty * 1.2
        WHERE forecast_qty_lower IS NULL
    """)

    op.execute("""
        UPDATE prediction
        SET created_at = generated_at
        WHERE created_at IS NULL
    """)

    op.execute("""
        DELETE FROM prediction p1
        USING prediction p2
        WHERE p1.id > p2.id
        AND p1.user_sid = p2.user_sid
        AND p1.product_sid = p2.product_sid
        AND p1.timeframe = p2.timeframe
        AND p1.period_start = p2.period_start
        AND p1.period_end = p2.period_end
    """)

    op.create_index('ix_prediction_product_user_timeframe', 'prediction',
                    ['product_sid', 'user_sid', 'timeframe', 'period_start'], unique=False)
    op.create_index('ix_prediction_user_created', 'prediction',
                    ['user_sid', 'created_at'], unique=False)

    op.create_check_constraint(
        'ck_prediction_forecast_bounds',
        'prediction',
        'forecast_qty_lower <= forecast_qty AND forecast_qty <= forecast_qty_upper'
    )

    op.create_check_constraint(
        'ck_prediction_positive_forecast',
        'prediction',
        'forecast_qty >= 0 AND forecast_qty_lower >= 0 AND forecast_qty_upper >= 0'
    )

    op.create_unique_constraint(
        'uq_prediction_user_product_period',
        'prediction',
        ['user_sid', 'product_sid', 'timeframe', 'period_start', 'period_end']
    )


def downgrade() -> None:
    op.drop_constraint('uq_prediction_user_product_period', 'prediction', type_='unique')
    op.drop_constraint('ck_prediction_positive_forecast', 'prediction', type_='check')
    op.drop_constraint('ck_prediction_forecast_bounds', 'prediction', type_='check')

    op.drop_index('ix_prediction_user_created', table_name='prediction')
    op.drop_index('ix_prediction_product_user_timeframe', table_name='prediction')

    connection = op.get_bind()
    result = connection.execute(sa.text(
        "SELECT constraint_name FROM information_schema.table_constraints "
        "WHERE table_name = 'prediction' AND constraint_name = 'fk_prediction_user_sid'"
    ))

    if result.fetchone():
        op.drop_index('ix_prediction_user_sid', table_name='prediction')
        op.drop_constraint('fk_prediction_user_sid', 'prediction', type_='foreignkey')
        op.drop_column('prediction', 'user_sid')

    op.drop_column('prediction', 'created_at')
    op.drop_column('prediction', 'forecast_qty_upper')
    op.drop_column('prediction', 'forecast_qty_lower')