"""add_total_failed_to_batches

Revision ID: a1b2c3d4e5f6
Revises: 709cb4e13da1
Create Date: 2026-11-16 20:33:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '709cb4e13da1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'voucher_batches',
        sa.Column(
            'total_failed',
            sa.INTEGER(),
            server_default='0',
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column('voucher_batches', 'total_failed')
