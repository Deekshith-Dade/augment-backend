"""article tags

Revision ID: 84fcc059a83e
Revises: 6ac02a530c5f
Create Date: 2025-06-15 14:28:19.366321

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '84fcc059a83e'
down_revision: Union[str, None] = '6ac02a530c5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('external_articles', sa.Column('tags', sa.ARRAY(sa.String()), nullable=True))
    op.drop_index(op.f('external_articles_embedding_idx'), table_name='external_articles', postgresql_with={'lists': '100'}, postgresql_using='ivfflat')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(op.f('external_articles_embedding_idx'), 'external_articles', ['embedding'], unique=False, postgresql_with={'lists': '100'}, postgresql_using='ivfflat')
    op.drop_column('external_articles', 'tags')
    # ### end Alembic commands ###
