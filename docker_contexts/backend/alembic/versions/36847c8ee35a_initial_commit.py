"""initial commit

Revision ID: 36847c8ee35a
Revises: 
Create Date: 2024-02-12 09:23:16.933346

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '36847c8ee35a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('doi',
    sa.Column('id_key', sa.Integer(), nullable=False),
    sa.Column('doi', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id_key')
    )
    op.create_table('study',
    sa.Column('id_key', sa.Integer(), nullable=False),
    sa.Column('study_id', sa.String(), nullable=False),
    sa.Column('data_contributor', sa.String(), nullable=False),
    sa.Column('data_contributor_contact_info', sa.String(), nullable=False),
    sa.Column('study_metadata', sa.String(), nullable=False),
    sa.Column('response_types', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id_key')
    )
    op.create_table('study_doi',
    sa.Column('study_id', sa.Integer(), nullable=False),
    sa.Column('doi_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['doi_id'], ['doi.id_key'], ),
    sa.ForeignKeyConstraint(['study_id'], ['study.id_key'], ),
    sa.PrimaryKeyConstraint('study_id', 'doi_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('study_doi')
    op.drop_table('study')
    op.drop_table('doi')
    # ### end Alembic commands ###
