"""train_test_not_meme_template

Revision ID: c0d038ed9891
Revises: c3680ed34084
Create Date: 2021-03-17 16:35:49.174502

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c0d038ed9891'
down_revision = 'c3680ed34084'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('meme_correct_test',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('meme_correct_train',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('meme_incorrect_test',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('meme_incorrect_train',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('not_a_meme_test',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('not_a_meme_train',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('not_a_template_test',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('not_a_template_train',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=400), nullable=False),
    sa.Column('path', sa.String(length=400), nullable=False),
    sa.Column('name_idx', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('meme_correct_training')
    op.drop_table('not_meme')
    op.drop_table('meme_correct_testing')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('meme_correct_testing',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.Column('path', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.Column('name_idx', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='meme_correct_testing_pkey')
    )
    op.create_table('not_meme',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.Column('path', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name='not_meme_pkey')
    )
    op.create_table('meme_correct_training',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.Column('path', sa.VARCHAR(length=400), autoincrement=False, nullable=False),
    sa.Column('name_idx', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='meme_correct_training_pkey')
    )
    op.drop_table('not_a_template_train')
    op.drop_table('not_a_template_test')
    op.drop_table('not_a_meme_train')
    op.drop_table('not_a_meme_test')
    op.drop_table('meme_incorrect_train')
    op.drop_table('meme_incorrect_test')
    op.drop_table('meme_correct_train')
    op.drop_table('meme_correct_test')
    # ### end Alembic commands ###
