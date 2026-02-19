"""Fix status constraint for ingested_file

Revision ID: fix_ingested_file_status
Revises: <previous_rev>
Create Date: 2025-11-16
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fix_ingested_file_status'
down_revision = '<previous_rev>'
branch_labels = None
depends_on = None


def upgrade():
    # Drop old broken constraint
    op.execute("""
        ALTER TABLE ingested_file
        DROP CONSTRAINT IF EXISTS ingested_file_status_check;
    """)

    # Add new correct constraint
    op.execute("""
        ALTER TABLE ingested_file
        ADD CONSTRAINT ingested_file_status_check
        CHECK (status IN (
            'uploaded',
            'processing',
            'ingested',
            'completed'
        ));
    """)

    # Force invalid status rows into a safe state
    op.execute("""
        UPDATE ingested_file
        SET status='uploaded'
        WHERE status NOT IN ('uploaded','processing','ingested','completed')
           OR status IS NULL;
    """)


def downgrade():
    # Remove new constraint (rollback)
    op.execute("""
        ALTER TABLE ingested_file
        DROP CONSTRAINT IF EXISTS ingested_file_status_check;
    """)

    # Re-add the old one (unknown constraints restored minimally)
    op.execute("""
        ALTER TABLE ingested_file
        ADD CONSTRAINT ingested_file_status_check
        CHECK (status IN ('processed'));
    """)
    