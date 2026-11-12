"""Alembic environment configuration for the Voucher OCR application.

Supports:
    - ``alembic upgrade head``  (online / offline)
    - ``alembic revision --autogenerate``  (reads models from app.db.models)

The DATABASE_URL environment variable is expected in asyncpg form
(``postgresql+asyncpg://…``).  Alembic uses a *synchronous* driver, so we
strip the ``+asyncpg`` dialect modifier before passing the URL to the engine.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so ``app`` can be imported when
# alembic is invoked from the project root (e.g. ``alembic upgrade head``).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Import models so that autogenerate can discover all table metadata.
# ---------------------------------------------------------------------------
from app.db.base import Base  # noqa: E402  (must come after sys.path tweak)
import app.db.models  # noqa: E402, F401  — registers models on Base.metadata

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Resolve the synchronous database URL from the environment.
# ---------------------------------------------------------------------------
_async_url: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://voucher_user:changeme@postgres:5432/voucher_ocr",
)
# Convert asyncpg → standard psycopg2/psycopg URL for Alembic
_sync_url: str = _async_url.replace("postgresql+asyncpg://", "postgresql://")

config.set_main_option("sqlalchemy.url", _sync_url)

# Target metadata for --autogenerate
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migrations (no DB connection required; generates SQL script)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine, skipping the
    creation of a connection — useful for generating a migration SQL script.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (uses a live DB connection)
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine, associates a connection with the context, and runs
    migrations against the live database.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
