import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Un seul endroit décide d'où vient la configuration : env_files.CANONICAL, le fichier
# que systemd passe au service. Cette boucle chargeait quatre fichiers dans l'ordre
# INVERSE en affirmant « le même ordre que main.py », qui n'en charge aucun : un DB_URL
# dans le .env du dépôt aurait donc migré une base pendant que l'API utilisait l'autre.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env_files import load_env  # noqa: E402

load_env()

DB_URL = os.environ.get("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL environment variable is required")

config = context.config
config.set_main_option("sqlalchemy.url", DB_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = DB_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()