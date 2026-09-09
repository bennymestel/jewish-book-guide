"""
Shared DB connection helper.
"""
import logging

import psycopg

import config

logger = logging.getLogger(__name__)

# A cold/paused DB must fail fast
# rather than hang the request indefinitely — connect_timeout bounds the TCP/auth
# handshake, statement_timeout bounds any individual query once connected.
DB_CONNECT_TIMEOUT_SECONDS = 10
DB_STATEMENT_TIMEOUT_MS = 15000


def connect(**kwargs) -> psycopg.Connection:
    """Open a DB connection with a bounded connect + statement timeout.
    Use this instead of calling psycopg.connect(config.DB_URL) directly."""
    return psycopg.connect(
        config.DB_URL,
        connect_timeout=DB_CONNECT_TIMEOUT_SECONDS,
        options=f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
        **kwargs,
    )


def ensure_search_extensions() -> None:
    """Turn on pg_trgm (needed for lexical search) if it isn't already on.
    Safe to call on every startup. Logs and continues on failure instead of
    raising, so a permissions issue degrades the lexical arm instead of
    breaking the whole server."""
    try:
        with connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.commit()
    except Exception as e:
        logger.warning("Could not enable pg_trgm extension: %s", e)
