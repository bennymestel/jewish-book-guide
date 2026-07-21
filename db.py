"""
Shared DB connection helper.
"""
import psycopg

import config

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
