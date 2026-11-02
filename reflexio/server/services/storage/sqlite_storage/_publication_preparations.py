"""Run-owned raw preparation; never a financial or retention owner."""

import sqlite3

_TABLE = """CREATE TABLE _agent_output_preparations (
 run_id TEXT PRIMARY KEY NOT NULL REFERENCES _agent_runs(id) ON DELETE CASCADE,
 org_id TEXT NOT NULL CHECK(trim(org_id)<>''),
 project_id TEXT NOT NULL DEFAULT '' CHECK(project_id=''),
 preparation TEXT CHECK(preparation IS NULL OR (json_valid(preparation) AND json_type(preparation)='object')),
 active_attempt TEXT CHECK(active_attempt IS NULL OR (json_valid(active_attempt) AND json_type(active_attempt)='object')),
 CHECK((preparation IS NULL) <> (active_attempt IS NULL))
)"""


def _installed(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='_agent_output_preparations'"
    ).fetchone()
    return row[0] if row else None


def migrate_publication_preparations(conn: sqlite3.Connection) -> None:
    """Current startup only reads schema; concurrent installers recheck in TX."""
    if conn.in_transaction:
        raise ValueError("Preparation migration requires its own transaction")
    if _installed(conn) == _TABLE:
        return
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        installed = _installed(conn)
        if installed is None:
            conn.execute(_TABLE)
        elif installed != _TABLE:
            raise ValueError("Publication preparation schema is unsupported")
