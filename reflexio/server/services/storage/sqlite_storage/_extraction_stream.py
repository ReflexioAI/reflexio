# ruff: noqa: S608 -- Migration identifiers below are fixed constants.
"""SQLite transaction adapter for durable extraction."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from reflexio.server.services.storage.storage_base._extraction_stream import (
    Kind,
    StreamSQL,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_work (
 org_id TEXT NOT NULL, user_id TEXT NOT NULL, highwater INTEGER NOT NULL DEFAULT 0,
 pending_since REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0, last_turn REAL NOT NULL DEFAULT 0, due_at REAL NOT NULL DEFAULT 0,
 lease_token TEXT, lease_owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
 PRIMARY KEY(org_id,user_id)
);
CREATE INDEX IF NOT EXISTS learning_work_due ON learning_work(due_at,lease_until);
CREATE TABLE IF NOT EXISTS extraction_cursors (
 user_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('profile','playbook')),
 completed_seq INTEGER NOT NULL DEFAULT 0, started INTEGER NOT NULL DEFAULT 0,
 project_id TEXT NOT NULL DEFAULT '', window_id TEXT, attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
 last_error TEXT, last_turn REAL NOT NULL DEFAULT 0, PRIMARY KEY(user_id,kind,project_id)
);
CREATE TABLE IF NOT EXISTS extraction_windows (
 window_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
 project_id TEXT NOT NULL DEFAULT '', predecessor INTEGER NOT NULL, end_seq INTEGER NOT NULL, manifest TEXT NOT NULL,
 policy TEXT NOT NULL, force INTEGER NOT NULL DEFAULT 0,
 skip_aggregation INTEGER NOT NULL DEFAULT 0, outcome TEXT,
 invalidated INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0, effects TEXT, effects_done INTEGER NOT NULL DEFAULT 0, derived_claimed INTEGER NOT NULL DEFAULT 0, effects_retry_at REAL NOT NULL DEFAULT 0,
 UNIQUE(user_id,kind,predecessor,project_id)
);
CREATE INDEX IF NOT EXISTS extraction_windows_effects ON extraction_windows(completed,effects_done);
"""


def migrate_extraction_stream(conn: sqlite3.Connection) -> None:
    for table, column, sql_type in (
        ("interactions", "ingestion_seq", "INTEGER"),
        ("requests", "learning_admission", "TEXT"),
    ):
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    conn.executescript(SCHEMA)
    conn.executescript("""
    CREATE TRIGGER IF NOT EXISTS invalidate_extraction_input BEFORE DELETE ON interactions
    BEGIN
      UPDATE extraction_windows SET invalidated=1,outcome=NULL,
        effects=CASE WHEN completed=1 THEN json_object('version',1,'skipped',json('true'),'billing',json_extract(effects,'$.billing'),'request_ids',json_extract(effects,'$.request_ids')) ELSE effects END
      WHERE (completed=0 OR effects_done=0) AND EXISTS (
        SELECT 1 FROM json_each(manifest) WHERE json_extract(value,'$.interaction_id')=OLD.interaction_id
      );
    END;
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_force_barrier ON requests(user_id,json_extract(learning_admission,'$.max_seq')) WHERE json_extract(learning_admission,'$.force')=1"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS interactions_arrival ON interactions(user_id,ingestion_seq)"
    )
    conn.commit()


class SQLiteStreamSQL:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()] if cur.description else []

    def table(self, name: str) -> str:
        if name not in {
            "learning_work",
            "extraction_cursors",
            "extraction_windows",
            "requests",
            "interactions",
            "_agent_runs",
        }:
            raise ValueError("Unknown extraction table")
        return name

    def now(self) -> float:
        return float(
            self.conn.execute("SELECT (julianday('now')-2440587.5)*86400").fetchone()[0]
        )

    def lock(self, *, skip_locked: bool = False) -> str:  # noqa: ARG002
        return ""  # commit_scope owns BEGIN IMMEDIATE

    def admission_value(self, field: str) -> str:
        if field not in {"force", "max_seq"}:
            raise ValueError("Unknown admission field")
        return f"json_extract(r.learning_admission, '$.{field}')"

    def input_lock(self) -> str:
        return ""

    def project_filter(self) -> str:
        return "? = ''"

    def eligible(self, kind: Kind) -> str:
        return f"json_extract(r.learning_admission, '$.{kind}.eligible')=1"


class SQLiteExtractionStreamMixin:
    conn: sqlite3.Connection
    _lock: Any
    commit_scope: Any

    @contextmanager
    def _stream_sql(
        self,
        *,
        discovery: bool = False,  # noqa: ARG002 -- SQLite has no separate discovery role.
        read_only: bool = False,
    ) -> Iterator[StreamSQL]:
        if read_only:
            with self._lock:
                yield SQLiteStreamSQL(self.conn)
        else:
            with self.commit_scope():
                yield SQLiteStreamSQL(self.conn)
