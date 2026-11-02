# ruff: noqa: S608 -- Migration identifiers below are fixed constants.
"""SQLite transaction adapter for durable extraction."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.storage.storage_base._extraction_stream import (
    Kind,
    StreamSQL,
    source_request_row,
)
from reflexio.server.services.storage.storage_base.agent_run._models import (
    PendingToolCallRecord,
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
 window_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, org_id TEXT,
 project_id TEXT NOT NULL DEFAULT '', predecessor INTEGER NOT NULL, end_seq INTEGER NOT NULL, manifest TEXT NOT NULL,
 policy TEXT NOT NULL, force INTEGER NOT NULL DEFAULT 0,
 skip_aggregation INTEGER NOT NULL DEFAULT 0, outcome TEXT,
 invalidated INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0, completed_at REAL, effects TEXT, effects_done INTEGER NOT NULL DEFAULT 0, derived_claimed INTEGER NOT NULL DEFAULT 0, effects_retry_at REAL NOT NULL DEFAULT 0,
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
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(extraction_windows)")
    }
    if "completed_at" not in columns:
        conn.execute("ALTER TABLE extraction_windows ADD COLUMN completed_at REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_admission_time ON requests("
        "json_extract(learning_admission,'$.admitted_at'),request_id,user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_window_completion ON extraction_windows("
        "user_id,kind,project_id,end_seq) WHERE completed=1"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_request_arrivals ON interactions("
        "user_id,request_id,ingestion_seq)"
    )
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_session_activity ON requests(user_id,session_id,json_extract(learning_admission,'$.max_seq') DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS extraction_completion_boundary ON requests(user_id,json_extract(learning_admission,'$.max_seq')) WHERE json_extract(learning_admission,'$.session_finished')=1"
    )
    conn.commit()
    _migrate_user_work_ownership(conn)


def _migrate_user_work_ownership(conn: sqlite3.Connection) -> None:
    """Add private owner/position state together without resetting live work.

    The legacy stream migration above has finished its executescript/commit
    boundaries. A completed schema needs no write lock. An upgrade rechecks the
    columns under the database write lock before any ALTER, so separate
    upgrading processes cannot race or observe a partially committed upgrade.
    """
    needed = (
        ("_agent_runs", "user_work_token"),
        ("learning_work", "position_json"),
        ("extraction_windows", "org_id"),
    )
    if all(
        column in {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for table, column in needed
    ):
        return
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        for table, column in needed:
            columns = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")


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
            "_agent_run_finalization_receipts",
            "_agent_output_receipts",
            "_agent_output_preparations",
            "_pending_tool_calls",
            "_run_tool_dependencies",
            "user_playbooks",
            "profiles",
        }:
            raise ValueError("Unknown extraction table")
        return name

    def publication_order(self) -> str:
        return "rowid"

    def resume_json_predicates(self) -> tuple[str, str]:
        return (
            "json_type(r.generation_request_snapshot, '$.overflow') IS NULL",
            "COALESCE(json_type(p.result, '$.not_applicable'), '') != 'true'",
        )

    def now(self) -> float:
        return float(
            self.conn.execute("SELECT (julianday('now')-2440587.5)*86400").fetchone()[0]
        )

    def lock(self, *, skip_locked: bool = False) -> str:  # noqa: ARG002
        return ""  # commit_scope owns BEGIN IMMEDIATE

    def validation_lock(self, *, alias: Literal["d"] | None = None) -> str:  # noqa: ARG002
        # The enclosing BEGIN IMMEDIATE already excludes competing writers.
        return ""

    def try_pending_tool_group(self, record: PendingToolCallRecord) -> bool:  # noqa: ARG002
        return True  # enclosing BEGIN IMMEDIATE serializes all question writers

    def admission_value(self, field: str) -> str:
        if field not in {
            "force",
            "max_seq",
            "min_seq",
            "session_finished",
            "admitted_at",
            "idle_timeout_seconds",
        }:
            raise ValueError("Unknown admission field")
        return f"json_extract(r.learning_admission, '$.{field}')"

    def expiry_question_columns(self) -> str:
        return "q.id,q.org_id,'' AS project_id"

    def erasure_run_columns(self) -> str:
        return "r.id,r.org_id,'' AS project_id,r.extractor_kind"

    def publication_binding_columns(self) -> str:
        return (
            "json_type(r.committed_output,'$.publication') IS NOT NULL AS has_publication,"
            "json_extract(r.committed_output,'$.publication.id') AS output_id"
        )

    def input_lock(self) -> str:
        return ""

    def interaction_input_lock(self) -> str:
        return ""  # BEGIN IMMEDIATE already serializes source writers

    def source_interaction(self, row: dict[str, Any]) -> Interaction:
        from ._base import _row_to_interaction

        return _row_to_interaction(row)

    def source_witness_bytes(
        self, row: dict[str, Any], *, request_override: Request | None = None
    ) -> bytes:
        from reflexio.server.services.extraction.input_preparation import (
            source_witness_bytes,
        )

        from ._base import _row_to_request

        request = (
            request_override
            if request_override is not None
            else _row_to_request(source_request_row(row))
        )
        return source_witness_bytes(request, self.source_interaction(row))

    def project_filter(self, alias: Literal["r", "i", "q"] = "r") -> str:  # noqa: ARG002 -- SQLite has no projects.
        return "? = ''"

    def eligible(self, kind: Kind) -> str:
        return f"json_extract(r.learning_admission, '$.{kind}.eligible')=1"


class SQLiteExtractionStreamMixin:
    db_path: str
    conn: sqlite3.Connection
    _lock: Any
    commit_scope: Any
    in_commit_scope: bool

    def _durable_work_route(self) -> object:
        # Existing file clients share a logical queue. In-memory clients do not.
        from pathlib import Path

        return self.conn if self.db_path == ":memory:" else Path(self.db_path).resolve()

    def _require_independent_writer(self) -> None:
        with self._lock:
            if self.in_commit_scope or self.conn.in_transaction:
                raise ValueError(
                    "Metadata and financial operations require an independent writer"
                )

    @contextmanager
    def _independent_writer_metadata_sql(self) -> Iterator[StreamSQL]:
        with self._stream_sql(read_only=True) as db:
            self._require_independent_writer()
            yield db

    @contextmanager
    def _publication_receipt_sql(
        self, *, read_only: bool = True
    ) -> Iterator[StreamSQL]:
        with self._lock:
            self._require_independent_writer()
            with self._stream_sql(read_only=read_only) as db:
                yield db

    @contextmanager
    def _stream_sql(
        self,
        *,
        discovery: bool = False,  # noqa: ARG002 -- SQLite has no separate discovery role.
        read_only: bool = False,
        coordination: bool = False,  # noqa: ARG002 -- Shared work has no separate SQLite role.
    ) -> Iterator[StreamSQL]:
        if read_only:
            with self._lock:
                yield SQLiteStreamSQL(self.conn)
        else:
            with self.commit_scope():
                yield SQLiteStreamSQL(self.conn)
