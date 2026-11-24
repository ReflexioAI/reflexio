# ruff: noqa: S608 -- Migration identifiers below are fixed constants.
"""SQLite transaction adapter for durable extraction."""

import json
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
from reflexio.server.work_scope import current_project_id

CURSOR_WINDOW_INDEX_NAME = "extraction_cursors_window_lookup"
CURSOR_WINDOW_INDEX_DDL = (
    f"CREATE INDEX IF NOT EXISTS {CURSOR_WINDOW_INDEX_NAME}\n"
    " ON extraction_cursors(window_id) WHERE window_id IS NOT NULL"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_work (
 org_id TEXT NOT NULL, user_id TEXT NOT NULL, highwater INTEGER NOT NULL DEFAULT 0,
 pending_since REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0, last_turn REAL NOT NULL DEFAULT 0, due_at REAL NOT NULL DEFAULT 0,
 lease_token TEXT, lease_owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
    position_json TEXT,
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


def register_window_delivery_validator(conn: sqlite3.Connection) -> None:
    """Install the inventory validator before the connection starts serving queries."""
    from reflexio.server.services.extraction.publication_receipts import (
        valid_window_delivery_payload,
    )

    conn.create_function(
        "reflexio_valid_window_delivery",
        2,
        valid_window_delivery_payload,
        deterministic=True,
    )


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
    @contextmanager
    def learning_transition_write_protocol(self) -> Iterator[None]:
        """SQLite serializes typed control writes through its writer owner."""
        yield

    @contextmanager
    def user_work_claim_protocol(self) -> Iterator[None]:
        """SQLite has no native protocol-floor trigger or session marker."""
        yield

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
            "_operation_state",
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

    def transition_run_witness_columns(self) -> str:
        return (
            "json_extract(r.generation_request_snapshot,'$.output_schema_name') AS schema_name,"
            "json_type(r.generation_request_snapshot,'$.overflow') IS NOT NULL AS overflow,"
            "json_extract(r.generation_request_snapshot,'$.overflow.checkpoint.extraction_instruction_contract') AS instruction_digest,"
            "json_extract(r.generation_request_snapshot,'$.overflow.checkpoint.extraction_policy') AS policy_digest"
        )

    def transition_admission_clock_missing(self) -> str:
        return "json_type(r.learning_admission,'$.admitted_at') IS NULL"

    def transition_admission_clock_numeric(self) -> str:
        return "json_type(r.learning_admission,'$.admitted_at') IN ('integer','real')"

    def transition_admission_input_stats(self) -> str:
        return (
            "json_object('n',COUNT(*),'distinct_n',COUNT(DISTINCT i.ingestion_seq),"
            "'lo',MIN(i.ingestion_seq),'hi',MAX(i.ingestion_seq))"
        )

    def transition_coverage_batch_limit(self) -> int:
        get_limit = getattr(self.conn, "getlimit", None)
        variables = (
            get_limit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) if get_limit else 999
        )
        return min(500, (variables - 3) // 5)

    def transition_inventory_snapshot(self) -> tuple[object, ...] | None:
        # SQLite does not expose a transaction generation. Use normal page gates.
        return None

    def transition_window_history_format(self) -> tuple[str, str]:
        """Compact ACK/retired formats for fixed window alias w, no credit."""
        request_ids = "json_each(w.effects,'$.request_ids')"
        common = (
            "json_type(w.effects,'$.version')='integer' AND json_extract(w.effects,'$.version')=1 "
            "AND json_type(w.effects,'$.skipped')='true' "
            "AND json_type(w.effects,'$.request_ids')='array' "
            f"AND NOT EXISTS(SELECT 1 FROM {request_ids} j WHERE j.type<>'text' OR length(trim(j.value,char(9)||char(10)||char(11)||char(12)||char(13)||' '))=0) "
            f"AND (SELECT count(*)=count(DISTINCT value) FROM {request_ids})"
        )

        def shape(keys: str, count: int, extra: str) -> str:
            return (
                "CASE WHEN json_valid(w.effects) THEN CASE WHEN json_type(w.effects)='object' THEN "
                f"({common}) AND (SELECT count(*) FROM json_each(w.effects))={count} "
                f"AND NOT EXISTS(SELECT 1 FROM json_each(w.effects) j WHERE j.key NOT IN ({keys})) "
                f"AND ({extra}) ELSE 0 END ELSE 0 END"
            )

        keys = "'version','skipped','billing','request_ids'"
        return (
            shape(keys, 4, "json_type(w.effects,'$.billing') IN ('object','null')"),
            shape(
                keys + ",'retired'",
                5,
                "json_type(w.effects,'$.retired')='true' AND json_type(w.effects,'$.billing')='null'",
            ),
        )

    def transition_window_history_request_match(self, request_sql: str) -> str:
        # Older JSON1 scalar extraction can truncate an escaped NUL.
        return (
            "EXISTS(SELECT 1 FROM json_each(CASE WHEN json_valid(w.effects) "
            "THEN w.effects ELSE '{}' END,'$.request_ids') item "
            f"WHERE item.type='text' AND item.value={request_sql} "
            "AND instr(w.effects -> item.fullkey,char(92)||'u0000')=0)"
        )

    def transition_window_history_no_billing(self) -> str:
        return (
            "CASE WHEN json_valid(w.effects) THEN json_type(w.effects,'$.billing')='null' "
            "ELSE 0 END"
        )

    def transition_window_billing_descriptor_valid(self) -> str:
        return (
            "CASE WHEN json_valid(w.effects) THEN reflexio_valid_window_delivery("
            "CASE w.kind WHEN 'profile' THEN 'profile' WHEN 'playbook' THEN 'playbook' END,"
            "CAST(w.effects -> '$.billing' AS BLOB))=1 ELSE 0 END"
        )

    def transition_window_history_parent_idle(self) -> str:
        return (
            "CASE WHEN json_valid(a.pending_tool_call_ids) THEN json_type(a.pending_tool_call_ids)='array' "
            "AND json_array_length(a.pending_tool_call_ids)=0 ELSE 0 END "
            "AND CASE WHEN json_valid(a.generation_request_snapshot) THEN "
            "(json_type(a.generation_request_snapshot,'$.overflow.checkpoint.active_child') IS NULL "
            "OR json_type(a.generation_request_snapshot,'$.overflow.checkpoint.active_child')='null') "
            "AND (json_type(a.generation_request_snapshot,'$.overflow.checkpoint.waiting_child') IS NULL "
            "OR json_type(a.generation_request_snapshot,'$.overflow.checkpoint.waiting_child')='null') ELSE 0 END"
        )

    def transition_project_scope(self, source: str) -> tuple[str, str]:  # noqa: ARG002 -- SQLite has no projects.
        return "'' AS project_id", "? = ''"

    def transition_unbound_project(self) -> str:
        return "0=1"  # Projects do not exist in OSS SQLite.

    def lock_learning_policy_authority(self, *, exclusive: bool) -> None:  # noqa: ARG002 -- BEGIN IMMEDIATE serializes SQLite authority writers.
        if not self.conn.in_transaction:
            raise ValueError("Policy authority lock requires an existing transaction")

    def read_learning_policy_authority(self) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT authority FROM _learning_policy_authority WHERE singleton=1"
        )
        return json.loads(rows[0]["authority"]) if rows else None

    def write_learning_policy_authority(self, payload: str, *, missing: bool) -> None:
        self.lock_learning_policy_authority(exclusive=True)
        if missing:
            self.query(
                "INSERT INTO _learning_policy_authority(singleton,authority) VALUES(1,?)",
                (payload,),
            )
        elif (
            len(
                self.query(
                    "UPDATE _learning_policy_authority SET authority=? WHERE singleton=1 RETURNING singleton",
                    (payload,),
                )
            )
            != 1
        ):
            raise ValueError("Dataset policy authority disappeared")

    @contextmanager
    def extraction_project_scope(self, project_id: str) -> Iterator[None]:
        """Validate SQLite's neutral scope; it has no native project carrier."""
        self.learning_transition_project(project_id)
        yield

    def learning_transition_project(self, project_id: str | None = None) -> str:
        if project_id or current_project_id():
            raise ValueError("SQLite learning transitions do not support project scope")
        return ""

    def lock_learning_transition(self, key: str) -> None:  # noqa: ARG002 -- BEGIN IMMEDIATE owns writer serialization.
        if not self.conn.in_transaction:
            raise ValueError("Learning transition requires a writer transaction")

    def insert_learning_transition(self, key: str, project: str, payload: str) -> None:  # noqa: ARG002 -- SQLite project scope was checked before writing.
        self.query(
            "INSERT INTO _operation_state(service_name,operation_state) VALUES(?,?)",
            (key, payload),
        )

    def transition_json_scalar(self, field: str) -> str:
        values = {
            "output_id": "json_extract(r.committed_output,'$.publication.id')",
            "preparation_id": "json_extract(r.preparation,'$.output_id')",
            "overflow_stage": "json_extract(r.generation_request_snapshot,'$.overflow.checkpoint.stage')",
        }
        return values[field]

    def transition_window_publication_proof(self) -> str:
        # Older JSON1 scalar extraction can truncate strings at an escaped NUL.
        # The first durable-window writer already emitted v1 envelopes. A raw
        # payload may be readable during recovery but cannot prove this lineage.
        return (
            "CASE WHEN json_valid(w.effects) AND json_valid(w.manifest) THEN ("
            "w.project_id='' AND typeof(w.predecessor)='integer' AND typeof(w.end_seq)='integer' "
            "AND (w.effects -> '$.acknowledged_window_run_id')=json_quote(r.id) "
            f"AND ({self.transition_window_billing_descriptor_valid()}) "
            "AND json_type(w.effects,'$.version')='integer' "
            "AND json_extract(w.effects,'$.version')=1 AND json_type(w.effects,'$.skipped')='true' "
            "AND json_type(w.manifest)='array' AND json_array_length(w.manifest)=0 "
            "AND json_type(r.generation_request_snapshot,'$.overflow.version')='integer' "
            "AND json_extract(r.generation_request_snapshot,'$.overflow.version')=1 "
            "AND (r.generation_request_snapshot -> '$.overflow.checkpoint.stage')=json_quote('recovered') "
            "AND json_type(r.generation_request_snapshot,'$.overflow.source_digest')='text' "
            "AND length(r.generation_request_snapshot -> '$.overflow.source_digest')=66 "
            "AND substr(r.generation_request_snapshot -> '$.overflow.source_digest',2,64) NOT GLOB '*[^a-f0-9]*' "
            "AND json_type(r.committed_output,'$._reflexio_envelope_version')='integer' "
            "AND json_extract(r.committed_output,'$._reflexio_envelope_version')=1 "
            "AND json_type(r.committed_output,'$.output')='object' "
            "AND json_type(r.committed_output,'$.publication') IS NULL) ELSE 0 END"
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
    def _learning_policy_authority_sql(self) -> Iterator[StreamSQL]:
        with self._lock:
            self._require_independent_writer()
            with self._stream_sql() as db:
                yield db

    @contextmanager
    def _independent_writer_metadata_sql(self) -> Iterator[StreamSQL]:
        with self._stream_sql(read_only=True) as db:
            self._require_independent_writer()
            self.conn.execute("BEGIN")
            try:
                yield db
            finally:
                self.conn.rollback()

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
