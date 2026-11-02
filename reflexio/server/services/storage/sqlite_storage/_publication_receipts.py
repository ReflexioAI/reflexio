"""Additive SQLite installation; historical run receipts remain unchanged."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from reflexio.server.services.extraction.publication_receipts import (
    valid_receipt_payload,
    window_handoff_matches,
)

# Receipt data constraints, not operator policy or retention/budget defaults.
_TABLE = """
CREATE TABLE _agent_output_receipts (
 receipt_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
 org_id TEXT NOT NULL CHECK(trim(org_id)<>''),
 project_id TEXT NOT NULL DEFAULT '' CHECK(project_id=''),
 run_id TEXT,
 output_id TEXT,
 row_kind TEXT NOT NULL DEFAULT 'output',
 entity_type TEXT NOT NULL,
 learning_ids TEXT NOT NULL,
 delivery_owner TEXT NOT NULL DEFAULT 'receipt',
 delivery_state TEXT NOT NULL,
 delivery_key TEXT,
 delivery_facts TEXT,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 FOREIGN KEY(run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE,
 CHECK(length(receipt_id)=32 AND receipt_id NOT GLOB '*[^0-9a-f]*'),
 CHECK(reflexio_valid_publication_receipt(row_kind,entity_type,learning_ids,delivery_state,delivery_facts)=1),
 CHECK(
  (row_kind='output' AND delivery_owner='receipt' AND delivery_key IS NULL AND
    ((run_id IS NULL AND output_id IS NULL) OR
     (run_id IS NOT NULL AND output_id IS NOT NULL AND length(output_id)=32 AND output_id NOT GLOB '*[^0-9a-f]*')))
  OR
  (row_kind='window_handoff' AND run_id IS NULL AND output_id IS NULL AND delivery_owner='receipt' AND delivery_key IS NOT NULL AND trim(delivery_key)<>''))
)
"""

_GUARDS = {
    "publication_output_identity": """CREATE UNIQUE INDEX IF NOT EXISTS publication_output_identity
 ON _agent_output_receipts(org_id,run_id,output_id)
 WHERE row_kind='output' AND run_id IS NOT NULL""",
    "publication_window_handoff_identity": """CREATE UNIQUE INDEX IF NOT EXISTS publication_window_handoff_identity
 ON _agent_output_receipts(org_id,project_id,delivery_key)
 WHERE row_kind='window_handoff'""",
    "publication_pending_delivery": """CREATE INDEX IF NOT EXISTS publication_pending_delivery
 ON _agent_output_receipts(org_id)
 WHERE delivery_state='pending'""",
    "publication_receipt_insert_guard": """CREATE TRIGGER IF NOT EXISTS publication_receipt_insert_guard
 BEFORE INSERT ON _agent_output_receipts BEGIN
  -- SQLite REPLACE can skip DELETE triggers when recursive_triggers is off.
  -- Refuse identity conflicts before its implicit deletion instead of relying
  -- on a connection-wide recursive-trigger mode.
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM _agent_output_receipts q
    WHERE q.receipt_id=NEW.receipt_id
      OR (q.row_kind='output' AND NEW.row_kind='output' AND q.org_id=NEW.org_id
        AND q.run_id=NEW.run_id AND q.output_id=NEW.output_id)
      OR (q.row_kind='window_handoff' AND NEW.row_kind='window_handoff'
        AND q.org_id=NEW.org_id AND q.project_id=NEW.project_id AND q.delivery_key=NEW.delivery_key)
  ) THEN RAISE(ABORT,'publication receipt identity already exists') END;
  SELECT CASE WHEN NEW.row_kind='legacy' THEN RAISE(ABORT,'historical receipt insertion is retired') END;
  SELECT CASE WHEN NEW.row_kind='output' AND
   (NEW.run_id IS NULL OR NEW.output_id IS NULL OR NOT EXISTS (
     SELECT 1 FROM _agent_runs r WHERE r.id=NEW.run_id AND r.org_id=NEW.org_id
      AND r.status IN ('agent_completed','finalizing')
      AND json_extract(r.committed_output,'$.publication.id')=NEW.output_id
      AND ((r.extractor_kind='profile' AND NEW.entity_type='profile') OR
           (r.extractor_kind='playbook' AND NEW.entity_type='user_playbook'))))
   THEN RAISE(ABORT,'publication output binding changed') END;
  SELECT CASE WHEN NEW.row_kind='output' AND
    ((json_array_length(NEW.learning_ids)=0 AND NEW.delivery_state<>'accepted') OR
     (json_array_length(NEW.learning_ids)>0 AND NEW.delivery_state<>'pending'))
   THEN RAISE(ABORT,'new publication delivery state invalid') END;
  SELECT CASE WHEN NEW.row_kind='window_handoff' AND NEW.delivery_state<>'pending'
   THEN RAISE(ABORT,'new window handoff must remain pending') END;
 END""",
    "publication_receipt_update_guard": """CREATE TRIGGER IF NOT EXISTS publication_receipt_update_guard
 BEFORE UPDATE ON _agent_output_receipts WHEN
 NOT (
   NEW.rowid IS OLD.rowid
   AND NEW.receipt_id IS OLD.receipt_id AND NEW.org_id IS OLD.org_id
   AND NEW.project_id IS OLD.project_id AND NEW.row_kind IS OLD.row_kind
   AND NEW.entity_type IS OLD.entity_type AND NEW.learning_ids IS OLD.learning_ids
   AND NEW.delivery_owner IS OLD.delivery_owner AND NEW.delivery_key IS OLD.delivery_key
   AND NEW.created_at IS OLD.created_at
   AND (
    (NEW.run_id IS OLD.run_id AND NEW.output_id IS OLD.output_id
      AND NEW.delivery_state IS OLD.delivery_state AND NEW.delivery_facts IS OLD.delivery_facts)
    OR (OLD.delivery_state='pending' AND NEW.delivery_state='accepted' AND NEW.delivery_facts IS NULL
      AND NEW.run_id IS OLD.run_id AND NEW.output_id IS OLD.output_id)
    OR (OLD.delivery_state='pending' AND OLD.row_kind='output' AND OLD.run_id IS NOT NULL
      AND NEW.run_id IS NULL AND NEW.output_id IS NULL AND NEW.delivery_state='pending'
      AND NEW.delivery_facts IS OLD.delivery_facts)
   )
 ) BEGIN SELECT RAISE(ABORT,'publication receipt is immutable'); END""",
    "publication_receipt_delete_guard": """CREATE TRIGGER IF NOT EXISTS publication_receipt_delete_guard
 BEFORE DELETE ON _agent_output_receipts WHEN OLD.delivery_state='pending'
 BEGIN SELECT RAISE(ABORT,'pending financial publication cannot be deleted'); END""",
    "publication_window_scope_guard": """CREATE TRIGGER IF NOT EXISTS publication_window_scope_guard
 BEFORE UPDATE OF org_id ON extraction_windows WHEN NEW.org_id IS NOT OLD.org_id
 BEGIN SELECT RAISE(ABORT,'window publication scope is immutable'); END""",
    "publication_window_delete_guard": """CREATE TRIGGER IF NOT EXISTS publication_window_delete_guard
 BEFORE DELETE ON extraction_windows
 WHEN OLD.effects_done=0 AND OLD.effects IS NOT NULL
   AND json_type(OLD.effects,'$.billing') IS NOT NULL
   AND json_type(OLD.effects,'$.billing')<>'null'
   AND NOT EXISTS (
    SELECT 1 FROM _agent_output_receipts q
    WHERE OLD.org_id=reflexio_publication_org() AND q.org_id=OLD.org_id
      AND q.project_id=OLD.project_id
      AND q.row_kind='window_handoff' AND q.delivery_key=OLD.window_id
      AND q.delivery_state='pending'
      AND q.entity_type=CASE OLD.kind WHEN 'profile' THEN 'profile'
                         WHEN 'playbook' THEN 'user_playbook' END
      AND reflexio_window_handoff_matches(OLD.kind,json_extract(OLD.effects,'$.billing'),
                                         q.learning_ids,q.delivery_facts)=1)
 BEGIN SELECT RAISE(ABORT,'pending window financial publication cannot be deleted'); END""",
    "publication_window_replace_guard": """CREATE TRIGGER IF NOT EXISTS publication_window_replace_guard
 BEFORE INSERT ON extraction_windows WHEN EXISTS (
   SELECT 1 FROM extraction_windows w WHERE
    (w.window_id=NEW.window_id OR (w.user_id=NEW.user_id AND w.kind=NEW.kind
      AND w.predecessor=NEW.predecessor AND w.project_id=NEW.project_id))
    AND w.effects_done=0 AND w.effects IS NOT NULL
    AND json_type(w.effects,'$.billing') IS NOT NULL
    AND json_type(w.effects,'$.billing')<>'null')
 BEGIN SELECT RAISE(ABORT,'pending window financial publication cannot be replaced'); END""",
}


def _installed_receipt_sql(
    conn: sqlite3.Connection,
) -> tuple[dict[str, str | None], bool]:
    installed = dict(
        conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE tbl_name IN "
            "('_agent_output_receipts','extraction_windows')"
        )
    )
    window_scope_ready = any(
        row[1] == "org_id"
        for row in conn.execute("PRAGMA table_info(extraction_windows)")
    )
    return installed, window_scope_ready


def _stored_sql(statement: str) -> str:
    # SQLite removes this CREATE prefix phrase in sqlite_master. Preserve all
    # other bytes: whitespace inside quoted SQL literals can change semantics.
    return statement.replace(" IF NOT EXISTS", "", 1).strip()


def _receipt_schema_current(
    installed: dict[str, str | None], window_scope_ready: bool
) -> bool:
    return installed.get("_agent_output_receipts") == _stored_sql(_TABLE) and all(
        installed.get(name) == _stored_sql(statement)
        for name, statement in _GUARDS.items()
        if not name.startswith("publication_window_") or window_scope_ready
    )


def _refuse_legacy_publication_debt(conn: sqlite3.Connection) -> None:
    # This relation belongs to the whole file, which may contain more
    # than one org. A first opener must not admit another org's debt.
    blocked = conn.execute("""SELECT r.id,r.org_id,r.status FROM _agent_runs r
      WHERE r.status IS NULL OR r.status NOT IN
        ('finalized','finalized_pending_tool','resume_ready','failed','expired','cancelled')
      OR (r.status IN ('failed','expired','cancelled') AND
        (r.committed_output IS NOT NULL OR EXISTS (
          SELECT 1 FROM _agent_run_finalization_receipts q WHERE q.run_id=r.id)))
      ORDER BY r.id LIMIT 1""").fetchone()
    if blocked is not None:
        raise ValueError(
            "Legacy publication cutover refused: undrained execution "
            f"org={blocked[1]} run={blocked[0]} status={blocked[2]}; "
            "drain with the old writer or establish an audited disposition. "
            "No historical output or financial evidence was changed."
        )


def validate_publication_cutover(
    org_id: str,
    *,
    db_path: str | None,
    base_dir: str | None,
) -> None:
    """Inspect persistent history before publishing an incompatible local file.

    No table, identity claim or directory is created. Runtime migration repeats
    the same debt check under its actual writer after the quiescent preflight.
    """
    from ._dataset_path import (
        IDENTITY_CLAIM_TABLE,
        LEGACY_DB_FILENAME,
        barrier_identity_labels,
        derive_db_path,
        stored_identity_labels,
        validate_dataset_identity,
    )

    def claimed_owner(path: Path) -> str | None:
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (IDENTITY_CLAIM_TABLE,),
                ).fetchone()
                is None
            ):
                return None
            row = conn.execute(
                f"SELECT org_id FROM {IDENTITY_CLAIM_TABLE} WHERE k=1 LIMIT 1"
            ).fetchone()  # noqa: S608 -- Fixed identity table.
            return str(row[0]) if row else None

    if db_path in ("", ":memory:"):
        return  # Explicit transient SQLite targets have no prior persistent debt.
    if db_path is not None:
        path = Path(db_path)
        if not path.is_absolute():
            raise ValueError(
                "Local storage admission requires an absolute SQLite path; preserve the original configuration and select its verified database before upgrading"
            )
    else:
        from reflexio.server import LOCAL_STORAGE_PATH

        validate_dataset_identity(org_id)
        root = Path(base_dir if base_dir is not None else LOCAL_STORAGE_PATH)
        derived = derive_db_path(root, org_id)
        legacy = root / LEGACY_DB_FILENAME
        path = derived
        if derived.exists():
            owner = claimed_owner(derived)
            if owner is not None and owner != org_id:
                raise ValueError(
                    "SQLite dataset identity conflicts with the selected organization"
                )
        elif legacy.exists():
            labels = stored_identity_labels(legacy)
            if not labels or org_id in labels:
                if (labels or set()) - {org_id} and barrier_identity_labels(legacy) - {
                    org_id
                }:
                    raise ValueError(
                        "SQLite legacy data contains foreign erasure barriers; split identities before upgrading"
                    )
                owner = claimed_owner(legacy)
                if owner is None or owner == org_id:
                    path = legacy
    if not path.exists():
        return
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('_agent_runs','_agent_run_finalization_receipts','_agent_output_receipts') LIMIT 3"
            )
        }
        if "_agent_output_receipts" in tables:
            installed, _ready = _installed_receipt_sql(conn)
            if installed.get("_agent_output_receipts") != _stored_sql(_TABLE):
                raise ValueError(
                    "Publication receipt schema is unsupported; original configuration was not upgraded"
                )
            return
        if "_agent_runs" not in tables:
            if (
                "_agent_run_finalization_receipts" in tables
                and conn.execute(
                    "SELECT 1 FROM _agent_run_finalization_receipts LIMIT 1"
                ).fetchone()
            ):
                raise ValueError("Historical publication receipt owner is unavailable")
            return
        if "_agent_run_finalization_receipts" not in tables:
            if conn.execute("SELECT 1 FROM _agent_runs LIMIT 1").fetchone():
                raise ValueError(
                    "Historical publication ledger is unavailable; original configuration was not upgraded"
                )
            return
        if conn.execute(
            "SELECT 1 FROM _agent_run_finalization_receipts q LEFT JOIN _agent_runs r ON r.id=q.run_id WHERE r.id IS NULL LIMIT 1"
        ).fetchone():
            raise ValueError("Historical publication receipt owner is unavailable")
        _refuse_legacy_publication_debt(conn)


def migrate_publication_receipts(conn: sqlite3.Connection, *, org_id: str) -> None:
    """Install atomically; current startup is read-only and never rewrites history."""
    conn.create_function(
        "reflexio_valid_publication_receipt",
        5,
        valid_receipt_payload,
        deterministic=True,
    )
    conn.create_function("reflexio_publication_org", 0, lambda: org_id)
    conn.create_function(
        "reflexio_window_handoff_matches", 4, window_handoff_matches, deterministic=True
    )
    if conn.in_transaction:
        raise ValueError("Publication receipt migration requires its own transaction")
    installed, window_scope_ready = _installed_receipt_sql(conn)
    if _receipt_schema_current(installed, window_scope_ready):
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-read AFTER the wait; a concurrent opener may already have installed
        # the schema and published debt. Never decide a mutation from old reads.
        installed, window_scope_ready = _installed_receipt_sql(conn)
        table = installed.get("_agent_output_receipts")
        if table is None:
            legacy_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(_agent_run_finalization_receipts)"
                )
            }
            if "receipt_id" in legacy_columns:
                raise ValueError("Historical publication receipt schema is unsupported")
            orphan = conn.execute("""SELECT 1 FROM _agent_run_finalization_receipts q
              LEFT JOIN _agent_runs r ON r.id=q.run_id
              WHERE r.id IS NULL LIMIT 1""").fetchone()
            if orphan is not None:
                raise ValueError("Historical publication receipt owner is unavailable")
            _refuse_legacy_publication_debt(conn)
            conn.execute(_TABLE)
        elif table != _stored_sql(_TABLE):
            raise ValueError("Publication receipt schema is unsupported")
        for name, statement in _GUARDS.items():
            if name.startswith("publication_window_") and not window_scope_ready:
                continue
            if name in installed and installed[name] != _stored_sql(statement):
                kind = "TRIGGER" if statement.startswith("CREATE TRIGGER") else "INDEX"
                # Fixed names owned by this module; no external SQL identifier.
                conn.execute(f'DROP {kind} "{name}"')  # noqa: S608
            conn.execute(statement)
        installed, window_scope_ready = _installed_receipt_sql(conn)
        if not _receipt_schema_current(installed, window_scope_ready):
            raise ValueError("Publication receipt schema is incomplete")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
