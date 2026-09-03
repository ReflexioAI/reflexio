"""SQLite subject-write gate.

This is the load-bearing sliver of the (now enterprise-only) governance /
erasure system that core OSS writers depend on directly: every core SQLite
writer (session_outcomes, requests, playbook, profiles, interactions) calls
``_assert_subject_writable_locked`` before writing, to refuse a write for a
subject with an active erasure barrier.

That check only ever *reads* ``subject_write_barriers`` — it does not create,
complete, or fail a barrier, does not touch ``purge_operations`` or
``audit_events``, and does not construct any governance domain model. The
public erasure orchestration (begin/complete/fail a barrier, the purge
lifecycle, the audit trail, ``GovernanceService`` itself) is an
enterprise-only surface with zero OSS routes/CLI/client consumers and moved
to ``reflexio_ext`` — see
``docs/superpowers/specs/2026-09-02-project-scoped-tenancy-design.md`` §9.1.
This mixin is what stayed behind, because core OSS writes structurally
depend on it regardless of whether an erasure feature is reachable at all.
"""

from __future__ import annotations

import sqlite3
import threading

from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError

_SUBJECT_WRITE_BARRIERS_DDL = """
CREATE TABLE IF NOT EXISTS subject_write_barriers (
    org_id TEXT NOT NULL,
    subject_ref TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    error_detail TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (org_id, subject_ref),
    CHECK (status IN ('erasing', 'erased', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_subject_write_barriers_org_purge
    ON subject_write_barriers(org_id, purge_id);
"""


def init_subject_write_barrier_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_SUBJECT_WRITE_BARRIERS_DDL)
    _ensure_governance_subject_ref_columns(conn)


def _ensure_governance_subject_ref_columns(conn: sqlite3.Connection) -> None:
    for table in (
        "requests",
        "interactions",
        "profiles",
        "user_playbooks",
        "agent_success_evaluation_result",
    ):
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if not columns:
            continue
        if "governance_subject_ref" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN governance_subject_ref TEXT")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_governance_subject_ref "
            f"ON {table}(governance_subject_ref)"
        )


class SubjectWriteGateMixin:
    """SQLite subject-write-gate primitives, consumed by every core writer."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _active_subject_barrier_locked(self, subject_ref: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM subject_write_barriers
               WHERE org_id = ? AND subject_ref = ? AND status IN ('erasing', 'erased')""",
            (self.org_id, subject_ref),
        ).fetchone()

    def _assert_subject_writable_locked(self, subject_ref: str) -> None:
        row = self._active_subject_barrier_locked(subject_ref)
        if row is not None:
            raise SubjectWriteBarrierError(
                f"subject {subject_ref} is blocked by erasure barrier {row['purge_id']}"
            )

    def _subject_ref_for_user_id(self, user_id: str) -> str:
        return governance_subject_ref(
            self.org_id,
            user_id,
            get_governance_ref_secret(),
        )
