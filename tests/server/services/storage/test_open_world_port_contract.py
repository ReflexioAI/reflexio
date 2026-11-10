"""``OpenWorldStoragePort`` contract on SQLite.

Binds the shared contract (``reflexio.test_support.open_world_port_contract``)
to a fresh SQLite database per test, with the store's clock pinned so every
case is exact. Also holds the SQLite-only cases: the one-clock-reading rule
across 00:00 UTC (a backend whose clock is not injectable cannot express it),
bringing a ledger created without integrity columns forward, and the
exposure-ledger paths Postgres refuses before reaching (a replay that rewrites
its references, an ownerless playbook, an event id re-recorded with other
content).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from reflexio.models.api_schema.domain import Interaction, Request, RetrievedLearning
from reflexio.models.api_schema.service_schemas import DeleteUserInteractionRequest
from reflexio.server.services.offline_tuner.exposure_receipts import (
    receipt_project_id,
)
from reflexio.server.services.playbook.publication import canonical_json_bytes
from reflexio.server.services.search_exposure import UserPlaybookExposureEvent
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import (
    SQLiteStorage,
    _open_world_evidence,
)
from reflexio.server.services.storage.sqlite_storage._open_world import (
    _INSERT_EXPOSURE_SQL,
    _exposure_row,
)
from reflexio.test_support.open_world_attempt_contract import AttemptRows
from reflexio.test_support.open_world_evidence_contract import (
    JobRow,
    call_once_before,
    jumping_clock,
)
from reflexio.test_support.open_world_exposure_contract import ExposureRow
from reflexio.test_support.open_world_port_contract import (
    OpenWorldPortContract,
    ProgressRow,
)
from reflexio.test_support.open_world_publication_contract import (
    LifecycleRow,
    PublicationRows,
)

#: A synthetic midday instant, far from either day boundary.
_PINNED_NOW = int(datetime(2026, 6, 15, 12, 0, tzinfo=UTC).timestamp())


class _SQLitePortHarness:
    """A fresh SQLite store whose sweep clock reads ``self.clock_value``."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage
        self.clock_value: float = _PINNED_NOW
        storage._sweep_progress_clock = lambda: self.clock_value  # noqa: SLF001
        storage._publication_clock = lambda: self.clock_value  # noqa: SLF001

    @property
    def storage(self) -> SQLiteStorage:
        return self._storage

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._storage.conn

    def now(self) -> int:
        return int(self.clock_value)

    def _write(self, sql: str, params: tuple[object, ...]) -> sqlite3.Cursor:
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return cursor

    def seed_playbook(self, *, current: bool = True) -> int:
        cursor = self._write(
            "INSERT INTO user_playbooks (created_at, request_id, content, status)"
            " VALUES (?, ?, 'synthetic playbook', ?)",
            (
                datetime.fromtimestamp(self.now(), tz=UTC).isoformat(),
                f"seed-{uuid4().hex[:10]}",
                None if current else "archived",
            ),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def seed_request(self, *, session_id: str, user_id: str, at_epoch: int) -> str:
        request_id = f"request-{uuid4().hex[:12]}"
        self._write(
            "INSERT INTO requests (request_id, user_id, created_at, session_id)"
            " VALUES (?, ?, ?, ?)",
            (
                request_id,
                user_id,
                datetime.fromtimestamp(at_epoch, tz=UTC).isoformat(),
                session_id,
            ),
        )
        return request_id

    def seed_session_outcome(
        self, *, session_id: str, user_id: str, at_epoch: int
    ) -> None:
        self._write(
            "INSERT INTO session_outcomes (outcome_id, outcome_revision, user_id,"
            " session_id, outcome, occurred_at, source, outcome_contract_digest,"
            " finalized_trajectory_digest, governance_subject_ref, created_at)"
            " VALUES (?, 1, ?, ?, 'success', ?, 'test', ?, ?, ?, ?)",
            (
                f"outcome-{uuid4().hex[:12]}",
                user_id,
                session_id,
                at_epoch,
                "0" * 64,
                "0" * 64,
                f"subref_v1_{uuid4().hex}",
                at_epoch,
            ),
        )

    def seed_exposure(
        self,
        *,
        user_playbook_id: int,
        request_id: str | None,
        session_id: str | None,
        user_id: str,
        exposed_at: int | None,
        ingested_at: int,
        published: bool = True,
    ) -> None:
        prefix = "publish-v1:" if published else "legacy-search:"
        self.insert_exposure(
            UserPlaybookExposureEvent(
                exposure_event_id=prefix + uuid4().hex,
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                playbook_owner_user_id="owner",
                user_playbook_id=user_playbook_id,
                served_semantic_digest=None,
                served_full_version_fingerprint=None,
                exposed_at=exposed_at,
                ingested_at=ingested_at,
                governance_subject_ref=None,
                playbook_owner_governance_subject_ref=None,
            )
        )

    # -- the exposure half of the harness

    @property
    def receipt_project_id(self) -> str:
        return receipt_project_id()

    refuses_unattributable_exposure = False

    def _exposure_rows(self, sql: str, key: str) -> list[ExposureRow]:
        rows = self._conn.execute(sql, (key,)).fetchall()
        return [
            ExposureRow(
                **{name: row[name] for name in ExposureRow._fields}
                | {"integrity_reasons": tuple(json.loads(row["integrity_reasons"]))}
            )
            for row in rows
        ]

    def exposure_rows(self, request_id: str) -> list[ExposureRow]:
        return self._exposure_rows(
            "SELECT * FROM user_playbook_exposure_events WHERE request_id = ?"
            " ORDER BY exposure_event_id",
            request_id,
        )

    def insert_exposure(self, event: UserPlaybookExposureEvent) -> ExposureRow:
        self._write(_INSERT_EXPOSURE_SQL, _exposure_row(event))
        (row,) = self._exposure_rows(
            "SELECT * FROM user_playbook_exposure_events WHERE exposure_event_id = ?",
            event.exposure_event_id,
        )
        return row

    def block_subject(self, user_id: str) -> None:
        self._write(
            "INSERT INTO subject_write_barriers (org_id, subject_ref, purge_id,"
            " status, created_at, updated_at) VALUES (?, ?, ?, 'erasing', 0, 0)",
            (
                self._storage.org_id,
                self._storage._subject_ref_for_user_id(user_id),  # noqa: SLF001
                f"purge-{uuid4().hex[:8]}",
            ),
        )

    # -- the evidence half of the harness

    governance_registry = False

    def register_subject(self, user_id: str, *, epoch: int = 0) -> str:
        assert epoch == 0, "SQLite holds every subject at epoch 0"
        return self._storage._subject_ref_for_user_id(user_id)  # noqa: SLF001

    def optimization_jobs(self, target_id: int) -> list[JobRow]:
        rows = self._conn.execute(
            "SELECT job_id, status, stage, terminal_outcome"
            " FROM playbook_optimization_jobs WHERE target_id = ? ORDER BY job_id",
            (target_id,),
        ).fetchall()
        return [JobRow(*tuple(row)) for row in rows]

    @contextmanager
    def fail_artifact_writes(self) -> Iterator[None]:
        self._conn.execute(
            "CREATE TEMP TRIGGER contract_refuse_artifact"
            " BEFORE INSERT ON playbook_optimization_artifacts"
            " BEGIN SELECT RAISE(ABORT, 'artifact write refused'); END"
        )
        try:
            yield
        finally:
            self._conn.execute("DROP TRIGGER IF EXISTS temp.contract_refuse_artifact")

    @contextmanager
    def second_storage(self) -> Iterator[SQLiteStorage]:
        other = SQLiteStorage(
            org_id=self._storage.org_id, db_path=self._storage.db_path
        )
        try:
            yield other
        finally:
            other.conn.close()

    @contextmanager
    def during_snapshot(self, write: Callable[[], None]) -> Iterator[None]:
        with call_once_before(
            _open_world_evidence, "resolve_exposure_correlations", write
        ):
            yield

    @contextmanager
    def source_read_budget(self, limit_bytes: int) -> Iterator[None]:
        with patch.object(
            _open_world_evidence, "_SOURCE_READ_BYTES_LIMIT", limit_bytes
        ):
            yield

    @contextmanager
    def source_read_clock_jumps(self) -> Iterator[None]:
        with jumping_clock(_open_world_evidence):
            yield

    # -- the attempt half of the harness

    clear_erases_jobs = True

    def attempt_rows(self, job_id: int) -> AttemptRows:
        def count(table: str) -> int:
            (value,) = self._conn.execute(
                f"SELECT count(*) FROM {table} WHERE job_id = ?",  # noqa: S608
                (job_id,),
            ).fetchone()
            return int(value)

        return AttemptRows(
            invocations=count("offline_tuner_open_world_invocations"),
            contexts=count("offline_tuner_open_world_attempt_context"),
            artifacts=count("playbook_optimization_artifacts"),
            compactions=count("offline_tuner_session_compactions"),
        )

    def corrupt_artifact_digest(
        self, job_id: int, artifact_kind: str, content_digest: str
    ) -> None:
        self._write(
            "UPDATE playbook_optimization_artifacts SET content_digest = ?"
            " WHERE job_id = ? AND artifact_kind = ?",
            (content_digest, job_id, artifact_kind),
        )

    # -- the publication half of the harness

    def publication_rows(self, job_id: int) -> PublicationRows:
        def count(sql: str) -> int:
            (value,) = self._conn.execute(sql, (job_id,)).fetchone()
            return int(value)

        return PublicationRows(
            claims=count(
                "SELECT count(*) FROM user_playbook_publication_claims WHERE job_id = ?"
            ),
            consumed_claims=count(
                "SELECT count(*) FROM user_playbook_publication_claims"
                " WHERE job_id = ? AND consumed = 1"
            ),
            staging=count(
                "SELECT count(*) FROM user_playbook_publication_staging"
                " WHERE job_id = ?"
            ),
            results=count(
                "SELECT count(*) FROM user_playbook_publication_results"
                " WHERE job_id = ?"
            ),
            lifecycles=count(
                "SELECT count(*) FROM user_playbook_deployment_lifecycles"
                " WHERE job_id = ?"
            ),
            publication_events=count(
                "SELECT count(*) FROM playbook_optimization_events"
                " WHERE job_id = ? AND event_type = 'provisional_publication'"
            ),
        )

    def lifecycle(self, job_id: int) -> LifecycleRow | None:
        row = self._conn.execute(
            "SELECT state, predecessor_id, successor_id, published_at,"
            " observation_deadline, terminal_reason, terminal_at,"
            " aggregation_obligation FROM user_playbook_deployment_lifecycles"
            " WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return None if row is None else LifecycleRow(*tuple(row))

    def lineage_ops(self, request_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT entity_id, op FROM lineage_event WHERE request_id = ?",
            (request_id,),
        ).fetchall()
        return sorted((str(row[0]), str(row[1])) for row in rows)

    def playbook_ids(self, user_id: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?"
            " ORDER BY user_playbook_id",
            (user_id,),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def drift_playbook(self, user_playbook_id: int, *, column: str) -> None:
        assert column in {"content", "rationale"}
        self._write(
            f"UPDATE user_playbooks SET {column} = COALESCE({column}, '')"  # noqa: S608
            " || ' (edited)' WHERE user_playbook_id = ?",
            (user_playbook_id,),
        )

    @contextmanager
    def fail_result_writes(self) -> Iterator[None]:
        self._conn.execute(
            "CREATE TEMP TRIGGER contract_refuse_result"
            " BEFORE INSERT ON user_playbook_publication_results"
            " BEGIN SELECT RAISE(ABORT, 'publication result write refused'); END"
        )
        try:
            yield
        finally:
            self._conn.execute("DROP TRIGGER IF EXISTS temp.contract_refuse_result")

    def rewrite_attempt_context(self, job_id: int) -> None:
        (context_json,) = self._conn.execute(
            "SELECT context_json FROM offline_tuner_open_world_attempt_context"
            " WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        rewritten = canonical_json_bytes(
            {**json.loads(context_json), "contract_probe": True}
        ).decode()
        self._write(
            "UPDATE offline_tuner_open_world_attempt_context"
            " SET context_json = ?, context_digest = ? WHERE job_id = ?",
            (rewritten, sha256(rewritten.encode()).hexdigest(), job_id),
        )

    def age_progress(self, user_playbook_id: int, *, seconds: int) -> None:
        self._write(
            "UPDATE open_world_sweep_progress SET attempted_at = attempted_at - ?"
            " WHERE user_playbook_id = ?",
            (seconds, user_playbook_id),
        )

    def backdate_progress(self, user_playbook_id: int, *, days: int) -> None:
        self._write(
            "UPDATE open_world_sweep_progress"
            " SET progress_day = date(progress_day, ?),"
            " attempted_at = attempted_at - ? * 86400"
            " WHERE user_playbook_id = ?",
            (f"-{days} days", days, user_playbook_id),
        )

    def progress_rows(self, user_playbook_id: int) -> list[ProgressRow]:
        rows = self._conn.execute(
            "SELECT progress_day, outcome, job_id, attempted_at, attempt_count"
            " FROM open_world_sweep_progress WHERE user_playbook_id = ?"
            " ORDER BY progress_day",
            (user_playbook_id,),
        ).fetchall()
        return [ProgressRow(*tuple(row)) for row in rows]


@pytest.fixture
def port_harness(tmp_path: Path) -> Iterator[_SQLitePortHarness]:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="contract_test", db_path=str(tmp_path / "r.db"))
        try:
            yield _SQLitePortHarness(storage)
        finally:
            storage.conn.close()


class TestSQLiteOpenWorldPort(OpenWorldPortContract):
    pass


def test_the_record_reads_the_clock_once_across_midnight(
    port_harness: _SQLitePortHarness,
) -> None:
    """A second clock reading would put the next day's instant on the old day."""
    midnight = _PINNED_NOW - _PINNED_NOW % 86_400 + 86_400
    readings = iter([midnight - 0.25, midnight + 0.25])

    def clock() -> float:
        return next(readings)

    port_harness.storage._sweep_progress_clock = clock  # noqa: SLF001
    playbook_id = port_harness.seed_playbook()

    record = port_harness.storage.record_open_world_sweep_progress(
        user_playbook_id=playbook_id, outcome="attempt_error", job_id=None
    )

    assert record.attempted_at == midnight - 1
    assert (
        record.progress_day
        == datetime.fromtimestamp(midnight - 1, tz=UTC).date().isoformat()
    )


def test_the_queue_reads_the_clock_once(
    port_harness: _SQLitePortHarness,
) -> None:
    calls: list[int] = []

    def clock() -> float:
        calls.append(1)
        return float(_PINNED_NOW)

    port_harness.storage._sweep_progress_clock = clock  # noqa: SLF001

    port_harness.storage.list_open_world_playbooks_without_progress(
        limit=10, window_seconds=86_400
    )

    assert len(calls) == 1


def test_migrate_is_idempotent_and_keeps_progress(
    port_harness: _SQLitePortHarness,
) -> None:
    playbook_id = port_harness.seed_playbook()
    port_harness.storage.record_open_world_sweep_progress(
        user_playbook_id=playbook_id, outcome="attempt_error", job_id=3
    )

    port_harness.storage.migrate()

    assert [row.job_id for row in port_harness.progress_rows(playbook_id)] == [3]


#: ``user_playbook_exposure_events`` as it was first created, before the
#: integrity columns. No released version shipped it; a database created from
#: that development revision still needs bringing forward.
_LEDGER_WITHOUT_INTEGRITY = """
DROP TABLE user_playbook_exposure_events;
CREATE TABLE user_playbook_exposure_events (
    exposure_event_id TEXT PRIMARY KEY,
    request_id TEXT,
    session_id TEXT,
    user_id TEXT,
    playbook_owner_user_id TEXT,
    user_playbook_id INTEGER,
    served_semantic_digest TEXT,
    served_full_version_fingerprint TEXT,
    exposed_at INTEGER,
    ingested_at INTEGER NOT NULL,
    governance_subject_ref TEXT,
    playbook_owner_governance_subject_ref TEXT
);
CREATE INDEX idx_up_exposure_events_playbook_window
    ON user_playbook_exposure_events(
        user_playbook_id, COALESCE(exposed_at, ingested_at)
    );
CREATE INDEX idx_up_exposure_events_request_id
    ON user_playbook_exposure_events(request_id);
INSERT INTO user_playbook_exposure_events VALUES
    ('publish-v1:kept', 'request', 'session', 'viewer', 'owner', 7,
     'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
     'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
     10, 20, 'subref_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
     'subref_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'),
    ('publish-v1:partial', 'request', 'session', 'viewer', 'owner', 7,
     NULL, NULL, NULL, 30, NULL, NULL);
"""


def test_migrate_brings_a_ledger_without_integrity_forward(
    port_harness: _SQLitePortHarness,
) -> None:
    conn = port_harness.storage.conn
    conn.executescript(_LEDGER_WITHOUT_INTEGRITY)

    port_harness.storage.migrate()
    port_harness.storage.migrate()

    kept, partial = port_harness.exposure_rows("request")
    assert (kept.exposure_event_id, kept.ingested_at) == ("publish-v1:kept", 20)
    assert (kept.integrity_state, kept.integrity_reasons) == ("complete", ())
    assert (partial.integrity_state, partial.integrity_reasons) == (
        "incomplete",
        (
            "missing_served_semantic_digest",
            "missing_served_full_version_fingerprint",
            "missing_exposed_at",
            "missing_governance_identity",
            "missing_playbook_owner_governance_identity",
        ),
    )
    indexes = {
        row[1]
        for row in conn.execute("PRAGMA index_list(user_playbook_exposure_events)")
    }
    assert {
        "idx_up_exposure_events_playbook_window",
        "idx_up_exposure_events_request_id",
    } <= indexes
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO user_playbook_exposure_events (exposure_event_id,"
            " ingested_at, integrity_state, integrity_reasons)"
            " VALUES ('publish-v1:incoherent', 0, 'complete', '[\"missing_user_id\"]')"
        )


def test_re_recording_an_event_id_with_other_content_is_refused(
    port_harness: _SQLitePortHarness,
) -> None:
    """A deleted interaction re-published under its id may not rewrite history."""
    storage = port_harness.storage
    first = port_harness.seed_playbook()
    second = port_harness.seed_playbook()
    storage.conn.execute(
        "UPDATE user_playbooks SET user_id = 'owner' WHERE user_playbook_id IN (?, ?)",
        (first, second),
    )
    storage.conn.commit()
    storage.add_request(Request(request_id="request", user_id="viewer", session_id="s"))

    def publish(playbook_id: int) -> None:
        interaction = Interaction(
            interaction_id=900,
            user_id="viewer",
            request_id="request",
            role="Assistant",
            content="synthetic answer",
            retrieved_learnings=[
                RetrievedLearning(kind="user_playbook", learning_id=str(playbook_id))
            ],
        )
        storage.add_user_interactions_bulk(
            "viewer", [interaction], embeddings_prepared=True
        )

    def forget_interaction() -> None:
        storage.delete_user_interaction(
            DeleteUserInteractionRequest(user_id="viewer", interaction_id=900)
        )

    publish(first)
    (recorded,) = port_harness.exposure_rows("request")
    forget_interaction()
    publish(first)
    assert port_harness.exposure_rows("request") == [recorded], (
        "the same content under the same id records nothing"
    )
    forget_interaction()

    with pytest.raises(StorageError, match="exposure event conflict"):
        publish(second)

    assert port_harness.exposure_rows("request") == [recorded]
    assert storage.get_user_interaction("viewer") == []


def test_a_replay_that_rewrites_its_references_records_nothing(
    port_harness: _SQLitePortHarness,
) -> None:
    """Only a new interaction records exposures; SQLite lets a replay rewrite it.

    Postgres refuses such a replay outright, so this case is SQLite's own.
    """
    storage = port_harness.storage
    playbook_id = port_harness.seed_playbook()
    storage.conn.execute(
        "UPDATE user_playbooks SET user_id = 'owner' WHERE user_playbook_id = ?",
        (playbook_id,),
    )
    storage.conn.commit()
    storage.add_request(Request(request_id="request", user_id="viewer", session_id="s"))
    interaction = Interaction(
        user_id="viewer",
        request_id="request",
        role="Assistant",
        content="synthetic answer",
        retrieved_learnings=[
            RetrievedLearning(kind="user_playbook", learning_id=str(playbook_id))
        ],
    )
    storage.add_user_interactions_bulk(
        "viewer", [interaction], embeddings_prepared=True
    )
    recorded = port_harness.exposure_rows("request")
    assert len(recorded) == 1

    interaction.retrieved_learnings = [
        *interaction.retrieved_learnings,
        RetrievedLearning(kind="user_playbook", learning_id=str(playbook_id)),
    ]
    storage.add_user_interactions_bulk(
        "viewer", [interaction], embeddings_prepared=True
    )

    assert port_harness.exposure_rows("request") == recorded


def test_an_ownerless_playbook_records_no_exposure(
    port_harness: _SQLitePortHarness,
) -> None:
    """Without an owner there is no subject to attribute; Postgres refuses."""
    storage = port_harness.storage
    ownerless = port_harness.seed_playbook()
    storage.add_request(Request(request_id="request", user_id="viewer", session_id="s"))

    storage.add_user_interactions_bulk(
        "viewer",
        [
            Interaction(
                user_id="viewer",
                request_id="request",
                role="Assistant",
                content="synthetic answer",
                retrieved_learnings=[
                    RetrievedLearning(kind="user_playbook", learning_id=str(ownerless))
                ],
            )
        ],
        embeddings_prepared=True,
    )

    assert len(storage.get_user_interaction("viewer")) == 1
    assert port_harness.exposure_rows("request") == []


def test_migrate_gives_owned_playbooks_their_subject_ref(
    port_harness: _SQLitePortHarness,
) -> None:
    """A playbook stored before its ref was written reads it after migrate (D4)."""
    storage = port_harness.storage
    owned = port_harness.seed_playbook()
    ownerless = port_harness.seed_playbook()
    storage.conn.execute(
        "UPDATE user_playbooks SET user_id = 'legacy-owner',"
        " governance_subject_ref = NULL WHERE user_playbook_id = ?",
        (owned,),
    )
    storage.conn.commit()
    before = storage.get_user_playbook_by_id(owned)
    assert before is not None and before.governance_subject_ref is None

    storage.migrate()
    storage.migrate()

    after = storage.get_user_playbook_by_id(owned)
    assert after is not None
    assert after.governance_subject_ref == storage._subject_ref_for_user_id(  # noqa: SLF001
        "legacy-owner"
    )
    unowned = storage.get_user_playbook_by_id(ownerless)
    assert unowned is not None and unowned.governance_subject_ref is None


def test_a_freeze_inside_a_commit_scope_rolls_back_with_it(
    port_harness: _SQLitePortHarness,
) -> None:
    """The freeze joins the caller's transaction instead of committing its own."""
    from reflexio.test_support.open_world_evidence_contract import (
        _freeze,
        _frozen_target,
    )

    playbook, request, now = _frozen_target(port_harness)

    with (
        pytest.raises(RuntimeError, match="abort the scope"),
        port_harness.storage.commit_scope(),
    ):
        assert _freeze(port_harness, request, now=now).claim is not None
        raise RuntimeError("abort the scope")

    assert port_harness.optimization_jobs(playbook.user_playbook_id) == []


def test_a_caught_freeze_failure_cannot_commit_the_callers_scope(
    port_harness: _SQLitePortHarness,
) -> None:
    """A failed freeze leaves no leased job behind, even if its caller swallows it."""
    from reflexio.server.services.storage.error import StorageError
    from reflexio.test_support.open_world_evidence_contract import (
        _freeze,
        _frozen_target,
    )

    playbook, request, now = _frozen_target(port_harness)

    with (
        pytest.raises(StorageError, match="marked transaction for rollback"),
        port_harness.storage.commit_scope(),
        port_harness.fail_artifact_writes(),
        suppress(Exception),
    ):
        _freeze(port_harness, request, now=now)

    assert port_harness.optimization_jobs(playbook.user_playbook_id) == []


#: The job table's ``terminal_outcome`` list as the previous release wrote it,
#: before the CHECK was derived from ``OptimizationTerminalOutcome``.
_PREVIOUS_TERMINAL_OUTCOMES = (
    "applied",
    "insufficient_negative_evidence",
    "insufficient_positive_evidence",
    "insufficient_coverage",
    "deployment_unsupported",
    "candidate_regressed",
    "candidate_did_not_improve",
    "incumbent_changed",
    "generation_failed",
    "publication_failed",
    "governance_erased",
    "no_grounded_hypothesis",
    "analyst_unqualified",
    "heldout_evidence_failed",
    "stale_incumbent",
    "governance_invalidated",
    "infrastructure_failure",
    "evidence_view_incomplete",
)


def _as_previous_release(conn: sqlite3.Connection) -> None:
    """Put the open-world schema back as the previous release left it.

    No attempt tables, and a job table whose ``terminal_outcome`` CHECK lists
    only the outcomes that release knew. Rows are kept.
    """
    from reflexio.server.services.storage.sqlite_storage._base import (
        _TERMINAL_OUTCOME_CHECK,
    )
    from reflexio.server.services.storage.sqlite_storage._open_world_attempt import (
        ATTEMPT_TABLES,
    )

    (table_sql,) = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'playbook_optimization_jobs'"
    ).fetchone()
    previous_check = (
        "terminal_outcome TEXT CHECK (terminal_outcome IS NULL OR terminal_outcome IN ("
        + ", ".join(f"'{outcome}'" for outcome in _PREVIOUS_TERMINAL_OUTCOMES)
        + "))"
    )
    assert _TERMINAL_OUTCOME_CHECK in table_sql
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in ATTEMPT_TABLES:
        conn.execute(f"DROP TABLE {table}")
    conn.execute(
        table_sql.replace(_TERMINAL_OUTCOME_CHECK, previous_check).replace(
            "playbook_optimization_jobs", "playbook_optimization_jobs_previous", 1
        )
    )
    conn.execute(
        "INSERT INTO playbook_optimization_jobs_previous"
        " SELECT * FROM playbook_optimization_jobs"
    )
    conn.execute("DROP TABLE playbook_optimization_jobs")
    conn.execute(
        "ALTER TABLE playbook_optimization_jobs_previous"
        " RENAME TO playbook_optimization_jobs"
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")


def test_migrate_brings_a_previous_release_database_forward(
    port_harness: _SQLitePortHarness,
) -> None:
    """The attempt tables appear, the outcome CHECK widens, and rows survive."""
    from reflexio.test_support.open_world_evidence_contract import (
        _freeze,
        _frozen_target,
    )

    storage = port_harness.storage
    playbook, request, now = _frozen_target(port_harness)
    record = _freeze(port_harness, request, now=now)
    assert record.claim is not None
    _as_previous_release(storage.conn)
    assert not storage.rule_evidence_method_ready()
    with pytest.raises(sqlite3.IntegrityError):
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET terminal_outcome = 'abandoned'"
        )
    storage.conn.rollback()

    storage.migrate()
    storage.migrate()

    assert storage.rule_evidence_method_ready()
    assert port_harness.optimization_jobs(playbook.user_playbook_id) == [
        JobRow(record.job.job_id, "running", "evidence_frozen", None)
    ]
    identity = "a" * 64
    storage.session_compaction_checkpoint(
        job_id=record.job.job_id,
        identity=identity,
        fence=record.claim.fence,
        now=now,
        receipt={
            "identity": identity,
            "batches": 1,
            "complete": False,
            "compaction": {
                "points": [
                    {
                        "category": "goal",
                        "detail": "synthetic",
                        "source_id": "source-1",
                        "exact_quote": "synthetic quote",
                    }
                ]
            },
        },
    )
    assert storage.advance_playbook_optimization_stage(
        record.job.job_id,
        record.claim.fence,
        "failed",
        terminal_outcome="abandoned",
        now=now,
    )
    assert port_harness.optimization_jobs(playbook.user_playbook_id) == [
        JobRow(record.job.job_id, "failed", "failed", "abandoned")
    ]
    storage.conn.execute(
        "DELETE FROM playbook_optimization_jobs WHERE job_id = ?",
        (record.job.job_id,),
    )
    storage.conn.commit()
    assert port_harness.attempt_rows(record.job.job_id) == AttemptRows(0, 0, 0, 0), (
        "child rows cascade from the job"
    )


def test_the_outcome_arms_are_the_attempt_vocabulary() -> None:
    """The stage-advance arms, against the outcomes the runner can write."""
    from typing import get_args

    from reflexio.models.api_schema.domain import OptimizationTerminalOutcome
    from reflexio.server.services.offline_tuner.evidence.open_world_models import (
        OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES,
    )
    from reflexio.server.services.offline_tuner.open_world.models import (
        AttemptTerminalFailureReason,
    )
    from reflexio.server.services.offline_tuner.open_world_records import (
        OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES,
        OPEN_WORLD_FAILED_TERMINAL_OUTCOMES,
    )

    outcomes = set(get_args(OptimizationTerminalOutcome))
    assert {
        *get_args(AttemptTerminalFailureReason),
        "abandoned",
    } == OPEN_WORLD_FAILED_TERMINAL_OUTCOMES
    assert (
        OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES <= OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES
    )
    assert outcomes >= OPEN_WORLD_FAILED_TERMINAL_OUTCOMES
    assert outcomes >= OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES


#: ``user_playbook_publication_results`` as the release before provisional
#: publication created it: GEPA outcomes only, no lifecycle or retry columns.
_RESULTS_BEFORE_PROVISIONAL = """
CREATE TABLE user_playbook_publication_results (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('applied', 'incumbent_changed')),
    successor_user_playbook_id INTEGER,
    staging_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
)
"""


def _as_release_before_provisional(conn: sqlite3.Connection, job_id: int) -> None:
    """Put the publication schema back as the release before P7 left it.

    No lifecycle table, staging without the provisional columns, and a result
    table whose CHECK admits only GEPA outcomes, holding one GEPA row.
    """
    from reflexio.server.services.storage.sqlite_storage._base import (
        PROVISIONAL_STAGING_COLUMNS,
    )

    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DROP TABLE user_playbook_deployment_lifecycles")
    for column in PROVISIONAL_STAGING_COLUMNS:
        conn.execute(
            f"ALTER TABLE user_playbook_publication_staging DROP COLUMN {column}"
        )
    conn.execute("DROP TABLE user_playbook_publication_results")
    conn.execute(_RESULTS_BEFORE_PROVISIONAL)
    conn.execute(
        "INSERT INTO user_playbook_publication_results VALUES"
        " ('gepa', ?, 'applied', 99, ?, 1)",
        (job_id, "c" * 64),
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    return [tuple(row)[1:] for row in conn.execute(f"PRAGMA table_info('{table}')")]


def test_migrate_brings_the_release_before_provisional_publication_forward(
    port_harness: _SQLitePortHarness,
    tmp_path: Path,
) -> None:
    """The lifecycle table appears, staging and results gain their columns,
    the result CHECK admits ``provisional``, a GEPA row survives, and the
    upgraded schema is the fresh one: an attempt then publishes."""
    from reflexio.test_support.open_world_attempt_contract import _accepted
    from reflexio.test_support.open_world_publication_contract import publish

    storage = port_harness.storage
    conn = storage.conn
    attempt = _accepted(port_harness)
    _as_release_before_provisional(conn, attempt.job_id)
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'user_playbook_deployment_lifecycles'"
        ).fetchone()
        is None
    )

    storage.migrate()
    storage.migrate()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        fresh = SQLiteStorage(org_id="contract_test", db_path=str(tmp_path / "f.db"))
    try:
        for table in (
            "user_playbook_publication_staging",
            "user_playbook_publication_results",
            "user_playbook_deployment_lifecycles",
        ):
            assert _columns(conn, table) == _columns(fresh.conn, table), table
        assert (
            conn.execute(
                "SELECT sql FROM sqlite_master"
                " WHERE name = 'user_playbook_publication_results'"
            ).fetchone()[0]
            == fresh.conn.execute(
                "SELECT sql FROM sqlite_master"
                " WHERE name = 'user_playbook_publication_results'"
            ).fetchone()[0]
        )
    finally:
        fresh.conn.close()
    assert [
        tuple(row)
        for row in conn.execute(
            "SELECT optimizer_kind, outcome, successor_user_playbook_id"
            " FROM user_playbook_publication_results WHERE job_id = ?",
            (attempt.job_id,),
        )
    ] == [("gepa", "applied", 99)]

    result = publish(attempt)

    assert result.outcome == "provisional"
    assert port_harness.publication_rows(attempt.job_id).lifecycles == 1
