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
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from reflexio.models.api_schema.domain import Interaction, Request, RetrievedLearning
from reflexio.models.api_schema.service_schemas import DeleteUserInteractionRequest
from reflexio.server.services.offline_tuner.exposure_receipts import (
    receipt_project_id,
)
from reflexio.server.services.search_exposure import UserPlaybookExposureEvent
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._open_world import (
    _INSERT_EXPOSURE_SQL,
    _exposure_row,
)
from reflexio.test_support.open_world_exposure_contract import ExposureRow
from reflexio.test_support.open_world_port_contract import (
    OpenWorldPortContract,
    ProgressRow,
)

#: A synthetic midday instant, far from either day boundary.
_PINNED_NOW = int(datetime(2026, 6, 15, 12, 0, tzinfo=UTC).timestamp())


class _SQLitePortHarness:
    """A fresh SQLite store whose sweep clock reads ``self.clock_value``."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage
        self.clock_value: float = _PINNED_NOW
        storage._sweep_progress_clock = lambda: self.clock_value  # noqa: SLF001

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
