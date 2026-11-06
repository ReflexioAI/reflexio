"""``OpenWorldStoragePort`` contract on SQLite.

Binds the shared contract (``reflexio.test_support.open_world_port_contract``)
to a fresh SQLite database per test, with the store's clock pinned so every
case is exact. Also holds the SQLite-only case the contract cannot express on
a backend whose clock is not injectable: the one-clock-reading rule across
00:00 UTC.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
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
        self._write(
            "INSERT INTO user_playbook_exposure_events (exposure_event_id,"
            " request_id, session_id, user_id, playbook_owner_user_id,"
            " user_playbook_id, exposed_at, ingested_at)"
            " VALUES (?, ?, ?, ?, 'owner', ?, ?, ?)",
            (
                prefix + uuid4().hex,
                request_id,
                session_id,
                user_id,
                user_playbook_id,
                exposed_at,
                ingested_at,
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
