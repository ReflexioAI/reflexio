"""SQLite contract for the immutable open-world qualification cache."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reflexio.models.api_schema import service_schemas as schemas
from reflexio.server.services.storage.error import (
    OpenWorldQualificationConflictError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration

_COMPONENT_DIGEST = "a" * 64
_SUITE_DIGEST = "b" * 64
_RESULT_DIGEST = "c" * 64

# Bounded concurrency for the independent-connection writer tests below: enough
# threads to exercise real SQLite lock contention without saturating the host.
_CONCURRENT_WRITERS = 6


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "reflexio.db")


@pytest.fixture
def storage(db_path: str) -> Generator[SQLiteStorage]:
    store = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        yield store
    finally:
        store.conn.close()


def _class_counts() -> tuple[schemas.OpenWorldQualificationClassCount, ...]:
    return tuple(
        schemas.OpenWorldQualificationClassCount(
            qualification_class=qualification_class,
            required=3,
            passed_required=3,
        )
        for qualification_class in schemas.OPEN_WORLD_QUALIFICATION_CLASSES
    )


def _record(**overrides: object) -> schemas.OpenWorldQualificationRecord:
    payload: dict[str, object] = {
        "component_identity_digest": _COMPONENT_DIGEST,
        "suite_digest": _SUITE_DIGEST,
        "result_digest": _RESULT_DIGEST,
        "class_counts": _class_counts(),
        "passed": True,
        "observation_digests": ("0" * 64, "1" * 64),
        "created_at": 1_700_000_000,
    }
    payload.update(overrides)
    return schemas.OpenWorldQualificationRecord(**payload)  # type: ignore[arg-type]


def test_persist_then_load_round_trips(storage: SQLiteStorage) -> None:
    persisted = storage.persist_open_world_qualification_record(_record())

    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )

    assert persisted == _record()
    assert loaded == persisted


def test_load_returns_none_for_unknown_key(storage: SQLiteStorage) -> None:
    storage.persist_open_world_qualification_record(_record())

    assert (
        storage.load_open_world_qualification_record(
            component_identity_digest=_COMPONENT_DIGEST,
            suite_digest="d" * 64,
        )
        is None
    )
    assert (
        storage.load_open_world_qualification_record(
            component_identity_digest="d" * 64,
            suite_digest=_SUITE_DIGEST,
        )
        is None
    )


def test_failed_result_is_persisted_and_loadable(storage: SQLiteStorage) -> None:
    failing = _class_counts()[:-1] + (
        schemas.OpenWorldQualificationClassCount(
            qualification_class="prompt_injection_resistance",
            required=3,
            passed_required=2,
        ),
    )
    record = _record(class_counts=failing, passed=False)

    storage.persist_open_world_qualification_record(record)

    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )
    assert loaded is not None
    assert loaded.passed is False
    assert loaded.class_counts[-1].passed_required == 2


def test_exact_replay_is_idempotent(storage: SQLiteStorage) -> None:
    first = storage.persist_open_world_qualification_record(_record())
    second = storage.persist_open_world_qualification_record(_record())

    assert second == first
    rows = storage.conn.execute(
        "SELECT COUNT(*) FROM offline_tuner_open_world_qualifications"
    ).fetchone()[0]
    assert rows == 1


def test_first_insert_controls_created_at(storage: SQLiteStorage) -> None:
    first = storage.persist_open_world_qualification_record(
        _record(created_at=1_700_000_000)
    )
    replay = storage.persist_open_world_qualification_record(
        _record(created_at=1_900_000_000)
    )

    assert first.created_at == 1_700_000_000
    assert replay.created_at == 1_700_000_000
    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )
    assert loaded is not None
    assert loaded.created_at == 1_700_000_000


@pytest.mark.parametrize(
    "conflicting",
    [
        {"result_digest": "d" * 64},
        {"passed": False},
        {"observation_digests": ("0" * 64,)},
        {
            "class_counts": _class_counts()[:-1]
            + (
                schemas.OpenWorldQualificationClassCount(
                    qualification_class="prompt_injection_resistance",
                    required=4,
                    passed_required=4,
                ),
            )
        },
    ],
)
def test_semantic_conflict_is_rejected(
    storage: SQLiteStorage,
    conflicting: dict[str, object],
) -> None:
    storage.persist_open_world_qualification_record(_record())

    with pytest.raises(OpenWorldQualificationConflictError):
        storage.persist_open_world_qualification_record(_record(**conflicting))

    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )
    assert loaded == _record()


def test_distinct_keys_are_independent(storage: SQLiteStorage) -> None:
    storage.persist_open_world_qualification_record(_record())
    other = _record(suite_digest="d" * 64, result_digest="e" * 64)

    storage.persist_open_world_qualification_record(other)

    assert (
        storage.load_open_world_qualification_record(
            component_identity_digest=_COMPONENT_DIGEST,
            suite_digest="d" * 64,
        )
        == other
    )


def test_record_survives_close_and_reopen(db_path: str) -> None:
    store = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        store.persist_open_world_qualification_record(_record())
    finally:
        store.conn.close()

    reopened = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        loaded = reopened.load_open_world_qualification_record(
            component_identity_digest=_COMPONENT_DIGEST,
            suite_digest=_SUITE_DIGEST,
        )
    finally:
        reopened.conn.close()

    assert loaded == _record()


def test_raw_sql_update_is_rejected(storage: SQLiteStorage) -> None:
    storage.persist_open_world_qualification_record(_record())

    with pytest.raises(sqlite3.IntegrityError):
        storage.conn.execute(
            "UPDATE offline_tuner_open_world_qualifications SET passed = 0"
        )
    storage.conn.rollback()

    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )
    assert loaded == _record()


def test_raw_sql_delete_is_rejected(storage: SQLiteStorage) -> None:
    storage.persist_open_world_qualification_record(_record())

    with pytest.raises(sqlite3.IntegrityError):
        storage.conn.execute("DELETE FROM offline_tuner_open_world_qualifications")
    storage.conn.rollback()

    loaded = storage.load_open_world_qualification_record(
        component_identity_digest=_COMPONENT_DIGEST,
        suite_digest=_SUITE_DIGEST,
    )
    assert loaded == _record()


def test_immutability_triggers_survive_reopen(db_path: str) -> None:
    store = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        store.persist_open_world_qualification_record(_record())
    finally:
        store.conn.close()

    reopened = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            reopened.conn.execute("DELETE FROM offline_tuner_open_world_qualifications")
        reopened.conn.rollback()
    finally:
        reopened.conn.close()


# ---------------------------------------------------------------------------
# Concurrent writers on INDEPENDENT SQLite connections (separate SQLiteStorage
# instances over the same db file, not one instance shared across threads) --
# real cross-connection lock contention, not just the in-process RLock.
# Synchronization is via ThreadPoolExecutor.submit()/.result(); no sleeps.
# ---------------------------------------------------------------------------


def test_concurrent_identical_writes_converge_on_independent_connections(
    db_path: str,
) -> None:
    stores = [
        SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
        for _ in range(_CONCURRENT_WRITERS)
    ]
    try:
        with ThreadPoolExecutor(max_workers=_CONCURRENT_WRITERS) as pool:
            futures = [
                pool.submit(store.persist_open_world_qualification_record, _record())
                for store in stores
            ]
            results = [future.result() for future in futures]
    finally:
        for store in stores:
            store.conn.close()

    assert all(result == _record() for result in results)

    verify_store = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        rows = verify_store.conn.execute(
            "SELECT COUNT(*) FROM offline_tuner_open_world_qualifications"
        ).fetchone()[0]
        loaded = verify_store.load_open_world_qualification_record(
            component_identity_digest=_COMPONENT_DIGEST,
            suite_digest=_SUITE_DIGEST,
        )
    finally:
        verify_store.conn.close()
    assert rows == 1
    assert loaded == _record()


def test_concurrent_conflicting_writes_yield_one_winner_and_one_intact_row(
    db_path: str,
) -> None:
    variants = [
        _record(result_digest=f"{index:064x}") for index in range(_CONCURRENT_WRITERS)
    ]
    stores = [
        SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
        for _ in variants
    ]
    try:
        with ThreadPoolExecutor(max_workers=_CONCURRENT_WRITERS) as pool:
            futures = [
                pool.submit(store.persist_open_world_qualification_record, variant)
                for store, variant in zip(stores, variants, strict=True)
            ]
            outcomes: list[tuple[str, object]] = []
            for future in futures:
                try:
                    outcomes.append(("winner", future.result()))
                except OpenWorldQualificationConflictError as exc:
                    outcomes.append(("conflict", exc))
    finally:
        for store in stores:
            store.conn.close()

    winners = [record for kind, record in outcomes if kind == "winner"]
    conflicts = [exc for kind, exc in outcomes if kind == "conflict"]

    assert len(winners) == 1
    assert len(conflicts) == len(variants) - 1
    assert all(
        isinstance(exc, OpenWorldQualificationConflictError) for exc in conflicts
    )
    assert winners[0] in variants

    verify_store = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        rows = verify_store.conn.execute(
            "SELECT COUNT(*) FROM offline_tuner_open_world_qualifications"
        ).fetchone()[0]
        loaded = verify_store.load_open_world_qualification_record(
            component_identity_digest=_COMPONENT_DIGEST,
            suite_digest=_SUITE_DIGEST,
        )
    finally:
        verify_store.conn.close()
    assert rows == 1
    assert loaded == winners[0]


# ---------------------------------------------------------------------------
# Upgrade from a pre-Task-9 schema: a DB that predates this table must
# self-upgrade on normal storage initialization, in place, with existing data
# intact. Dropping the table (SQLite auto-drops its triggers with it) on a
# fully-initialized modern schema reproduces exactly what an old DB on disk
# looks like, without hand-reconstructing legacy DDL.
# ---------------------------------------------------------------------------


def _qualification_schema_objects(
    conn: sqlite3.Connection,
) -> tuple[set[str], set[str]]:
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type = 'table' AND name = 'offline_tuner_open_world_qualifications'"
        )
    }
    triggers = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type = 'trigger'"
            " AND name LIKE 'offline_tuner_open_world_qualifications_%'"
        )
    }
    return tables, triggers


def test_upgrade_from_pre_task_9_schema_installs_table_and_triggers(
    db_path: str,
) -> None:
    baseline = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        # optimizer_kind="gepa" (not a legacy kind) so re-running migrate() on
        # the second, upgraded instance does not itself retire this sentinel
        # job as part of unrelated legacy-optimizer cleanup -- that would
        # confound this test's "unrelated data survives" assertion below.
        sentinel_job = baseline.create_playbook_optimization_job(
            schemas.PlaybookOptimizationJob(
                optimizer_kind="gepa", target_kind="agent_playbook", target_id=1
            )
        )
    finally:
        baseline.conn.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DROP TABLE offline_tuner_open_world_qualifications")
        raw.commit()
        assert _qualification_schema_objects(raw) == (set(), set())
    finally:
        raw.close()

    upgraded = SQLiteStorage(org_id="open-world-qualification", db_path=db_path)
    try:
        tables, triggers = _qualification_schema_objects(upgraded.conn)
        assert tables == {"offline_tuner_open_world_qualifications"}
        assert triggers == {
            "offline_tuner_open_world_qualifications_no_update",
            "offline_tuner_open_world_qualifications_no_delete",
        }

        # Pre-existing, unrelated data survives the upgrade untouched.
        assert (
            upgraded.get_playbook_optimization_job(sentinel_job.job_id) == sentinel_job
        )

        # The reinstalled table is immediately usable and immutable.
        persisted = upgraded.persist_open_world_qualification_record(_record())
        assert persisted == _record()
        with pytest.raises(sqlite3.IntegrityError):
            upgraded.conn.execute(
                "UPDATE offline_tuner_open_world_qualifications SET passed = 0"
            )
        upgraded.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            upgraded.conn.execute("DELETE FROM offline_tuner_open_world_qualifications")
        upgraded.conn.rollback()
    finally:
        upgraded.conn.close()


def test_qualification_cache_surface_is_narrow() -> None:
    for method_name in (
        "persist_open_world_qualification_record",
        "load_open_world_qualification_record",
    ):
        assert hasattr(BaseStorage, method_name)
        assert hasattr(SQLiteStorage, method_name)
    for method_name in (
        "delete_open_world_qualification_record",
        "update_open_world_qualification_record",
        "list_open_world_qualification_records",
    ):
        assert not hasattr(BaseStorage, method_name)
        assert not hasattr(SQLiteStorage, method_name)
