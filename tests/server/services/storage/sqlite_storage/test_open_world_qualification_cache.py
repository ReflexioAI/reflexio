"""SQLite contract for the immutable open-world qualification cache."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
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
