"""Expected search degradations log at WARNING; real faults stay at ERROR.

A search request's own deadline and a tenant whose vector index is not yet
qualified are handled outcomes -- search degrades to FTS or to its deadline
answer -- so they must not reach error tracking as ERROR events. Anything else
still must.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from reflexio.server import search_runtime
from reflexio.server.services import unified_search_service as service
from reflexio.server.services.storage.error import (
    ReadinessUnavailableError,
    StorageError,
)
from reflexio.server.services.storage.sqlite_storage._base import SQLiteStorageBase

_LOGGER = "reflexio.server.services.unified_search_service"


def _storage_raising(error: BaseException) -> MagicMock:
    storage = MagicMock()
    storage._query_embedding_cache_identity = object()
    storage.embedding_model_name = "test-embedding-model"
    storage.embedding_dimensions = 0
    storage._get_embedding.side_effect = error
    return storage


def _run_phase_a(storage: MagicMock) -> tuple[object, list[float] | None, bool]:
    return service._run_phase_a(
        "a query",
        storage,
        llm_client=MagicMock(),
        prompt_manager=MagicMock(),
        enable_reformulation=False,
    )


@pytest.mark.parametrize(
    "error",
    [
        search_runtime.SearchDeadlineError(),
        ReadinessUnavailableError("embedding_readiness_unavailable"),
    ],
    ids=["search-deadline", "vector-readiness"],
)
def test_expected_embedding_unavailability_is_a_warning(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER)

    _, embedding, embedding_failed = _run_phase_a(_storage_raising(error))

    assert embedding is None
    assert embedding_failed is True
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "event=search_query_embedding_unavailable" in records[0].getMessage()
    assert type(error).__name__ in records[0].getMessage()


@pytest.mark.parametrize(
    "error",
    [RuntimeError("provider exploded"), StorageError("embedding backend down")],
    ids=["unexpected", "storage-error"],
)
def test_unexpected_embedding_failure_stays_an_error(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=_LOGGER)

    _, embedding, embedding_failed = _run_phase_a(_storage_raising(error))

    assert embedding is None
    assert embedding_failed is True
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert [r.levelno for r in records] == [logging.ERROR]
    assert "Embedding generation failed" in records[0].getMessage()


class _FakeSQLiteStorage:
    def __init__(self) -> None:
        self.rolled_back = 0

    def _rollback_owned_transaction(self) -> None:
        self.rolled_back += 1


def test_sqlite_storage_does_not_wrap_the_search_deadline() -> None:
    """A wrapped deadline would miss the caller's deadline branch and log ERROR."""

    @SQLiteStorageBase.handle_exceptions
    def read(_self: _FakeSQLiteStorage) -> None:
        raise search_runtime.SearchDeadlineError()

    storage = _FakeSQLiteStorage()
    with pytest.raises(search_runtime.SearchDeadlineError):
        read(storage)
    assert storage.rolled_back == 1


def test_sqlite_storage_still_wraps_other_failures() -> None:
    @SQLiteStorageBase.handle_exceptions
    def read(_self: _FakeSQLiteStorage) -> None:
        raise RuntimeError("disk on fire")

    storage = _FakeSQLiteStorage()
    with pytest.raises(StorageError):
        read(storage)
    assert storage.rolled_back == 1
