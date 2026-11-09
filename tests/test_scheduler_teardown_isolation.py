"""Library schedulers stop before test finalizers touch the storage they poll.

A library engine starts a durable-learning scheduler whose discovery reads the
context's SQLite connection on its own thread. Fixtures commonly close that
connection in their finalizer without the storage lock; closing it during a
concurrent query segfaults CPython's sqlite3 and takes the xdist worker down
(``_extraction_stream.py`` ``now``). The root conftest therefore stops library
schedulers before any finalizer runs, and fails a test that leaves a durable
scheduler thread alive.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning import local
from reflexio.server.services.durable_learning.scheduler import (
    DurableLearningScheduler,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from tests import conftest


def _sqlite_context(base_dir: Path, org_id: str) -> RequestContext:
    configurator = DefaultConfigurator(org_id=org_id, base_dir=str(base_dir))
    configurator.set_config(
        Config(storage_config=StorageConfigSQLite(db_path=str(base_dir / "lib.db")))
    )
    return RequestContext(
        org_id=org_id, storage_base_dir=str(base_dir), configurator=configurator
    )


@pytest.fixture
def library_context(tmp_path: Path) -> Iterator[RequestContext]:
    """Start the library scheduler, then close its storage the way fixtures do."""
    context = _sqlite_context(tmp_path, f"teardown-{tmp_path.name}")
    local.ensure_local_extraction(context)
    scheduler = local._schedulers[context.storage_base_dir]
    assert scheduler.is_running()
    yield context
    # Runs as a finalizer: the scheduler polling this connection must already
    # be joined, or this unlocked close races its discovery query.
    assert not scheduler.is_running(), "scheduler still polling at finalizer time"
    assert isinstance(context.storage, SQLiteStorage)
    context.storage.conn.close()


def test_library_scheduler_is_stopped_before_test_finalizers(
    library_context: RequestContext,
) -> None:
    scheduler = local._schedulers[library_context.storage_base_dir]
    assert scheduler.is_running()


def test_guard_reports_a_live_durable_scheduler_thread_once() -> None:
    scheduler = DurableLearningScheduler(
        request_context_factory=lambda org_id: RequestContext(org_id=org_id),
        org_ids_provider=lambda: [],
    )
    scheduler.start()
    try:
        assert scheduler._thread is not None
        leaked = conftest._new_leaked_durable_scheduler_threads()
        assert scheduler._thread in leaked
        assert scheduler._thread not in conftest._new_leaked_durable_scheduler_threads()
    finally:
        scheduler.stop(timeout_seconds=5)
    assert not scheduler.is_running()
