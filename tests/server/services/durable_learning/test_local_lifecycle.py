"""A library-local scheduler outlives its handles for no longer than one call."""

import gc
import threading

import pytest

from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning import local

_THREAD_NAME = "reflexio-durable-learning-scheduler"


def _scheduler_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == _THREAD_NAME)


@pytest.fixture
def isolated_registry():
    """Swap the module registry for an empty one and stop whatever it collects."""
    with local._lock:
        saved_schedulers = local._schedulers
        saved_contexts = local._contexts
        saved_server = local._server_scheduler
        local._schedulers = {}
        local._contexts = type(saved_contexts)()
        local._server_scheduler = None
    yield
    with local._lock:
        created = list(local._schedulers.values())
        local._schedulers = saved_schedulers
        local._contexts = saved_contexts
        local._server_scheduler = saved_server
    for scheduler in created:
        scheduler.stop()


def _context(base_dir, org_id: str) -> RequestContext:
    base_dir.mkdir(parents=True, exist_ok=True)
    configurator = DefaultConfigurator(org_id=org_id, base_dir=str(base_dir))
    configurator.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(base_dir / "lifecycle.db")),
            window_size=1,
            stride_size=1,
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Preferences"
            ),
            user_playbook_extractor_config=None,
        )
    )
    return RequestContext(
        org_id=org_id, storage_base_dir=str(base_dir), configurator=configurator
    )


def test_dropped_handles_do_not_accumulate_scheduler_threads(
    tmp_path, isolated_registry
):
    baseline = _scheduler_threads()
    for index in range(6):
        context = _context(tmp_path / f"dir{index}", f"lifecycle{index}")
        local.ensure_local_extraction(context)
        del context
        gc.collect()
    # Only the most recently registered directory may still hold a scheduler:
    # it is retired by the next caller, never by the one that created it.
    assert len(local._schedulers) == 1
    assert _scheduler_threads() <= baseline + 1


def test_scheduler_survives_while_its_context_is_reachable(tmp_path, isolated_registry):
    kept = _context(tmp_path / "kept", "kept-org")
    local.ensure_local_extraction(kept)
    kept_scheduler = local._schedulers[kept.storage_base_dir]

    transient = _context(tmp_path / "other", "other-org")
    local.ensure_local_extraction(transient)
    del transient
    gc.collect()

    local.ensure_local_extraction(_context(tmp_path / "third", "third-org"))
    gc.collect()

    assert local._schedulers.get(kept.storage_base_dir) is kept_scheduler
    assert kept_scheduler.is_running()
    assert str(tmp_path / "other") not in local._schedulers


def test_two_handles_on_one_key_both_keep_the_scheduler_alive(
    tmp_path, isolated_registry
):
    """A second handle on the same org+directory must not unregister the first.

    `ReflexioBase` builds an independent `RequestContext` per handle, so two
    handles sharing an org and a directory are two distinct objects. A registry
    keyed by `(org_id, storage_base_dir)` collapses them: the second
    registration evicts the first, and collecting the second empties the key
    while the first handle is still alive and using its scheduler. The sweep
    then retires that scheduler and extraction stops silently under a live
    caller.

    Measured on the keyed version: with both handles alive the registry held
    only the second. This is the regression test for that.
    """
    shared = tmp_path / "shared"
    first = _context(shared, "same-org")
    local.ensure_local_extraction(first)
    scheduler = local._schedulers[first.storage_base_dir]

    second = _context(shared, "same-org")
    assert second is not first, "precondition: the handles are distinct objects"
    local.ensure_local_extraction(second)
    # The first must still be represented -- the whole point of the set.
    assert first in local._live[first.storage_base_dir]

    del second
    gc.collect()

    # Any later caller triggers the sweep; the first handle is still alive, so
    # its scheduler must survive it.
    local.ensure_local_extraction(_context(tmp_path / "elsewhere", "other-org"))
    gc.collect()

    assert first in local._live[first.storage_base_dir]
    assert local._schedulers.get(first.storage_base_dir) is scheduler
    assert scheduler.is_running()
