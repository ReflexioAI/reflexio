"""Config loading, revision stamps and invalidation agree across request races."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from reflexio.lib._config import ConfigMixin
from reflexio.models.config_schema import Config
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.cache import reflexio_cache as cache
from reflexio.server.services.configurator.config_storage import ConfigStorage
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.fixture
def policy(monkeypatch):
    """Exercise the real configurator and config accessors without services."""
    durable: dict[str, Any] = {"value": False, "version": 1}
    storage = Mock(spec=ConfigStorage)
    storage.get_version.side_effect = lambda: ("db", durable["version"])
    storage.load_config.side_effect = lambda: Config(
        storage_config=None, enable_document_expansion=durable["value"]
    )
    monkeypatch.setattr(
        DefaultConfigurator, "_select_config_storage", lambda *_args: storage
    )
    builds = []

    def construct(org_id, storage_base_dir=None):
        configurator = DefaultConfigurator(org_id, base_dir=storage_base_dir)
        instance = object.__new__(ConfigMixin)
        instance.org_id = org_id
        instance.request_context = cast(
            RequestContext, SimpleNamespace(configurator=configurator)
        )
        builds.append(instance)
        return instance

    monkeypatch.setattr(cache, "Reflexio", construct)
    cache.clear_reflexio_cache()
    yield SimpleNamespace(
        durable=durable, storage=storage, builds=builds, construct=construct
    )
    cache.clear_reflexio_cache()


class _ObservedLock:
    """Signal actual lock contention rather than depending on thread timing."""

    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            assert self.lock.acquire(timeout=5)
        return self

    def __exit__(self, *args):
        self.lock.release()


def _pause_first_construction(policy, monkeypatch):
    loaded = threading.Event()
    release = threading.Event()
    observed = _ObservedLock()
    index = hash(("race-org", None)) % cache._REFLEXIO_CONSTRUCTION_LOCK_STRIPES
    stripes: list[Any] = list(cache._reflexio_construction_locks)
    stripes[index] = observed
    monkeypatch.setattr(cache, "_reflexio_construction_locks", tuple(stripes))

    def construct(**kwargs):
        instance = policy.construct(**kwargs)
        if len(policy.builds) == 1:
            loaded.set()
            assert release.wait(timeout=5)
        return instance

    monkeypatch.setattr(cache, "Reflexio", construct)
    return loaded, release, observed.waiting


@pytest.mark.parametrize("clear_all", [False, True], ids=["invalidate", "clear"])
def test_invalidation_fences_inflight_construction(policy, monkeypatch, clear_all):
    loaded, release, waiting = _pause_first_construction(policy, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cache.get_reflexio, "race-org")
        try:
            assert loaded.wait(timeout=5)
            policy.durable.update(value=True, version=2)
            invalidate = executor.submit(
                cache.clear_reflexio_cache
                if clear_all
                else lambda: cache.invalidate_reflexio_cache("race-org")
            )
            assert waiting.wait(timeout=5)
            assert not invalidate.done()
        finally:
            release.set()
        assert first.result(timeout=5).get_config().enable_document_expansion is False
        result = invalidate.result(timeout=5)
        if not clear_all:
            assert result is True

    assert cache.get_cache_stats()["current_size"] == 0
    assert cache.get_reflexio("race-org").get_config().enable_document_expansion is True
    assert len(policy.builds) == 2


def test_sibling_write_during_construction_retains_older_stamp(policy, monkeypatch):
    loaded, release, _waiting = _pause_first_construction(policy, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(cache.get_reflexio, "race-org")
        try:
            assert loaded.wait(timeout=5)
            policy.durable.update(value=True, version=2)
        finally:
            release.set()
        old = first.result(timeout=5)

    assert old.get_config().enable_document_expansion is False
    assert cache._reflexio_cache[("race-org", None)].cached_version == ("db", 1)
    fresh = cache.get_reflexio("race-org")
    assert fresh.get_config().enable_document_expansion is True
    assert len(policy.builds) == 2


def test_waiting_get_validates_just_constructed_entry(policy, monkeypatch):
    loaded, release, waiting = _pause_first_construction(policy, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cache.get_reflexio, "race-org")
        try:
            assert loaded.wait(timeout=5)
            second = executor.submit(cache.get_reflexio, "race-org")
            assert waiting.wait(timeout=5)
            policy.durable.update(value=True, version=2)
        finally:
            release.set()
        assert first.result(timeout=5).get_config().enable_document_expansion is False
        fresh = second.result(timeout=5)

    assert fresh.get_config().enable_document_expansion is True
    assert len(policy.builds) == 2


def test_stable_cold_start_probes_once_and_constructs_once(policy):
    first = cache.get_reflexio("stable-org")
    assert policy.storage.get_version.call_count == 1
    assert first.loaded_config_version() == ("db", 1)
    assert cache.get_reflexio("stable-org") is first
    assert policy.storage.get_version.call_count == 2
    assert len(policy.builds) == 1


def test_unsupported_version_remains_cacheable(policy):
    policy.storage.get_version.side_effect = None
    policy.storage.get_version.return_value = None
    first = cache.get_reflexio("unsupported-org")
    assert cache.get_reflexio("unsupported-org") is first
    policy.storage.get_version.assert_called_once()
    assert len(policy.builds) == 1


def test_failed_preload_probe_retries_and_preserves_later_eviction(policy):
    policy.storage.get_version.side_effect = [
        OSError("temporary version read failure"),
        ("db", 1),
        ("db", 2),
        ("db", 2),
    ]
    cache.get_reflexio("retry-org")
    assert cache.get_cache_stats()["current_size"] == 0
    second = cache.get_reflexio("retry-org")
    assert cache.get_cache_stats()["current_size"] == 1
    assert cache.get_reflexio("retry-org") is not second
    assert len(policy.builds) == 3


def test_missing_file_stamp_keeps_initialization_version_aware(
    policy, monkeypatch, tmp_path
):
    storage = LocalFileConfigStorage("new-org", base_dir=str(tmp_path))
    monkeypatch.setattr(
        DefaultConfigurator, "_select_config_storage", lambda *_args: storage
    )
    first = cache.get_reflexio("new-org")
    assert first.loaded_config_version() == ("file", None)
    second = cache.get_reflexio("new-org")
    assert second is not first
    assert second.loaded_config_version() == storage.get_version()
    assert cache.get_reflexio("new-org") is second
    assert len(policy.builds) == 2


def test_supplied_config_has_no_persistent_stamp_or_probe(policy):
    supplied = Config(storage_config=None, enable_document_expansion=True)
    configurator = DefaultConfigurator("supplied-org", config=supplied)
    assert configurator.config is supplied
    assert configurator.loaded_config_version is None
    policy.storage.get_version.assert_not_called()
    policy.storage.load_config.assert_not_called()
