"""Uncertain revisions never certify cached policy as current."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from reflexio.server.cache import reflexio_cache as cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear_reflexio_cache()
    yield
    cache.clear_reflexio_cache()


def _instance(value, version):
    return SimpleNamespace(
        org_id="cache-regression-org",
        value=value,
        loaded_config_version=lambda: version,
        current_config_version=lambda: version,
    )


def test_unsupported_revision_loads_current_policy_instead_of_sharing_it(monkeypatch):
    durable = {"value": "before"}
    monkeypatch.setattr(
        cache, "Reflexio", lambda **_: _instance(durable["value"], None)
    )
    first = cast(SimpleNamespace, cache.get_reflexio("unsupported"))
    durable["value"] = "after"
    second = cast(SimpleNamespace, cache.get_reflexio("unsupported"))
    assert first.value == "before"
    assert second.value == "after"
    assert first is not second
    assert cache.get_cache_stats()["current_size"] == 0


def test_failed_hit_probe_reloads_policy_instead_of_certifying_old_revision(
    monkeypatch,
):
    first = _instance("before", ("db", 1))
    second = _instance("after", ("db", 2))
    builds = iter((first, second))
    monkeypatch.setattr(cache, "Reflexio", lambda **_: next(builds))
    assert cache.get_reflexio("probe-failure") is first

    def fail():
        raise OSError("synthetic revision probe unavailable")

    first.current_config_version = fail
    assert cache.get_reflexio("probe-failure") is second
    assert cache.get_reflexio("probe-failure") is second


def test_failed_refresh_propagates_instead_of_returning_old_policy(monkeypatch):
    first = _instance("before", ("db", 1))
    monkeypatch.setattr(cache, "Reflexio", lambda **_: first)
    assert cache.get_reflexio("failed-refresh") is first

    def fail_probe():
        raise OSError("synthetic probe unavailable")

    def fail_load(**_):
        raise OSError("synthetic config load unavailable")

    first.current_config_version = fail_probe
    monkeypatch.setattr(cache, "Reflexio", fail_load)
    with pytest.raises(OSError, match="config load unavailable"):
        cache.get_reflexio("failed-refresh")
    assert cache.get_cache_stats()["current_size"] == 0
    recovered = _instance("after", ("db", 2))
    monkeypatch.setattr(cache, "Reflexio", lambda **_: recovered)
    assert cache.get_reflexio("failed-refresh") is recovered
    assert cache.get_reflexio("failed-refresh") is recovered
