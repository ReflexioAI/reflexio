"""Tests for the Braintrust recurring sync scheduler (Plan C-cron)."""

from __future__ import annotations

import os
import threading
import time

import pytest

from reflexio.server.services.braintrust import _cron
from reflexio.server.services.braintrust._cron import (
    BraintrustSyncScheduler,
    _interval_seconds,
    _resolve_base_url,
    get_instance,
    make_client_factory_with_base_url,
    trigger_tick_now,
)


def test_interval_seconds_default_is_15min(monkeypatch) -> None:
    monkeypatch.delenv("IS_TEST_ENV", raising=False)
    assert _interval_seconds() == 15 * 60


def test_interval_seconds_test_env_shortens_to_5s(monkeypatch) -> None:
    monkeypatch.setenv("IS_TEST_ENV", "true")
    assert _interval_seconds() == 5


def test_resolve_base_url_default(monkeypatch) -> None:
    monkeypatch.delenv("BRAINTRUST_BASE_URL", raising=False)
    assert _resolve_base_url() == "https://api.braintrust.dev"


def test_resolve_base_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("BRAINTRUST_BASE_URL", "http://localhost:9000")
    assert _resolve_base_url() == "http://localhost:9000"


def test_client_factory_with_env_override_carries_base_url(monkeypatch) -> None:
    monkeypatch.setenv("BRAINTRUST_BASE_URL", "http://example.com")
    factory = make_client_factory_with_base_url()
    client = factory("sk-test")
    assert client.base_url == "http://example.com"


def test_trigger_tick_now_calls_per_org_sync() -> None:
    """The synchronous test helper runs the loop body once."""
    calls: list[str] = []
    scheduler = BraintrustSyncScheduler(
        interval_seconds=999,
        list_connected_orgs=lambda: ["org_a", "org_b"],
        run_sync_for_org=lambda org: calls.append(org),
    )
    trigger_tick_now(scheduler)
    assert calls == ["org_a", "org_b"]


def test_trigger_tick_handles_zero_orgs() -> None:
    calls: list[str] = []
    scheduler = BraintrustSyncScheduler(
        interval_seconds=999,
        list_connected_orgs=lambda: [],
        run_sync_for_org=lambda org: calls.append(org),
    )
    trigger_tick_now(scheduler)
    assert calls == []


def test_start_spawns_daemon_then_stop_signals_exit(monkeypatch) -> None:
    """Smoke: the loop runs, syncs at least once, and stops cleanly."""
    monkeypatch.setenv("IS_TEST_ENV", "true")  # 5s interval
    ran = threading.Event()
    calls: list[str] = []

    def list_orgs() -> list[str]:
        return ["org_only"]

    def run_sync(org: str) -> None:
        calls.append(org)
        ran.set()

    scheduler = BraintrustSyncScheduler(
        interval_seconds=1,  # override for fast test
        list_connected_orgs=list_orgs,
        run_sync_for_org=run_sync,
    )
    scheduler.start()
    assert ran.wait(timeout=3.0), "sync did not run within 3s"
    scheduler.stop()
    # Thread should exit shortly
    assert scheduler._thread is not None
    scheduler._thread.join(timeout=3.0)
    assert not scheduler._thread.is_alive()
    assert len(calls) >= 1


def test_start_is_idempotent(monkeypatch) -> None:
    """Calling start() twice doesn't spawn two daemons."""
    monkeypatch.setenv("IS_TEST_ENV", "true")
    scheduler = BraintrustSyncScheduler(interval_seconds=10)
    scheduler.start()
    first_thread = scheduler._thread
    scheduler.start()
    assert scheduler._thread is first_thread
    scheduler.stop()


def test_loop_swallows_per_org_exceptions(monkeypatch) -> None:
    """A throwing sync for one org doesn't prevent the next org's sync."""
    monkeypatch.setenv("IS_TEST_ENV", "true")
    calls: list[str] = []

    def maybe_throw(org: str) -> None:
        if org == "org_bad":
            raise RuntimeError("simulated failure")
        calls.append(org)

    scheduler = BraintrustSyncScheduler(
        interval_seconds=10,
        list_connected_orgs=lambda: ["org_bad", "org_good"],
        run_sync_for_org=maybe_throw,
    )
    trigger_tick_now(scheduler)
    assert "org_good" in calls


def test_singleton_returns_same_instance() -> None:
    _cron._reset_for_test()
    s1 = get_instance()
    s2 = get_instance()
    try:
        assert s1 is s2
    finally:
        _cron._reset_for_test()


def test_singleton_starts_on_first_access() -> None:
    _cron._reset_for_test()
    scheduler = get_instance()
    try:
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
    finally:
        _cron._reset_for_test()


def test_reset_for_test_clears_singleton(monkeypatch) -> None:
    monkeypatch.setenv("IS_TEST_ENV", "true")
    s1 = get_instance()
    _cron._reset_for_test()
    s2 = get_instance()
    try:
        assert s1 is not s2
    finally:
        _cron._reset_for_test()


def test_cleanup_envs() -> None:
    """Belt-and-suspenders: ensure module-level state is reset."""
    os.environ.pop("IS_TEST_ENV", None)
    os.environ.pop("BRAINTRUST_BASE_URL", None)
    _cron._reset_for_test()


# Bonus: verify the default factory in BraintrustConnectorService honors the env
def test_service_default_factory_honors_braintrust_base_url(monkeypatch) -> None:
    """When BRAINTRUST_BASE_URL is set, the service's default factory uses it."""
    monkeypatch.setenv("BRAINTRUST_BASE_URL", "http://staging-bt.example.com")
    from reflexio.server.services.braintrust.service import _default_client_factory

    client = _default_client_factory("sk")
    assert client.base_url == "http://staging-bt.example.com"

    # And without the env, falls back to api.braintrust.dev
    monkeypatch.delenv("BRAINTRUST_BASE_URL", raising=False)
    client2 = _default_client_factory("sk")
    assert client2.base_url == "https://api.braintrust.dev"


# Quiet the linter about importing time (used in scheduler module)
_ = time
_ = pytest
