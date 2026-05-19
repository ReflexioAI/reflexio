"""Tests for openclaw_smart.runtime."""

from __future__ import annotations

import pytest

from openclaw_smart import runtime


@pytest.fixture(autouse=True)
def _reset_host(monkeypatch):
    """Ensure each test starts with the module-level host unset."""
    monkeypatch.setattr(runtime, "_current_host", None)
    monkeypatch.delenv(runtime.HOST_ENV, raising=False)
    yield


def test_default_host_is_openclaw():
    assert runtime.host() == runtime.HOST_OPENCLAW
    assert runtime.is_openclaw() is True


def test_set_host_sets_module_state():
    runtime.set_host("openclaw")
    assert runtime.host() == "openclaw"
    assert runtime.is_openclaw()


def test_set_host_falls_back_for_unknown_value():
    runtime.set_host("definitely-not-real")
    assert runtime.host() == runtime.HOST_OPENCLAW


def test_set_host_writes_env_var():
    runtime.set_host("openclaw")
    import os

    assert os.environ.get(runtime.HOST_ENV) == "openclaw"


def test_agent_version_is_openclaw():
    assert runtime.agent_version() == "openclaw"
