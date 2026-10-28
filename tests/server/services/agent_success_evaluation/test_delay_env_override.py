"""Tests that GROUP_EVALUATION_DELAY_SECONDS is configurable via the environment.

The delay is resolved at import time, so these tests monkeypatch the environment
and `importlib.reload` the module to re-evaluate the module-level globals.
"""

import importlib

import pytest

from reflexio.server.services.agent_success_evaluation import scheduler


@pytest.fixture(autouse=True)
def restore_module(monkeypatch):
    """Reload the module after each test so import-time globals match the real env."""
    yield
    monkeypatch.undo()
    importlib.reload(scheduler)


def test_default_when_env_unset(monkeypatch):
    """Unset env var -> default 600s."""
    monkeypatch.delenv("GROUP_EVALUATION_DELAY_SECONDS", raising=False)
    importlib.reload(scheduler)
    assert scheduler.GROUP_EVALUATION_DELAY_SECONDS == 600


def test_env_override_applies(monkeypatch):
    """Explicit env var overrides the default and flows to the effective delay."""
    monkeypatch.delenv("IS_TEST_ENV", raising=False)
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", "120")
    importlib.reload(scheduler)
    assert scheduler.GROUP_EVALUATION_DELAY_SECONDS == 120
    assert scheduler._EFFECTIVE_DELAY_SECONDS == 120


def test_test_env_cannot_override_canonical_policy(monkeypatch):
    """IS_TEST_ENV has no authority over the canonical inactivity policy."""
    monkeypatch.setenv("IS_TEST_ENV", "true")
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", "120")
    importlib.reload(scheduler)
    assert scheduler.GROUP_EVALUATION_DELAY_SECONDS == 120
    assert scheduler._EFFECTIVE_DELAY_SECONDS == 120


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_delay_uses_default(monkeypatch, value):
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", value)
    importlib.reload(scheduler)
    assert scheduler._EFFECTIVE_DELAY_SECONDS == 600


@pytest.mark.parametrize("value", ["private-invalid-delay", "-1", str(10**100)])
def test_invalid_delay_refused_without_echo(monkeypatch, value):
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", value)
    with pytest.raises(ValueError, match="GROUP_EVALUATION_DELAY_SECONDS") as error:
        importlib.reload(scheduler)
    assert value not in str(error.value)
