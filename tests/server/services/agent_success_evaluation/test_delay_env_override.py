"""Tests that GROUP_EVALUATION_DELAY_SECONDS is configurable via the environment.

The delay is resolved at import time, so these tests monkeypatch the environment
and `importlib.reload` the module to re-evaluate the module-level globals.
"""

import importlib

import pytest

from reflexio.server.services.agent_success_evaluation import delayed_group_evaluator


@pytest.fixture(autouse=True)
def restore_module():
    """Reload the module after each test so import-time globals match the real env."""
    yield
    importlib.reload(delayed_group_evaluator)


def test_default_when_env_unset(monkeypatch):
    """Unset env var -> default 600s."""
    monkeypatch.delenv("GROUP_EVALUATION_DELAY_SECONDS", raising=False)
    importlib.reload(delayed_group_evaluator)
    assert delayed_group_evaluator.GROUP_EVALUATION_DELAY_SECONDS == 600


def test_env_override_applies(monkeypatch):
    """Explicit env var overrides the default and flows to the effective delay."""
    monkeypatch.delenv("IS_TEST_ENV", raising=False)
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", "120")
    importlib.reload(delayed_group_evaluator)
    assert delayed_group_evaluator.GROUP_EVALUATION_DELAY_SECONDS == 120
    assert delayed_group_evaluator._EFFECTIVE_DELAY_SECONDS == 120


def test_test_env_wins_over_override(monkeypatch):
    """IS_TEST_ENV=true forces the 30s effective delay regardless of the override."""
    monkeypatch.setenv("IS_TEST_ENV", "true")
    monkeypatch.setenv("GROUP_EVALUATION_DELAY_SECONDS", "120")
    importlib.reload(delayed_group_evaluator)
    assert delayed_group_evaluator.GROUP_EVALUATION_DELAY_SECONDS == 120
    assert delayed_group_evaluator._EFFECTIVE_DELAY_SECONDS == 30
