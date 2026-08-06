"""Regression guard for the local xdist worker cap in ``tests/conftest.py``.

xdist dispatches ``pytest_xdist_auto_num_workers`` for **both** ``-n auto`` and
``-n logical``, and for neither when an explicit ``-n <N>`` is given::

    # xdist/plugin.py
    if config.option.numprocesses in ("auto", "logical"):
        auto_num_cpus = config.hook.pytest_xdist_auto_num_workers(config=config)

Capping ``logical`` as well as ``auto`` is deliberate, not an oversight: ``logical`` asks
for *more* workers than physical cores, so exempting it would reopen the all-cores local
run the cap exists to prevent. This file exists because "only cap ``auto``" is a natural
looking change to propose.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

_CONFTEST_PATH = Path(__file__).resolve().parent / "conftest.py"


def _conftest_plugin(config: pytest.Config):
    """The already-registered conftest module — not a fresh import (it has import-time side effects)."""
    for plugin in config.pluginmanager.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and Path(plugin_file).resolve() == _CONFTEST_PATH:
            return plugin
    pytest.fail(f"conftest at {_CONFTEST_PATH} is not a registered plugin")


def _fake_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(option=SimpleNamespace(numprocesses=mode))


@pytest.mark.parametrize("mode", ["auto", "logical"])
def test_local_run_is_capped(
    request: pytest.FixtureRequest, monkeypatch, mode: str
) -> None:
    monkeypatch.delenv("CI", raising=False)
    conftest = _conftest_plugin(request.config)

    workers = conftest.pytest_xdist_auto_num_workers(_fake_config(mode))

    assert workers is not None, (
        f"-n {mode} must be capped locally, not deferred to xdist"
    )
    assert workers <= conftest._LOCAL_MAX_XDIST_WORKERS


@pytest.mark.parametrize("mode", ["auto", "logical"])
def test_ci_run_defers_to_xdist(
    request: pytest.FixtureRequest, monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("CI", "true")
    conftest = _conftest_plugin(request.config)

    # None lets xdist's own impl win the firstresult chain, i.e. CI keeps every core.
    assert conftest.pytest_xdist_auto_num_workers(_fake_config(mode)) is None
