"""The mutation stats hooks refuse an engine whose internals they do not know."""

import importlib
from types import SimpleNamespace

import pytest

from tests import conftest

# ``mutmut.state`` is shadowed by a function of the same name on the package.
engine_state = importlib.import_module("mutmut.state")


def test_supported_engine_state_is_returned() -> None:
    assert conftest._mutmut_state() is engine_state.state()


@pytest.mark.parametrize("installed", ["4.0.0", "2.5.1"])
def test_other_major_fails_with_an_upgrade_hint(
    monkeypatch: pytest.MonkeyPatch, installed: str
) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _name: installed)
    with pytest.raises(RuntimeError, match=f"Unsupported mutmut {installed}"):
        conftest._mutmut_state()


def test_changed_state_layout_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_state, "state", lambda: SimpleNamespace(duration_by_test={})
    )
    with pytest.raises(RuntimeError, match="missing") as error:
        conftest._mutmut_state()
    assert "tests_by_mangled_function_name" in str(error.value)


def test_stats_hook_uses_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "4.0.0")
    report = SimpleNamespace(when="setup", nodeid="tests/x.py::t", duration=0.1)
    with pytest.raises(RuntimeError, match="Unsupported mutmut 4.0.0"):
        conftest.pytest_runtest_logreport(report)  # type: ignore[arg-type]
