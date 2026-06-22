"""Tests for openclaw_smart.hook dispatch."""

from __future__ import annotations

import io
import json

import pytest
from openclaw_smart import hook


@pytest.fixture(autouse=True)
def _reset_handlers():
    """Reload handler dispatch dict for each test (state is module-global)."""
    hook._HANDLERS = None
    yield
    hook._HANDLERS = None


def test_main_dispatches_to_handler(monkeypatch):
    called = {}

    def fake_handler(payload):
        called["payload"] = payload

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"sessionKey": "s1", "agentId": "a"})),
    )
    monkeypatch.setattr(hook, "_HANDLERS", {"session-start": fake_handler})

    rc = hook.main(["session-start"])
    assert rc == 0
    assert called["payload"] == {"sessionKey": "s1", "agentId": "a"}


def test_main_accepts_host_arg(monkeypatch):
    called = {}

    def fake_handler(payload):
        called["payload"] = payload

    monkeypatch.setattr("sys.stdin", io.StringIO('{"sessionKey":"s1"}'))
    monkeypatch.setattr(hook, "_HANDLERS", {"session-start": fake_handler})

    rc = hook.main(["openclaw", "session-start"])
    assert rc == 0
    assert called


def test_main_silent_when_recursion_guard_fires(monkeypatch, capsys):
    monkeypatch.setenv("OPENCLAW_SMART_INTERNAL", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    called = {"count": 0}

    def boom(payload):
        called["count"] += 1
        raise RuntimeError("should not be called")

    monkeypatch.setattr(hook, "_HANDLERS", {"session-start": boom})
    rc = hook.main(["session-start"])
    assert rc == 0
    assert called["count"] == 0
    assert capsys.readouterr().out == ""


def test_main_silent_on_handler_exception(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    def boom(payload):
        raise RuntimeError("x")

    monkeypatch.setattr(hook, "_HANDLERS", {"session-start": boom})
    rc = hook.main(["session-start"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_silent_on_unknown_event(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = hook.main(["no-such-event"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_silent_with_no_event(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = hook.main([])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_handles_malformed_stdin(monkeypatch):
    called = {}

    def fake_handler(payload):
        called["payload"] = payload

    monkeypatch.setattr("sys.stdin", io.StringIO("not json {{{"))
    monkeypatch.setattr(hook, "_HANDLERS", {"session-start": fake_handler})
    rc = hook.main(["session-start"])
    assert rc == 0
    # Malformed → empty dict, handler still called
    assert called["payload"] == {}


def test_load_handlers_includes_all_events():
    handlers = hook._load_handlers()
    expected = {
        "session-start",
        "before-prompt-build",
        "before-tool-call",
        "after-tool-call",
        "agent-end",
        "session-end",
    }
    assert set(handlers) == expected
