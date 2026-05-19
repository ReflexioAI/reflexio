"""Tests for openclaw_smart.events.before_tool_call (observe-only stub)."""

from __future__ import annotations

from openclaw_smart.events import before_tool_call as btc


def test_handle_is_silent(capsys):
    btc.handle(
        {
            "sessionKey": "s1",
            "tool_name": "Edit",
            "tool_input": {"file_path": "x.py", "new_string": "..."},
            "agentId": "a",
        }
    )
    assert capsys.readouterr().out == ""


def test_handle_accepts_empty_payload(capsys):
    btc.handle({})
    assert capsys.readouterr().out == ""


def test_handle_accepts_camelcase_payload(capsys):
    """openClaw delivers camelCase fields; handler should ignore them silently."""
    btc.handle(
        {
            "sessionKey": "s1",
            "toolName": "Bash",
            "params": {"command": "ls"},
            "agentId": "a",
        }
    )
    assert capsys.readouterr().out == ""
