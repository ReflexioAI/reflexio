"""Tests for openclaw_smart.events.after_tool_call."""

from __future__ import annotations

import json

import pytest
from openclaw_smart.events import after_tool_call as atc


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))
    return sessions


def _read_records(sessions_dir, session_id="s1"):
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_handle_records_tool_invocation_camelcase(isolate_state_dir):
    """openClaw delivers camelCase fields; handler translates to snake_case."""
    atc.handle(
        {
            "sessionKey": "s1",
            "toolName": "Bash",
            "params": {"command": "ls"},
            "result": "file1.txt",
            "agentId": "a",
        }
    )
    records = _read_records(isolate_state_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec["role"] == "Assistant_tool"
    assert rec["tool_name"] == "Bash"
    assert rec["tool_input"] == {"command": "ls"}
    assert rec["tool_output"] == "file1.txt"
    assert rec["status"] == "success"


def test_handle_accepts_snake_case_fallback(isolate_state_dir):
    """Already-translated payloads work too."""
    atc.handle(
        {
            "sessionKey": "s2",
            "tool_name": "Edit",
            "tool_input": {"file_path": "x.py", "new_string": "hi"},
            "tool_response": "ok",
            "status": "success",
        }
    )
    records = _read_records(isolate_state_dir, "s2")
    assert records[0]["tool_name"] == "Edit"
    assert records[0]["tool_input"]["file_path"] == "x.py"


def test_handle_redacts_high_entropy_secrets(isolate_state_dir):
    secret_command = "API_KEY=AbCdEfGhIj1234567890QwErTyUiOp curl ..."
    atc.handle(
        {
            "sessionKey": "s3",
            "toolName": "Bash",
            "params": {"command": secret_command},
            "result": "",
        }
    )
    records = _read_records(isolate_state_dir, "s3")
    command = records[0]["tool_input"]["command"]
    assert "AbCdEfGhIj1234567890QwErTyUiOp" not in command
    assert "<redacted:" in command


def test_handle_does_not_redact_simple_assignments(isolate_state_dir):
    """LOG_LEVEL=INFO should not be redacted — too short and not entropy-y."""
    atc.handle(
        {
            "sessionKey": "s4",
            "toolName": "Bash",
            "params": {"command": "LOG_LEVEL=INFO some_cmd"},
            "result": "",
        }
    )
    records = _read_records(isolate_state_dir, "s4")
    assert "LOG_LEVEL=INFO" in records[0]["tool_input"]["command"]


def test_handle_classifies_error_status(isolate_state_dir):
    atc.handle(
        {
            "sessionKey": "s5",
            "toolName": "Bash",
            "params": {"command": "false"},
            "result": {"is_error": True, "error": "exit 1"},
        }
    )
    records = _read_records(isolate_state_dir, "s5")
    assert records[0]["status"] == "error"
    assert "exit 1" in records[0]["tool_output"]


def test_handle_flattens_bash_dict_output(isolate_state_dir):
    atc.handle(
        {
            "sessionKey": "s6",
            "toolName": "Bash",
            "params": {"command": "echo hi"},
            "result": {"stdout": "hi", "stderr": ""},
        }
    )
    records = _read_records(isolate_state_dir, "s6")
    assert records[0]["tool_output"] == "hi"


def test_handle_skips_when_no_session_id(isolate_state_dir):
    atc.handle({"toolName": "Bash", "params": {"command": "ls"}})
    assert _read_records(isolate_state_dir) == []


def test_handle_skips_when_no_tool_name(isolate_state_dir):
    atc.handle({"sessionKey": "s7", "params": {"command": "ls"}})
    assert _read_records(isolate_state_dir, "s7") == []


def test_handle_truncates_long_output(isolate_state_dir):
    long_output = "x" * 5000
    atc.handle(
        {
            "sessionKey": "s8",
            "toolName": "Bash",
            "params": {"command": "cat large_file"},
            "result": long_output,
        }
    )
    records = _read_records(isolate_state_dir, "s8")
    output = records[0]["tool_output"]
    assert "truncated" in output
    assert len(output) < 5000
