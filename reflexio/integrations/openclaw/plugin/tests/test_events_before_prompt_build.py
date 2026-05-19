"""Tests for openclaw_smart.events.before_prompt_build."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openclaw_smart import state
from openclaw_smart.events import before_prompt_build as bpb


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))
    return sessions


def test_handle_buffers_prompt(isolate_state_dir):
    with patch(
        "openclaw_smart.events.before_prompt_build.context_inject"
    ) as ci, patch(
        "openclaw_smart.events.before_prompt_build.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        ci.emit_context.return_value = False
        bpb.handle(
            {
                "sessionKey": "s1",
                "prompt": "implement OAuth",
                "agentId": "agent-x",
                "workspaceDir": "/tmp/x",
            }
        )

    path = isolate_state_dir / "s1.jsonl"
    assert path.exists()
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    assert record["role"] == "User"
    assert record["content"] == "implement OAuth"
    assert record["user_id"] == "proj-x"


def test_handle_calls_emit_context_with_top_k():
    with patch(
        "openclaw_smart.events.before_prompt_build.context_inject"
    ) as ci, patch(
        "openclaw_smart.events.before_prompt_build.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        bpb.handle(
            {
                "sessionKey": "s1",
                "prompt": "implement OAuth",
                "agentId": "agent-x",
                "workspaceDir": "/tmp/x",
            }
        )
        ci.emit_context.assert_called_once()
        kwargs = ci.emit_context.call_args[1]
        assert kwargs["session_id"] == "s1"
        assert kwargs["project_id"] == "proj-x"
        assert kwargs["query"] == "implement OAuth"
        assert kwargs["top_k"] == 3


def test_handle_skips_when_no_session_id(isolate_state_dir):
    with patch("openclaw_smart.events.before_prompt_build.context_inject") as ci:
        bpb.handle({"prompt": "x", "agentId": "a"})
        ci.emit_context.assert_not_called()
    assert not isolate_state_dir.exists() or not any(isolate_state_dir.iterdir())


def test_handle_skips_when_no_prompt(isolate_state_dir):
    with patch("openclaw_smart.events.before_prompt_build.context_inject") as ci:
        bpb.handle({"sessionKey": "s1", "prompt": "", "agentId": "a"})
        ci.emit_context.assert_not_called()


def test_handle_swallows_emit_exceptions(isolate_state_dir):
    with patch(
        "openclaw_smart.events.before_prompt_build.context_inject"
    ) as ci, patch(
        "openclaw_smart.events.before_prompt_build.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        ci.emit_context.side_effect = ConnectionError("reflexio down")
        # Must not raise
        bpb.handle(
            {
                "sessionKey": "s1",
                "prompt": "implement OAuth",
                "agentId": "a",
                "workspaceDir": "/tmp/x",
            }
        )
    # Prompt still buffered
    path = isolate_state_dir / "s1.jsonl"
    assert path.exists()


def test_handle_falls_back_to_session_id():
    with patch(
        "openclaw_smart.events.before_prompt_build.context_inject"
    ) as ci, patch(
        "openclaw_smart.events.before_prompt_build.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        bpb.handle({"sessionId": "s2", "prompt": "hello", "agentId": "a"})
        ci.emit_context.assert_called_once()
        assert ci.emit_context.call_args[1]["session_id"] == "s2"
