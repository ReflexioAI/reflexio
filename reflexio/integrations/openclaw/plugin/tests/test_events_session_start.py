"""Tests for openclaw_smart.events.session_start."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openclaw_smart.events import session_start


@pytest.fixture
def fake_adapter():
    adapter = MagicMock()
    adapter.fetch_stall_state.return_value = None
    adapter.apply_extraction_defaults.return_value = True
    adapter.apply_optimizer_defaults.return_value = True
    return adapter


def test_handle_no_session_id_returns_silently(fake_adapter, capsys):
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({})
    assert capsys.readouterr().out == ""
    fake_adapter.apply_extraction_defaults.assert_not_called()


def test_handle_applies_extraction_defaults(fake_adapter, capsys):
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1", "workspaceDir": "/tmp"})
    fake_adapter.apply_extraction_defaults.assert_called_once_with(
        window_size=5, stride_size=3
    )


def test_handle_applies_optimizer_defaults_by_default(fake_adapter, monkeypatch):
    monkeypatch.delenv("OPENCLAW_SMART_ENABLE_OPTIMIZER", raising=False)
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1"})
    fake_adapter.apply_optimizer_defaults.assert_called_once()
    kwargs = fake_adapter.apply_optimizer_defaults.call_args[1]
    assert "openclaw-smart-optimizer-assistant" in kwargs["script_path"]


def test_handle_skips_optimizer_when_disabled(fake_adapter, monkeypatch):
    monkeypatch.setenv("OPENCLAW_SMART_ENABLE_OPTIMIZER", "0")
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1"})
    fake_adapter.apply_optimizer_defaults.assert_not_called()


def test_handle_emits_banner_when_stall_active(fake_adapter, capsys):
    fake_adapter.fetch_stall_state.return_value = SimpleNamespace(
        stalled=True,
        notified_in_cc=False,
        reason="auth_error",
        reset_estimate=None,
    )
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1"})
    out = capsys.readouterr().out.strip()
    assert out, "expected stdout envelope when stall is active"
    parsed = json.loads(out)
    assert "prependContext" in parsed
    assert "openclaw-smart" in parsed["prependContext"]
    fake_adapter.mark_stall_notified.assert_called_once()


def test_handle_skips_banner_when_already_notified(fake_adapter, capsys):
    fake_adapter.fetch_stall_state.return_value = SimpleNamespace(
        stalled=True,
        notified_in_cc=True,
        reason="auth_error",
        reset_estimate=None,
    )
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1"})
    assert capsys.readouterr().out == ""
    fake_adapter.mark_stall_notified.assert_not_called()


def test_handle_skips_banner_when_not_stalled(fake_adapter, capsys):
    fake_adapter.fetch_stall_state.return_value = SimpleNamespace(
        stalled=False,
        notified_in_cc=False,
        reason=None,
        reset_estimate=None,
    )
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionKey": "s1"})
    assert capsys.readouterr().out == ""


def test_handle_swallows_stall_state_exceptions(fake_adapter, capsys):
    fake_adapter.fetch_stall_state.side_effect = RuntimeError("boom")
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        # Must not raise
        session_start.handle({"sessionKey": "s1"})
    assert capsys.readouterr().out == ""
    # Defaults still applied
    fake_adapter.apply_extraction_defaults.assert_called_once()


def test_handle_falls_back_to_session_id_key(fake_adapter):
    """If only ``sessionId`` is present (openClaw sometimes), it is used."""
    with patch.object(session_start, "_adapter", return_value=fake_adapter):
        session_start.handle({"sessionId": "sess-x"})
    fake_adapter.apply_extraction_defaults.assert_called_once()
