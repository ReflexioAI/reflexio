"""End-to-end: drive all 6 hooks with synthetic payloads, assert buffer state.

Exercises the hook dispatcher → event-handler → state buffer pipeline in
order, the same sequence openClaw would trigger during a real session. The
reflexio adapter is patched to a no-op so the test never depends on a live
backend; failures inside ``publish`` would otherwise be tolerated silently
(per ``reflexio_adapter.Adapter.publish``) but explicit patching keeps the
test deterministic across CI runs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Force state.py to use a tmp directory so the test never touches ~/.openclaw-smart/."""
    state_dir = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(state_dir))
    return state_dir


@pytest.fixture
def stub_adapter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace every imported ``Adapter`` symbol with a MagicMock that returns success.

    Each consumer module does ``from openclaw_smart.reflexio_adapter import Adapter``
    at import time, so patching the source module alone leaves stale bindings in
    consumers. We patch the consumer-side aliases as well so the test is
    isolated regardless of import order.
    """
    from openclaw_smart import (  # noqa: F401 — force-import the consumers
        context_inject,
        publish,
        reflexio_adapter,
    )
    from openclaw_smart.events import session_start

    fake = MagicMock()
    fake.publish.return_value = True
    fake.apply_extraction_defaults.return_value = True
    fake.apply_optimizer_defaults.return_value = True
    fake.fetch_stall_state.return_value = None
    fake.search_all.return_value = ([], [], [])

    factory = lambda *a, **kw: fake  # noqa: E731 — single-line lambda is clearer here
    monkeypatch.setattr(reflexio_adapter, "Adapter", factory)
    monkeypatch.setattr(publish, "Adapter", factory)
    monkeypatch.setattr(context_inject, "Adapter", factory)
    monkeypatch.setattr(session_start, "Adapter", factory)
    return fake


def _drive(monkeypatch: pytest.MonkeyPatch, event: str, payload: dict) -> int:
    from openclaw_smart import hook

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return hook.main(["openclaw", event])


def test_full_session_loop_records_all_roles(
    monkeypatch: pytest.MonkeyPatch,
    isolated_state: Path,
    stub_adapter: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Walk the 6 hooks; assert User, Assistant_tool, Assistant records land."""
    from openclaw_smart import state

    session = "e2e-1"
    project = "test-project"

    # session-start: pushes defaults + emits a stall banner (or empty stdout).
    assert _drive(monkeypatch, "session-start", {"sessionKey": session, "agentId": project}) == 0
    capsys.readouterr()  # drain

    # before-prompt-build: appends a "User" record.
    assert (
        _drive(
            monkeypatch,
            "before-prompt-build",
            {
                "sessionKey": session,
                "prompt": "implement OAuth flow",
                "agentId": project,
            },
        )
        == 0
    )
    capsys.readouterr()

    # before-tool-call: observe-only stub; no record, no stdout.
    assert (
        _drive(
            monkeypatch,
            "before-tool-call",
            {
                "sessionKey": session,
                "toolName": "Edit",
                "params": {"file_path": "x.py"},
                "agentId": project,
            },
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == ""

    # after-tool-call: appends an "Assistant_tool" record (camelCase translated).
    assert (
        _drive(
            monkeypatch,
            "after-tool-call",
            {
                "sessionKey": session,
                "toolName": "Edit",
                "params": {"file_path": "x.py", "new_string": "..."},
                "result": "wrote 10 lines",
                "agentId": project,
            },
        )
        == 0
    )
    capsys.readouterr()

    # agent-end: extracts the assistant text and appends an "Assistant" record.
    assert (
        _drive(
            monkeypatch,
            "agent-end",
            {
                "sessionKey": session,
                "agentId": project,
                "messages": [
                    {"role": "user", "content": "implement OAuth flow"},
                    {"role": "assistant", "content": "Done."},
                ],
            },
        )
        == 0
    )
    capsys.readouterr()

    # session-end: force-publish; should not crash.
    assert _drive(monkeypatch, "session-end", {"sessionKey": session, "agentId": project}) == 0
    capsys.readouterr()

    records = state.read_all(session)
    roles = [r.get("role") for r in records if "role" in r]
    assert "User" in roles
    assert "Assistant_tool" in roles
    assert "Assistant" in roles

    # session_end publishes; we asserted publish was called via the stub.
    assert stub_adapter.publish.called

    # The agent_end / session_end path writes a published_up_to watermark
    # once a publish succeeds. The stub returns True, so we should see it.
    assert any("published_up_to" in r for r in records), (
        "expected published_up_to watermark after session_end"
    )


def test_unknown_event_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    isolated_state: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unrecognized hook name short-circuits with empty stdout, never crashes."""
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    from openclaw_smart import hook

    rc = hook.main(["openclaw", "this-does-not-exist"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
