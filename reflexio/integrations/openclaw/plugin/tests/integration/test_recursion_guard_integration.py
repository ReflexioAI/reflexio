"""Integration: hook silently no-ops when OPENCLAW_SMART_INTERNAL=1.

This guard prevents an infinite feedback loop where the reflexio backend's
own openclaw subprocess (the openclaw LLM provider) re-enters openclaw-smart
and republishes the extractor's system prompt back into reflexio.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def isolated_state(monkeypatch, tmp_path: Path) -> Path:
    """Force state.py to write under tmp_path so the test never touches ~/.openclaw-smart/."""
    state_dir = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(state_dir))
    return state_dir


def test_hook_short_circuits_when_internal_env_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    isolated_state: Path,
) -> None:
    """With OPENCLAW_SMART_INTERNAL=1, every handler is bypassed and stdout is empty."""
    monkeypatch.setenv("OPENCLAW_SMART_INTERNAL", "1")

    from openclaw_smart import hook

    payload = {
        "event": {"prompt": "anything"},
        "ctx": {"sessionKey": "internal-1", "agentId": "a"},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    rc = hook.main(["openclaw", "before-prompt-build"])
    assert rc == 0

    # No session JSONL should have been created — the handler bailed before
    # ever touching state.append().
    assert not isolated_state.exists() or not list(isolated_state.glob("*.jsonl"))

    out = capsys.readouterr().out
    # The TS shim treats empty stdout as "no return value" per the design.
    assert out.strip() == ""


def test_hook_short_circuits_when_workspace_inside_reflexio(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    isolated_state: Path,
    tmp_path: Path,
) -> None:
    """A workspaceDir inside the reflexio install dir is also treated as internal.

    The reflexio backend cwd's into its own checkout when it spawns the openclaw
    subprocess for the LLM provider. internal_call.is_internal_invocation honours
    this so the recursion guard fires even if the env var got stripped.
    """
    import reflexio

    reflexio_pkg_dir = Path(reflexio.__file__).resolve().parent
    payload = {
        "event": {"prompt": "anything"},
        "ctx": {
            "sessionKey": "internal-2",
            "agentId": "a",
            "workspaceDir": str(reflexio_pkg_dir / "sub"),
        },
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    from openclaw_smart import hook

    rc = hook.main(["openclaw", "before-prompt-build"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() == ""
    assert not isolated_state.exists() or not list(isolated_state.glob("*.jsonl"))
