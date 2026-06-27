"""Integration: search reflexio, render the hits, assert the prependContext envelope.

Two flavours of coverage:

* **live-backend path** — when reflexio is reachable at ``REFLEXIO_URL``, run
  ``context_inject.emit_context`` end-to-end and assert the expected envelope
  shape. Skipped automatically when the backend isn't running.
* **stub path** — replace ``Adapter.search_all`` with a fixed return value so
  the rendering + envelope-emit logic is exercised regardless of environment.
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def _reflexio_url() -> str:
    return os.environ.get("REFLEXIO_URL", "http://localhost:8061/")


def _backend_alive(url: str) -> bool:
    import urllib.error
    import urllib.request

    health = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=1.5) as resp:  # nosec — local only
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.fixture
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    state_dir = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(state_dir))
    return state_dir


def _parse_emitted(captured: StringIO) -> dict | None:
    raw = captured.getvalue().strip()
    if not raw:
        return None
    return json.loads(raw)


def test_emit_context_returns_false_when_search_empty(
    monkeypatch: pytest.MonkeyPatch, isolated_state: Path
) -> None:
    """No hits → no envelope, no JSONL side-effect."""
    from openclaw_smart import context_inject

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    fake_adapter = MagicMock()
    fake_adapter.search_all.return_value = ([], [], [])

    emitted = context_inject.emit_context(
        session_id="sess-empty",
        project_id="proj-1",
        query="anything",
        top_k=3,
        adapter=fake_adapter,
    )
    assert emitted is False
    assert _parse_emitted(captured) is None
    assert not (isolated_state / "sess-empty.injected.jsonl").exists()


def test_emit_context_wraps_hits_in_prependContext_envelope(
    monkeypatch: pytest.MonkeyPatch, isolated_state: Path
) -> None:
    """A stubbed user playbook is rendered as markdown under ``prependContext``."""
    from openclaw_smart import context_inject

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    user_playbook = {
        "content": "Always validate user input before persistence.",
        "trigger": "any external string lands in the persistence layer",
        "rationale": "blocks a whole class of injection bugs",
        "user_playbook_id": "pb-1",
    }
    fake_adapter = MagicMock()
    fake_adapter.search_all.return_value = ([user_playbook], [], [])

    emitted = context_inject.emit_context(
        session_id="sess-hit",
        project_id="proj-1",
        query="how do I validate user input",
        top_k=3,
        adapter=fake_adapter,
    )

    assert emitted is True
    payload = _parse_emitted(captured)
    assert payload is not None
    md = payload.get("prependContext")
    assert md, f"expected prependContext markdown, got {payload!r}"
    # Title text shows up inside the rendered markdown.
    assert "validate user input" in md.lower()
    # Citation registry was persisted for the agent_end hook to consume.
    assert (isolated_state / "sess-hit.injected.jsonl").exists()


def test_emit_context_against_live_backend(
    monkeypatch: pytest.MonkeyPatch, isolated_state: Path
) -> None:
    """When a real backend is running, the full pipeline executes without error.

    No fixture data is seeded — the backend may legitimately return zero hits;
    the assertion is that the call completes, not that there's data to inject.
    """
    if not _backend_alive(_reflexio_url()):
        pytest.skip(f"reflexio backend not reachable at {_reflexio_url()}")

    from openclaw_smart import context_inject

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    # No assertion on emitted bool — backend state is environment-dependent.
    context_inject.emit_context(
        session_id="sess-live",
        project_id="proj-live",
        query="ping",
        top_k=3,
    )
    # If markdown was emitted it should at least be valid JSON with the right key.
    raw = captured.getvalue().strip()
    if raw:
        payload = json.loads(raw)
        assert "prependContext" in payload
