"""Integration: publish buffered turns to a local SQLite-backed reflexio.

Requires a reflexio backend reachable at ``REFLEXIO_URL`` (default
``http://localhost:8081/``). Skipped automatically when the backend is
unreachable so the suite is portable across machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _reflexio_url() -> str:
    return os.environ.get("REFLEXIO_URL", "http://localhost:8081/")


def _backend_alive(url: str) -> bool:
    """True iff ``GET <url>health`` returns 200 within a short window."""
    import urllib.error
    import urllib.request

    health = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=1.5) as resp:  # nosec — local only
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.fixture
def require_backend() -> None:
    if not _backend_alive(_reflexio_url()):
        pytest.skip(f"reflexio backend not reachable at {_reflexio_url()}")


@pytest.fixture
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Force state.py to write under tmp_path so the test doesn't leak fixtures."""
    state_dir = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(state_dir))
    return state_dir


def test_publish_lands_in_reflexio_storage(
    require_backend: None,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state: Path,
    worker_id: str = "test",
) -> None:
    """Append two turns to the JSONL buffer → publish → assert success status + count."""
    from openclaw_smart import state
    from openclaw_smart.publish import publish_unpublished

    session_id = f"oc-int-{worker_id}"
    project_id = f"oc-test-proj-{worker_id}"

    state.append(session_id, {"role": "User", "content": "implement auth"})
    state.append(
        session_id,
        {"role": "Assistant", "content": "Wrote login/logout endpoints."},
    )

    status, count = publish_unpublished(
        session_id=session_id,
        project_id=project_id,
        force_extraction=False,
        skip_aggregation=False,
    )

    assert status == "ok", f"publish returned {status!r} — check backend logs"
    assert count == 2

    # The watermark is now stamped — a second publish with no new turns is a no-op.
    status2, count2 = publish_unpublished(
        session_id=session_id,
        project_id=project_id,
        force_extraction=False,
        skip_aggregation=False,
    )
    assert (status2, count2) == ("nothing", 0)


def test_publish_failed_when_backend_url_invalid(
    monkeypatch: pytest.MonkeyPatch, isolated_state: Path
) -> None:
    """Pointing the adapter at a dead port surfaces 'failed' without stamping."""
    from openclaw_smart import state
    from openclaw_smart.publish import publish_unpublished
    from openclaw_smart.reflexio_adapter import Adapter

    # Pick a port that's almost certainly closed.
    bogus = Adapter(url="http://127.0.0.1:9/")
    session_id = "oc-int-bogus"
    state.append(session_id, {"role": "User", "content": "hi"})

    status, count = publish_unpublished(
        session_id=session_id,
        project_id="proj-bogus",
        force_extraction=False,
        skip_aggregation=False,
        adapter=bogus,
    )
    assert status == "failed"
    assert count == 1
    # No watermark stamped — a retry would re-publish the same record.
    records = state.read_all(session_id)
    assert not any("published_up_to" in r for r in records)
