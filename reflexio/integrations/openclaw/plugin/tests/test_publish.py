"""Tests for openclaw_smart.publish."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from openclaw_smart import publish, state
from openclaw_smart.reflexio_adapter import Adapter


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))
    return sessions


class _Adapter(Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def publish(
        self,
        *,
        session_id: str,
        project_id: str,
        interactions: Sequence[dict[str, Any]],
        force_extraction: bool = False,
        skip_aggregation: bool = False,
    ) -> bool:
        del session_id, project_id, interactions, force_extraction, skip_aggregation
        self.calls += 1
        return True


def test_publish_unpublished_serializes_with_lock_and_stamps_watermark(
    isolate_state_dir,
):
    state.append("s1", {"role": "User", "content": "hi"})
    adapter = _Adapter()

    status, count = publish.publish_unpublished(
        session_id="s1",
        project_id="proj",
        force_extraction=False,
        skip_aggregation=False,
        adapter=adapter,
    )

    assert (status, count) == ("ok", 1)
    assert adapter.calls == 1
    assert (isolate_state_dir / "s1.publish.lock").exists()
    records = [
        json.loads(line)
        for line in (isolate_state_dir / "s1.jsonl").read_text().splitlines()
    ]
    assert records[-1] == {"published_up_to": 1}
