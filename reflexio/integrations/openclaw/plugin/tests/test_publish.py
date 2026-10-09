"""Tests for openclaw_smart.publish."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from copy import deepcopy
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
        self.publications: list[list[dict[str, Any]]] = []
        self.during_publish: Callable[[], None] | None = None

    def publish(
        self,
        *,
        session_id: str,
        project_id: str,
        interactions: Sequence[dict[str, Any]],
        force_extraction: bool = False,
        skip_aggregation: bool = False,
    ) -> bool:
        del session_id, project_id, force_extraction, skip_aggregation
        self.publications.append(deepcopy(list(interactions)))
        if self.during_publish is not None:
            callback, self.during_publish = self.during_publish, None
            callback()
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
    assert records[-1] == {"published_up_to": 1, "published_retrieved_learnings": 0}


def _publish(adapter: _Adapter):
    return publish.publish_unpublished(
        session_id="s1",
        project_id="proj",
        force_extraction=False,
        skip_aggregation=False,
        adapter=adapter,
    )


def _refs(learning_id: str):
    return {
        "retrieved_learning_refs": [{"kind": "profile", "learning_id": learning_id}]
    }


def test_abandoned_context_does_not_reach_the_published_response():
    for record in [
        {"role": "User", "content": "abandoned"},
        _refs("old"),
        {"role": "Assistant_tool", "tool_name": "old_tool"},
        {"role": "User", "content": "current"},
        _refs("new"),
        {"role": "Assistant_tool", "tool_name": "new_tool"},
        {"role": "Assistant", "content": "response"},
    ]:
        state.append("s1", record)
    adapter = _Adapter()
    assert _publish(adapter) == ("ok", 3)
    response = adapter.publications[0][-1]
    assert response["retrieved_learnings"] == [
        {"kind": "profile", "learning_id": "new"}
    ]
    assert [tool["tool_name"] for tool in response["tools_used"]] == ["new_tool"]
    assert state.read_all("s1")[-1]["published_retrieved_learnings"] == 1


def test_context_appended_during_publish_survives_and_counts_across_batches(
    monkeypatch,
):
    monkeypatch.setattr(state, "_RETRIEVED_LEARNINGS_SESSION_CAP", 2)
    for record in [
        {"role": "User", "content": "first"},
        _refs("first"),
        {"role": "Assistant", "content": "first answer"},
    ]:
        state.append("s1", record)
    adapter = _Adapter()

    def racing_hook():
        state.append("s1", {"role": "User", "content": "second"})
        state.append("s1", _refs("second"))

    adapter.during_publish = racing_hook
    assert _publish(adapter) == ("ok", 2)
    assert state.read_all("s1")[-1] == {
        "published_up_to": 3,
        "published_retrieved_learnings": 1,
    }
    state.append("s1", {"role": "Assistant", "content": "second answer"})
    assert _publish(adapter) == ("ok", 2)
    assert [turn["content"] for turn in adapter.publications[1]] == [
        "second",
        "second answer",
    ]
    assert adapter.publications[1][-1]["retrieved_learnings"] == [
        {"kind": "profile", "learning_id": "second"},
    ]
    assert state.read_all("s1")[-1]["published_retrieved_learnings"] == 2
    state.append("s1", _refs("over cap"))
    state.append("s1", {"role": "Assistant", "content": "third answer"})
    assert _publish(adapter) == ("ok", 1)
    assert "retrieved_learnings" not in adapter.publications[2][-1]
