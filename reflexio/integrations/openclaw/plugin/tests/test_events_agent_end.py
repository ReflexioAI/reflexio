"""Tests for openclaw_smart.events.agent_end."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from openclaw_smart import state
from openclaw_smart.events import agent_end


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


def test_extract_assistant_text_plain_string():
    text = agent_end._extract_assistant_text(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert text == "hello"


def test_extract_assistant_text_block_form():
    text = agent_end._extract_assistant_text(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            },
        ]
    )
    assert "first" in text and "second" in text


def test_extract_assistant_text_concatenates_trailing_assistant_turns():
    text = agent_end._extract_assistant_text(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "step 1"},
            {"role": "assistant", "content": "step 2"},
        ]
    )
    assert "step 1" in text
    assert "step 2" in text


def test_extract_assistant_text_stops_at_user_boundary():
    text = agent_end._extract_assistant_text(
        [
            {"role": "assistant", "content": "old turn"},
            {"role": "user", "content": "new q"},
            {"role": "assistant", "content": "new answer"},
        ]
    )
    assert "new answer" in text
    assert "old turn" not in text


def test_extract_assistant_text_handles_nested_message_shape():
    """Accept Claude-Code-style ``{message: {role, content}}`` shape too."""
    text = agent_end._extract_assistant_text(
        [{"message": {"role": "assistant", "content": "hi"}}]
    )
    assert text == "hi"


def test_extract_assistant_text_empty_for_no_messages():
    assert agent_end._extract_assistant_text([]) == ""
    assert agent_end._extract_assistant_text(None) == ""


def test_handle_no_session_id_returns_silently(isolate_state_dir):
    with patch("openclaw_smart.events.agent_end.publish") as pub:
        agent_end.handle({})
        pub.publish_unpublished.assert_not_called()


def test_handle_appends_assistant_record(isolate_state_dir):
    with patch("openclaw_smart.events.agent_end.publish") as pub, patch(
        "openclaw_smart.events.agent_end.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        agent_end.handle(
            {
                "sessionKey": "s1",
                "agentId": "a",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "Done."},
                ],
            }
        )

    records = _read_records(isolate_state_dir)
    assert records[-1]["role"] == "Assistant"
    assert records[-1]["content"] == "Done."
    assert records[-1]["user_id"] == "proj-x"


def test_handle_publishes_unpublished_slice(isolate_state_dir):
    with patch("openclaw_smart.events.agent_end.publish") as pub, patch(
        "openclaw_smart.events.agent_end.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        agent_end.handle(
            {
                "sessionKey": "s1",
                "messages": [{"role": "assistant", "content": "Done."}],
            }
        )
        pub.publish_unpublished.assert_called_once()
        kwargs = pub.publish_unpublished.call_args[1]
        assert kwargs["session_id"] == "s1"
        assert kwargs["project_id"] == "proj-x"
        assert kwargs["force_extraction"] is False
        assert kwargs["skip_aggregation"] is False


def test_handle_resolves_citations(isolate_state_dir, monkeypatch):
    # Seed the injected registry so cited tags resolve.
    state.append_injected(
        "s1",
        [
            {
                "id": "s1-ab12",
                "kind": "playbook",
                "title": "Use OAuth2",
                "real_id": "abc",
            }
        ],
    )

    with patch("openclaw_smart.events.agent_end.publish") as pub, patch(
        "openclaw_smart.events.agent_end.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        agent_end.handle(
            {
                "sessionKey": "s1",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "Implemented OAuth.\n"
                            "✨ 1 openclaw-smart learning applied [oc:s1-ab12]"
                        ),
                    }
                ],
            }
        )

    records = _read_records(isolate_state_dir)
    assert "cited_items" in records[-1]
    cited = records[-1]["cited_items"]
    assert cited[0]["id"] == "s1-ab12"
    assert cited[0]["title"] == "Use OAuth2"
    assert cited[0]["real_id"] == "abc"


def test_handle_skips_unknown_citations(isolate_state_dir):
    """Citations for ids not in the registry are dropped, not raised."""
    with patch("openclaw_smart.events.agent_end.publish") as pub, patch(
        "openclaw_smart.events.agent_end.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        agent_end.handle(
            {
                "sessionKey": "s1",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "Done.\n"
                            "✨ 1 openclaw-smart learning applied [oc:s9-9999]"
                        ),
                    }
                ],
            }
        )
    records = _read_records(isolate_state_dir)
    assert "cited_items" not in records[-1]


def test_handle_falls_back_to_session_id(isolate_state_dir):
    with patch("openclaw_smart.events.agent_end.publish") as pub, patch(
        "openclaw_smart.events.agent_end.ids.resolve_project_id_with_fallback",
        return_value="proj-x",
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        agent_end.handle(
            {
                "sessionId": "sess-y",
                "messages": [{"role": "assistant", "content": "ok"}],
            }
        )
        kwargs = pub.publish_unpublished.call_args[1]
        assert kwargs["session_id"] == "sess-y"
