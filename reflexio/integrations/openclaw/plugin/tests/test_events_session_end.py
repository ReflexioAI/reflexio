"""Tests for openclaw_smart.events.session_end."""

from __future__ import annotations

from unittest.mock import patch

from openclaw_smart.events import session_end


def test_handle_no_session_id_returns_silently():
    with patch("openclaw_smart.events.session_end.publish") as pub:
        session_end.handle({})
        pub.publish_unpublished.assert_not_called()


def test_handle_force_extracts():
    with (
        patch("openclaw_smart.events.session_end.publish") as pub,
        patch(
            "openclaw_smart.events.session_end.ids.resolve_project_id_with_fallback",
            return_value="proj-x",
        ),
    ):
        session_end.handle({"sessionKey": "s1", "agentId": "a"})
        pub.publish_unpublished.assert_called_once()
        kwargs = pub.publish_unpublished.call_args[1]
        assert kwargs["session_id"] == "s1"
        assert kwargs["project_id"] == "proj-x"
        assert kwargs["force_extraction"] is True
        assert kwargs["skip_aggregation"] is False


def test_handle_falls_back_to_session_id_key():
    with (
        patch("openclaw_smart.events.session_end.publish") as pub,
        patch(
            "openclaw_smart.events.session_end.ids.resolve_project_id_with_fallback",
            return_value="proj-y",
        ),
    ):
        session_end.handle({"sessionId": "s2"})
        pub.publish_unpublished.assert_called_once()
        kwargs = pub.publish_unpublished.call_args[1]
        assert kwargs["session_id"] == "s2"
        assert kwargs["project_id"] == "proj-y"
        assert kwargs["force_extraction"] is True
        assert kwargs["skip_aggregation"] is False
