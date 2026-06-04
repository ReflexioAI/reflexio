"""Tests for applied-learnings metering at search endpoints.

Verifies that ``_meter_applied_learnings`` emits exactly one ``learning_applied``
usage event when a production-agent caller surfaces >= 1 result, and nothing for
dashboard callers or empty result sets.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from reflexio.models.api_schema.ui.entities import AgentPlaybookView, ProfileView
from reflexio.server.api import create_app
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder


def _make_profile_view(user_id: str = "u1") -> ProfileView:
    return ProfileView(
        profile_id="p1",
        user_id=user_id,
        content="content",
        last_modified_timestamp=0,
        generated_from_request_id="r1",
    )


def _make_agent_playbook_view() -> AgentPlaybookView:
    return AgentPlaybookView(agent_version="v1", content="content")


def _client(caller_type: str) -> TestClient:
    app = create_app(get_org_id=lambda: "test-org", get_caller_type=lambda: caller_type)
    return TestClient(app, raise_server_exceptions=False)


@contextmanager
def _patch_unified_search(
    profiles: list,
    agent_playbooks: list,
    user_playbooks: list,
):
    """Patch get_reflexio so the unified_search method returns a canned service response.

    The mock response carries properly-sized lists so the view converters succeed.
    get_config() returns None so platform_llm_from_config(None) returns True without
    iterating MagicMock values.
    """
    mock_reflexio = MagicMock()
    # Set up the service-level response (not the view response).
    # The endpoint calls unified_search then wraps each item with to_*_view().
    # We need the response's list attributes to hold properly-typed objects.
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.msg = "OK"
    mock_response.reformulated_query = None
    mock_response.agent_trace = None
    mock_response.rehydrated_text = None
    mock_response.profiles = profiles
    mock_response.agent_playbooks = agent_playbooks
    mock_response.user_playbooks = user_playbooks
    mock_reflexio.unified_search.return_value = mock_response
    # Prevent platform_llm_from_config from iterating MagicMock.values()
    mock_reflexio.request_context.configurator.get_config.return_value = None

    with patch("reflexio.server.api.get_reflexio", return_value=mock_reflexio):
        yield


def _capture() -> list[UsageEvent]:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    return events


def test_production_agent_search_meters_surfaced_count() -> None:
    """A production-agent call with results emits one learning_applied event."""
    events = _capture()
    profiles = [_make_profile_view("u1"), _make_profile_view("u2")]
    agent_playbooks = [_make_agent_playbook_view()]
    user_playbooks: list = []
    try:
        with _patch_unified_search(profiles, agent_playbooks, user_playbooks):
            resp = _client("production_agent").post(
                "/api/search", json={"query": "x", "user_id": "u1"}
            )
        assert resp.status_code == 200
    finally:
        configure_usage_event_recorder(None)

    applied = [e for e in events if e.event_name == "learning_applied"]
    assert len(applied) == 1
    assert applied[0].count_value == 3  # 2 profiles + 1 agent_playbook + 0 user_playbooks
    assert applied[0].caller_type == "production_agent"


def test_dashboard_search_meters_nothing() -> None:
    """A dashboard (JWT) caller never emits learning_applied regardless of results."""
    events = _capture()
    try:
        with _patch_unified_search([_make_profile_view()], [], []):
            _client("dashboard").post("/api/search", json={"query": "x", "user_id": "u1"})
    finally:
        configure_usage_event_recorder(None)

    assert [e for e in events if e.event_name == "learning_applied"] == []


def test_empty_result_meters_nothing() -> None:
    """A production-agent call that surfaces zero results emits nothing."""
    events = _capture()
    try:
        with _patch_unified_search([], [], []):
            _client("production_agent").post(
                "/api/search", json={"query": "x", "user_id": "u1"}
            )
    finally:
        configure_usage_event_recorder(None)

    assert [e for e in events if e.event_name == "learning_applied"] == []
