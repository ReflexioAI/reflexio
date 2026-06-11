"""Tests for applied-learnings metering at search endpoints.

Verifies that ``_meter_applied_learnings`` emits exactly one ``learning_applied``
usage event when a production-agent caller surfaces >= 1 result, and nothing for
dashboard callers or empty result sets.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.ui.entities import (
    AgentPlaybookView,
    ProfileView,
    UserPlaybookView,
)
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


def _make_user_playbook_view() -> UserPlaybookView:
    return UserPlaybookView(agent_version="v1", request_id="r1", content="content")


def _client(caller_type: str, get_billing_gate=None) -> TestClient:
    app = create_app(
        get_org_id=lambda: "test-org",
        get_caller_type=lambda: caller_type,
        get_billing_gate=get_billing_gate,
    )
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
    assert (
        applied[0].count_value == 3
    )  # 2 profiles + 1 agent_playbook + 0 user_playbooks
    assert applied[0].caller_type == "production_agent"


def test_dashboard_search_meters_nothing() -> None:
    """A dashboard (JWT) caller never emits learning_applied regardless of results."""
    events = _capture()
    try:
        with _patch_unified_search([_make_profile_view()], [], []):
            _client("dashboard").post(
                "/api/search", json={"query": "x", "user_id": "u1"}
            )
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


def test_get_billing_gate_override_seam() -> None:
    """create_app wires the get_billing_gate override for every billing line.

    The enterprise enforcement gate is injected through this seam: create_app must
    register an override for each line ("application" + "learnings_generated"),
    keyed by the lru_cached ``default_billing_gate`` sentinel the routes depend on,
    and the override must actually fire when an application route is served. This
    pins the core DI contract so it can't silently regress.
    """
    from reflexio.server.api import default_billing_gate

    fired: list[str] = []

    def gate_factory(line: str):
        def _gate() -> None:
            fired.append(line)

        return _gate

    app = create_app(
        get_org_id=lambda: "test-org",
        get_caller_type=lambda: "production_agent",
        get_billing_gate=gate_factory,
    )

    # Both billing lines are overridden, keyed by the exact sentinel the routes use.
    overrides = app.dependency_overrides
    assert default_billing_gate("application") in overrides
    assert default_billing_gate("learnings_generated") in overrides

    # The "application" override fires for real on an application route.
    client = TestClient(app, raise_server_exceptions=False)
    with _patch_unified_search([_make_profile_view()], [], []):
        resp = client.post("/api/search", json={"query": "x", "user_id": "u1"})
    assert resp.status_code == 200
    assert "application" in fired


def test_metering_failure_does_not_break_search_response() -> None:
    """A get_reflexio error inside the metering helper must not turn a 200 into a 500."""
    events = _capture()
    profiles = [_make_profile_view("u1")]
    try:
        # The unified_search mock is wired via _patch_unified_search (first get_reflexio call).
        # Inside the helper, get_reflexio is called a second time; we make *that* call's
        # get_config raise so the metering path fails while the search response succeeds.
        mock_reflexio_search = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.msg = "OK"
        mock_response.reformulated_query = None
        mock_response.agent_trace = None
        mock_response.rehydrated_text = None
        mock_response.profiles = profiles
        mock_response.agent_playbooks = []
        mock_response.user_playbooks = []
        mock_reflexio_search.unified_search.return_value = mock_response
        # Make get_config raise so metering blows up after the search completes.
        mock_reflexio_search.request_context.configurator.get_config.side_effect = (
            RuntimeError("boom")
        )

        with patch(
            "reflexio.server.api.get_reflexio", return_value=mock_reflexio_search
        ):
            resp = _client("production_agent").post(
                "/api/search", json={"query": "x", "user_id": "u1"}
            )
        assert resp.status_code == 200
    finally:
        configure_usage_event_recorder(None)

    # Metering failed silently — no learning_applied event should have been emitted.
    assert [e for e in events if e.event_name == "learning_applied"] == []


# --- Per-endpoint metering (the four non-unified routes) -----------------------
#
# Each of these endpoints calls a distinct service method on get_reflexio and
# derives surfaced_count from a distinct response list attribute. The cases below
# exercise the real endpoint handler + view conversion + _meter_applied_learnings
# wiring for each, asserting the emitted surfaced_count matches that route's shape.


@contextmanager
def _patch_service_method(method_name: str, response_attr: str, items: list):
    """Patch get_reflexio so ``method_name`` returns a canned service response.

    The response carries ``items`` on ``response_attr`` (e.g. ``user_profiles``)
    so the endpoint's view conversion and surfaced_count computation run for real.
    get_config() returns None so platform_llm_from_config(None) is True without
    iterating a MagicMock.
    """
    mock_reflexio = MagicMock()
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.msg = "OK"
    setattr(mock_response, response_attr, items)
    getattr(mock_reflexio, method_name).return_value = mock_response
    mock_reflexio.request_context.configurator.get_config.return_value = None

    with patch("reflexio.server.api.get_reflexio", return_value=mock_reflexio):
        yield


# (path, payload, service method, response attribute, surfaced item factory)
_ENDPOINT_CASES = [
    pytest.param(
        "/api/search_profiles",
        {"user_id": "u1", "query": "x"},
        "search_user_profiles",
        "user_profiles",
        _make_profile_view,
        id="search_profiles",
    ),
    pytest.param(
        "/api/search_user_playbooks",
        {"query": "x"},
        "search_user_playbooks",
        "user_playbooks",
        _make_user_playbook_view,
        id="search_user_playbooks",
    ),
    pytest.param(
        "/api/search_agent_playbooks",
        {"query": "x"},
        "search_agent_playbooks",
        "agent_playbooks",
        _make_agent_playbook_view,
        id="search_agent_playbooks",
    ),
    pytest.param(
        "/api/get_agent_playbooks",
        {},
        "get_agent_playbooks",
        "agent_playbooks",
        _make_agent_playbook_view,
        id="get_agent_playbooks",
    ),
]


@pytest.mark.parametrize(
    ("path", "payload", "method_name", "response_attr", "make_item"),
    _ENDPOINT_CASES,
)
def test_production_agent_per_endpoint_meters_surfaced_count(
    path: str,
    payload: dict,
    method_name: str,
    response_attr: str,
    make_item,
) -> None:
    """Each non-unified route emits one learning_applied event with its own count."""
    events = _capture()
    items = [make_item(), make_item()]
    try:
        with _patch_service_method(method_name, response_attr, items):
            resp = _client("production_agent").post(path, json=payload)
        assert resp.status_code == 200
    finally:
        configure_usage_event_recorder(None)

    applied = [e for e in events if e.event_name == "learning_applied"]
    assert len(applied) == 1
    assert applied[0].count_value == 2  # len(items) for this endpoint's response shape
    assert applied[0].caller_type == "production_agent"


@pytest.mark.parametrize(
    ("path", "payload", "method_name", "response_attr", "make_item"),
    _ENDPOINT_CASES,
)
def test_dashboard_per_endpoint_meters_nothing(
    path: str,
    payload: dict,
    method_name: str,
    response_attr: str,
    make_item,
) -> None:
    """A dashboard caller never meters, regardless of the route or result size."""
    events = _capture()
    try:
        with _patch_service_method(method_name, response_attr, [make_item()]):
            resp = _client("dashboard").post(path, json=payload)
        assert resp.status_code == 200
    finally:
        configure_usage_event_recorder(None)

    assert [e for e in events if e.event_name == "learning_applied"] == []


@pytest.mark.parametrize(
    ("path", "payload", "method_name", "response_attr", "make_item"),
    _ENDPOINT_CASES,
)
def test_empty_result_per_endpoint_meters_nothing(
    path: str,
    payload: dict,
    method_name: str,
    response_attr: str,
    make_item,
) -> None:
    """A production-agent call surfacing zero results meters nothing on any route."""
    events = _capture()
    try:
        with _patch_service_method(method_name, response_attr, []):
            resp = _client("production_agent").post(path, json=payload)
        assert resp.status_code == 200
    finally:
        configure_usage_event_recorder(None)

    assert [e for e in events if e.event_name == "learning_applied"] == []
