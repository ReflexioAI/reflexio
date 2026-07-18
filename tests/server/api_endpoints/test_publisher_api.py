"""Behavioral contracts for publisher endpoint delegation and failures."""

from collections.abc import Callable
from typing import Protocol
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.api_endpoints.publisher_api import (
    add_agent_playbook,
    add_user_interaction,
    add_user_playbook,
    add_user_profile,
    clear_user_data,
    delete_agent_playbook,
    delete_agent_playbooks_by_ids_bulk,
    delete_all_interactions_bulk,
    delete_all_playbooks_bulk,
    delete_all_profiles_bulk,
    delete_profiles_by_ids,
    delete_request,
    delete_requests_by_ids,
    delete_session,
    delete_user_interaction,
    delete_user_playbook,
    delete_user_playbooks_by_ids_bulk,
    delete_user_profile,
    run_playbook_aggregation,
    update_agent_playbook_status,
)

MODULE = "reflexio.server.api_endpoints.publisher_api"
ORG_ID = "test-org"


class EndpointResponse(Protocol):
    success: bool


Endpoint = Callable[..., EndpointResponse]


@pytest.fixture
def mock_reflexio():
    with patch(f"{MODULE}.get_reflexio") as mock_get:
        reflexio = MagicMock()
        mock_get.return_value = reflexio
        yield reflexio


@patch(f"{MODULE}.validate_publish_user_interaction_request")
def test_add_user_interaction_forwards_publish_options(mock_validate, mock_reflexio):
    mock_validate.return_value = (True, "")
    request = MagicMock()
    expected = MagicMock()
    mock_reflexio.publish_interaction.return_value = expected

    result = add_user_interaction(ORG_ID, request)

    mock_reflexio.publish_interaction.assert_called_once_with(
        request=request,
        use_publish_limiter=True,
        publish_limiter_wait_forever=True,
        defer_learning=False,
    )
    assert result is expected


@patch(f"{MODULE}.validate_publish_user_interaction_request")
def test_add_user_interaction_returns_validation_message(mock_validate, mock_reflexio):
    mock_validate.return_value = (False, "No interaction data provided")
    request = MagicMock()

    result = add_user_interaction(ORG_ID, request)

    assert result.success is False
    assert result.message == "No interaction data provided"
    mock_reflexio.publish_interaction.assert_not_called()


_DELEGATION_CASES: tuple[tuple[str, Endpoint, str, str], ...] = (
    ("add-user-playbook", add_user_playbook, "add_user_playbook", "keyword"),
    ("add-agent-playbook", add_agent_playbook, "add_agent_playbook", "keyword"),
    ("add-user-profile", add_user_profile, "add_user_profile", "keyword"),
    (
        "delete-user-interaction",
        delete_user_interaction,
        "delete_interaction",
        "positional",
    ),
    ("delete-request", delete_request, "delete_request", "positional"),
    ("delete-session", delete_session, "delete_session", "positional"),
    (
        "delete-agent-playbook",
        delete_agent_playbook,
        "delete_agent_playbook",
        "positional",
    ),
    (
        "delete-user-playbook",
        delete_user_playbook,
        "delete_user_playbook",
        "positional",
    ),
    (
        "delete-all-interactions",
        delete_all_interactions_bulk,
        "delete_all_interactions_bulk",
        "none",
    ),
    (
        "delete-all-profiles",
        delete_all_profiles_bulk,
        "delete_all_profiles_bulk",
        "none",
    ),
    (
        "delete-all-playbooks",
        delete_all_playbooks_bulk,
        "delete_all_playbooks_bulk",
        "none",
    ),
    (
        "delete-requests-by-id",
        delete_requests_by_ids,
        "delete_requests_by_ids",
        "positional",
    ),
    (
        "delete-profiles-by-id",
        delete_profiles_by_ids,
        "delete_profiles_by_ids",
        "positional",
    ),
    (
        "delete-agent-playbooks-by-id",
        delete_agent_playbooks_by_ids_bulk,
        "delete_agent_playbooks_by_ids_bulk",
        "positional",
    ),
    (
        "delete-user-playbooks-by-id",
        delete_user_playbooks_by_ids_bulk,
        "delete_user_playbooks_by_ids_bulk",
        "positional",
    ),
    ("clear-user-data", clear_user_data, "clear_user_data", "positional"),
    (
        "update-agent-playbook-status",
        update_agent_playbook_status,
        "update_agent_playbook_status",
        "positional",
    ),
)


@pytest.mark.parametrize(
    ("endpoint", "method_name", "argument_style"),
    [(endpoint, method, style) for _, endpoint, method, style in _DELEGATION_CASES],
    ids=[case[0] for case in _DELEGATION_CASES],
)
def test_endpoint_delegation_contract(
    mock_reflexio,
    endpoint: Endpoint,
    method_name: str,
    argument_style: str,
):
    request = MagicMock()
    expected = MagicMock()
    method = getattr(mock_reflexio, method_name)
    method.return_value = expected

    result = endpoint(ORG_ID) if argument_style == "none" else endpoint(ORG_ID, request)

    if argument_style == "none":
        method.assert_called_once_with()
    elif argument_style == "keyword":
        method.assert_called_once_with(request=request)
    else:
        method.assert_called_once_with(request)
    assert result is expected


def test_add_user_profile_uses_org_id_to_select_reflexio():
    with patch(f"{MODULE}.get_reflexio") as mock_get:
        reflexio = MagicMock()
        mock_get.return_value = reflexio
        request = MagicMock()

        add_user_profile(ORG_ID, request)

    mock_get.assert_called_once_with(org_id=ORG_ID)
    reflexio.add_user_profile.assert_called_once_with(request=request)


@patch(f"{MODULE}.validate_delete_user_profile_request")
def test_delete_user_profile_delegates_after_validation(mock_validate, mock_reflexio):
    mock_validate.return_value = (True, "")
    request = MagicMock()
    expected = MagicMock()
    mock_reflexio.delete_profile.return_value = expected

    result = delete_user_profile(ORG_ID, request)

    mock_reflexio.delete_profile.assert_called_once_with(request)
    assert result is expected


@patch(f"{MODULE}.validate_delete_user_profile_request")
def test_delete_user_profile_returns_validation_message(mock_validate, mock_reflexio):
    mock_validate.return_value = (False, "Profile id or search query is required")
    request = MagicMock()

    result = delete_user_profile(ORG_ID, request)

    assert result.success is False
    assert result.message == "Profile id or search query is required"
    mock_reflexio.delete_profile.assert_not_called()


@patch(f"{MODULE}.validate_delete_user_profile_request")
def test_delete_user_profile_translates_storage_exception(mock_validate, mock_reflexio):
    mock_validate.return_value = (True, "")
    mock_reflexio.delete_profile.side_effect = RuntimeError("storage error")

    result = delete_user_profile(ORG_ID, MagicMock())

    assert result.success is False
    assert result.message == "storage error"


_EXCEPTION_CASES: tuple[tuple[str, Endpoint, str, str], ...] = (
    (
        "delete-user-interaction",
        delete_user_interaction,
        "delete_interaction",
        "message",
    ),
    ("delete-request", delete_request, "delete_request", "message"),
    ("delete-session", delete_session, "delete_session", "message"),
    (
        "delete-agent-playbook",
        delete_agent_playbook,
        "delete_agent_playbook",
        "message",
    ),
    (
        "delete-user-playbook",
        delete_user_playbook,
        "delete_user_playbook",
        "message",
    ),
    (
        "update-agent-playbook-status",
        update_agent_playbook_status,
        "update_agent_playbook_status",
        "msg",
    ),
)


@pytest.mark.parametrize(
    ("endpoint", "method_name", "error_field"),
    [(endpoint, method, field) for _, endpoint, method, field in _EXCEPTION_CASES],
    ids=[case[0] for case in _EXCEPTION_CASES],
)
def test_endpoint_exception_contract(
    mock_reflexio,
    endpoint: Endpoint,
    method_name: str,
    error_field: str,
):
    getattr(mock_reflexio, method_name).side_effect = RuntimeError("operation failed")

    result = endpoint(ORG_ID, MagicMock())

    assert result.success is False
    assert getattr(result, error_field) == "operation failed"


def test_run_playbook_aggregation_forwards_distinct_fields(mock_reflexio):
    request = MagicMock(agent_version="v1", playbook_name="quality")

    result = run_playbook_aggregation(ORG_ID, request)

    mock_reflexio.run_playbook_aggregation.assert_called_once_with("v1", "quality")
    assert result.success is True


def test_run_playbook_aggregation_translates_exception(mock_reflexio):
    request = MagicMock(agent_version="v1", playbook_name="quality")
    mock_reflexio.run_playbook_aggregation.side_effect = RuntimeError("llm error")

    result = run_playbook_aggregation(ORG_ID, request)

    assert result.success is False
    assert result.message == "llm error"
