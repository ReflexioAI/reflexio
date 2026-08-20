"""Tests for retriever_schema — UnifiedSearchResponse msg field round-trips.

The agentic search orchestrator relies on ``UnifiedSearchResponse.msg``
being an accepted, round-trippable field so it can surface partial-failure
context. These tests pin the contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)


def test_unified_search_response_accepts_msg():
    r = UnifiedSearchResponse(
        success=True,
        profiles=[],
        user_playbooks=[],
        agent_playbooks=[],
        reformulated_query="q",
        msg="partial",
    )
    assert r.msg == "partial"


def test_unified_search_response_msg_defaults_to_none():
    r = UnifiedSearchResponse(
        success=True,
        profiles=[],
        user_playbooks=[],
        agent_playbooks=[],
        reformulated_query="q",
    )
    assert r.msg is None


def test_unified_search_response_msg_roundtrips_through_json():
    r = UnifiedSearchResponse(
        success=True,
        profiles=[],
        user_playbooks=[],
        agent_playbooks=[],
        reformulated_query="q",
        msg="partial: some agents timed out",
    )
    restored = UnifiedSearchResponse.model_validate_json(r.model_dump_json())
    assert restored.msg == "partial: some agents timed out"
    assert restored.reformulated_query == "q"


@pytest.mark.parametrize(
    "search_request",
    [
        SearchUserPlaybookRequest(source="api"),
        SearchAgentPlaybookRequest(source="webhook"),
        UnifiedSearchRequest(query="billing", source="workflow:v1"),
    ],
)
def test_search_requests_accept_valid_source(search_request):
    assert search_request.source is not None


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (SearchUserPlaybookRequest, {}),
        (SearchAgentPlaybookRequest, {}),
        (UnifiedSearchRequest, {"query": "billing"}),
    ],
)
def test_search_requests_reject_invalid_source(model, kwargs):
    with pytest.raises(ValidationError):
        model(source="Contains Spaces", **kwargs)


@pytest.mark.parametrize(
    "search_request",
    [
        SearchUserPlaybookRequest(source=""),
        SearchAgentPlaybookRequest(source=""),
        UnifiedSearchRequest(query="billing", source=""),
    ],
)
def test_empty_search_source_is_accepted_as_no_filter(search_request):
    assert search_request.source == ""
