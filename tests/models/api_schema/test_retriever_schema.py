"""Tests for retriever_schema — UnifiedSearchResponse msg field round-trips.

The agentic search orchestrator relies on ``UnifiedSearchResponse.msg``
being an accepted, round-trippable field so it can surface partial-failure
context. These tests pin the contract.
"""

from __future__ import annotations

from reflexio.models.api_schema.retriever_schema import UnifiedSearchResponse


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
