"""Integration test for AgenticSearchService end-to-end wiring.

Uses real ``SQLiteStorage`` in a tmp_path + mocked LiteLLM so we exercise
the full orchestrator path (6 agents → 2 synthesizers → optional
reconciler) without real LLM calls. Exhaustive agent-flow coverage is
handled by the Phase 5 golden-set suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.search.agentic_search_service import (
    AgenticSearchService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _build_request_context(storage: SQLiteStorage) -> MagicMock:
    """Build a request_context stand-in with real storage + mocked prompt_manager."""
    pm = MagicMock()
    pm.render_prompt.return_value = "stub prompt"
    ctx = MagicMock()
    ctx.storage = storage
    ctx.prompt_manager = pm
    return ctx


@pytest.fixture
def real_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    return LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))


def test_agentic_search_returns_unified_response_shape(
    tmp_path, real_client, tool_call_completion
):
    """Every agent submits empty, both synthesizers rank empty → empty response."""
    store = SQLiteStorage(org_id="u1-org", db_path=str(tmp_path / "reflexio.db"))
    make_tc, _ = tool_call_completion
    # 6 agents each call submit_candidates; 2 synthesizers each call rank + finish.
    responses = [make_tc("submit_candidates", {"ids": [], "why": "none"})] * 6 + [
        make_tc("rank", {"ordered_ids": []}),
        make_tc("finish", {}),
    ] * 2

    svc = AgenticSearchService(
        llm_client=real_client, request_context=_build_request_context(store)
    )
    req = UnifiedSearchRequest(query="polars preference", user_id="u1")

    with patch("litellm.completion", side_effect=responses):
        resp = svc.search(req)

    assert isinstance(resp, UnifiedSearchResponse)
    assert resp.success is True
    assert resp.profiles == []
    assert resp.user_playbooks == []
    assert resp.agent_playbooks == []
    assert resp.reformulated_query == "polars preference"
    assert resp.msg is None


def test_agentic_search_skips_reformulation_when_disabled(
    tmp_path, real_client, tool_call_completion
):
    """enable_reformulation=False → reformulated_query is the raw query."""
    store = SQLiteStorage(org_id="u1-org", db_path=str(tmp_path / "reflexio.db"))
    make_tc, _ = tool_call_completion
    responses = [make_tc("submit_candidates", {"ids": [], "why": "none"})] * 6 + [
        make_tc("rank", {"ordered_ids": []}),
        make_tc("finish", {}),
    ] * 2
    svc = AgenticSearchService(
        llm_client=real_client, request_context=_build_request_context(store)
    )
    req = UnifiedSearchRequest(query="q", user_id="u1", enable_reformulation=False)

    with patch("litellm.completion", side_effect=responses):
        resp = svc.search(req)

    assert resp.reformulated_query == "q"


def test_agentic_search_constructor_stores_client_and_context():
    """Constructor wiring matches UnifiedSearchService so the dispatcher can swap."""
    client = MagicMock()
    rc = MagicMock()
    svc = AgenticSearchService(llm_client=client, request_context=rc)
    assert svc.client is client
    assert svc.request_context is rc
    assert svc.storage is rc.storage
    assert svc.prompt_manager is rc.prompt_manager
