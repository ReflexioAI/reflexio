"""Integration test for AgenticExtractionService end-to-end wiring.

Uses real SqliteStorage in a tmp_path + mocked LiteLLM so we exercise the
full orchestrator path (readers → critics → reconciler) without real LLM
calls. Exhaustive candidate-flow coverage is handled by the Phase 5
golden-set suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.agentic_extraction_service import (
    AgenticExtractionService,
    ExtractionResult,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@dataclass
class _FakeExtractionRequest:
    """Minimal request object — matches the _HasExtractionInputs protocol."""

    user_id: str
    sessions: str


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


def test_agentic_extraction_end_to_end_empty_candidates(
    tmp_path, real_client, tool_call_completion
):
    """Readers + critics all finish immediately; orchestrator returns empty lanes."""
    store = SQLiteStorage(org_id="u1-org", db_path=str(tmp_path / "reflexio.db"))
    make_tc, _ = tool_call_completion
    # 6 readers + 2 critics = 8 LLM calls minimum; provide extras to be safe.
    responses = [make_tc("finish", {})] * 10

    request_context = _build_request_context(store)
    svc = AgenticExtractionService(
        llm_client=real_client, request_context=request_context
    )
    req = _FakeExtractionRequest(user_id="u1", sessions="USER: noop")

    with patch("litellm.completion", side_effect=responses):
        result = svc.run(req)

    assert isinstance(result, ExtractionResult)
    assert result.skipped_reason is None
    assert result.profiles == []
    assert result.playbooks == []


def test_agentic_extraction_skips_when_no_sessions(tmp_path, real_client):
    """No sessions string → skipped result with reason, no LLM calls needed."""
    store = SQLiteStorage(org_id="u1-org", db_path=str(tmp_path / "reflexio.db"))
    request_context = _build_request_context(store)
    svc = AgenticExtractionService(
        llm_client=real_client, request_context=request_context
    )
    req = _FakeExtractionRequest(user_id="u1", sessions="")

    result = svc.run(req)

    assert result.skipped_reason == "no sessions to extract"
    assert result.profiles == []
    assert result.playbooks == []


def test_agentic_extraction_constructor_stores_client_and_context():
    """Constructor wiring matches ProfileGenerationService so the dispatcher can swap."""
    client = MagicMock()
    rc = MagicMock()
    svc = AgenticExtractionService(llm_client=client, request_context=rc)
    assert svc.client is client
    assert svc.request_context is rc
    assert svc.storage is rc.storage
    assert svc.prompt_manager is rc.prompt_manager
