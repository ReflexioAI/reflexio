"""End-to-end smoke: config(extraction=agentic, search=agentic) — full pipeline.

Wires both agentic services via the dispatcher factories, runs one
extraction and one search cycle with a mocked LiteLLM, and asserts the
pipelines terminate cleanly. Exhaustive per-stage coverage lives in the
extraction + search integration tests; this smoke test exists to prove the
two factories return the expected service classes and that the full
reader/critic/agent/synth chain runs end-to-end on real SQLite storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.agentic_extraction_service import (
    AgenticExtractionService,
)
from reflexio.server.services.generation_service import (
    build_extraction_service,
    build_search_service,
)
from reflexio.server.services.search.agentic_search_service import (
    AgenticSearchService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@dataclass
class _FakeExtractionRequest:
    user_id: str
    sessions: str


def _request_context(storage: SQLiteStorage) -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = "stub"
    ctx = MagicMock()
    ctx.storage = storage
    ctx.prompt_manager = pm
    return ctx


@pytest.fixture
def real_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    return LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))


def test_agentic_backend_full_pipeline(tmp_path, real_client, tool_call_completion):
    """Factories pick agentic when configured; extraction + search both complete."""
    store = SQLiteStorage(org_id="u1-org", db_path=str(tmp_path / "reflexio.db"))
    cfg = Config(
        storage_config=StorageConfigSQLite(),
        extraction_backend="agentic",
        search_backend="agentic",
    )
    rc = _request_context(store)

    extract_svc_raw = build_extraction_service(
        cfg, llm_client=real_client, request_context=rc
    )
    search_svc_raw = build_search_service(
        cfg, llm_client=real_client, request_context=rc
    )

    assert isinstance(extract_svc_raw, AgenticExtractionService)
    assert isinstance(search_svc_raw, AgenticSearchService)
    extract_svc = cast(AgenticExtractionService, extract_svc_raw)
    search_svc = cast(AgenticSearchService, search_svc_raw)

    make_tc, _ = tool_call_completion
    # Extraction: 6 readers finish + 2 critics finish = 8 LLM calls (give extras).
    extract_responses = [make_tc("finish", {})] * 10
    # Search: 6 agents submit empty + 2 synths rank empty + finish.
    search_responses = [
        make_tc("submit_candidates", {"ids": [], "why": "none"})
    ] * 6 + [make_tc("rank", {"ordered_ids": []}), make_tc("finish", {})] * 2

    extract_req = _FakeExtractionRequest(user_id="u1", sessions="USER: noop")
    search_req = UnifiedSearchRequest(query="q", user_id="u1")

    with patch("litellm.completion", side_effect=extract_responses + search_responses):
        e_res = extract_svc.run(extract_req)
        s_res = search_svc.search(search_req)

    assert e_res.skipped_reason is None
    assert e_res.profiles == []
    assert e_res.playbooks == []
    assert s_res.success is True
    assert s_res.reformulated_query == "q"
    assert s_res.profiles == []
    assert s_res.user_playbooks == []
    assert s_res.agent_playbooks == []
