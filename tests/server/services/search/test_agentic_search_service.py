"""Integration tests for AgenticSearchService — populated entity lists + agent_answer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import AgentPlaybook, PlaybookStatus
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.tools import ToolLoopTrace, ToolLoopTurn
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import (
    SearchAgentPlaybooksArgs,
    _handle_search_agent_playbooks,
)
from reflexio.server.services.search.plan import SearchResult


def _mk_tc(id_, name, args):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mk_resp(tool_calls):
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = None
    return r


@pytest.fixture
def temp_storage(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(org_id="svc-test", db_path=str(tmp_path / "svc.db"))


def test_agentic_search_populates_profiles_from_trace(temp_storage, monkeypatch):
    """Agent searches profiles; service fetches and returns matching profile objects."""
    from reflexio.models.api_schema.domain.entities import (
        NEVER_EXPIRES_TIMESTAMP,
        UserProfile,
    )
    from reflexio.models.api_schema.domain.enums import ProfileTimeToLive

    monkeypatch.setattr(temp_storage, "_get_embedding", lambda _text: [0.1] * 512)
    monkeypatch.setattr(
        "reflexio.server.llm.tools.supports_tool_calling", lambda _: True
    )

    temp_storage.add_user_profile(
        "u_1",
        [
            UserProfile(
                profile_id="p_seed_1",
                user_id="u_1",
                content="user likes sushi",
                last_modified_timestamp=0,
                generated_from_request_id="r_1",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                extractor_names=["test"],
            ),
        ],
    )

    client = MagicMock()
    client.config = MagicMock()
    client.config.api_key_config = None
    client.generate_chat_response.side_effect = [
        _mk_resp(
            [_mk_tc("c1", "search_user_profiles", {"query": "sushi", "top_k": 10})]
        ),
        _mk_resp([_mk_tc("c2", "finish", {"answer": "sushi lover"})]),
    ]

    import tempfile

    from reflexio.server.api_endpoints.request_context import RequestContext

    with tempfile.TemporaryDirectory() as d:
        rc = RequestContext(org_id="svc-test", storage_base_dir=d)
        rc.storage = temp_storage  # type: ignore[attr-defined]

        from reflexio.server.services.search.agentic_search_service import (
            AgenticSearchService,
        )

        svc = AgenticSearchService(llm_client=client, request_context=rc)

        request = UnifiedSearchRequest(
            query="what does user like?",
            user_id="u_1",
            agent_version="v1",
            top_k=5,
            enable_agent_answer=True,
        )
        response = svc.search(request)

    assert response.success is True
    assert response.agent_answer == "sushi lover"
    assert response.msg is None
    assert len(response.profiles) == 1
    assert response.profiles[0].profile_id == "p_seed_1"
    assert response.user_playbooks == []
    assert response.agent_playbooks == []


def test_agentic_search_empty_when_agent_searches_nothing(temp_storage, monkeypatch):
    """Agent finishes without searching; service returns empty entity lists."""
    monkeypatch.setattr(
        "reflexio.server.llm.tools.supports_tool_calling", lambda _: True
    )

    client = MagicMock()
    client.config = MagicMock()
    client.config.api_key_config = None
    client.generate_chat_response.side_effect = [
        _mk_resp([_mk_tc("c1", "finish", {"answer": "no evidence"})]),
    ]

    import tempfile

    from reflexio.server.api_endpoints.request_context import RequestContext

    with tempfile.TemporaryDirectory() as d:
        rc = RequestContext(org_id="svc-test", storage_base_dir=d)
        rc.storage = temp_storage  # type: ignore[attr-defined]

        from reflexio.server.services.search.agentic_search_service import (
            AgenticSearchService,
        )

        svc = AgenticSearchService(llm_client=client, request_context=rc)

        request = UnifiedSearchRequest(
            query="anything?",
            user_id="u_nobody",
            agent_version="v1",
            top_k=5,
            enable_agent_answer=True,
        )
        response = svc.search(request)

    assert response.success is True
    assert response.agent_answer == "no evidence"
    assert response.profiles == []
    assert response.user_playbooks == []
    assert response.agent_playbooks == []


def test_agentic_agent_playbook_tool_excludes_rejected_by_default():
    """Default agentic playbook search only asks storage for approved/pending."""
    seen_statuses: list[PlaybookStatus] = []

    def search_agent_playbooks(request, options=None):
        seen_statuses.append(request.playbook_status_filter)
        if request.playbook_status_filter == PlaybookStatus.REJECTED:
            return [
                AgentPlaybook(
                    agent_playbook_id=3,
                    agent_version="v1",
                    content="rejected rule",
                    playbook_status=PlaybookStatus.REJECTED,
                )
            ]
        return [
            AgentPlaybook(
                agent_playbook_id=len(seen_statuses),
                agent_version="v1",
                content=f"{request.playbook_status_filter.value} rule",
                playbook_status=request.playbook_status_filter,
            )
        ]

    result = _handle_search_agent_playbooks(
        SearchAgentPlaybooksArgs(query="formatting", top_k=5),
        SimpleNamespace(search_agent_playbooks=search_agent_playbooks),
        ExtractionCtx(user_id="u1", agent_version="v1"),
    )

    assert seen_statuses == [PlaybookStatus.APPROVED, PlaybookStatus.PENDING]
    assert {hit["playbook_status"] for hit in result["hits"]} == {
        PlaybookStatus.APPROVED,
        PlaybookStatus.PENDING,
    }


def test_agentic_fetch_entities_excludes_rejected_by_default():
    from reflexio.server.services.search.agentic_search_service import (
        AgenticSearchService,
    )

    storage = SimpleNamespace(
        get_agent_playbooks=lambda _agent_version: [
            AgentPlaybook(
                agent_playbook_id=1,
                agent_version="v1",
                content="approved rule",
                playbook_status=PlaybookStatus.APPROVED,
            ),
            AgentPlaybook(
                agent_playbook_id=2,
                agent_version="v1",
                content="rejected rule",
                playbook_status=PlaybookStatus.REJECTED,
            ),
        ]
    )
    svc = AgenticSearchService(
        llm_client=MagicMock(),
        request_context=cast(
            RequestContext,
            SimpleNamespace(storage=storage, prompt_manager=MagicMock()),
        ),
    )
    trace = ToolLoopTrace(
        turns=[
            ToolLoopTurn(
                tool_name="search_agent_playbooks",
                args={},
                result={"hits": [{"id": "1"}, {"id": "2"}]},
                latency_ms=0,
            )
        ],
        finished=True,
    )
    result = SearchResult(
        answer=None,
        outcome="finish_tool",
        budget_exceeded=False,
        trace=trace,
    )

    _, _, agent_playbooks = svc._fetch_entities(
        UnifiedSearchRequest(query="formatting", user_id="u1", agent_version="v1"),
        result,
    )

    assert [p.agent_playbook_id for p in agent_playbooks] == [1]
