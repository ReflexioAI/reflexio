"""Final selection stays inside native search and before served-state writes."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    UnifiedSearchRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode
from reflexio.server import search_runtime
from reflexio.server.extensions import register_service
from reflexio.server.services import unified_search_service as service
from reflexio.server.services.retrieval.agent_selection import AGENT_PLAYBOOK_SELECTOR
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.storage.storage_base import BaseStorage


@pytest.fixture
def native(monkeypatch):
    vector = [0.1] * 512
    agents = [
        AgentPlaybook(agent_playbook_id=i, agent_version="v1", content=f"a{i}")
        for i in range(1, 4)
    ]
    users = [
        UserPlaybook(
            user_playbook_id=11,
            user_id="u",
            agent_version="v1",
            request_id="request",
            content="linked source",
        ),
        UserPlaybook(
            user_playbook_id=12,
            user_id="u",
            agent_version="v1",
            request_id="request",
            content="independent user",
        ),
    ]
    profiles = [
        UserProfile(
            profile_id="p",
            user_id="u",
            content="profile",
            last_modified_timestamp=1,
            generated_from_request_id="request",
        )
    ]
    storage = SimpleNamespace(
        org_id="org",
        embedding_model_name="local/multilingual-e5-small",
        supports_embedding=True,
        _query_embedding_cache_identity=object(),
        _get_embedding=MagicMock(return_value=vector),
        get_source_user_playbook_ids_for_agent_playbooks=MagicMock(
            return_value={1: [11]}
        ),
    )
    seen = {}

    def retrieve(**kwargs):
        seen.update(kwargs)
        return profiles, agents, users

    monkeypatch.setattr(service, "_EMBEDDING_CACHE_TTL_SECONDS", 0)
    monkeypatch.setattr(service, "_run_phase_b", retrieve)
    reformulator = MagicMock()
    reformulator.rewrite.return_value = ReformulationResult(standalone_query="q")
    monkeypatch.setattr(service, "QueryReformulator", lambda **_kw: reformulator)
    session_seen_cache.clear()
    yield SimpleNamespace(
        storage=storage,
        vector=vector,
        agents=agents,
        users=users,
        profiles=profiles,
        seen=seen,
    )
    session_seen_cache.clear()


def run(native, **overrides):
    request = UnifiedSearchRequest(query="q", user_id="u", top_k=3, **overrides)
    return service.run_unified_search(
        request=request,
        org_id="org",
        storage=cast(BaseStorage, native.storage),
        llm_client=MagicMock(),
        prompt_manager=MagicMock(),
    )


def test_exact_native_query_vector_used_once_and_only_agents_change(native):
    received = {}

    class Selector:
        def select(self, **kwargs):
            received.update(kwargs)
            return [kwargs["items"][0], kwargs["items"][2]]

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    response = run(native)
    assert response.success
    assert native.storage._get_embedding.call_count == 1
    assert received["query_embedding"] is native.vector
    assert native.seen["embedding"] is received["query_embedding"]
    assert received["storage"] is native.storage and received["org_id"] == "org"
    assert [a.agent_playbook_id for a in response.agent_playbooks] == [1, 3]
    assert response.profiles == native.profiles
    assert [p.user_playbook_id for p in response.user_playbooks] == [12]


def test_source_suppression_precedes_gate_and_removed_agents_are_not_seen(native):
    class Selector:
        def select(self, **kwargs):
            native.storage.get_source_user_playbook_ids_for_agent_playbooks.assert_called_once()
            return [kwargs["items"][1]]

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    response = run(native, session_id="session")
    assert [p.user_playbook_id for p in response.user_playbooks] == [12]
    assert [a.agent_playbook_id for a in response.agent_playbooks] == [2]
    assert session_seen_cache.seen("org", "session") == frozenset(
        {("profile", "p"), ("user_playbook", "12"), ("agent_playbook", "2")}
    )


def test_provider_refusal_never_serves_native_agents_or_records_seen(native):
    class Selector:
        def select(self, **kwargs):
            raise ReadinessUnavailableError("selection source stale")

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    with pytest.raises(ReadinessUnavailableError, match="stale"):
        run(native, session_id="session")
    assert not session_seen_cache.seen("org", "session")


@pytest.mark.parametrize("choice", ["reverse", "duplicate", "new", "mutate_list"])
def test_selector_cannot_reorder_duplicate_or_insert_new_instances(native, choice):
    class Selector:
        def select(self, **kwargs):
            items = kwargs["items"]
            if choice == "reverse":
                return list(reversed(items))
            if choice == "duplicate":
                return [items[0], items[0]]
            if choice == "mutate_list":
                items.reverse()
                return items
            return [items[0].model_copy(update={"content": "rewritten"})]

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    with pytest.raises(ReadinessUnavailableError, match="stable subsequence"):
        run(native)
    assert [item.agent_playbook_id for item in native.agents] == [1, 2, 3]


def test_provider_absence_preserves_bare_oss(native):
    response = run(native)
    assert response.agent_playbooks == native.agents


def test_empty_native_agents_do_not_require_features(native, monkeypatch):
    monkeypatch.setattr(service, "_run_phase_b", lambda **_kw: ([], [], []))
    provider = SimpleNamespace(
        select=MagicMock(side_effect=AssertionError("empty gate"))
    )
    register_service(AGENT_PLAYBOOK_SELECTOR, provider)
    assert run(native).agent_playbooks == []
    provider.select.assert_not_called()


def test_explicit_fts_passes_missing_vector_and_original_request_to_provider(native):
    received = {}

    class Selector:
        def select(self, **kwargs):
            received.update(kwargs)
            assert kwargs["request"].search_mode is SearchMode.FTS
            return kwargs["items"]

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    response = run(native, search_mode=SearchMode.FTS)
    assert response.success and not response.degraded
    assert received["query_embedding"] is None
    native.storage._get_embedding.assert_not_called()


def test_embedding_failure_is_not_hidden_from_selected_provider(native):
    native.storage._get_embedding.side_effect = RuntimeError("offline encoder")

    class Selector:
        def select(self, **kwargs):
            assert kwargs["query_embedding"] is None
            assert kwargs["request"].search_mode is SearchMode.HYBRID
            raise ReadinessUnavailableError("E5 selection needs its original vector")

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    with pytest.raises(ReadinessUnavailableError, match="original vector"):
        run(native)


def test_profile_arm_degradation_is_visible_to_selected_provider(native, monkeypatch):
    monkeypatch.setattr(
        service, "_run_phase_b", lambda **_kw: (None, native.agents, [])
    )

    class Selector:
        def select(self, **kwargs):
            assert kwargs["degraded"] is True
            assert kwargs["query_embedding"] is native.vector
            raise ReadinessUnavailableError("partial native result")

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    with pytest.raises(ReadinessUnavailableError, match="partial"):
        run(native)


def test_deadline_consumed_by_selector_prevents_served_state(native):
    import time

    scope = search_runtime.SearchScope(deadline=time.monotonic() + 30)

    class Selector:
        def select(self, **kwargs):
            scope.deadline = time.monotonic() - 1
            return kwargs["items"]

    register_service(AGENT_PLAYBOOK_SELECTOR, Selector())
    token = search_runtime._scope.set(scope)
    try:
        with pytest.raises(search_runtime.SearchDeadlineError):
            run(native, session_id="session")
        assert not scope.success_callbacks
        assert not session_seen_cache.seen("org", "session")
    finally:
        search_runtime._scope.reset(token)


def test_total_native_failure_does_not_invoke_selector(native, monkeypatch):
    monkeypatch.setattr(service, "_run_phase_b", lambda **_kw: (None, None, None))
    provider = SimpleNamespace(
        select=MagicMock(side_effect=AssertionError("failed native search"))
    )
    register_service(AGENT_PLAYBOOK_SELECTOR, provider)
    response = run(native)
    assert not response.success
    provider.select.assert_not_called()
