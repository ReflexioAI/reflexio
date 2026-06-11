"""Phase B single-RPC routing: combined storage call, fallback, kill switch."""

from typing import Any, cast

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.storage.storage_base import BaseStorage


def _agent_playbook(playbook_id: int, content: str) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=playbook_id, agent_version="v1", content=content
    )


class _CombinedStorage:
    """Fake storage advertising the combined Phase B capability."""

    supports_embedding = True
    supports_unified_hybrid_search = True

    def __init__(
        self,
        result: tuple[list[Any], list[Any], list[Any]] = ([], [], []),
        raise_on_combined: bool = False,
    ) -> None:
        self.result = result
        self.raise_on_combined = raise_on_combined
        self.combined_calls: list[dict[str, Any]] = []
        self.fanout_calls: list[str] = []

    def unified_hybrid_search(self, **kwargs: Any):
        self.combined_calls.append(kwargs)
        if self.raise_on_combined:
            raise RuntimeError("function public.unified_hybrid_search does not exist")
        return self.result

    # Per-arm methods used by the fan-out fallback path.
    def search_user_profile(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("profiles")
        return []

    def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("agent_playbooks")
        return []

    def search_user_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("user_playbooks")
        return []


class _MissingCombinedMethodStorage:
    supports_embedding = True
    supports_unified_hybrid_search = True

    def __init__(self) -> None:
        self.fanout_calls: list[str] = []

    def search_user_profile(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("profiles")
        return []

    def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("agent_playbooks")
        return []

    def search_user_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("user_playbooks")
        return []


def _run_phase_b(storage: _CombinedStorage, *, user_id: str | None = "u"):
    return uss._run_phase_b(
        request=UnifiedSearchRequest(query="q", user_id=user_id, top_k=5),
        org_id="o",
        storage=cast(BaseStorage, storage),
        embedding=[0.1, 0.2],
        query="q",
        top_k=5,
        threshold=0.3,
    )


def test_single_rpc_used_when_supported(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    playbooks = [
        _agent_playbook(1, "a"),
        _agent_playbook(1, "dup"),
        _agent_playbook(2, "b"),
    ]
    user_playbooks = [UserPlaybook(agent_version="v1", request_id="r1", content="up")]
    storage = _CombinedStorage(result=([], playbooks, user_playbooks))

    profiles, agent_playbooks, returned_user_playbooks = _run_phase_b(storage)

    assert len(storage.combined_calls) == 1
    assert storage.fanout_calls == []
    call = storage.combined_calls[0]
    assert call["include_profiles"] is True
    assert call["include_agent_playbooks"] is True
    assert call["include_user_playbooks"] is True
    assert call["agent_playbook_statuses"] == [
        PlaybookStatus.APPROVED,
        PlaybookStatus.PENDING,
    ]
    # Duplicate agent playbook ids are deduped, mirroring the fan-out path.
    assert agent_playbooks is not None
    assert [p.agent_playbook_id for p in agent_playbooks] == [1, 2]
    assert profiles == []
    assert returned_user_playbooks == user_playbooks


def test_single_rpc_skips_profiles_without_user_id(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage()

    _run_phase_b(storage, user_id=None)

    assert storage.combined_calls[0]["include_profiles"] is False


def test_single_rpc_failure_falls_back_to_fanout(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage(raise_on_combined=True)

    profiles, agent_playbooks, user_playbooks = _run_phase_b(storage)

    assert len(storage.combined_calls) == 1
    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
    assert (profiles, agent_playbooks, user_playbooks) == ([], [], [])


def test_single_rpc_missing_callable_falls_back_to_fanout(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _MissingCombinedMethodStorage()

    profiles, agent_playbooks, user_playbooks = uss._run_phase_b(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
        org_id="o",
        storage=cast(BaseStorage, storage),
        embedding=[0.1, 0.2],
        query="q",
        top_k=5,
        threshold=0.3,
    )

    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
    assert (profiles, agent_playbooks, user_playbooks) == ([], [], [])


def test_single_rpc_kill_switch_disables_combined_path(monkeypatch):
    monkeypatch.setenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", "0")
    storage = _CombinedStorage()

    _run_phase_b(storage)

    assert storage.combined_calls == []
    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
