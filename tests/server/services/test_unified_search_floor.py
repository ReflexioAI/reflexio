from typing import cast
from unittest.mock import patch

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import RetrievalFloorConfig
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.storage.storage_base import BaseStorage


def _fake_user_playbook(content: str) -> UserPlaybook:
    # run_unified_search builds a UnifiedSearchResponse, which validates
    # user_playbooks as real UserPlaybook instances, so back the test items
    # with the real entity (still keyed on .content for floor scoring).
    return UserPlaybook(agent_version="v1", request_id="r1", content=content)


class _FakeStorage:
    # run_unified_search reads storage.supports_embedding before _run_phase_a runs,
    # so the stub must expose it even though the phases are monkeypatched.
    supports_embedding = False


def test_floor_applied_per_arm(monkeypatch):
    pbs = [
        _fake_user_playbook("good"),
        _fake_user_playbook("ok"),
        _fake_user_playbook("junk"),
    ]
    monkeypatch.setattr(uss, "_run_phase_a", lambda **_kw: ("q", None))
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([], [], pbs))

    score = {"good": 2.0, "ok": -1.0, "junk": -9.0}

    def fake_score(query, docs):  # noqa: ARG001
        return [score[d] for d in docs]

    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=fake_score,
    ):
        resp = uss.run_unified_search(
            request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
            org_id="o",
            storage=cast(BaseStorage, _FakeStorage()),
            llm_client=cast(LiteLLMClient, object()),
            prompt_manager=cast(PromptManager, object()),
            retrieval_floor=RetrievalFloorConfig(user_playbook_floor=-5.0),
        )

    assert resp.success is True
    # junk dropped, sorted desc
    assert [p.content for p in resp.user_playbooks] == ["good", "ok"]


def test_floor_disabled_returns_all(monkeypatch):
    pbs = [_fake_user_playbook("a"), _fake_user_playbook("b")]
    monkeypatch.setattr(uss, "_run_phase_a", lambda **_kw: ("q", None))
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([], [], pbs))

    resp = uss.run_unified_search(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
        org_id="o",
        storage=cast(BaseStorage, _FakeStorage()),
        llm_client=cast(LiteLLMClient, object()),
        prompt_manager=cast(PromptManager, object()),
        retrieval_floor=RetrievalFloorConfig(enabled=False),
    )
    assert [p.content for p in resp.user_playbooks] == ["a", "b"]
