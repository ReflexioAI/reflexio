from typing import cast
from unittest.mock import patch

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import RetrievalFloorConfig
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    CrossEncoderUnavailableError,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem
from reflexio.server.services.storage.storage_base import BaseStorage


def _fake_user_playbook(content: str, *, created_at: int | None = None) -> UserPlaybook:
    # run_unified_search builds a UnifiedSearchResponse, which validates
    # user_playbooks as real UserPlaybook instances, so back the test items
    # with the real entity (still keyed on .content for floor scoring).
    if created_at is not None:
        return UserPlaybook(
            agent_version="v1",
            request_id="r1",
            content=content,
            created_at=created_at,
        )
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
            retrieval_floor=RetrievalFloorConfig(
                enabled=True, user_playbook_floor=-5.0
            ),
        )

    assert resp.success is True
    # junk dropped, sorted desc
    assert [p.content for p in resp.user_playbooks] == ["good", "ok"]


def test_floor_recency_applied_after_logits(monkeypatch):
    old = _fake_user_playbook("old", created_at=1)
    fresh = _fake_user_playbook("fresh", created_at=4_102_444_800)
    monkeypatch.setattr(uss, "_run_phase_a", lambda **_kw: ("q", None))
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([], [], [old, fresh]))

    def fake_score(query, docs):  # noqa: ARG001
        return [2.0 if doc == "old" else 1.9 for doc in docs]

    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=fake_score,
    ):
        resp = uss.run_unified_search(
            request=UnifiedSearchRequest(query="q", user_id="u", top_k=2),
            org_id="o",
            storage=cast(BaseStorage, _FakeStorage()),
            llm_client=cast(LiteLLMClient, object()),
            prompt_manager=cast(PromptManager, object()),
            retrieval_floor=RetrievalFloorConfig(
                enabled=True, user_playbook_floor=-5.0
            ),
            recency=RecencyConfig(enabled=True, max_penalty_logit=1.0, pool_size=2),
        )

    assert [p.content for p in resp.user_playbooks] == ["fresh", "old"]


def test_floor_recency_does_not_overtake_clearly_more_relevant(monkeypatch):
    # Invariant: at the default penalty, recency must NOT reorder when the logit
    # gap exceeds max_penalty_logit. An ancient but clearly-more-relevant item
    # stays ahead of a brand-new but weaker one.
    old = _fake_user_playbook("old", created_at=1)  # ancient -> max penalty
    fresh = _fake_user_playbook("fresh", created_at=4_102_444_800)  # brand new
    monkeypatch.setattr(uss, "_run_phase_a", lambda **_kw: ("q", None))
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([], [], [old, fresh]))

    def fake_score(query, docs):  # noqa: ARG001
        # 1.0 logit gap >> max_penalty_logit (0.2)
        return [2.0 if doc == "old" else 1.0 for doc in docs]

    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=fake_score,
    ):
        resp = uss.run_unified_search(
            request=UnifiedSearchRequest(query="q", user_id="u", top_k=2),
            org_id="o",
            storage=cast(BaseStorage, _FakeStorage()),
            llm_client=cast(LiteLLMClient, object()),
            prompt_manager=cast(PromptManager, object()),
            retrieval_floor=RetrievalFloorConfig(
                enabled=True, user_playbook_floor=-5.0
            ),
            recency=RecencyConfig(enabled=True, max_penalty_logit=0.2, pool_size=2),
        )

    assert [p.content for p in resp.user_playbooks] == ["old", "fresh"]


def test_floor_recency_falls_back_to_combined_score_when_unavailable(monkeypatch):
    # When the cross-encoder is unavailable, the floor pass returns the pool
    # untruncated with no logits; recency must fall back to the combined_score
    # (multiplicative) arm rather than crash or no-op.
    old = ScoredItem(_fake_user_playbook("old", created_at=1), 1.0)
    fresh = ScoredItem(_fake_user_playbook("fresh", created_at=4_102_444_800), 0.9)
    monkeypatch.setattr(uss, "_run_phase_a", lambda **_kw: ("q", None))
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([], [], [old, fresh]))

    def boom(query, docs):  # noqa: ARG001
        raise CrossEncoderUnavailableError("no model")

    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=boom,
    ):
        resp = uss.run_unified_search(
            request=UnifiedSearchRequest(query="q", user_id="u", top_k=2),
            org_id="o",
            storage=cast(BaseStorage, _FakeStorage()),
            llm_client=cast(LiteLLMClient, object()),
            prompt_manager=cast(PromptManager, object()),
            retrieval_floor=RetrievalFloorConfig(
                enabled=True, user_playbook_floor=-5.0
            ),
            recency=RecencyConfig(enabled=True, max_penalty_frac=1.0, pool_size=2),
        )

    assert [p.content for p in resp.user_playbooks] == ["fresh", "old"]


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
