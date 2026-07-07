"""Unit tests for the wants_current handling (Phase 3 temporal enrichment)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import DeepSearchConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.search.deep_search_schemas import (
    PlannedSubquery,
    ReflectVerdict,
    SearchPlan,
)
from reflexio.server.services.search.deep_search_service import (
    AgenticUnifiedSearchService,
    _filter_current,
    _freshness_collapse,
)
from reflexio.server.services.search.executor import Candidate

_NOW = int(datetime.now(UTC).timestamp())
_DAY = 86_400


def _profile_candidate(
    pid: str,
    age_days: int,
    *,
    content: str | None = None,
    superseded_by: str | None = None,
    expiration_timestamp: int | None = None,
) -> Candidate:
    entity = UserProfile(
        profile_id=pid,
        user_id="u1",
        content=content or f"content-{pid}",
        last_modified_timestamp=_NOW - age_days * _DAY,
        generated_from_request_id="r1",
        superseded_by=superseded_by,
    )
    if expiration_timestamp is not None:
        entity.expiration_timestamp = expiration_timestamp
    return Candidate(key=f"P:{pid}", arm="profiles", entity=entity)


def _playbook_candidate(
    upid: int, age_days: int, *, trigger: str, content: str
) -> Candidate:
    entity = UserPlaybook(
        user_playbook_id=upid,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        trigger=trigger,
        content=content,
        created_at=_NOW - age_days * _DAY,
    )
    return Candidate(key=f"UP:{upid}", arm="user_playbooks", entity=entity)


# ---------------------------------------------------------------------------
# _filter_current
# ---------------------------------------------------------------------------


def test_filter_current_drops_superseded_and_expired():
    live = _profile_candidate("live", 5)
    superseded = _profile_candidate("old", 50, superseded_by="live")
    expired = _profile_candidate("gone", 50, expiration_timestamp=_NOW - _DAY)
    kept = _filter_current([live, superseded, expired], _NOW)
    assert [c.key for c in kept] == ["P:live"]


def test_filter_current_keeps_never_expiring_and_no_expiry_entities():
    profile = _profile_candidate("p", 5)  # NEVER_EXPIRES default
    playbook = _playbook_candidate(1, 5, trigger="t", content="c")
    kept = _filter_current([profile, playbook], _NOW)
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# _freshness_collapse
# ---------------------------------------------------------------------------


def test_freshness_collapse_promotes_fresh_near_duplicate():
    stale = _playbook_candidate(
        1, 200, trigger="user says ship", content="Skip tests and deploy immediately."
    )
    fresh = _playbook_candidate(
        2, 5, trigger="user says ship", content="Run tests then deploy."
    )
    unrelated = _playbook_candidate(
        3, 100, trigger="code review", content="Require two approvals."
    )
    # Ranked order puts the stale near-duplicate first (the failure mode).
    collapsed = _freshness_collapse([stale, fresh, unrelated])
    assert [c.key for c in collapsed] == ["UP:2", "UP:1", "UP:3"]


def test_freshness_collapse_leaves_distinct_facts_alone():
    a = _profile_candidate("a", 90, content="User prefers postgres for OLTP work.")
    b = _profile_candidate("b", 1, content="User plays tennis on weekends.")
    collapsed = _freshness_collapse([a, b])
    assert [c.key for c in collapsed] == ["P:a", "P:b"]


def test_freshness_collapse_group_anchored_at_best_rank():
    unrelated_top = _profile_candidate("top", 10, content="User is a backend engineer.")
    stale = _profile_candidate(
        "stale", 90, content="User uses pip for Python package management."
    )
    fresh = _profile_candidate(
        "fresh", 2, content="User switched to uv for Python package management."
    )
    collapsed = _freshness_collapse([unrelated_top, stale, fresh])
    # The duplicate group stays anchored where its best-ranked member was
    # (position 2), internally newest-first; the unrelated leader is untouched.
    assert [c.key for c in collapsed] == ["P:top", "P:fresh", "P:stale"]


# ---------------------------------------------------------------------------
# wants_current end-to-end through the service.
# ---------------------------------------------------------------------------


def test_wants_current_arm_prefers_fresh_duplicate_despite_reflect_order():
    stale = UserPlaybook(
        user_playbook_id=1,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        trigger="user says ship",
        content="Skip tests and deploy immediately.",
        created_at=_NOW - 200 * _DAY,
    )
    fresh = UserPlaybook(
        user_playbook_id=2,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        trigger="user says ship",
        content="Run tests then deploy.",
        created_at=_NOW - 5 * _DAY,
    )
    storage = MagicMock()
    storage.supports_embedding = False
    storage.search_user_playbooks.return_value = [stale, fresh]
    storage.search_user_profile.return_value = []
    storage.search_agent_playbooks.return_value = []
    storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}

    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(
                arm="user_playbooks", query="do we skip tests?", wants_current=True
            )
        ]
    )
    # Reflect (wrongly) ranks the stale rule first — wants_current must fix it.
    verdict = ReflectVerdict(
        sufficiency="sufficient", ranked_candidate_ids=["UP:1", "UP:2"]
    )
    llm = MagicMock()
    llm.generate_chat_response.side_effect = [plan, verdict]
    ctx = MagicMock()
    ctx.storage = storage
    ctx.prompt_manager = PromptManager()
    service = AgenticUnifiedSearchService(
        llm_client=llm, request_context=ctx, config=DeepSearchConfig()
    )

    response = service.search(
        UnifiedSearchRequest(
            query="do we skip tests on ship?", user_id="u1", search_depth="deep"
        ),
        org_id="org",
    )

    assert [p.user_playbook_id for p in response.user_playbooks] == [2, 1]
