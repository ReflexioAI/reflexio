"""Contract: search never returns non-current rows.

Pins the invariant the unified-search pipeline relies on instead of runtime
filtering: TTL-expired profiles and soft-superseded rows (tombstone status +
``superseded_by``) are excluded by storage search itself, in every arm. Any
backend added to the parametrized ``storage`` fixture is covered.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus, ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
)
from reflexio.models.config_schema import SearchMode

pytestmark = pytest.mark.integration

_NOW = int(datetime.now(UTC).timestamp())
_DAY = 86_400
_USER = "u-currency"


def _profile(pid: str, *, expired: bool = False) -> UserProfile:
    last_modified = _NOW - 40 * _DAY
    return UserProfile(
        profile_id=pid,
        user_id=_USER,
        content=f"currency contract fact {pid}",
        last_modified_timestamp=last_modified,
        generated_from_request_id="r-currency",
        profile_time_to_live=(
            ProfileTimeToLive.ONE_MONTH if expired else ProfileTimeToLive.INFINITY
        ),
        expiration_timestamp=(last_modified + 30 * _DAY if expired else 4_102_444_800),
    )


def _search_profiles(storage) -> list[str]:
    hits = storage.search_user_profile(
        SearchUserProfileRequest(
            user_id=_USER,
            query="currency contract fact",
            top_k=10,
            threshold=0.0,
            search_mode=SearchMode.FTS,
        ),
        status_filter=[None],
    )
    return [p.profile_id for p in hits]


def test_search_excludes_ttl_expired_profiles(storage):
    storage.add_user_profile(_USER, [_profile("live"), _profile("dead", expired=True)])

    assert _search_profiles(storage) == ["live"]


def test_search_excludes_superseded_profiles_and_couples_status(storage):
    storage.add_user_profile(_USER, [_profile("successor"), _profile("predecessor")])
    superseded_ids = storage.supersede_profiles_by_ids(
        _USER, ["predecessor"], request_id="r-supersede"
    )
    assert superseded_ids == ["predecessor"]

    assert _search_profiles(storage) == ["successor"]


def test_search_excludes_superseded_user_playbooks(storage):
    playbooks = [
        UserPlaybook(
            user_id=_USER,
            agent_version="v1",
            request_id="r-currency",
            trigger="currency contract trigger",
            content=f"currency contract rule {name}",
            created_at=_NOW - 10 * _DAY,
        )
        for name in ("keep", "retire")
    ]
    storage.save_user_playbooks(playbooks)
    keep, retire = playbooks
    assert (
        storage.supersede_user_playbooks_by_ids(
            [retire.user_playbook_id], request_id="r-supersede"
        )
        == 1
    )

    hits = storage.search_user_playbooks(
        SearchUserPlaybookRequest(
            query="currency contract rule",
            user_id=_USER,
            top_k=10,
            threshold=0.0,
            search_mode=SearchMode.FTS,
            status_filter=None,  # unified search passes None → current-only
        ),
        None,
    )
    assert [p.user_playbook_id for p in hits] == [keep.user_playbook_id]


def test_search_excludes_superseded_agent_playbooks(storage):
    # APPROVED playbooks are protected from supersede; seed PENDING (still
    # searchable under the default APPROVED+PENDING allow-list).
    saved = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content=f"currency contract team rule {name}",
                created_at=_NOW - 10 * _DAY,
                playbook_status=PlaybookStatus.PENDING,
            )
            for name in ("keep", "retire")
        ]
    )
    keep, retire = saved
    assert (
        storage.supersede_agent_playbooks_by_ids(
            [retire.agent_playbook_id], request_id="r-supersede"
        )
        == 1
    )

    hits = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="currency contract team rule",
            top_k=10,
            threshold=0.0,
            search_mode=SearchMode.FTS,
            status_filter=[None],
            playbook_status_filter=[PlaybookStatus.APPROVED, PlaybookStatus.PENDING],
        ),
        None,
    )
    assert [p.agent_playbook_id for p in hits] == [keep.agent_playbook_id]
