"""Native search keeps admitted FTS candidates until temporal ordering."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
)
from reflexio.models.config_schema import SearchMode
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.storage.storage_base import BaseStorage


@pytest.mark.integration
def test_native_fts_candidate_beyond_default_pool_reaches_temporal_ordering(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = int(datetime.now(UTC).timestamp())
    profiles = [
        UserProfile(
            profile_id=f"old-{index}",
            user_id="owned-user",
            content="package manager package manager package manager pip",
            last_modified_timestamp=now - (100 + index) * 86_400,
            generated_from_request_id="owned-request",
        )
        for index in range(29)
    ]
    profiles.append(
        UserProfile(
            profile_id="newest",
            user_id="owned-user",
            content="User selected uv as the package manager for all Python projects.",
            last_modified_timestamp=now - 86_400,
            generated_from_request_id="owned-request",
        )
    )
    storage.add_user_profile("owned-user", profiles, skip_embedding=True)
    retrieved = storage.search_user_profile(
        SearchUserProfileRequest(
            query="package manager",
            user_id="owned-user",
            top_k=30,
            search_mode=SearchMode.FTS,
        )
    )
    assert len(retrieved) == 30
    assert [profile.profile_id for profile in retrieved].index("newest") >= 20

    monkeypatch.setattr(
        uss,
        "_run_phase_a",
        lambda **_kw: (
            ReformulationResult(
                standalone_query="package manager", recency_dominant=True
            ),
            None,
            False,
        ),
    )
    monkeypatch.setattr(
        uss.session_seen_cache,
        "seen",
        lambda *_: frozenset(("profile", f"unrelated-{index}") for index in range(10)),
    )
    monkeypatch.setattr(uss.session_seen_cache, "record", lambda *_: None)
    result = uss.run_unified_search(
        request=UnifiedSearchRequest(
            query="latest package manager",
            user_id="owned-user",
            top_k=2,
            entity_types=["profiles"],
            search_mode=SearchMode.FTS,
            session_id="owned-session",
        ),
        org_id="contract_test",
        storage=storage,
        llm_client=MagicMock(),
        prompt_manager=MagicMock(),
    )
    assert result.success
    assert len(result.profiles) == 2
    assert result.profiles[0].profile_id == "newest"
