"""Integration tests for Reflexio.rerank_user_profiles.

Uses real SQLite storage in a temp dir and the real cross-encoder model — slow
on first run (model download) but cached afterwards under
``~/.cache/huggingface/``. The model is a 22M-param MS-MARCO MiniLM, ~50 ms for
K=30 on CPU, so steady-state cost is small enough to keep these tests in the
default integration tier (no ``@skip_in_precommit``).
"""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import (
    NEVER_EXPIRES_TIMESTAMP,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import RerankUserProfilesRequest

pytestmark = pytest.mark.integration


_RELEVANT_CONTENTS = [
    "User loves Italian pasta and pizza",
    "User prefers spicy ramen with pork broth",
    "User is allergic to peanuts",
]
_IRRELEVANT_CONTENTS = [
    "User uses a Linux laptop for development",
    "User commutes by bicycle on weekdays",
    "User watches NBA games on Sunday evenings",
]


@pytest.fixture
def reflexio_with_seeded_profiles(tmp_path):
    """Reflexio instance with three relevant + three irrelevant profiles."""
    from reflexio.lib.reflexio_lib import Reflexio

    reflexio = Reflexio(org_id="rerank-test", storage_base_dir=str(tmp_path))
    storage = reflexio._get_storage()
    profiles = []
    for idx, content in enumerate(_RELEVANT_CONTENTS):
        profiles.append(
            UserProfile(
                user_id="u_rerank",
                profile_id=f"rel_{idx}",
                content=content,
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_000 + idx,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="req_test",
            )
        )
    for idx, content in enumerate(_IRRELEVANT_CONTENTS):
        profiles.append(
            UserProfile(
                user_id="u_rerank",
                profile_id=f"irr_{idx}",
                content=content,
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_100 + idx,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="req_test",
            )
        )
    storage.add_user_profile("u_rerank", profiles)
    return reflexio


def test_rerank_surfaces_relevant_profile_above_irrelevant(
    reflexio_with_seeded_profiles,
):
    """Cross-encoder must rank a food-related profile above unrelated profiles."""
    all_ids = [f"rel_{i}" for i in range(3)] + [f"irr_{i}" for i in range(3)]
    response = reflexio_with_seeded_profiles.rerank_user_profiles(
        RerankUserProfilesRequest(
            user_id="u_rerank",
            query="What food does the user enjoy?",
            profile_ids=all_ids,
            top_k=3,
        )
    )
    assert response.success is True
    ids = [p.profile_id for p in response.user_profiles]
    assert len(ids) == 3, f"top_k=3 should return 3 hits, got {ids!r}"
    assert ids[0].startswith("rel_"), (
        f"expected food-related profile at top, got id={ids[0]!r}; all={ids!r}"
    )


def test_rerank_silently_drops_unknown_ids(reflexio_with_seeded_profiles):
    """Unknown profile_ids must be dropped without error."""
    response = reflexio_with_seeded_profiles.rerank_user_profiles(
        RerankUserProfilesRequest(
            user_id="u_rerank",
            query="pasta",
            profile_ids=["rel_0", "does-not-exist", "neither-does-this"],
            top_k=10,
        )
    )
    assert {p.profile_id for p in response.user_profiles} == {"rel_0"}


def test_rerank_respects_top_k(reflexio_with_seeded_profiles):
    """top_k must cap the number of returned hits even with more candidates."""
    all_ids = [f"rel_{i}" for i in range(3)] + [f"irr_{i}" for i in range(3)]
    response = reflexio_with_seeded_profiles.rerank_user_profiles(
        RerankUserProfilesRequest(
            user_id="u_rerank",
            query="food preferences",
            profile_ids=all_ids,
            top_k=2,
        )
    )
    assert len(response.user_profiles) == 2


def test_rerank_empty_input_returns_empty(reflexio_with_seeded_profiles):
    """Empty profile_ids must short-circuit without calling the model."""
    response = reflexio_with_seeded_profiles.rerank_user_profiles(
        RerankUserProfilesRequest(
            user_id="u_rerank", query="anything", profile_ids=[], top_k=5
        )
    )
    assert response.success is True
    assert response.user_profiles == []
    assert response.msg == "No profile_ids provided"
