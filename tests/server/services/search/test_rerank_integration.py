"""Integration tests for the cross-encoder rerank tool + Reflexio.rerank_user_profiles.

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
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import (
    RerankUserProfilesArgs,
    _handle_rerank_user_profiles,
)

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
def seeded_storage(tmp_path):
    """SQLite storage with three relevant + three irrelevant profiles."""
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id="rerank-test", db_path=str(tmp_path / "rerank.db"))
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
    return storage


@pytest.fixture
def ctx():
    return ExtractionCtx(user_id="u_rerank", agent_version="v1")


def test_rerank_handler_surfaces_relevant_profile_above_irrelevant(seeded_storage, ctx):
    """Cross-encoder must rank a food-related profile above unrelated profiles."""
    all_ids = [f"rel_{i}" for i in range(3)] + [f"irr_{i}" for i in range(3)]
    result = _handle_rerank_user_profiles(
        RerankUserProfilesArgs(
            query="What food does the user enjoy?",
            profile_ids=all_ids,
            top_k=3,
        ),
        seeded_storage,
        ctx,
    )
    hit_ids = [hit["id"] for hit in result["hits"]]
    assert len(hit_ids) == 3, f"top_k=3 should return 3 hits, got {hit_ids!r}"
    # The top hit should be one of the food-related profiles.
    assert hit_ids[0].startswith("rel_"), (
        f"expected food-related profile at top, got id={hit_ids[0]!r}; all={hit_ids!r}"
    )
    # The handler bumps search_count for budgeting parity with search.
    assert ctx.search_count == 1


def test_rerank_handler_silently_drops_unknown_ids(seeded_storage, ctx):
    """Unknown profile_ids must be dropped without error."""
    result = _handle_rerank_user_profiles(
        RerankUserProfilesArgs(
            query="pasta",
            profile_ids=["rel_0", "does-not-exist", "neither-does-this"],
            top_k=10,
        ),
        seeded_storage,
        ctx,
    )
    hit_ids = {hit["id"] for hit in result["hits"]}
    assert hit_ids == {"rel_0"}


def test_rerank_handler_respects_top_k(seeded_storage, ctx):
    """top_k must cap the number of returned hits even with more candidates."""
    all_ids = [f"rel_{i}" for i in range(3)] + [f"irr_{i}" for i in range(3)]
    result = _handle_rerank_user_profiles(
        RerankUserProfilesArgs(
            query="food preferences",
            profile_ids=all_ids,
            top_k=2,
        ),
        seeded_storage,
        ctx,
    )
    assert len(result["hits"]) == 2


def test_rerank_handler_empty_input_returns_empty(seeded_storage, ctx):
    """Empty profile_ids must short-circuit without calling the model."""
    result = _handle_rerank_user_profiles(
        RerankUserProfilesArgs(query="anything", profile_ids=[], top_k=5),
        seeded_storage,
        ctx,
    )
    assert result == {"hits": []}
    assert ctx.search_count == 1


def test_reflexio_rerank_user_profiles_returns_response(tmp_path):
    """The Reflexio facade method should wire request -> handler -> response."""
    from reflexio.lib.reflexio_lib import Reflexio

    reflexio = Reflexio(org_id="rerank-facade", storage_base_dir=str(tmp_path))
    storage = reflexio._get_storage()
    storage.add_user_profile(
        "u_facade",
        [
            UserProfile(
                user_id="u_facade",
                profile_id="food",
                content="user loves Italian pasta",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_000,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="req",
            ),
            UserProfile(
                user_id="u_facade",
                profile_id="commute",
                content="user bikes to work",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_001,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source="test",
                generated_from_request_id="req",
            ),
        ],
    )

    response = reflexio.rerank_user_profiles(
        RerankUserProfilesRequest(
            user_id="u_facade",
            query="What food does the user like?",
            profile_ids=["food", "commute"],
            top_k=2,
        )
    )
    assert response.success is True
    ids = [p.profile_id for p in response.user_profiles]
    assert ids[0] == "food", f"expected food profile first, got {ids!r}"
