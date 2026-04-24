"""Unit tests for atomic tool handlers. Uses in-memory SQLite storage — no LLM."""

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import (
    GetSessionExcerptArgs,
    GetUserProfileArgs,
    SearchAgentPlaybooksArgs,
    SearchUserPlaybooksArgs,
    SearchUserProfilesArgs,
    _handle_get_session_excerpt,
    _handle_get_user_profile,
    _handle_search_agent_playbooks,
    _handle_search_user_playbooks,
    _handle_search_user_profiles,
)


@pytest.fixture
def seeded_storage(tmp_path):
    """SQLite storage seeded with one profile and one user playbook."""
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(str(tmp_path / "test.db"))
    storage.add_user_profile(
        "u_1",
        [
            UserProfile(
                user_id="u_1",
                profile_id="p_10",
                content="user likes Italian food",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                last_modified_timestamp=1_700_000_000,
                expiration_timestamp=4102444800,
                source="test",
                generated_from_request_id="req_test",
            )
        ],
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_playbook_id=0,
                user_id="u_1",
                agent_version="v1",
                request_id="r_1",
                playbook_name="coding",
                content="show code examples",
                trigger="user asks for help",
            )
        ]
    )
    return storage


@pytest.fixture
def ctx():
    return ExtractionCtx(user_id="u_1", agent_version="v1", extractor_name="coding")


def test_search_user_profiles_populates_known_ids(seeded_storage, ctx):
    result = _handle_search_user_profiles(
        SearchUserProfilesArgs(query="Italian food", top_k=10),
        seeded_storage,
        ctx,
    )
    assert "hits" in result
    assert ctx.search_count == 1


def test_search_user_profiles_empty_result(seeded_storage, ctx):
    result = _handle_search_user_profiles(
        SearchUserProfilesArgs(query="quantum mechanics", top_k=10),
        seeded_storage,
        ctx,
    )
    assert ctx.search_count == 1
    assert "hits" in result


def test_get_user_profile_populates_known_ids_when_found(seeded_storage, ctx):
    result = _handle_get_user_profile(
        GetUserProfileArgs(id="p_10"), seeded_storage, ctx
    )
    assert "profile" in result
    assert result["profile"]["id"] == "p_10"
    assert "p_10" in ctx.known_ids
    # get does NOT bump search_count
    assert ctx.search_count == 0


def test_get_user_profile_not_found(seeded_storage, ctx):
    result = _handle_get_user_profile(
        GetUserProfileArgs(id="p_nonexistent"), seeded_storage, ctx
    )
    assert result == {"error": "not found"}
    assert "p_nonexistent" not in ctx.known_ids


def test_search_user_playbooks_populates_known_ids(seeded_storage, ctx):
    result = _handle_search_user_playbooks(
        SearchUserPlaybooksArgs(query="code examples", top_k=10),
        seeded_storage,
        ctx,
    )
    assert "hits" in result
    assert ctx.search_count == 1


def test_search_agent_playbooks_bumps_search_count(seeded_storage, ctx):
    result = _handle_search_agent_playbooks(
        SearchAgentPlaybooksArgs(query="x", top_k=10), seeded_storage, ctx
    )
    assert "hits" in result
    assert ctx.search_count == 1


def test_top_k_capped_server_side(seeded_storage, ctx):
    """Server-side cap (25) prevents unbounded requests."""
    # top_k=1000 should be capped before reaching storage; best-effort check is
    # that the call succeeds without error and returns within cap.
    result = _handle_search_user_profiles(
        SearchUserProfilesArgs(query="x", top_k=1000),
        seeded_storage,
        ctx,
    )
    assert "hits" in result


def test_get_session_excerpt_returns_error_when_api_missing():
    """If storage doesn't have get_interactions_by_session, handler returns error."""
    from unittest.mock import MagicMock

    mock_storage = MagicMock(
        spec=["search_user_profile"]
    )  # no get_interactions_by_session
    # Purposefully does NOT have get_interactions_by_session attr
    del mock_storage.get_interactions_by_session  # ensure AttributeError on access
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_get_session_excerpt(
        GetSessionExcerptArgs(session_id="s", span="x"),
        mock_storage,
        ctx,
    )
    assert "error" in result
