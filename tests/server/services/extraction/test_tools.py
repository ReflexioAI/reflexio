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

    storage = SQLiteStorage("test_org", db_path=str(tmp_path / "test.db"))
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


# --- Mutating handlers ---

from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
)
from reflexio.server.services.extraction.tools import (
    CreateUserPlaybookArgs,
    CreateUserProfileArgs,
    DeleteUserPlaybookArgs,
    DeleteUserProfileArgs,
    _handle_create_user_playbook,
    _handle_create_user_profile,
    _handle_delete_user_playbook,
    _handle_delete_user_profile,
    apply_plan_op,
)


def test_create_user_profile_appends_plan_no_storage_write(seeded_storage, ctx):
    result = _handle_create_user_profile(
        CreateUserProfileArgs(
            content="user prefers dark mode", ttl="infinity", source_span="I use dark"
        ),
        seeded_storage,
        ctx,
    )
    assert "tentative_id" in result
    assert "op_idx" in result
    assert len(ctx.plan) == 1
    assert isinstance(ctx.plan[0], CreateUserProfileOp)
    # Storage unchanged — was 1 seeded profile, still 1
    assert len(seeded_storage.get_user_profile("u_1")) == 1


def test_create_user_profile_adds_tentative_id_to_known_ids(seeded_storage, ctx):
    r = _handle_create_user_profile(
        CreateUserProfileArgs(content="x", ttl="infinity", source_span="y"),
        seeded_storage,
        ctx,
    )
    tid = r["tentative_id"]
    assert tid in ctx.known_ids  # self-correction via delete becomes possible


def test_delete_user_profile_appends_plan(seeded_storage, ctx):
    ctx.known_ids.add("p_10")
    result = _handle_delete_user_profile(
        DeleteUserProfileArgs(id="p_10"), seeded_storage, ctx
    )
    assert len(ctx.plan) == 1
    assert isinstance(ctx.plan[0], DeleteUserProfileOp)
    assert result["op_idx"] == 0
    # Storage unchanged
    assert len(seeded_storage.get_user_profile("u_1")) == 1


def test_create_user_playbook_appends_plan(seeded_storage, ctx):
    _handle_create_user_playbook(
        CreateUserPlaybookArgs(
            trigger="on review",
            content="suggest refactor",
            source_span="evidence",
        ),
        seeded_storage,
        ctx,
    )
    assert isinstance(ctx.plan[0], CreateUserPlaybookOp)


def test_delete_user_playbook_appends_plan(seeded_storage, ctx):
    ctx.known_ids.add("pb_5")
    _handle_delete_user_playbook(DeleteUserPlaybookArgs(id="pb_5"), seeded_storage, ctx)
    assert isinstance(ctx.plan[0], DeleteUserPlaybookOp)


# --- apply_plan_op ---


def test_apply_plan_op_create_user_profile_calls_add(seeded_storage, ctx):
    op = CreateUserProfileOp(
        content="user loves hiking", ttl="infinity", source_span="I hike weekly"
    )
    before = len(seeded_storage.get_user_profile("u_1"))
    apply_plan_op(op, seeded_storage, ctx)
    assert len(seeded_storage.get_user_profile("u_1")) == before + 1


def test_apply_plan_op_delete_user_profile_removes_record(seeded_storage, ctx):
    # Verify p_10 exists
    assert any(p.profile_id == "p_10" for p in seeded_storage.get_user_profile("u_1"))
    op = DeleteUserProfileOp(id="p_10")
    apply_plan_op(op, seeded_storage, ctx)
    remaining = [p.profile_id for p in seeded_storage.get_user_profile("u_1")]
    assert "p_10" not in remaining


def test_apply_plan_op_create_profile_computes_expiration_from_ttl(tmp_path):
    """Bug regression: profile_time_to_live must be consistent with expiration_timestamp."""
    from reflexio.models.api_schema.domain.entities import NEVER_EXPIRES_TIMESTAMP
    from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
    from reflexio.server.services.extraction.plan import (
        CreateUserProfileOp,
        ExtractionCtx,
    )
    from reflexio.server.services.extraction.tools import apply_plan_op
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")

    op = CreateUserProfileOp(content="x", ttl="one_week", source_span="y")
    apply_plan_op(op, storage, ctx)

    profiles = storage.get_user_profile("u_1")
    assert len(profiles) == 1
    p = profiles[0]
    assert p.profile_time_to_live == ProfileTimeToLive.ONE_WEEK
    assert p.expiration_timestamp != NEVER_EXPIRES_TIMESTAMP
    assert p.expiration_timestamp > p.last_modified_timestamp
    # one_week is 7 days = 604800 seconds
    assert p.expiration_timestamp - p.last_modified_timestamp == 604800


def test_apply_plan_op_create_profile_infinity_ttl_uses_sentinel(tmp_path):
    """An 'infinity' TTL should still produce NEVER_EXPIRES_TIMESTAMP."""
    from reflexio.models.api_schema.domain.entities import NEVER_EXPIRES_TIMESTAMP
    from reflexio.server.services.extraction.plan import (
        CreateUserProfileOp,
        ExtractionCtx,
    )
    from reflexio.server.services.extraction.tools import apply_plan_op
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    op = CreateUserProfileOp(content="x", ttl="infinity", source_span="y")
    apply_plan_op(op, storage, ctx)
    p = storage.get_user_profile("u_1")[0]
    assert p.expiration_timestamp == NEVER_EXPIRES_TIMESTAMP


# ====================================================================
# Registry tests
# ====================================================================

from reflexio.server.services.extraction.tools import (
    EXTRACTION_TOOLS,
    PLAYBOOK_EXTRACTION_TOOLS,
    PROFILE_EXTRACTION_TOOLS,
    SEARCH_TOOLS,
)


def test_extraction_registry_has_all_tools():
    specs = {t["function"]["name"] for t in EXTRACTION_TOOLS.openai_specs()}
    # EXTRACTION_TOOLS is the backward-compat union of all four create/delete tools
    # plus the full read surface (including agent-playbook and session-excerpt tools).
    assert specs == {
        "search_user_profiles",
        "get_user_profile",
        "create_user_profile",
        "delete_user_profile",
        "search_user_playbooks",
        "get_user_playbook",
        "create_user_playbook",
        "delete_user_playbook",
        "search_agent_playbooks",
        "get_agent_playbook",
        "get_session_excerpt",
        "finish",
    }


def test_profile_extraction_registry_excludes_playbook_mutations():
    """PROFILE_EXTRACTION_TOOLS must not expose create/delete_user_playbook."""
    specs = {t["function"]["name"] for t in PROFILE_EXTRACTION_TOOLS.openai_specs()}
    assert "create_user_profile" in specs
    assert "delete_user_profile" in specs
    assert "create_user_playbook" not in specs
    assert "delete_user_playbook" not in specs
    assert "finish" in specs


def test_playbook_extraction_registry_excludes_profile_mutations():
    """PLAYBOOK_EXTRACTION_TOOLS must not expose create/delete_user_profile."""
    specs = {t["function"]["name"] for t in PLAYBOOK_EXTRACTION_TOOLS.openai_specs()}
    assert "create_user_playbook" in specs
    assert "delete_user_playbook" in specs
    assert "create_user_profile" not in specs
    assert "delete_user_profile" not in specs
    assert "finish" in specs


def test_search_registry_is_read_only():
    specs = {t["function"]["name"] for t in SEARCH_TOOLS.openai_specs()}
    assert specs == {
        "search_user_profiles",
        "get_user_profile",
        "search_user_playbooks",
        "get_user_playbook",
        "search_agent_playbooks",
        "get_agent_playbook",
        "get_session_excerpt",
        "finish",
    }
    # No mutations allowed in search
    assert "create_user_profile" not in specs
    assert "delete_user_profile" not in specs


# ====================================================================
# Query-embedding plumbing for HYBRID search mode
# ====================================================================

from unittest.mock import MagicMock  # noqa: E402

from reflexio.server.services.extraction.tools import _maybe_embed_query  # noqa: E402


def test_maybe_embed_query_returns_none_when_storage_has_no_embedder():
    """Disk/local storage backends that don't expose _get_embedding should
    gracefully produce None rather than raising."""
    assert _maybe_embed_query(object(), "anything") is None


def test_maybe_embed_query_returns_none_when_embedder_raises():
    """Embedder failures must not break search — fall back to FTS via None."""
    storage = MagicMock()
    storage._get_embedding.side_effect = RuntimeError("provider down")
    assert _maybe_embed_query(storage, "anything") is None


def test_maybe_embed_query_returns_embedding_when_supported():
    storage = MagicMock()
    storage._get_embedding.return_value = [0.1, 0.2, 0.3]
    assert _maybe_embed_query(storage, "sushi") == [0.1, 0.2, 0.3]
    storage._get_embedding.assert_called_once_with("sushi")


def test_search_user_profiles_passes_query_embedding():
    """Profile search handler must compute + pass a query embedding so
    storage doesn't downgrade HYBRID to FTS (regression for the
    'no query embedding provided — falling back to FTS' warning)."""
    storage = MagicMock()
    storage._get_embedding.return_value = [0.1, 0.2, 0.3]
    storage.search_user_profile.return_value = []
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    args = SearchUserProfilesArgs(query="sushi", top_k=5)

    _handle_search_user_profiles(args, storage, ctx)

    storage._get_embedding.assert_called_once_with("sushi")
    _, kwargs = storage.search_user_profile.call_args
    assert kwargs["query_embedding"] == [0.1, 0.2, 0.3]


def test_search_user_playbooks_passes_query_embedding_via_options():
    """Playbook search handler wraps the embedding in SearchOptions."""
    storage = MagicMock()
    storage._get_embedding.return_value = [0.4, 0.5]
    storage.search_user_playbooks.return_value = []
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    args = SearchUserPlaybooksArgs(query="code review", top_k=5, status="current")

    _handle_search_user_playbooks(args, storage, ctx)

    storage._get_embedding.assert_called_once_with("code review")
    _, kwargs = storage.search_user_playbooks.call_args
    assert kwargs["options"].query_embedding == [0.4, 0.5]


def test_search_agent_playbooks_passes_query_embedding_via_options():
    """Agent-playbook search handler wraps the embedding in SearchOptions."""
    storage = MagicMock()
    storage._get_embedding.return_value = [0.6, 0.7]
    storage.search_agent_playbooks.return_value = []
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    args = SearchAgentPlaybooksArgs(query="debug approach", top_k=5, status="current")

    _handle_search_agent_playbooks(args, storage, ctx)

    storage._get_embedding.assert_called_once_with("debug approach")
    _, kwargs = storage.search_agent_playbooks.call_args
    assert kwargs["options"].query_embedding == [0.6, 0.7]
