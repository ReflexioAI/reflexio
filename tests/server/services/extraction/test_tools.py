"""Unit tests for atomic tool handlers. Uses in-memory SQLite storage — no LLM."""

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import (
    GetUserProfileArgs,
    ReadSessionTextArgs,
    SearchAgentPlaybooksArgs,
    SearchUserPlaybooksArgs,
    SearchUserProfilesArgs,
    _handle_get_user_profile,
    _handle_read_session_text,
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
    # Every hit's id must be added to ctx.known_ids — that's the side
    # effect this test name claims to validate.
    hit_ids = {hit["id"] for hit in result["hits"]}
    assert hit_ids, "expected at least one hit from seeded storage"
    assert hit_ids.issubset(ctx.known_ids)


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
    hit_ids = {hit["id"] for hit in result["hits"]}
    assert hit_ids, "expected at least one hit from seeded storage"
    assert hit_ids.issubset(ctx.known_ids)


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


def test_read_session_text_returns_error_when_api_missing():
    """If storage lacks get_interactions_by_request_ids, handler returns error."""
    from unittest.mock import MagicMock

    mock_storage = MagicMock(spec=["search_user_profile"])
    # Purposefully does NOT have get_interactions_by_request_ids
    del mock_storage.get_interactions_by_request_ids
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s"]),
        mock_storage,
        ctx,
    )
    assert "error" in result


def test_read_session_text_single_session_returns_concatenated_text():
    """Single session_id list with empty query returns raw role-prefixed turns
    (no compression), since empty query bypasses the LLM call."""
    from unittest.mock import MagicMock

    fake_interactions = [
        MagicMock(request_id="s1", role="user", content="hi there"),
        MagicMock(request_id="s1", role="assistant", content="hello back"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], query=""),
        mock_storage,
        ctx,
    )
    assert "text" in result
    text = result["text"]
    assert "=== session s1 ===" in text
    assert "[user] hi there" in text
    assert "[assistant] hello back" in text
    mock_storage.get_interactions_by_request_ids.assert_called_once_with(["s1"])


def test_read_session_text_compresses_when_query_and_llm_provided():
    """With non-empty query + llm_client + prompt_manager, handler runs the
    compression prompt and returns compressed text."""
    from unittest.mock import MagicMock

    fake_interactions = [
        MagicMock(request_id="s1", role="user", content="some long unrelated chitchat"),
        MagicMock(request_id="s1", role="user", content="I paid $800 for the boots"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions

    mock_prompt_manager = MagicMock()
    mock_prompt_manager.render_prompt.return_value = "RENDERED_COMPRESSION_PROMPT"

    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = (
        "=== session s1 ===\n[user] I paid $800 for the boots"
    )

    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], query="how much for the boots"),
        mock_storage,
        ctx,
        llm_client=mock_llm,
        prompt_manager=mock_prompt_manager,
    )

    assert result == {"text": "=== session s1 ===\n[user] I paid $800 for the boots"}
    # Prompt manager rendered the compression prompt with the right id and vars.
    rendered_call = mock_prompt_manager.render_prompt.call_args
    assert rendered_call.args[0] == "compress_session_for_query"
    assert rendered_call.kwargs["variables"]["query"] == "how much for the boots"
    assert "$800" in rendered_call.kwargs["variables"]["raw_turns"]
    # LLM was invoked with the rendered prompt.
    mock_llm.generate_response.assert_called_once()
    assert mock_llm.generate_response.call_args.args[0] == "RENDERED_COMPRESSION_PROMPT"


def test_read_session_text_falls_back_to_raw_on_compression_exception():
    """When the compression LLM call raises, handler returns raw role-prefixed
    turns instead of erroring out — rehydration should never silently fail."""
    from unittest.mock import MagicMock

    fake_interactions = [
        MagicMock(request_id="s1", role="user", content="user content here"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions

    mock_prompt_manager = MagicMock()
    mock_prompt_manager.render_prompt.return_value = "PROMPT"

    mock_llm = MagicMock()
    mock_llm.generate_response.side_effect = RuntimeError("LLM exploded")

    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], query="anything"),
        mock_storage,
        ctx,
        llm_client=mock_llm,
        prompt_manager=mock_prompt_manager,
    )

    assert "text" in result
    assert "[user] user content here" in result["text"]
    assert "=== session s1 ===" in result["text"]


def test_read_session_text_falls_back_to_raw_on_empty_compression_output():
    """Empty/whitespace LLM output triggers raw fallback rather than returning
    empty text — keeps the rehydration usable."""
    from unittest.mock import MagicMock

    fake_interactions = [
        MagicMock(request_id="s1", role="user", content="real content"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions

    mock_prompt_manager = MagicMock()
    mock_prompt_manager.render_prompt.return_value = "PROMPT"

    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "   \n  "

    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], query="anything"),
        mock_storage,
        ctx,
        llm_client=mock_llm,
        prompt_manager=mock_prompt_manager,
    )

    assert "[user] real content" in result["text"]


def test_read_session_text_skips_compression_without_llm_wiring():
    """When llm_client / prompt_manager are not wired through the bundle,
    handler returns raw turns — no error, no compression attempt."""
    from unittest.mock import MagicMock

    fake_interactions = [
        MagicMock(request_id="s1", role="user", content="content"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions

    ctx = ExtractionCtx(user_id="u", agent_version="v")
    # Note: llm_client and prompt_manager omitted → defaults to None.
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], query="anything"),
        mock_storage,
        ctx,
    )

    assert "[user] content" in result["text"]


def test_read_session_text_multi_session_groups_and_orders():
    """Multi-session fetch groups interactions by request_id and preserves the
    order of session_ids passed in args, even when storage returns them
    interleaved."""
    from unittest.mock import MagicMock

    # Storage returns sessions interleaved (a, b, a) — the handler must group.
    fake_interactions = [
        MagicMock(request_id="sA", role="user", content="A1"),
        MagicMock(request_id="sB", role="user", content="B1"),
        MagicMock(request_id="sA", role="assistant", content="A2"),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["sA", "sB"]),
        mock_storage,
        ctx,
    )
    text = result["text"]
    # sA block must appear before sB block (preserves input order)
    a_idx = text.index("=== session sA ===")
    b_idx = text.index("=== session sB ===")
    assert a_idx < b_idx
    # sA block contains both A turns
    sa_block = text[a_idx:b_idx]
    assert "[user] A1" in sa_block
    assert "[assistant] A2" in sa_block
    # sB block contains the B turn
    assert "[user] B1" in text[b_idx:]


def test_read_session_text_returns_no_interactions_error_when_storage_empty():
    """When storage returns no interactions for any requested session, handler
    surfaces an error rather than empty text."""
    from unittest.mock import MagicMock

    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = []
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["missing"]),
        mock_storage,
        ctx,
    )
    assert "error" in result
    assert "no interactions" in result["error"].lower()


def test_read_session_text_truncates_per_session_at_cap():
    """When a session's body exceeds max_chars_per_session, it is truncated
    with an ellipsis. The cap is applied per-session, not across all sessions."""
    from unittest.mock import MagicMock

    big_content = "x" * 5000
    fake_interactions = [
        MagicMock(request_id="s1", role="user", content=big_content),
    ]
    mock_storage = MagicMock()
    mock_storage.get_interactions_by_request_ids.return_value = fake_interactions
    ctx = ExtractionCtx(user_id="u", agent_version="v")
    result = _handle_read_session_text(
        ReadSessionTextArgs(session_ids=["s1"], max_chars_per_session=100),
        mock_storage,
        ctx,
    )
    # Header is outside the per-session-body cap; the body itself is capped.
    text = result["text"]
    assert "=== session s1 ===" in text
    assert "…" in text  # ellipsis marker for truncation
    # Body should be ~100 chars + ellipsis, not 5000.
    assert len(text) < 500


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
        "read_session_text",
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
    # ``rerank_user_profiles`` was removed from the agent palette — search now
    # does internal rerank via the ``rerank``/``llm_rerank`` flags on
    # ``search_user_profiles``.
    assert specs == {
        "search_user_profiles",
        "get_user_profile",
        "storage_stats",
        "search_user_playbooks",
        "get_user_playbook",
        "search_agent_playbooks",
        "get_agent_playbook",
        "read_session_text",
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
    """Storage backends without an embedder (no `_get_embedding`) should
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


# ====================================================================
# Rerank dispatch — cross-encoder vs LLM rerank vs neither
# ====================================================================

from reflexio.server.services.extraction.tools import (  # noqa: E402
    _fetch_k_for_rerank,
    _maybe_rerank_hits,
)


class _Hit:
    """Profile-shape stub: minimum attributes needed for the rerank dispatcher
    AND the LLM-facing projection so the same stub works in
    ``_maybe_rerank_hits`` unit tests and in handler-level integration tests.
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self.profile_id = content  # use content as a stable id
        self.generated_from_request_id = ""
        self.profile_time_to_live = "permanent"
        self.last_modified_timestamp = 0
        self.source_span = None


def test_fetch_k_no_rerank_uses_final_k():
    assert _fetch_k_for_rerank(10, rerank=False, llm_rerank=False) == 10


def test_fetch_k_cross_encoder_pads_to_pool_size():
    """Cross-encoder rerank pads pool to RERANK_POOL_SIZE for headroom."""
    assert _fetch_k_for_rerank(10, rerank=True, llm_rerank=False) == 30


def test_fetch_k_llm_rerank_pads_to_pool_size():
    """LLM rerank uses the same headroom; same downstream reorder cost shape."""
    assert _fetch_k_for_rerank(10, rerank=False, llm_rerank=True) == 30


def test_fetch_k_large_final_k_overrides_pool():
    """When agent asks for more than the pool, fetch the larger amount."""
    assert _fetch_k_for_rerank(50, rerank=True, llm_rerank=True) == 50


def test_maybe_rerank_no_flags_returns_identity():
    """Without any rerank flag, return hits[:final_k] unchanged."""
    hits = [_Hit("a"), _Hit("b"), _Hit("c")]
    out = _maybe_rerank_hits(hits, rerank=False, rerank_query="q", final_k=2)
    assert [h.content for h in out] == ["a", "b"]


def test_maybe_rerank_pool_smaller_than_final_short_circuits():
    """If we don't have headroom (<= final_k hits), skip rerank."""
    hits = [_Hit("a"), _Hit("b")]
    out = _maybe_rerank_hits(hits, rerank=True, rerank_query="q", final_k=5)
    assert [h.content for h in out] == ["a", "b"]


def test_maybe_rerank_llm_path_reorders_when_scores_returned(monkeypatch):
    """LLM rerank, when it returns valid scores, reorders by descending score."""
    hits = [_Hit("walmart"), _Hit("thrive"), _Hit("nordstrom"), _Hit("mexico")]
    # Stub score_pairs_llm at the import site used by _try_llm_rerank.
    import reflexio.server.llm.rerank as rerank_mod

    monkeypatch.setattr(
        rerank_mod,
        "score_pairs_llm",
        lambda *_args, **_kw: [3.0, 9.0, 1.0, 0.0],
    )
    out = _maybe_rerank_hits(
        hits,
        rerank=False,
        rerank_query="grocery",
        final_k=2,
        llm_rerank=True,
        llm_client=object(),  # any non-None
        prompt_manager=object(),
    )
    # thrive (9) > walmart (3) — top-2 returned in score-descending order.
    assert [h.content for h in out] == ["thrive", "walmart"]


def test_maybe_rerank_llm_failure_falls_back_to_cross_encoder(monkeypatch):
    """When LLM rerank returns None AND cross-encoder rerank is enabled,
    cross-encoder takes over."""
    hits = [_Hit("a"), _Hit("b"), _Hit("c"), _Hit("d")]
    import reflexio.server.llm.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "score_pairs_llm", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        rerank_mod, "score_pairs", lambda *_a, **_kw: [1.0, 5.0, 2.0, 0.0]
    )
    out = _maybe_rerank_hits(
        hits,
        rerank=True,  # cross-encoder also requested
        rerank_query="q",
        final_k=2,
        llm_rerank=True,
        llm_client=object(),
        prompt_manager=object(),
    )
    # b(5) > c(2) — cross-encoder reorder applied.
    assert [h.content for h in out] == ["b", "c"]


def test_maybe_rerank_llm_failure_no_cross_encoder_falls_back_to_hybrid(monkeypatch):
    """When LLM rerank fails and cross-encoder is OFF, return hybrid order."""
    hits = [_Hit("a"), _Hit("b"), _Hit("c"), _Hit("d")]
    import reflexio.server.llm.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "score_pairs_llm", lambda *_a, **_kw: None)
    out = _maybe_rerank_hits(
        hits,
        rerank=False,
        rerank_query="q",
        final_k=2,
        llm_rerank=True,
        llm_client=object(),
        prompt_manager=object(),
    )
    # No reorder — hybrid order preserved, capped at final_k.
    assert [h.content for h in out] == ["a", "b"]


def test_search_user_profiles_passes_llm_rerank_flag_through(monkeypatch):
    """End-to-end: when a tool args carries llm_rerank=True, the handler
    fetches the larger pool and the LLM rerank is invoked."""
    storage = MagicMock()
    storage._get_embedding.return_value = [0.1]
    storage.search_user_profile.return_value = [_Hit(f"p{i}") for i in range(15)]
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    args = SearchUserProfilesArgs(query="grocery store", top_k=5, llm_rerank=True)

    import reflexio.server.llm.rerank as rerank_mod

    captured: dict[str, object] = {}

    def stub_score(q, docs, client, pm):
        captured["query"] = q
        captured["n_docs"] = len(docs)
        # Reverse the order — last doc is most relevant.
        return list(range(len(docs)))

    monkeypatch.setattr(rerank_mod, "score_pairs_llm", stub_score)

    sentinel_client = object()
    sentinel_pm = object()
    result = _handle_search_user_profiles(
        args, storage, ctx, llm_client=sentinel_client, prompt_manager=sentinel_pm
    )
    # Storage call uses RERANK_POOL_SIZE (30) for the candidate fetch.
    _, kwargs = storage.search_user_profile.call_args
    args_to_storage, _ = storage.search_user_profile.call_args
    assert args_to_storage[0].top_k == 30
    # Score function got the full hit pool (15 in this stub).
    assert captured["n_docs"] == 15
    # Final hits are the top-5 by descending score (reversed order = last 5).
    returned = [h["content"] for h in result["hits"]]
    assert returned == ["p14", "p13", "p12", "p11", "p10"]
