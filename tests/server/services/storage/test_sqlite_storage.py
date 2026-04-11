import hashlib
import json
import tempfile
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    ProfileTimeToLive,
    StructuredData,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode, SearchOptions
from reflexio.server.services.storage.sqlite_storage import (
    SQLiteStorage,
    _cosine_similarity,
    _effective_search_mode,
    _sanitize_fts_query,
    _true_rrf_merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0]),
    ):
        yield SQLiteStorage(org_id="0", db_path=f"{temp_dir}/reflexio.db")


# ---------------------------------------------------------------------------
# _sanitize_fts_query tests
# ---------------------------------------------------------------------------


class TestSanitizeFtsQuery:
    def test_bare_tokens_not_quoted(self):
        result = _sanitize_fts_query("user login")
        assert '"' not in result
        assert "user" in result
        assert "login" in result.replace("*", "")

    def test_or_default_between_tokens(self):
        result = _sanitize_fts_query("user login problem")
        assert "OR" in result

    def test_explicit_or_preserved(self):
        result = _sanitize_fts_query("agent failed OR error")
        assert "OR" in result
        # Should not add extra ORs when explicit operators exist
        assert result.count("OR") == 1

    def test_explicit_and_preserved(self):
        result = _sanitize_fts_query("agent AND error")
        assert "AND" in result

    def test_explicit_not_preserved(self):
        result = _sanitize_fts_query("agent NOT error")
        assert "NOT" in result

    def test_prefix_matching_on_last_token(self):
        result = _sanitize_fts_query("user login")
        assert result.endswith("*")

    def test_empty_input(self):
        assert _sanitize_fts_query("") == '""'

    def test_special_chars_stripped(self):
        result = _sanitize_fts_query('hello "world" (test)')
        assert '"' not in result or result == '""'
        assert "(" not in result
        assert ")" not in result

    def test_near_reserved_word_skipped(self):
        result = _sanitize_fts_query("agent NEAR error")
        assert "NEAR" not in result

    def test_leading_operator_skipped(self):
        result = _sanitize_fts_query("OR agent error")
        assert not result.startswith("OR")

    def test_consecutive_operators_collapsed(self):
        result = _sanitize_fts_query("agent OR AND error")
        # Should not have consecutive operators
        tokens = result.split()
        for i in range(len(tokens) - 1):
            assert not (
                tokens[i] in {"OR", "AND", "NOT"}
                and tokens[i + 1] in {"OR", "AND", "NOT"}
            )

    def test_trailing_operator_stripped(self):
        result = _sanitize_fts_query("agent OR")
        assert not result.endswith("OR")

    def test_apostrophe_in_query(self):
        result = _sanitize_fts_query("don't stop")
        assert "'" not in result
        assert "don" in result
        assert "stop" in result.replace("*", "")

    def test_only_apostrophe(self):
        assert _sanitize_fts_query("'") == '""'


# ---------------------------------------------------------------------------
# Stemming end-to-end
# ---------------------------------------------------------------------------


def test_stemming_works_end_to_end(storage):
    """Insert playbook with 'logging errors', search 'logged error', verify match."""
    from reflexio.models.api_schema.service_schemas import StructuredData

    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="logging errors in production",
                structured_data=StructuredData(
                    trigger="when the system encounters logging errors",
                ),
            )
        ]
    )

    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="logged error", top_k=10)
    )
    assert len(results) >= 1
    assert "logging" in results[0].structured_data.trigger.lower()


# ---------------------------------------------------------------------------
# OR recall
# ---------------------------------------------------------------------------


def test_or_recall_returns_multiple_matches(storage):
    """Search multi-term query, verify results matching any term appear."""
    from reflexio.models.api_schema.service_schemas import StructuredData

    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="authentication failed",
                structured_data=StructuredData(
                    trigger="when user authentication fails",
                ),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="timeout occurred",
                structured_data=StructuredData(
                    trigger="when request timeout occurs",
                ),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="unrelated playbook",
                structured_data=StructuredData(
                    trigger="when something unrelated happens",
                ),
            ),
        ]
    )

    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="authentication timeout", top_k=10)
    )
    # Should match both authentication and timeout playbooks
    assert len(results) >= 2
    contents = {r.content for r in results}
    assert "authentication failed" in contents
    assert "timeout occurred" in contents


# ---------------------------------------------------------------------------
# SQL filter pushdown
# ---------------------------------------------------------------------------


def test_search_user_playbooks_with_sql_filters(storage):
    """Verify equality filters work correctly with FTS."""
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="user1",
                agent_version="v1",
                request_id="r1",
                playbook_name="test_fb",
                content="handle errors gracefully",
                structured_data=StructuredData(
                    trigger="when errors occur in production"
                ),
            ),
            UserPlaybook(
                user_id="user2",
                agent_version="v2",
                request_id="r2",
                playbook_name="test_fb",
                content="handle errors loudly",
                structured_data=StructuredData(trigger="when errors occur in staging"),
            ),
        ]
    )

    # Search with agent_version filter
    results = storage.search_user_playbooks(
        SearchUserPlaybookRequest(query="errors", agent_version="v1", top_k=10)
    )
    assert len(results) == 1
    assert results[0].agent_version == "v1"

    # Search with user_id filter
    results = storage.search_user_playbooks(
        SearchUserPlaybookRequest(query="errors", user_id="user2", top_k=10)
    )
    assert len(results) == 1
    assert results[0].user_id == "user2"


def test_search_agent_playbooks_with_agent_version_filter(storage):
    """Verify agent_version filter works with FTS."""
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="be polite",
                structured_data=StructuredData(trigger="when talking to users"),
            ),
            AgentPlaybook(
                agent_version="v2",
                content="be polite always",
                structured_data=StructuredData(trigger="when talking to customers"),
            ),
        ]
    )

    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="talking customers", agent_version="v2", top_k=10
        )
    )
    assert len(results) == 1
    assert results[0].agent_version == "v2"


# ---------------------------------------------------------------------------
# when_condition-based FTS for user playbooks
# ---------------------------------------------------------------------------


def test_user_playbook_searchable_by_when_condition(storage):
    """Insert user playbook with trigger, search by content, verify match."""
    from reflexio.models.api_schema.service_schemas import StructuredData

    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="user1",
                agent_version="v1",
                request_id="r1",
                playbook_name="cond_test",
                content="When the deployment pipeline stalls, restart the build agent",
                structured_data=StructuredData(
                    trigger="when the deployment pipeline stalls",
                    instruction="restart the build agent",
                ),
            ),
        ]
    )

    results = storage.search_user_playbooks(
        SearchUserPlaybookRequest(query="deployment pipeline", top_k=10)
    )
    assert len(results) == 1
    assert "deployment" in results[0].content.lower()


# ---------------------------------------------------------------------------
# Existing test (preserved)
# ---------------------------------------------------------------------------


def test_search_user_profile_queryless_respects_time_window():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0]):
            storage = SQLiteStorage(org_id="0", db_path=f"{temp_dir}/reflexio.db")

            storage.add_user_profile(
                "user1",
                [
                    UserProfile(
                        user_id="user1",
                        profile_id="1",
                        content="old profile",
                        last_modified_timestamp=100,
                        generated_from_request_id="request_1",
                        profile_time_to_live=ProfileTimeToLive.INFINITY,
                    ),
                    UserProfile(
                        user_id="user1",
                        profile_id="2",
                        content="new profile",
                        last_modified_timestamp=200,
                        generated_from_request_id="request_2",
                        profile_time_to_live=ProfileTimeToLive.INFINITY,
                    ),
                ],
            )

            search_request = SearchUserProfileRequest(
                user_id="user1",
                start_time=datetime.fromtimestamp(150, tz=UTC),
                end_time=datetime.fromtimestamp(250, tz=UTC),
                top_k=10,
            )

            profiles = storage.search_user_profile(search_request)

        assert [profile.profile_id for profile in profiles] == ["2"]


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestEffectiveSearchMode:
    def test_fts_always_honored(self):
        assert _effective_search_mode(SearchMode.FTS, [1.0]) == SearchMode.FTS

    def test_hybrid_with_embedding(self):
        assert _effective_search_mode(SearchMode.HYBRID, [1.0]) == SearchMode.HYBRID

    def test_vector_with_embedding(self):
        assert _effective_search_mode(SearchMode.VECTOR, [1.0]) == SearchMode.VECTOR

    def test_hybrid_falls_back_to_fts_without_embedding(self):
        assert _effective_search_mode(SearchMode.HYBRID, None) == SearchMode.FTS

    def test_vector_falls_back_to_fts_without_embedding(self):
        assert _effective_search_mode(SearchMode.VECTOR, None) == SearchMode.FTS

    def test_fts_without_embedding(self):
        assert _effective_search_mode(SearchMode.FTS, None) == SearchMode.FTS


# ---------------------------------------------------------------------------
# Hybrid search integration tests
# ---------------------------------------------------------------------------


def _deterministic_embedding(text: str) -> list[float]:
    """Generate a deterministic 8-dim embedding from text for testing."""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:8]]


@pytest.fixture
def hybrid_storage():
    """Storage fixture with deterministic distinct embeddings per text."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(
            SQLiteStorage, "_get_embedding", side_effect=_deterministic_embedding
        ),
    ):
        yield SQLiteStorage(org_id="0", db_path=f"{temp_dir}/reflexio.db")


def test_hybrid_search_agent_playbooks_uses_embedding(hybrid_storage):
    """Hybrid mode should use embedding similarity in ranking."""
    # Insert playbooks with distinct content
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="optimize database queries for performance",
                structured_data=StructuredData(
                    trigger="when database queries are slow"
                ),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="handle network timeout errors gracefully",
                structured_data=StructuredData(trigger="when network requests timeout"),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="improve caching strategy for queries",
                structured_data=StructuredData(
                    trigger="when query cache misses are frequent"
                ),
            ),
        ]
    )

    # Search with embedding — should return results (basic smoke test)
    query_emb = _deterministic_embedding("database query optimization")
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="queries", top_k=10),
        options=SearchOptions(query_embedding=query_emb),
    )
    assert len(results) >= 1


def test_hybrid_search_falls_back_to_fts_without_embedding(hybrid_storage):
    """Without embedding, hybrid mode should degrade to FTS."""
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="handle errors gracefully",
                structured_data=StructuredData(trigger="when errors occur"),
            ),
        ]
    )

    # No embedding provided — should still work via FTS
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="errors", top_k=10),
    )
    assert len(results) == 1


def test_explicit_fts_mode_ignores_embedding(hybrid_storage):
    """Explicit FTS mode should not use embedding even when provided."""
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="testing fts mode",
                structured_data=StructuredData(trigger="when testing search modes"),
            ),
        ]
    )

    query_emb = _deterministic_embedding("testing")
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="testing", top_k=10, search_mode=SearchMode.FTS
        ),
        options=SearchOptions(query_embedding=query_emb),
    )
    assert len(results) == 1


def test_vector_only_search_agent_playbooks(hybrid_storage):
    """Vector-only search should work without query text."""
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="optimize database queries for performance",
                structured_data=StructuredData(
                    trigger="when database queries are slow"
                ),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="handle network timeout errors gracefully",
                structured_data=StructuredData(trigger="when network requests timeout"),
            ),
        ]
    )

    query_emb = _deterministic_embedding("database query optimization")
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(top_k=10),
        options=SearchOptions(query_embedding=query_emb),
    )
    assert len(results) >= 1


def test_explicit_vector_mode_bypasses_fts_filter(storage):
    """VECTOR mode should rank by embedding even when query text has no semantic match."""
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="lexical-only match",
                structured_data=StructuredData(trigger="lexical-only match"),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="semantic target",
                structured_data=StructuredData(trigger="semantic target"),
            ),
        ]
    )

    storage._execute(
        "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
        (json.dumps([0.0, 1.0]), 1),
    )
    storage._execute(
        "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
        (json.dumps([1.0, 0.0]), 2),
    )

    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="lexical-only",
            top_k=1,
            search_mode=SearchMode.VECTOR,
        ),
        options=SearchOptions(query_embedding=[1.0, 0.0]),
    )

    assert len(results) == 1
    assert results[0].agent_playbook_id == 2


def test_vector_search_ranks_full_filtered_set(storage):
    """Queryless vector search should consider candidates beyond the recency overfetch window."""
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                created_at=1,
                content="old but best match",
                structured_data=StructuredData(trigger="old but best match"),
            ),
            AgentPlaybook(
                agent_version="v1",
                created_at=2,
                content="candidate 2",
                structured_data=StructuredData(trigger="candidate 2"),
            ),
            AgentPlaybook(
                agent_version="v1",
                created_at=3,
                content="candidate 3",
                structured_data=StructuredData(trigger="candidate 3"),
            ),
            AgentPlaybook(
                agent_version="v1",
                created_at=4,
                content="candidate 4",
                structured_data=StructuredData(trigger="candidate 4"),
            ),
            AgentPlaybook(
                agent_version="v1",
                created_at=5,
                content="candidate 5",
                structured_data=StructuredData(trigger="candidate 5"),
            ),
            AgentPlaybook(
                agent_version="v1",
                created_at=6,
                content="candidate 6",
                structured_data=StructuredData(trigger="candidate 6"),
            ),
        ]
    )

    storage._execute(
        "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
        (json.dumps([1.0, 0.0]), 1),
    )
    for playbook_id in range(2, 7):
        storage._execute(
            "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
            (json.dumps([0.0, 1.0]), playbook_id),
        )

    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(top_k=1),
        options=SearchOptions(query_embedding=[1.0, 0.0]),
    )

    assert len(results) == 1
    assert results[0].agent_playbook_id == 1


def test_hybrid_search_user_playbooks(hybrid_storage):
    """Hybrid search should work for user playbooks."""
    hybrid_storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="user1",
                agent_version="v1",
                request_id="r1",
                playbook_name="test",
                content="improve error handling",
                structured_data=StructuredData(trigger="when errors occur"),
            ),
        ]
    )

    query_emb = _deterministic_embedding("error handling")
    results = hybrid_storage.search_user_playbooks(
        SearchUserPlaybookRequest(query="error", top_k=10),
        options=SearchOptions(query_embedding=query_emb),
    )
    assert len(results) == 1


def test_hybrid_search_with_null_embeddings(storage):
    """Rows with NULL embeddings should gracefully degrade to FTS-only ranking."""
    # The default `storage` fixture returns [0.0] embedding — effectively a zero vector.
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="handle errors",
                structured_data=StructuredData(trigger="when errors occur"),
            ),
        ]
    )

    # Provide a real embedding for the query — should still return results
    query_emb = [1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="errors", top_k=10),
        options=SearchOptions(query_embedding=query_emb),
    )
    assert len(results) == 1


def test_hybrid_mode_self_generates_embedding(hybrid_storage):
    """When search_mode=HYBRID but no query_embedding provided, storage should self-generate."""
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="optimize database queries",
                structured_data=StructuredData(trigger="when queries are slow"),
            ),
        ]
    )

    # Request HYBRID mode but do NOT provide query_embedding — search by trigger text
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="queries slow", top_k=10, search_mode=SearchMode.HYBRID
        ),
    )
    assert len(results) >= 1


def test_vector_mode_self_generates_embedding(hybrid_storage):
    """When search_mode=VECTOR but no query_embedding provided, storage should self-generate."""
    hybrid_storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="optimize database queries",
                structured_data=StructuredData(trigger="when queries are slow"),
            ),
        ]
    )

    # Request VECTOR mode — storage should generate embedding and rank by similarity
    results = hybrid_storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="queries slow", top_k=10, search_mode=SearchMode.VECTOR
        ),
    )
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# _true_rrf_merge tests
# ---------------------------------------------------------------------------


class _FakeRow(dict):
    """dict subclass that behaves like sqlite3.Row for testing."""

    def keys(self):
        return list(super().keys())


class TestTrueRrfMerge:
    def test_disjoint_sets(self):
        """Two non-overlapping sets should both contribute results."""
        fts = [_FakeRow(id=1, name="a"), _FakeRow(id=2, name="b")]
        vec = [_FakeRow(id=3, name="c"), _FakeRow(id=4, name="d")]
        result = _true_rrf_merge(fts, vec, "id", match_count=4)
        result_ids = {r["id"] for r in result}
        assert result_ids == {1, 2, 3, 4}

    def test_overlapping_sets(self):
        """Rows in both sets should rank higher than rows in only one."""
        shared = _FakeRow(id=1, name="shared")
        fts_only = _FakeRow(id=2, name="fts_only")
        vec_only = _FakeRow(id=3, name="vec_only")
        fts = [shared, fts_only]
        vec = [shared, vec_only]
        result = _true_rrf_merge(fts, vec, "id", match_count=3)
        # Shared row appears in both → highest RRF score → should be first
        assert result[0]["id"] == 1

    def test_empty_fts_returns_vec_results(self):
        """When FTS set is empty, vec results should still surface."""
        vec = [_FakeRow(id=1, name="a"), _FakeRow(id=2, name="b")]
        result = _true_rrf_merge([], vec, "id", match_count=2)
        assert len(result) == 2

    def test_empty_vec_returns_fts_results(self):
        """When vec set is empty, FTS results should still surface."""
        fts = [_FakeRow(id=1, name="a")]
        result = _true_rrf_merge(fts, [], "id", match_count=1)
        assert len(result) == 1

    def test_both_empty(self):
        result = _true_rrf_merge([], [], "id", match_count=5)
        assert result == []

    def test_match_count_limits(self):
        fts = [_FakeRow(id=i) for i in range(10)]
        vec = [_FakeRow(id=i + 10) for i in range(10)]
        result = _true_rrf_merge(fts, vec, "id", match_count=3)
        assert len(result) == 3

    def test_weights_affect_ranking(self):
        """High vector_weight should favor vec-ranked items."""
        fts = [_FakeRow(id=1), _FakeRow(id=2)]
        vec = [_FakeRow(id=2), _FakeRow(id=1)]  # reversed order
        # With very high vector weight, vec ordering should dominate
        result = _true_rrf_merge(
            fts, vec, "id", match_count=2, vector_weight=100.0, fts_weight=0.01
        )
        assert result[0]["id"] == 2  # vec #1


# ---------------------------------------------------------------------------
# True hybrid search integration: semantic-but-not-lexical matches
# ---------------------------------------------------------------------------


def test_hybrid_surfaces_semantic_only_match(storage):
    """True hybrid search should surface docs that match semantically but NOT lexically.

    This is the key regression test for the _true_rrf_merge refactor.
    """
    # Save two playbooks:
    # 1. "improve caching strategy" — will NOT match FTS query "database optimization"
    # 2. "optimize database queries" — will match FTS query
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="improve caching strategy for web requests",
                structured_data=StructuredData(trigger="cache miss rate is high"),
            ),
            AgentPlaybook(
                agent_version="v1",
                content="optimize database queries for better performance",
                structured_data=StructuredData(trigger="database queries are slow"),
            ),
        ]
    )

    # Manually set embeddings so that playbook #1 (caching) is closer to query embedding
    # while playbook #2 (database) matches lexically
    storage._execute(
        "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
        (json.dumps([1.0, 0.0, 0.0, 0.0]), 1),  # closest to query
    )
    storage._execute(
        "UPDATE agent_playbooks SET embedding = ? WHERE agent_playbook_id = ?",
        (json.dumps([0.0, 1.0, 0.0, 0.0]), 2),  # farther from query
    )

    # Query: "database" matches #2 lexically, but embedding is closest to #1
    query_emb = [0.9, 0.1, 0.0, 0.0]  # very close to playbook #1's embedding
    results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="database", top_k=10, search_mode=SearchMode.HYBRID
        ),
        options=SearchOptions(query_embedding=query_emb),
    )

    result_ids = [r.agent_playbook_id for r in results]
    # Both should appear — #1 via vector, #2 via FTS
    assert 1 in result_ids, "Semantic-only match (playbook #1) should appear in results"
    assert 2 in result_ids, "Lexical match (playbook #2) should appear in results"


def test_hybrid_surfaces_semantic_only_user_playbook(storage):
    """Same test for user_playbooks — semantic-only match should surface."""
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="u1",
                agent_version="v1",
                request_id="r1",
                playbook_name="test",
                content="improve caching strategy",
                structured_data=StructuredData(trigger="cache miss"),
            ),
            UserPlaybook(
                user_id="u1",
                agent_version="v1",
                request_id="r1",
                playbook_name="test",
                content="optimize database queries",
                structured_data=StructuredData(trigger="database queries"),
            ),
        ]
    )

    storage._execute(
        "UPDATE user_playbooks SET embedding = ? WHERE user_playbook_id = ?",
        (json.dumps([1.0, 0.0]), 1),
    )
    storage._execute(
        "UPDATE user_playbooks SET embedding = ? WHERE user_playbook_id = ?",
        (json.dumps([0.0, 1.0]), 2),
    )

    results = storage.search_user_playbooks(
        SearchUserPlaybookRequest(
            query="database", top_k=10, search_mode=SearchMode.HYBRID
        ),
        options=SearchOptions(query_embedding=[0.9, 0.1]),
    )

    result_ids = [r.user_playbook_id for r in results]
    assert 1 in result_ids, "Semantic-only match should surface via true hybrid"
    assert 2 in result_ids


# ---------------------------------------------------------------------------
# Embedding prefix tests
# ---------------------------------------------------------------------------


def test_embedding_prefix_applied():
    """_get_embedding should prefix text with 'search_document:' or 'search_query:' based on purpose."""
    captured_texts = []

    def mock_llm_get_embedding(text, model, dimensions):
        captured_texts.append(text)
        return [0.0]

    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch(
            "reflexio.server.services.storage.sqlite_storage._base.LiteLLMClient"
        ) as mock_cls,
    ):
        mock_cls.return_value.get_embedding = mock_llm_get_embedding
        with patch.object(SQLiteStorage, "_try_load_sqlite_vec", return_value=False):
            s = SQLiteStorage(org_id="0", db_path=f"{temp_dir}/reflexio.db")

        # Document purpose
        s._get_embedding("hello world", purpose="document")
        assert captured_texts[-1] == "search_document: hello world"

        # Query purpose
        s._get_embedding("hello world", purpose="query")
        assert captured_texts[-1] == "search_query: hello world"

        # Default purpose is document
        s._get_embedding("hello world")
        assert captured_texts[-1] == "search_document: hello world"


# ---------------------------------------------------------------------------
# sqlite-vec fallback test
# ---------------------------------------------------------------------------


def test_sqlite_vec_fallback_graceful():
    """When sqlite-vec is not installed, _has_sqlite_vec should be False and search still works."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0]),
    ):
        s = SQLiteStorage(org_id="0", db_path=f"{temp_dir}/reflexio.db")
        # Whether sqlite-vec is available or not, the storage should work
        assert isinstance(s._has_sqlite_vec, bool)
        # Save and search should work regardless
        s.save_agent_playbooks(
            [
                AgentPlaybook(
                    agent_version="v1",
                    content="test content",
                    structured_data=StructuredData(trigger="test"),
                ),
            ]
        )
        results = s.search_agent_playbooks(
            SearchAgentPlaybookRequest(query="test", top_k=5),
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# JSONB round-trip tests
# ---------------------------------------------------------------------------


def test_structured_data_jsonb_round_trip(storage):
    """Save and retrieve StructuredData with all fields populated, partial fields, and on aggregated AgentPlaybook."""
    from reflexio.models.api_schema.service_schemas import (
        BlockingIssue,
        BlockingIssueKind,
    )

    # -- Full StructuredData on UserPlaybook --
    full_sd = StructuredData(
        rationale="Users need context before code",
        trigger="User asks for help debugging an error trace",
        instruction="Outline high-level debugging strategy first",
        pitfall="Jumping straight to code fixes",
        blocking_issue=BlockingIssue(
            kind=BlockingIssueKind.MISSING_TOOL,
            details="No upload tool available",
        ),
        embedding_text="custom embedding override text",
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="u1",
                agent_version="v1",
                request_id="r_full",
                playbook_name="jsonb_full",
                content="full structured data test",
                structured_data=full_sd,
            ),
        ]
    )

    retrieved_full = storage.get_user_playbooks(playbook_name="jsonb_full")
    assert len(retrieved_full) == 1
    sd = retrieved_full[0].structured_data
    assert sd.rationale == "Users need context before code"
    assert sd.trigger == "User asks for help debugging an error trace"
    assert sd.instruction == "Outline high-level debugging strategy first"
    assert sd.pitfall == "Jumping straight to code fixes"
    assert sd.blocking_issue is not None
    assert sd.blocking_issue.kind == BlockingIssueKind.MISSING_TOOL
    assert sd.blocking_issue.details == "No upload tool available"
    assert sd.embedding_text == "custom embedding override text"

    # -- Partial StructuredData (only trigger set, others None) --
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="u2",
                agent_version="v1",
                request_id="r_partial",
                playbook_name="jsonb_partial",
                content="partial structured data test",
                structured_data=StructuredData(trigger="only trigger set"),
            ),
        ]
    )

    retrieved_partial = storage.get_user_playbooks(playbook_name="jsonb_partial")
    assert len(retrieved_partial) == 1
    sd_partial = retrieved_partial[0].structured_data
    assert sd_partial.trigger == "only trigger set"
    assert sd_partial.rationale is None
    assert sd_partial.instruction is None
    assert sd_partial.pitfall is None
    assert sd_partial.blocking_issue is None
    assert sd_partial.embedding_text is None

    # -- StructuredData on aggregated AgentPlaybook --
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                content="aggregated playbook with structured data",
                structured_data=StructuredData(
                    rationale="Users need context before code",
                    trigger="User asks for help debugging an error trace",
                    instruction="Outline high-level debugging strategy first",
                    pitfall="Jumping straight to code fixes",
                    blocking_issue=BlockingIssue(
                        kind=BlockingIssueKind.MISSING_TOOL,
                        details="No upload tool available",
                    ),
                    embedding_text="custom embedding override text",
                ),
            ),
        ]
    )

    retrieved_fb = storage.get_agent_playbooks()
    assert len(retrieved_fb) == 1
    sd_fb = retrieved_fb[0].structured_data
    assert sd_fb.rationale == "Users need context before code"
    assert sd_fb.trigger == "User asks for help debugging an error trace"
    assert sd_fb.instruction == "Outline high-level debugging strategy first"
    assert sd_fb.pitfall == "Jumping straight to code fixes"
    assert sd_fb.blocking_issue is not None
    assert sd_fb.blocking_issue.kind == BlockingIssueKind.MISSING_TOOL
    assert sd_fb.blocking_issue.details == "No upload tool available"
    assert sd_fb.embedding_text == "custom embedding override text"


def test_structured_data_with_extra_fields_round_trip(storage):
    """Extra unknown fields on StructuredData should survive save/retrieve (model_config extra='allow')."""
    sd = StructuredData(
        **{"trigger": "test", "priority": "high", "confidence_score": 0.95}
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="u_extra",
                agent_version="v1",
                request_id="r_extra",
                playbook_name="jsonb_extra",
                content="extra fields test",
                structured_data=sd,
            ),
        ]
    )

    retrieved = storage.get_user_playbooks(playbook_name="jsonb_extra")
    assert len(retrieved) == 1
    sd_out = retrieved[0].structured_data
    assert sd_out.trigger == "test"
    assert sd_out.model_extra["priority"] == "high"
    assert sd_out.model_extra["confidence_score"] == 0.95
