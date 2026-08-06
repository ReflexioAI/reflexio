"""Unit tests for the unified search service.

Tests the critical orchestration logic: empty query, embedding failure,
and reformulated_query propagation.
"""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
)
from reflexio.models.config_schema import (
    RetrievalFloorConfig,
    SearchMode,
    SearchOptions,
)
from reflexio.server.services.pre_retrieval import ReformulationResult
from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.unified_search_service import (
    _run_phase_b,
    _search_agent_playbooks_via_storage,
    run_unified_search,
)


def _mock_storage(embedding=None):
    """Create a mock storage with configurable embedding."""
    storage = MagicMock()
    storage.embedding_model_name = "local/minilm-l6-v2"
    storage._get_embedding.return_value = embedding or [0.1] * 1536
    # Storage search methods return empty lists by default
    storage.search_user_profile.return_value = []
    storage.search_agent_playbooks.return_value = []
    storage.search_user_playbooks.return_value = []
    return storage


class TestRunUnifiedSearch(unittest.TestCase):
    """Tests for the top-level run_unified_search function."""

    def test_storage_must_declare_embedding_model_name(self):
        storage = _mock_storage()
        del storage.embedding_model_name

        with self.assertRaisesRegex(AttributeError, "embedding_model_name"):
            run_unified_search(
                request=UnifiedSearchRequest(query="test query"),
                org_id="test-org",
                storage=storage,
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
            )

    def test_empty_query_rejected_by_validation(self):
        """Empty query is now rejected at the Pydantic validation level."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            UnifiedSearchRequest(query="")

    def test_whitespace_query_rejected_by_validation(self):
        """Whitespace-only query is rejected at the Pydantic validation level."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            UnifiedSearchRequest(query="   ")

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_embedding_failure_degrades_to_text_search(self, _reformulator_cls):
        """When embedding generation fails, should degrade to text-only search (not crash)."""
        storage = _mock_storage()
        storage._get_embedding.side_effect = RuntimeError("Embedding API down")

        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )

        request = UnifiedSearchRequest(query="test query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        # Default agent-playbook status filtering uses a single storage call
        # with the full APPROVED+PENDING allow-list; REJECTED is excluded
        # server-side.
        assert storage.search_agent_playbooks.call_count == 1
        sent_filters = [
            call.args[0].playbook_status_filter
            for call in storage.search_agent_playbooks.call_args_list
        ]
        assert sent_filters == [[PlaybookStatus.APPROVED, PlaybookStatus.PENDING]]

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_empty_embedding_degrades_to_fts(self, _reformulator_cls):
        """An empty embedding result (e.g. a storage backend that swallows
        EmbeddingUnavailableError and returns []) degrades to FTS and is flagged
        as degraded — not run as a vector search with an empty embedding."""
        storage = _mock_storage()
        storage._get_embedding.return_value = []  # no usable query vector

        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )

        request = UnifiedSearchRequest(query="test query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertTrue(result.degraded)
        self.assertEqual(result.search_mode_effective, "fts")

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_local_storage_without_get_embedding(self, _reformulator_cls):
        """Storage without _get_embedding should not crash and should use text-only search."""
        storage = _mock_storage()
        del storage._get_embedding  # Simulate a storage backend that lacks this method

        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )

        request = UnifiedSearchRequest(query="test query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        # Default agent-playbook status filtering uses a single storage call
        # with the full APPROVED+PENDING allow-list; REJECTED is excluded
        # server-side.
        assert storage.search_agent_playbooks.call_count == 1
        sent_filters = [
            call.args[0].playbook_status_filter
            for call in storage.search_agent_playbooks.call_args_list
        ]
        assert sent_filters == [[PlaybookStatus.APPROVED, PlaybookStatus.PENDING]]

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_reformulated_query_populated_when_changed(self, _reformulator_cls):
        """reformulated_query field should only be set when query was actually reformulated."""
        expanded = ReformulationResult(
            standalone_query="agent failed OR error to refund OR return"
        )
        _reformulator_cls.return_value.rewrite.return_value = expanded

        storage = _mock_storage()
        request = UnifiedSearchRequest(
            query="agent failed to refund", enable_reformulation=True
        )
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.reformulated_query,
            "agent failed OR error to refund OR return",
        )

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_reformulated_query_none_when_unchanged(self, _reformulator_cls):
        """reformulated_query should be None when query was not reformulated."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="same query"
        )

        storage = _mock_storage()
        request = UnifiedSearchRequest(query="same query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.reformulated_query)

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_recency_uses_pool_and_combined_score_after_phase_b(
        self, _reformulator_cls
    ):
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="same query"
        )
        storage = _mock_storage()
        old = UserPlaybook(
            user_playbook_id=1,
            user_id="user-1",
            agent_version="v1",
            request_id="r1",
            playbook_name="pb",
            content="old",
            created_at=1,
        )
        fresh = UserPlaybook(
            user_playbook_id=2,
            user_id="user-1",
            agent_version="v1",
            request_id="r2",
            playbook_name="pb",
            content="fresh",
            created_at=4_102_444_800,
        )
        seen_top_k = []

        def fake_phase_b(**kwargs):
            seen_top_k.append(kwargs["top_k"])
            return ([], [], [ScoredItem(old, 1.0), ScoredItem(fresh, 0.9)])

        with patch(
            "reflexio.server.services.unified_search_service._run_phase_b",
            side_effect=fake_phase_b,
        ):
            result = run_unified_search(
                request=UnifiedSearchRequest(
                    query="same query", user_id="user-1", top_k=1
                ),
                org_id="test-org",
                storage=storage,
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
                retrieval_floor=RetrievalFloorConfig(enabled=False),
                recency=RecencyConfig(enabled=True, max_penalty_frac=1.0, pool_size=2),
            )

        self.assertTrue(result.success)
        self.assertEqual(seen_top_k, [2])
        self.assertEqual([pb.content for pb in result.user_playbooks], ["fresh"])

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_recency_does_not_overtake_clearly_more_relevant_combined_score(
        self, _reformulator_cls
    ):
        # Invariant on the default (combined_score) arm: at the default penalty
        # fraction, an ancient but clearly-more-relevant item is never overtaken
        # by a fresher, weaker one (0.040 vs 0.024 is a 1.67x gap >> 15%).
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="q"
        )
        storage = _mock_storage()
        relevant = UserPlaybook(
            user_playbook_id=1,
            user_id="user-1",
            agent_version="v1",
            request_id="r1",
            playbook_name="pb",
            content="relevant",
            created_at=1,
        )
        fresh = UserPlaybook(
            user_playbook_id=2,
            user_id="user-1",
            agent_version="v1",
            request_id="r2",
            playbook_name="pb",
            content="fresh",
            created_at=4_102_444_800,
        )

        def fake_phase_b(**_kwargs):
            return ([], [], [ScoredItem(relevant, 0.040), ScoredItem(fresh, 0.024)])

        with patch(
            "reflexio.server.services.unified_search_service._run_phase_b",
            side_effect=fake_phase_b,
        ):
            result = run_unified_search(
                request=UnifiedSearchRequest(query="q", user_id="user-1", top_k=2),
                org_id="test-org",
                storage=storage,
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
                retrieval_floor=RetrievalFloorConfig(enabled=False),
                recency=RecencyConfig(enabled=True, max_penalty_frac=0.15, pool_size=2),
            )

        self.assertEqual(
            [pb.content for pb in result.user_playbooks], ["relevant", "fresh"]
        )

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_recency_routes_scored_single_rpc_without_support_flag(
        self, _reformulator_cls
    ):
        # Native-Postgres shape: supports_unified_hybrid_search=False but the
        # inherited unified_hybrid_search_scored is present. Recency must still
        # route through the scored single-RPC path (not silently no-op).
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="q"
        )

        class _PgLikeStorage:
            supports_embedding = False
            supports_unified_hybrid_search = False
            embedding_model_name = "local/minilm-l6-v2"

            def unified_hybrid_search_scored(self, **_kwargs):
                return ([], [], [])

        seen: dict[str, object] = {}

        def fake_single_rpc(**kwargs):
            seen["recency_on"] = kwargs.get("recency_on")
            return ([], [], [])

        with (
            patch(
                "reflexio.server.services.unified_search_service._run_phase_a",
                return_value=(ReformulationResult(standalone_query="q"), None, False),
            ),
            patch(
                "reflexio.server.services.unified_search_service._run_phase_b_single_rpc",
                side_effect=fake_single_rpc,
            ),
        ):
            run_unified_search(
                request=UnifiedSearchRequest(query="q", user_id="u", top_k=2),
                org_id="o",
                storage=cast(BaseStorage, _PgLikeStorage()),
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
                retrieval_floor=RetrievalFloorConfig(enabled=False),
                recency=RecencyConfig(enabled=True, pool_size=2),
            )

        self.assertTrue(seen.get("recency_on"))


def _agent_playbook(agent_playbook_id: int, status: PlaybookStatus) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=agent_playbook_id,
        agent_version="claude-code",
        content=f"rule {agent_playbook_id}",
        playbook_status=status,
    )


def test_search_agent_playbooks_allows_pending_and_approved_but_not_rejected() -> None:
    """One storage call carries the full allow-list; REJECTED is not in it."""
    seen_filters: list[list[PlaybookStatus] | PlaybookStatus | None] = []

    def search_agent_playbooks(request, _options):
        seen_filters.append(request.playbook_status_filter)
        return [
            _agent_playbook(1, PlaybookStatus.PENDING),
            _agent_playbook(2, PlaybookStatus.APPROVED),
        ]

    storage = SimpleNamespace(search_agent_playbooks=search_agent_playbooks)

    results = _search_agent_playbooks_via_storage(
        storage=cast(BaseStorage, storage),
        query="formatting",
        top_k=5,
        threshold=0.3,
        agent_version="claude-code",
        playbook_name=None,
        allowed_statuses=[PlaybookStatus.PENDING, PlaybookStatus.APPROVED],
        options=SearchOptions(),
    )

    assert seen_filters == [[PlaybookStatus.PENDING, PlaybookStatus.APPROVED]]
    assert [p.playbook_status for p in results] == [
        PlaybookStatus.PENDING,
        PlaybookStatus.APPROVED,
    ]


def test_search_agent_playbooks_default_excludes_rejected() -> None:
    """When the caller omits ``allowed_statuses``, the single storage call
    passes APPROVED + PENDING as the filter; REJECTED never appears so a
    dashboard rejection suppresses the playbook for every consumer."""
    seen_filters: list[list[PlaybookStatus] | PlaybookStatus | None] = []

    def search_agent_playbooks(request, _options):
        seen_filters.append(request.playbook_status_filter)
        return []

    storage = SimpleNamespace(search_agent_playbooks=search_agent_playbooks)

    _search_agent_playbooks_via_storage(
        storage=cast(BaseStorage, storage),
        query="formatting",
        top_k=5,
        threshold=0.3,
        agent_version="claude-code",
        playbook_name=None,
        allowed_statuses=None,
        options=SearchOptions(),
    )

    assert seen_filters == [[PlaybookStatus.APPROVED, PlaybookStatus.PENDING]]
    assert all(PlaybookStatus.REJECTED not in (f or []) for f in seen_filters)


class TestEntityTypesFiltering(unittest.TestCase):
    """``UnifiedSearchRequest.entity_types`` should gate which storage calls fire."""

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_only_requested_entity_types_query_storage(self, _reformulator_cls):
        """When ``entity_types=["agent_playbooks"]``, profile and user-playbook
        storage methods must NOT be invoked. Without this gate, callers asking
        for a single entity would silently incur the cost of all three legs."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="q"
        )
        storage = _mock_storage()

        request = UnifiedSearchRequest(query="q", entity_types=["agent_playbooks"])
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        storage.search_agent_playbooks.assert_called()
        storage.search_user_playbooks.assert_not_called()
        storage.search_user_profile.assert_not_called()

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_excluded_entity_types_return_empty_in_response(self, _reformulator_cls):
        """A leg that wasn't requested must come back as an empty list, not None."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="q"
        )
        storage = _mock_storage()

        request = UnifiedSearchRequest(query="q", entity_types=["profiles"])
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.agent_playbooks, [])
        self.assertEqual(result.user_playbooks, [])

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_opt_out_keeps_agent_playbooks_and_skips_user_context(
        self, _reformulator_cls
    ):
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="rewritten query"
        )
        storage = _mock_storage()

        result = run_unified_search(
            request=UnifiedSearchRequest(
                query="Prepare a general answer without using my profile.",
                user_id="user1",
                enable_reformulation=True,
            ),
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.profiles, [])
        self.assertEqual(result.user_playbooks, [])
        storage.search_user_profile.assert_not_called()
        storage.search_user_playbooks.assert_not_called()
        storage.search_agent_playbooks.assert_called_once()

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_chinese_opt_out_suppresses_user_context(self, _reformulator_cls):
        storage = _mock_storage()

        result = run_unified_search(
            request=UnifiedSearchRequest(
                query="不要使用我的个人资料",
                user_id="user1",
                entity_types=["profiles", "user_playbooks"],
            ),
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.profiles, [])
        self.assertEqual(result.user_playbooks, [])
        storage.search_user_profile.assert_not_called()
        storage.search_user_playbooks.assert_not_called()

    @patch("reflexio.server.services.unified_search_service._run_phase_a")
    def test_user_only_opt_out_returns_before_phase_a(self, phase_a):
        storage = _mock_storage()

        result = run_unified_search(
            request=UnifiedSearchRequest(
                query="Do not use my saved preferences for this answer.",
                user_id="user1",
                entity_types=["profiles", "user_playbooks"],
                enable_reformulation=True,
            ),
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.profiles, [])
        self.assertEqual(result.user_playbooks, [])
        phase_a.assert_not_called()
        storage.search_user_profile.assert_not_called()
        storage.search_user_playbooks.assert_not_called()
        storage.search_agent_playbooks.assert_not_called()

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_public_audience_alone_does_not_suppress_user_context(
        self, _reformulator_cls
    ):
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="Create this for a public audience."
        )
        storage = _mock_storage()

        result = run_unified_search(
            request=UnifiedSearchRequest(
                query="Create this for a public audience.",
                user_id="user1",
                entity_types=["profiles", "user_playbooks"],
            ),
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        storage.search_user_profile.assert_called_once()
        storage.search_user_playbooks.assert_called_once()


_SERVICE_LOGGER = "reflexio.server.services.unified_search_service"


def _fanout_storage(embedding=None):
    """Mock storage pinned to the per-arm fan-out (single-RPC disabled).

    Lets a test inspect the per-leg storage requests (and their
    ``search_mode``) directly instead of the opaque combined RPC.
    """
    storage = _mock_storage(embedding)
    storage.supports_embedding = True
    storage.supports_unified_hybrid_search = False
    return storage


def _leg_search_modes(storage):
    """The ``search_mode`` each fan-out leg was invoked with."""
    return {
        "profiles": storage.search_user_profile.call_args.args[0].search_mode,
        "agent_playbooks": storage.search_agent_playbooks.call_args.args[0].search_mode,
        "user_playbooks": storage.search_user_playbooks.call_args.args[0].search_mode,
    }


def test_phase_b_passes_tag_filter_to_every_fanout_leg() -> None:
    storage = _fanout_storage()

    _run_phase_b(
        request=UnifiedSearchRequest(
            query="billing", user_id="user-1", tags=["billing", "support"]
        ),
        org_id="test-org",
        storage=storage,
        embedding=[0.1] * 1536,
        query="billing",
        top_k=5,
        threshold=0.3,
    )

    assert storage.search_user_profile.call_args.args[0].tags == [
        "billing",
        "support",
    ]
    assert storage.search_agent_playbooks.call_args.args[0].tags == [
        "billing",
        "support",
    ]
    assert storage.search_user_playbooks.call_args.args[0].tags == [
        "billing",
        "support",
    ]


class TestEmbeddingFailureDegradesToFTS(unittest.TestCase):
    """D2/D3: embedding-generation failure degrades the whole search to FTS."""

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_embedding_failure_degrades_to_fts_visibly_and_counted(
        self, _reformulator_cls
    ):
        """A raising query-embedder must (a) still return results via FTS,
        (b) surface degraded=True / effective mode fts, (c) drive every storage
        leg with FTS so storage takes the placeholder branch and never
        re-embeds, and (d) emit the counted degrade signal."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )
        storage = _fanout_storage()
        storage._get_embedding.side_effect = RuntimeError("Embedding API down")

        request = UnifiedSearchRequest(query="test query", user_id="user-1")
        with self.assertLogs(_SERVICE_LOGGER, level="WARNING") as logs:
            result = run_unified_search(
                request=request,
                org_id="test-org",
                storage=storage,
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
            )

        # (a) success via FTS, not an exception
        self.assertTrue(result.success)
        # (b) caller-visible degrade
        self.assertTrue(result.degraded)
        self.assertEqual(result.search_mode_effective, SearchMode.FTS.value)
        # (c) every leg driven with FTS; the embedder was hit exactly once
        # (the failed Phase-A attempt) — storage never re-embeds.
        self.assertEqual(
            _leg_search_modes(storage),
            {
                "profiles": SearchMode.FTS,
                "agent_playbooks": SearchMode.FTS,
                "user_playbooks": SearchMode.FTS,
            },
        )
        self.assertEqual(storage._get_embedding.call_count, 1)
        # profile leg received no embedding to force a re-embed downstream
        self.assertIsNone(
            storage.search_user_profile.call_args.kwargs["query_embedding"]
        )
        # (d) counted signal fired with a stable event key
        self.assertTrue(
            any("event=search_degraded_to_fts" in line for line in logs.output),
            logs.output,
        )

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_successful_embedding_is_not_degraded(self, _reformulator_cls):
        """Healthy embedding: no degrade, requested mode honored, vector path
        used (embedding threaded into the profile leg)."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )
        embedding = [0.42] * 1536
        storage = _fanout_storage(embedding=embedding)

        request = UnifiedSearchRequest(query="test query", user_id="user-1")
        with self.assertNoLogs(_SERVICE_LOGGER, level="WARNING"):
            result = run_unified_search(
                request=request,
                org_id="test-org",
                storage=storage,
                llm_client=MagicMock(),
                prompt_manager=MagicMock(),
            )

        self.assertTrue(result.success)
        self.assertFalse(result.degraded)
        self.assertIsNone(result.search_mode_effective)
        self.assertEqual(
            _leg_search_modes(storage),
            {
                "profiles": SearchMode.HYBRID,
                "agent_playbooks": SearchMode.HYBRID,
                "user_playbooks": SearchMode.HYBRID,
            },
        )
        self.assertEqual(
            storage.search_user_profile.call_args.kwargs["query_embedding"], embedding
        )

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_fts_mode_from_start_is_not_degraded(self, _reformulator_cls):
        """A request that asks for FTS outright never embedded, so it is a
        benign None — not a degrade."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )
        storage = _fanout_storage()

        request = UnifiedSearchRequest(
            query="test query", user_id="user-1", search_mode=SearchMode.FTS
        )
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.degraded)
        self.assertIsNone(result.search_mode_effective)
        # FTS-only mode skips the embedder entirely.
        storage._get_embedding.assert_not_called()
        self.assertEqual(_leg_search_modes(storage)["agent_playbooks"], SearchMode.FTS)

    def test_empty_query_guard_is_not_degraded(self):
        """The empty-query early return is a benign no-op, never a degrade."""
        # query is NonEmptyStr at the API boundary; construct past validation to
        # exercise the defensive guard inside run_unified_search.
        request = UnifiedSearchRequest.model_construct(query="")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=_fanout_storage(),
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.degraded)
        self.assertIsNone(result.search_mode_effective)


if __name__ == "__main__":
    unittest.main()
