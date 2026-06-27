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
from reflexio.models.config_schema import RetrievalFloorConfig, SearchOptions
from reflexio.server.services.pre_retrieval import ReformulationResult
from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.unified_search_service import (
    _search_agent_playbooks_via_storage,
    configure_retrieval_capture_hook,
    run_unified_search,
)


def _mock_storage(embedding=None):
    """Create a mock storage with configurable embedding."""
    storage = MagicMock()
    storage._get_embedding.return_value = embedding or [0.1] * 1536
    # Storage search methods return empty lists by default
    storage.search_user_profile.return_value = []
    storage.search_agent_playbooks.return_value = []
    storage.search_user_playbooks.return_value = []
    return storage


class TestRunUnifiedSearch(unittest.TestCase):
    """Tests for the top-level run_unified_search function."""

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
    def test_capture_hook_receives_final_shaped_response(
        self, _reformulator_cls
    ) -> None:
        """The capture seam must see post-floor, post-suppression results only."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="same query"
        )
        storage = _mock_storage()

        pre_floor_agent = _agent_playbook(10, PlaybookStatus.PENDING)
        object.__setattr__(
            pre_floor_agent, "_source_user_playbook_ids", frozenset({101})
        )
        kept_user_playbook = UserPlaybook(
            user_playbook_id=102,
            user_id="user-1",
            agent_version="claude-code",
            request_id="req-102",
            playbook_name="pb",
            content="kept",
        )
        suppressed_user_playbook = UserPlaybook(
            user_playbook_id=101,
            user_id="user-1",
            agent_version="claude-code",
            request_id="req-101",
            playbook_name="pb",
            content="suppressed",
        )
        captured = []

        with (
            patch(
                "reflexio.server.services.unified_search_service._run_phase_b",
                return_value=(
                    [],
                    [pre_floor_agent, _agent_playbook(11, PlaybookStatus.APPROVED)],
                    [suppressed_user_playbook, kept_user_playbook],
                ),
            ),
            patch(
                "reflexio.server.services.unified_search_service._apply_floors",
                return_value=(
                    [],
                    [pre_floor_agent],
                    [suppressed_user_playbook, kept_user_playbook],
                ),
            ),
        ):
            configure_retrieval_capture_hook(
                lambda request, response, _storage, org_id: captured.append(
                    (
                        request.request_id,
                        org_id,
                        [pb.agent_playbook_id for pb in response.agent_playbooks],
                        [pb.user_playbook_id for pb in response.user_playbooks],
                    )
                )
            )
            try:
                result = run_unified_search(
                    request=UnifiedSearchRequest(
                        query="same query",
                        user_id="user-1",
                        request_id="req-1",
                        session_id="sess-1",
                    ),
                    org_id="test-org",
                    storage=storage,
                    llm_client=MagicMock(),
                    prompt_manager=MagicMock(),
                    retrieval_floor=RetrievalFloorConfig(enabled=True, pool_size=5),
                )
            finally:
                configure_retrieval_capture_hook(None)

        self.assertTrue(result.success)
        self.assertEqual(captured, [("req-1", "test-org", [10], [102])])

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

            def unified_hybrid_search_scored(self, **_kwargs):
                return ([], [], [])

        seen: dict[str, object] = {}

        def fake_single_rpc(**kwargs):
            seen["recency_on"] = kwargs.get("recency_on")
            return ([], [], [])

        with (
            patch(
                "reflexio.server.services.unified_search_service._run_phase_a",
                return_value=("q", None),
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


if __name__ == "__main__":
    unittest.main()
