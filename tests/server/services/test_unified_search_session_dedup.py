"""Session-scoped result dedup in unified search.

A request that carries a ``session_id`` must not re-return items already
served to that (org, session); the widened fetch pool backfills next-best
matches instead. Requests without a ``session_id`` are unaffected and record
nothing. Covered on both Phase B routing paths (per-arm fan-out and the
combined single RPC).
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.server.services.pre_retrieval import ReformulationResult
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.unified_search_service import (
    configure_retrieval_capture_hook,
    run_unified_search,
)

_ORG = "test-org"


def _profiles(count: int) -> list[UserProfile]:
    return [
        UserProfile(
            profile_id=f"p{i}",
            user_id="u1",
            content=f"profile {i}",
            last_modified_timestamp=1000 + i,
            generated_from_request_id="req-1",
        )
        for i in range(1, count + 1)
    ]


def _user_playbooks(count: int) -> list[UserPlaybook]:
    return [
        UserPlaybook(
            user_playbook_id=10 + i,
            user_id="u1",
            agent_version="agent-v1",
            request_id="req-1",
            content=f"user playbook {10 + i}",
        )
        for i in range(1, count + 1)
    ]


def _agent_playbooks(count: int) -> list[AgentPlaybook]:
    return [
        AgentPlaybook(
            agent_playbook_id=i,
            agent_version="agent-v1",
            content=f"agent playbook {i}",
        )
        for i in range(1, count + 1)
    ]


def _mock_storage(*, single_rpc: bool) -> MagicMock:
    storage = MagicMock()
    storage._get_embedding.return_value = [0.1] * 8
    storage.supports_unified_hybrid_search = single_rpc
    storage.search_user_profile.return_value = _profiles(5)
    storage.search_agent_playbooks.return_value = _agent_playbooks(5)
    storage.search_user_playbooks.return_value = _user_playbooks(5)
    storage.unified_hybrid_search.return_value = (
        _profiles(5),
        _agent_playbooks(5),
        _user_playbooks(5),
    )
    # The source-playbook suppression lookup must return a real mapping.
    storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    return storage


def _search(storage: MagicMock, session_id: str | None) -> tuple:
    request = UnifiedSearchRequest(
        query="how to refund",
        user_id="u1",
        top_k=2,
        session_id=session_id,
    )
    response = run_unified_search(
        request=request,
        org_id=_ORG,
        storage=storage,
        llm_client=MagicMock(),
        prompt_manager=MagicMock(),
    )
    assert response.success
    return (
        [p.profile_id for p in response.profiles or []],
        [p.agent_playbook_id for p in response.agent_playbooks or []],
        [p.user_playbook_id for p in response.user_playbooks or []],
    )


@patch("reflexio.server.services.unified_search_service.QueryReformulator")
class TestUnifiedSearchSessionDedup(unittest.TestCase):
    def setUp(self):
        session_seen_cache.clear()
        configure_retrieval_capture_hook(None)

    def tearDown(self):
        session_seen_cache.clear()

    def _prep(self, reformulator_cls) -> None:
        reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="how to refund"
        )

    def _assert_dedup_behavior(self, reformulator_cls, storage: MagicMock) -> None:
        self._prep(reformulator_cls)

        first = _search(storage, "s1")
        self.assertEqual(first, (["p1", "p2"], [1, 2], [11, 12]))

        # Same session: previously served items skipped, next-best backfilled.
        second = _search(storage, "s1")
        self.assertEqual(second, (["p3", "p4"], [3, 4], [13, 14]))

        # A concurrent session is unaffected by s1's history.
        other = _search(storage, "s2")
        self.assertEqual(other, (["p1", "p2"], [1, 2], [11, 12]))

        # Exhausted pool: everything remaining, then nothing.
        third = _search(storage, "s1")
        self.assertEqual(third, (["p5"], [5], [15]))
        fourth = _search(storage, "s1")
        self.assertEqual(fourth, ([], [], []))

    @patch.dict(os.environ, {"REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC": "0"})
    def test_dedup_and_backfill_fanout_path(self, reformulator_cls):
        self._assert_dedup_behavior(reformulator_cls, _mock_storage(single_rpc=False))

    @patch.dict(os.environ, {"REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC": "1"})
    def test_dedup_and_backfill_single_rpc_path(self, reformulator_cls):
        storage = _mock_storage(single_rpc=True)
        self._assert_dedup_behavior(reformulator_cls, storage)
        storage.unified_hybrid_search.assert_called()

    @patch.dict(os.environ, {"REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC": "0"})
    def test_no_session_id_is_unaffected_and_records_nothing(self, reformulator_cls):
        self._prep(reformulator_cls)
        storage = _mock_storage(single_rpc=False)

        self.assertEqual(_search(storage, None), (["p1", "p2"], [1, 2], [11, 12]))
        self.assertEqual(_search(storage, None), (["p1", "p2"], [1, 2], [11, 12]))
        self.assertEqual(len(session_seen_cache._sessions), 0)

        # Whitespace-only session ids are treated as absent.
        self.assertEqual(_search(storage, "   "), (["p1", "p2"], [1, 2], [11, 12]))
        self.assertEqual(len(session_seen_cache._sessions), 0)

    @patch.dict(os.environ, {"REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC": "0"})
    def test_items_with_default_ids_are_never_filtered_or_recorded(
        self, reformulator_cls
    ):
        """Rows carrying model-default ids (0 / "") have no stable identity.

        They must pass through dedup untouched on every call and must not be
        recorded — otherwise all id-0 rows would count as mutually "seen".
        """
        self._prep(reformulator_cls)
        storage = _mock_storage(single_rpc=False)
        unidentified = UserPlaybook(
            user_playbook_id=0,
            user_id="u1",
            agent_version="agent-v1",
            request_id="req-1",
            content="user playbook without a stable id",
        )
        storage.search_user_playbooks.return_value = [unidentified]
        storage.search_user_profile.return_value = []
        storage.search_agent_playbooks.return_value = []

        for _ in range(2):
            _, _, user_playbook_ids = _search(storage, "s1")
            self.assertEqual(user_playbook_ids, [0])
        self.assertEqual(session_seen_cache.seen(_ORG, "s1"), frozenset())

    @patch.dict(os.environ, {"REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC": "0"})
    def test_fetch_pool_widens_by_seen_count(self, reformulator_cls):
        self._prep(reformulator_cls)
        storage = _mock_storage(single_rpc=False)

        _search(storage, "s1")
        first_top_k = storage.search_user_playbooks.call_args.args[0].top_k
        self.assertEqual(first_top_k, 2)

        _search(storage, "s1")
        # 6 items (2 per arm) were recorded, so the pool widens by 6.
        second_top_k = storage.search_user_playbooks.call_args.args[0].top_k
        self.assertEqual(second_top_k, 8)


if __name__ == "__main__":
    unittest.main()
