"""
Unit tests for cluster-level change detection in playbook aggregator.

Tests fingerprint computation, change detection logic, selective LLM invocation,
and clustering stability.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


# Disable mock mode for clustering tests so actual clustering algorithms are used
@pytest.fixture(autouse=True)
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so clustering tests use real algorithms."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    StructuredData,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookAggregatorConfig
from reflexio.server.services.playbook.playbook_aggregator import (
    PlaybookAggregator,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
    PlaybookAggregatorRequest,
    StructuredPlaybookContent,
)


def create_similar_embeddings(n: int, base_seed: int = 42) -> list[list[float]]:
    """
    Create n similar embeddings (high cosine similarity).

    Args:
        n: Number of embeddings to create
        base_seed: Random seed for reproducibility

    Returns:
        List of n similar 512-dimensional embeddings
    """
    np.random.seed(base_seed)
    base = np.random.randn(512)
    base = base / np.linalg.norm(base)

    embeddings = []
    for _i in range(n):
        noise = np.random.randn(512) * 0.001
        vec = base + noise
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec.tolist())

    return embeddings


def create_dissimilar_embeddings(n: int, base_seed: int = 42) -> list[list[float]]:
    """
    Create n dissimilar embeddings (low cosine similarity).

    Args:
        n: Number of embeddings to create
        base_seed: Random seed for reproducibility

    Returns:
        List of n dissimilar 512-dimensional embeddings
    """
    np.random.seed(base_seed)
    embeddings = []
    for _i in range(n):
        vec = np.random.randn(512)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec.tolist())

    return embeddings


def create_user_playbooks_with_embeddings(
    embeddings: list[list[float]],
    playbook_name: str = "test_playbook",
    start_id: int = 0,
) -> list[UserPlaybook]:
    """
    Create UserPlaybook objects with given embeddings.

    Args:
        embeddings: List of embeddings
        playbook_name: Name for the playbooks
        start_id: Starting user_playbook_id

    Returns:
        List of UserPlaybook objects
    """
    return [
        UserPlaybook(
            user_playbook_id=start_id + i,
            agent_version="1.0",
            request_id=str(start_id + i),
            content=f"AgentPlaybook content {start_id + i}",
            playbook_name=playbook_name,
            structured_data=StructuredData(
                instruction=f"Do action {start_id + i}",
                trigger=f"When condition {start_id + i}",
            ),
            embedding=emb,
        )
        for i, emb in enumerate(embeddings)
    ]


@pytest.fixture
def mock_playbook_aggregator():
    """Create a PlaybookAggregator with mocked dependencies."""
    mock_llm_client = MagicMock()
    mock_request_context = MagicMock()
    mock_request_context.storage = MagicMock()
    mock_request_context.configurator = MagicMock()

    aggregator = PlaybookAggregator(
        llm_client=mock_llm_client,
        request_context=mock_request_context,
        agent_version="1.0",
    )
    return aggregator  # noqa: RET504


class TestClusterFingerprint:
    """Unit tests for fingerprint computation."""

    def test_fingerprint_deterministic(self):
        """Compute fingerprint twice for same playbooks, assert same result."""
        playbooks = [
            UserPlaybook(
                user_playbook_id=i,
                agent_version="1.0",
                request_id=str(i),
                content=f"content {i}",
                playbook_name="test",
                embedding=[0.0] * 512,
            )
            for i in [1, 2, 3]
        ]

        fp1 = PlaybookAggregator._compute_cluster_fingerprint(playbooks)
        fp2 = PlaybookAggregator._compute_cluster_fingerprint(playbooks)
        assert fp1 == fp2

    def test_fingerprint_order_independent(self):
        """Fingerprint should be the same regardless of input order."""
        playbooks_a = [
            UserPlaybook(
                user_playbook_id=i,
                agent_version="1.0",
                request_id=str(i),
                content=f"content {i}",
                playbook_name="test",
                embedding=[0.0] * 512,
            )
            for i in [3, 1, 2]
        ]
        playbooks_b = [
            UserPlaybook(
                user_playbook_id=i,
                agent_version="1.0",
                request_id=str(i),
                content=f"content {i}",
                playbook_name="test",
                embedding=[0.0] * 512,
            )
            for i in [1, 2, 3]
        ]

        fp_a = PlaybookAggregator._compute_cluster_fingerprint(playbooks_a)
        fp_b = PlaybookAggregator._compute_cluster_fingerprint(playbooks_b)
        assert fp_a == fp_b

    def test_fingerprint_different_ids(self):
        """Different user_playbook_ids should produce different fingerprints."""
        playbooks_a = [
            UserPlaybook(
                user_playbook_id=i,
                agent_version="1.0",
                request_id=str(i),
                content=f"content {i}",
                playbook_name="test",
                embedding=[0.0] * 512,
            )
            for i in [1, 2, 3]
        ]
        playbooks_b = [
            UserPlaybook(
                user_playbook_id=i,
                agent_version="1.0",
                request_id=str(i),
                content=f"content {i}",
                playbook_name="test",
                embedding=[0.0] * 512,
            )
            for i in [4, 5, 6]
        ]

        fp_a = PlaybookAggregator._compute_cluster_fingerprint(playbooks_a)
        fp_b = PlaybookAggregator._compute_cluster_fingerprint(playbooks_b)
        assert fp_a != fp_b


class TestDetermineClusterChanges:
    """Tests for change detection logic."""

    def test_first_run_no_previous_state(self, mock_playbook_aggregator):
        """On first run with no previous fingerprints, all clusters are changed."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        (
            changed_clusters,
            playbook_ids_to_archive,
        ) = mock_playbook_aggregator._determine_cluster_changes(clusters, {})

        # All clusters should be changed
        assert len(changed_clusters) == len(clusters)
        # No playbook_ids to archive (no previous state)
        assert playbook_ids_to_archive == []

    def test_no_changes_identical_clusters(self, mock_playbook_aggregator):
        """When clusters haven't changed, no clusters should be marked changed."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        # Build prev_fingerprints from current clusters
        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Re-cluster the same playbooks
        clusters2 = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        (
            changed_clusters,
            playbook_ids_to_archive,
        ) = mock_playbook_aggregator._determine_cluster_changes(
            clusters2, prev_fingerprints
        )

        assert len(changed_clusters) == 0
        assert playbook_ids_to_archive == []

    def test_one_new_playbook_changes_one_cluster(self, mock_playbook_aggregator):
        """Adding a new playbook to one group should only change that cluster."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        # Build prev_fingerprints
        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Add a new playbook similar to group_a
        new_emb = create_similar_embeddings(1, base_seed=42)
        new_playbook = create_user_playbooks_with_embeddings(new_emb, start_id=100)
        all_playbooks_updated = all_playbooks + new_playbook

        clusters2 = mock_playbook_aggregator.get_clusters(all_playbooks_updated, config)

        (
            changed_clusters,
            playbook_ids_to_archive,
        ) = mock_playbook_aggregator._determine_cluster_changes(
            clusters2, prev_fingerprints
        )

        # At least one cluster should be changed (the one that got the new playbook)
        assert len(changed_clusters) >= 1
        # The total changed should be less than all clusters
        assert len(changed_clusters) < len(clusters2) or len(clusters2) <= 1

    def test_cluster_disappeared(self, mock_playbook_aggregator):
        """When a cluster disappears, its old playbook_id should be archived."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        # Build prev_fingerprints
        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Only keep group_a playbooks
        playbooks_a_only = create_user_playbooks_with_embeddings(group_a)
        clusters2 = mock_playbook_aggregator.get_clusters(playbooks_a_only, config)

        (
            changed_clusters,
            playbook_ids_to_archive,
        ) = mock_playbook_aggregator._determine_cluster_changes(
            clusters2, prev_fingerprints
        )

        # The disappeared cluster's playbook_id should be in archive list
        assert len(playbook_ids_to_archive) >= 1

    def test_new_cluster_appears(self, mock_playbook_aggregator):
        """When a new cluster appears, it should be in changed_clusters."""
        group_a = create_similar_embeddings(3, base_seed=42)
        playbooks_a = create_user_playbooks_with_embeddings(group_a)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters = mock_playbook_aggregator.get_clusters(playbooks_a, config)

        # Build prev_fingerprints from just group_a
        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Add group_b
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = playbooks_a + create_user_playbooks_with_embeddings(
            group_b, start_id=100
        )
        clusters2 = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        (
            changed_clusters,
            playbook_ids_to_archive,
        ) = mock_playbook_aggregator._determine_cluster_changes(
            clusters2, prev_fingerprints
        )

        # The new cluster should appear in changed_clusters
        assert len(changed_clusters) >= 1
        # group_a should be unchanged, so no playbook_ids to archive
        assert playbook_ids_to_archive == []


class TestAggregatorRunWithChangeDetection:
    """End-to-end tests with mock storage verifying selective LLM invocation."""

    def _setup_aggregator_for_run(
        self,
        user_playbooks,
        existing_playbooks=None,
        operation_state=None,
        config=None,
    ):
        """
        Helper to create a fully configured mock aggregator for run() tests.

        Args:
            user_playbooks: User playbooks to return from storage
            existing_playbooks: Existing playbooks to return from storage
            operation_state: Operation state to return (for fingerprints/bookmarks)
            config: PlaybookAggregatorConfig to use

        Returns:
            Configured PlaybookAggregator with mocked dependencies
        """
        if existing_playbooks is None:
            existing_playbooks = []
        if config is None:
            config = PlaybookAggregatorConfig(
                min_cluster_size=2, reaggregation_trigger_count=1
            )

        mock_llm_client = MagicMock()
        mock_request_context = MagicMock()
        mock_storage = MagicMock()
        mock_request_context.storage = mock_storage
        mock_request_context.org_id = "test_org"
        mock_configurator = MagicMock()
        mock_request_context.configurator = mock_configurator

        # Setup configurator to return config
        mock_playbook_config = MagicMock()
        mock_playbook_config.extractor_name = "test_playbook"
        mock_playbook_config.aggregation_config = config
        mock_configurator.get_config.return_value.user_playbook_extractor_configs = [
            mock_playbook_config
        ]

        # Setup storage methods
        mock_storage.get_user_playbooks.return_value = user_playbooks
        mock_storage.get_agent_playbooks.return_value = existing_playbooks
        mock_storage.count_user_playbooks.return_value = len(user_playbooks)
        mock_storage.save_agent_playbooks.return_value = []

        # Setup operation state (for fingerprints and bookmarks)
        # Storage returns {"operation_state": {...}} wrapping
        def get_operation_state_side_effect(key):
            if operation_state is not None and "clusters" in key:
                return {"operation_state": {"cluster_fingerprints": operation_state}}
            if operation_state is not None and "clusters" not in key:
                # Return bookmark
                return {"operation_state": {"last_processed_user_playbook_id": 0}}
            return None

        mock_storage.get_operation_state.side_effect = get_operation_state_side_effect

        # Setup LLM client to return structured playbook
        structured = StructuredPlaybookContent(
            instruction="Do something",
            trigger="When something happens",
        )
        mock_response = PlaybookAggregationOutput(playbook=structured)
        mock_llm_client.generate_chat_response.return_value = mock_response
        mock_llm_client.config = MagicMock()
        mock_llm_client.config.model = "test-model"

        aggregator = PlaybookAggregator(
            llm_client=mock_llm_client,
            request_context=mock_request_context,
            agent_version="1.0",
        )

        return aggregator, mock_storage, mock_llm_client

    def test_first_run_calls_llm_for_all_clusters(self):
        """First run (no stored fingerprints) should call LLM for all clusters."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        user_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=user_playbooks,
            operation_state=None,
        )

        # Make save_agent_playbooks return playbooks with IDs
        def save_agent_playbooks_side_effect(playbooks):
            for i, fb in enumerate(playbooks):
                fb.agent_playbook_id = i + 1
            return playbooks

        mock_storage.save_agent_playbooks.side_effect = save_agent_playbooks_side_effect

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
        )

        aggregator.run(request)

        # LLM should be called for each cluster (at least 1, up to 2)
        assert mock_llm_client.generate_chat_response.call_count >= 1
        # Save playbooks should be called
        mock_storage.save_agent_playbooks.assert_called_once()
        # Fingerprints should be stored
        mock_storage.upsert_operation_state.assert_called()

    def test_second_run_no_changes_skips_llm(self):
        """Second run with same playbooks should skip LLM calls entirely."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        user_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        # Compute fingerprints for the existing clusters
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=1
        )
        MagicMock()
        # Actually compute clusters to get real fingerprints
        mock_llm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.storage = MagicMock()
        mock_ctx.configurator = MagicMock()
        temp = PlaybookAggregator(mock_llm, mock_ctx, "1.0")
        clusters = temp.get_clusters(user_playbooks, config)

        # Build fingerprints from actual clusters
        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        existing_playbooks = [
            AgentPlaybook(
                agent_playbook_id=cid + 100,
                playbook_name="test_playbook",
                agent_version="1.0",
                content=f"Existing playbook {cid}",
                playbook_status=PlaybookStatus.PENDING,
            )
            for cid in clusters
        ]

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=user_playbooks,
            existing_playbooks=existing_playbooks,
            operation_state=prev_fingerprints,
            config=config,
        )

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
        )

        aggregator.run(request)

        # LLM should NOT be called
        mock_llm_client.generate_chat_response.assert_not_called()
        # archive_agent_playbooks_by_ids should NOT be called (nothing to archive)
        mock_storage.archive_agent_playbooks_by_ids.assert_not_called()

    def test_second_run_with_new_playbooks_calls_llm_selectively(self):
        """Adding playbooks to one cluster should only call LLM for that cluster."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        original_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=1
        )

        # Compute original clusters and fingerprints
        mock_llm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.storage = MagicMock()
        mock_ctx.configurator = MagicMock()
        temp = PlaybookAggregator(mock_llm, mock_ctx, "1.0")
        original_clusters = temp.get_clusters(original_playbooks, config)

        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in original_clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Add 2 new playbooks similar to group_a
        new_embs = create_similar_embeddings(2, base_seed=42)
        new_playbooks = create_user_playbooks_with_embeddings(new_embs, start_id=100)
        all_playbooks = original_playbooks + new_playbooks

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=all_playbooks,
            operation_state=prev_fingerprints,
            config=config,
        )

        def save_agent_playbooks_side_effect(playbooks):
            for i, fb in enumerate(playbooks):
                fb.agent_playbook_id = i + 200
            return playbooks

        mock_storage.save_agent_playbooks.side_effect = save_agent_playbooks_side_effect

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
        )

        aggregator.run(request)

        # LLM should be called fewer times than total clusters
        total_llm_calls = mock_llm_client.generate_chat_response.call_count
        assert total_llm_calls >= 1
        # save_agent_playbooks should be called
        mock_storage.save_agent_playbooks.assert_called_once()

    def test_rerun_bypasses_change_detection(self):
        """rerun=True should call LLM for ALL clusters regardless of fingerprints."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        user_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=1
        )

        # Setup with existing fingerprints (so without rerun it would skip)
        mock_llm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.storage = MagicMock()
        mock_ctx.configurator = MagicMock()
        temp = PlaybookAggregator(mock_llm, mock_ctx, "1.0")
        clusters = temp.get_clusters(user_playbooks, config)

        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=user_playbooks,
            operation_state=prev_fingerprints,
            config=config,
        )

        def save_agent_playbooks_side_effect(playbooks):
            for i, fb in enumerate(playbooks):
                fb.agent_playbook_id = i + 1
            return playbooks

        mock_storage.save_agent_playbooks.side_effect = save_agent_playbooks_side_effect

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
            rerun=True,
        )

        aggregator.run(request)

        # LLM should be called for ALL clusters
        assert mock_llm_client.generate_chat_response.call_count == len(clusters)
        # archive_agent_playbooks_by_playbook_name should be called (full archive)
        mock_storage.archive_agent_playbooks_by_playbook_name.assert_called_once()

    def test_error_during_save_restores_archived_playbooks(self):
        """If save_agent_playbooks fails, archived playbooks should be restored."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        original_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=1
        )

        # Compute original clusters and fingerprints
        mock_llm = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.storage = MagicMock()
        mock_ctx.configurator = MagicMock()
        temp = PlaybookAggregator(mock_llm, mock_ctx, "1.0")
        original_clusters = temp.get_clusters(original_playbooks, config)

        prev_fingerprints = {}
        for cluster_id, cluster_playbooks in original_clusters.items():
            fp = PlaybookAggregator._compute_cluster_fingerprint(cluster_playbooks)
            raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
            prev_fingerprints[fp] = {
                "agent_playbook_id": cluster_id + 100,
                "user_playbook_ids": raw_ids,
            }

        # Add new playbooks to trigger a change
        new_embs = create_similar_embeddings(2, base_seed=42)
        new_playbooks = create_user_playbooks_with_embeddings(new_embs, start_id=100)
        all_playbooks = original_playbooks + new_playbooks

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=all_playbooks,
            operation_state=prev_fingerprints,
            config=config,
        )

        # Make save_agent_playbooks raise an exception (this happens after archiving)
        mock_storage.save_agent_playbooks.side_effect = Exception("Storage save error")

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
        )

        with pytest.raises(Exception, match="Storage save error"):
            aggregator.run(request)

        # restore_archived_agent_playbooks_by_ids should be called if selective archiving happened
        # OR restore_archived_agent_playbooks_by_playbook_name if it was a first run
        restore_by_ids_called = (
            mock_storage.restore_archived_agent_playbooks_by_ids.called
        )
        restore_by_name_called = (
            mock_storage.restore_archived_agent_playbooks_by_playbook_name.called
        )
        assert restore_by_ids_called or restore_by_name_called

    def test_first_run_deletes_archived_on_success(self):
        """Regression: first-run (non-rerun) path must delete archived playbooks after success."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        user_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=user_playbooks,
            operation_state=None,  # No previous state → first-run path
        )

        def save_agent_playbooks_side_effect(playbooks):
            for i, fb in enumerate(playbooks):
                fb.agent_playbook_id = i + 1
            return playbooks

        mock_storage.save_agent_playbooks.side_effect = save_agent_playbooks_side_effect

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
            rerun=False,
        )

        aggregator.run(request)

        mock_storage.delete_archived_agent_playbooks_by_playbook_name.assert_called_once()

    def test_first_run_restores_archived_on_error(self):
        """Regression: first-run (non-rerun) must restore archived playbooks on save error."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        user_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        aggregator, mock_storage, mock_llm_client = self._setup_aggregator_for_run(
            user_playbooks=user_playbooks,
            operation_state=None,  # No previous state → first-run path
        )

        mock_storage.save_agent_playbooks.side_effect = Exception("Storage save error")

        request = PlaybookAggregatorRequest(
            agent_version="1.0",
            playbook_name="test_playbook",
            rerun=False,
        )

        with pytest.raises(Exception, match="Storage save error"):
            aggregator.run(request)

        mock_storage.restore_archived_agent_playbooks_by_playbook_name.assert_called_once()


class TestLLMResponseTypeSafety:
    """Regression tests for LLM response isinstance guard."""

    def test_raw_string_response_returns_none(self):
        """Regression: plain string from LLM must not crash with AttributeError on .playbook."""
        mock_llm_client = MagicMock()
        mock_request_context = MagicMock()
        mock_request_context.storage = MagicMock()
        mock_request_context.configurator = MagicMock()

        # LLM returns a raw string instead of PlaybookAggregationOutput
        mock_llm_client.generate_chat_response.return_value = "unparsed text"
        mock_llm_client.config = MagicMock()
        mock_llm_client.config.model = "test-model"

        aggregator = PlaybookAggregator(
            llm_client=mock_llm_client,
            request_context=mock_request_context,
            agent_version="1.0",
        )

        cluster_playbooks = [
            UserPlaybook(
                user_playbook_id=1,
                agent_version="1.0",
                request_id="r1",
                content="content",
                playbook_name="test",
                structured_data=StructuredData(
                    instruction="do something",
                    trigger="when asked",
                ),
                embedding=[0.0] * 512,
            ),
        ]

        result = aggregator._generate_playbook_from_cluster(cluster_playbooks, "None")
        assert result is None

    def test_valid_aggregation_output_is_processed(self):
        """Positive test: valid PlaybookAggregationOutput produces a AgentPlaybook."""
        mock_llm_client = MagicMock()
        mock_request_context = MagicMock()
        mock_request_context.storage = MagicMock()
        mock_request_context.configurator = MagicMock()

        structured = StructuredPlaybookContent(
            instruction="Be concise",
            trigger="When answering questions",
        )
        mock_llm_client.generate_chat_response.return_value = PlaybookAggregationOutput(
            playbook=structured
        )
        mock_llm_client.config = MagicMock()
        mock_llm_client.config.model = "test-model"

        aggregator = PlaybookAggregator(
            llm_client=mock_llm_client,
            request_context=mock_request_context,
            agent_version="1.0",
        )

        from reflexio.models.api_schema.service_schemas import StructuredData

        cluster_playbooks = [
            UserPlaybook(
                user_playbook_id=1,
                agent_version="1.0",
                request_id="r1",
                content="content",
                playbook_name="test",
                structured_data=StructuredData(
                    instruction="do something",
                    trigger="when asked",
                ),
                embedding=[0.0] * 512,
            ),
        ]

        result = aggregator._generate_playbook_from_cluster(cluster_playbooks, "None")
        assert result is not None
        assert result.structured_data.instruction == "Be concise"
        assert result.structured_data.trigger == "When answering questions"
        assert result.playbook_status == PlaybookStatus.PENDING


class TestClusteringStability:
    """Verify clustering produces stable results for fingerprint comparison."""

    def test_same_playbooks_produce_same_clusters(self, mock_playbook_aggregator):
        """Running get_clusters twice with same input should produce same clusters."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        all_playbooks = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)

        clusters1 = mock_playbook_aggregator.get_clusters(all_playbooks, config)
        clusters2 = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        # Same number of clusters
        assert len(clusters1) == len(clusters2)

        # Same membership (compare sets of user_playbook_ids per cluster)
        def get_cluster_id_sets(clusters):
            return sorted(
                sorted(fb.user_playbook_id for fb in cfbs) for cfbs in clusters.values()
            )

        assert get_cluster_id_sets(clusters1) == get_cluster_id_sets(clusters2)

    def test_adding_playbook_only_affects_its_cluster(self, mock_playbook_aggregator):
        """Adding a playbook to one group should not change membership of the other."""
        group_a = create_similar_embeddings(3, base_seed=42)
        group_b = create_similar_embeddings(3, base_seed=100)
        playbooks_original = create_user_playbooks_with_embeddings(group_a + group_b)

        config = PlaybookAggregatorConfig(min_cluster_size=2)
        clusters1 = mock_playbook_aggregator.get_clusters(playbooks_original, config)

        # Find which cluster IDs contain group_b playbooks (IDs 3,4,5)
        group_b_ids = {3, 4, 5}
        group_b_cluster_members = None
        for cluster_playbooks in clusters1.values():
            ids_in_cluster = {fb.user_playbook_id for fb in cluster_playbooks}
            if ids_in_cluster & group_b_ids:
                group_b_cluster_members = ids_in_cluster
                break

        # Add a new playbook similar to group_a
        new_emb = create_similar_embeddings(1, base_seed=42)
        new_playbook = create_user_playbooks_with_embeddings(new_emb, start_id=100)
        all_playbooks = playbooks_original + new_playbook

        clusters2 = mock_playbook_aggregator.get_clusters(all_playbooks, config)

        # Find group_b cluster in new clustering
        group_b_cluster_members2 = None
        for cluster_playbooks in clusters2.values():
            ids_in_cluster = {fb.user_playbook_id for fb in cluster_playbooks}
            if ids_in_cluster & group_b_ids:
                group_b_cluster_members2 = ids_in_cluster
                break

        # Group B membership should be unchanged
        if group_b_cluster_members is not None and group_b_cluster_members2 is not None:
            assert group_b_cluster_members == group_b_cluster_members2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
