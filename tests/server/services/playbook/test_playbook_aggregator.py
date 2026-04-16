"""
Unit tests for PlaybookAggregator private helpers and run() orchestration.

Targets coverage gaps in:
- _should_run_aggregation (reaggregation_trigger_count defaults, threshold logic)
- _determine_cluster_changes (no previous clusters, fingerprint match/mismatch)
- _build_change_log (empty changes, full archive, incremental with updates/removals)
- _update_operation_state (empty list, normal update)
- _get_playbook_aggregator_config (match, no match, no configs)
- _compute_cluster_fingerprint (deterministic, order-independent)
- run() (rerun mode, no user playbooks, incremental no changes, save exception,
         change log exception, full archive delete path, incremental archive delete)
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    PlaybookAggregatorConfig,
    PlaybookConfig,
)
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aggregator(
    storage: MagicMock | None = None,
    configurator: MagicMock | None = None,
) -> PlaybookAggregator:
    """Build an aggregator with fully mocked dependencies."""
    llm = MagicMock()
    ctx = MagicMock()
    ctx.storage = storage or MagicMock()
    ctx.configurator = configurator or MagicMock()
    ctx.org_id = "test-org"
    return PlaybookAggregator(
        llm_client=llm,
        request_context=ctx,
        agent_version="v1",
    )


def _raw(
    rid: int = 1,
    name: str = "test_fb",
    when: str | None = "when cond",
    do: str | None = "do action",
    dont: str | None = None,
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=rid,
        agent_version="v1",
        request_id=f"req-{rid}",
        playbook_name=name,
        content=f"content-{rid}",
        trigger=when,
    )


def _agent_playbook(
    fid: int = 1, name: str = "test_fb", content: str = "c"
) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=fid,
        playbook_name=name,
        agent_version="v1",
        content=content,
        playbook_status=PlaybookStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# _should_run_aggregation
# ---------------------------------------------------------------------------


class TestShouldRunAggregation:
    """Tests for _should_run_aggregation."""

    def test_reaggregation_trigger_count_zero_defaults_to_two(self):
        """When reaggregation_trigger_count <= 0 the method should default to 2."""
        agg = _make_aggregator()
        # Bypass Pydantic ge=1 validation to hit the <= 0 guard in source
        config = PlaybookAggregatorConfig.model_construct(
            min_cluster_size=2, reaggregation_trigger_count=0
        )
        agg.storage.count_user_playbooks.return_value = 2

        result = agg._should_run_aggregation("fb", config)

        assert result is True
        # count >= default(2) -> True

    def test_reaggregation_trigger_count_negative_defaults_to_two(self):
        """Negative reaggregation_trigger_count also defaults to 2."""
        agg = _make_aggregator()
        # Bypass Pydantic ge=1 validation to hit the <= 0 guard in source
        config = PlaybookAggregatorConfig.model_construct(
            min_cluster_size=2, reaggregation_trigger_count=-1
        )
        agg.storage.count_user_playbooks.return_value = 2

        result = agg._should_run_aggregation("fb", config)

        assert result is True

    def test_enough_new_playbooks_returns_true(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=3
        )
        agg.storage.count_user_playbooks.return_value = 5

        assert agg._should_run_aggregation("fb", config) is True

    def test_not_enough_new_playbooks_returns_false(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=3
        )
        agg.storage.count_user_playbooks.return_value = 1

        assert agg._should_run_aggregation("fb", config) is False

    def test_rerun_flag_passed_to_count(self):
        """rerun=True should be forwarded so all playbooks are counted."""
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=2
        )
        agg.storage.count_user_playbooks.return_value = 10

        agg._should_run_aggregation("fb", config, rerun=True)

        # rerun=True -> last_processed_id=0
        call_kwargs = agg.storage.count_user_playbooks.call_args
        assert (
            call_kwargs.kwargs.get("min_user_playbook_id") == 0
            or call_kwargs[1].get("min_user_playbook_id") == 0
        )


# ---------------------------------------------------------------------------
# _get_new_user_playbooks_count
# ---------------------------------------------------------------------------


class TestGetNewUserPlaybooksCount:
    def test_rerun_uses_zero_as_last_processed(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 7

        result = agg._get_new_user_playbooks_count("fb", rerun=True)

        assert result == 7
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 0
        )

    def test_non_rerun_reads_bookmark(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 3

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_aggregator_bookmark.return_value = 42
            mock_csm.return_value = mgr

            result = agg._get_new_user_playbooks_count("fb", rerun=False)

        assert result == 3
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 42
        )

    def test_non_rerun_bookmark_none_defaults_to_zero(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 5

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_aggregator_bookmark.return_value = None
            mock_csm.return_value = mgr

            result = agg._get_new_user_playbooks_count("fb", rerun=False)

        assert result == 5
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 0
        )


# ---------------------------------------------------------------------------
# _update_operation_state
# ---------------------------------------------------------------------------


class TestUpdateOperationState:
    def test_empty_list_returns_early(self):
        agg = _make_aggregator()
        agg._update_operation_state("fb", [])
        # No state manager interaction expected

    def test_updates_with_max_id(self):
        agg = _make_aggregator()
        raws = [_raw(rid=3), _raw(rid=10), _raw(rid=7)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mock_csm.return_value = mgr

            agg._update_operation_state("fb", raws)

        mgr.update_aggregator_bookmark.assert_called_once_with(
            name="fb", version="v1", last_processed_id=10
        )


# ---------------------------------------------------------------------------
# _compute_cluster_fingerprint
# ---------------------------------------------------------------------------


class TestComputeClusterFingerprint:
    def test_deterministic(self):
        raws = [_raw(rid=1), _raw(rid=2), _raw(rid=3)]
        fp1 = PlaybookAggregator._compute_cluster_fingerprint(raws)
        fp2 = PlaybookAggregator._compute_cluster_fingerprint(raws)
        assert fp1 == fp2

    def test_order_independent(self):
        raws_a = [_raw(rid=1), _raw(rid=3), _raw(rid=2)]
        raws_b = [_raw(rid=3), _raw(rid=1), _raw(rid=2)]
        assert PlaybookAggregator._compute_cluster_fingerprint(
            raws_a
        ) == PlaybookAggregator._compute_cluster_fingerprint(raws_b)

    def test_different_ids_produce_different_fingerprint(self):
        fp_a = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=1)])
        fp_b = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=2)])
        assert fp_a != fp_b

    def test_fingerprint_length(self):
        fp = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=1)])
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# _determine_cluster_changes
# ---------------------------------------------------------------------------


class TestDetermineClusterChanges:
    def test_no_previous_fingerprints(self):
        """Empty prev_fingerprints => all clusters are changed, none to archive."""
        agg = _make_aggregator()
        clusters = {0: [_raw(rid=1), _raw(rid=2)]}

        changed, to_archive = agg._determine_cluster_changes(clusters, {})

        assert changed == clusters
        assert to_archive == []

    def test_fingerprint_match_no_changes(self):
        """Matching fingerprint => no changed clusters, none to archive."""
        agg = _make_aggregator()
        raws = [_raw(rid=1), _raw(rid=2)]
        clusters = {0: raws}
        fp = PlaybookAggregator._compute_cluster_fingerprint(raws)
        prev = {fp: {"agent_playbook_id": 10, "user_playbook_ids": [1, 2]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert changed == {}
        assert to_archive == []

    def test_fingerprint_mismatch_detects_change(self):
        """New fingerprint => cluster is changed; old fingerprint archived."""
        agg = _make_aggregator()
        raws_new = [_raw(rid=1), _raw(rid=2), _raw(rid=3)]
        clusters = {0: raws_new}
        prev = {"old_fp_hash": {"agent_playbook_id": 5, "user_playbook_ids": [1, 2]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 in changed
        assert 5 in to_archive

    def test_disappeared_cluster_with_no_playbook_id(self):
        """Disappeared fingerprint with agent_playbook_id=None should not be archived."""
        agg = _make_aggregator()
        clusters = {0: [_raw(rid=99)]}
        prev = {"gone_fp": {"agent_playbook_id": None, "user_playbook_ids": [1]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 in changed
        assert to_archive == []

    def test_multiple_clusters_mixed(self):
        """Some clusters match, some do not."""
        agg = _make_aggregator()
        raws_unchanged = [_raw(rid=1)]
        raws_new = [_raw(rid=5), _raw(rid=6)]
        clusters = {0: raws_unchanged, 1: raws_new}

        fp_unchanged = PlaybookAggregator._compute_cluster_fingerprint(raws_unchanged)
        prev = {
            fp_unchanged: {"agent_playbook_id": 10, "user_playbook_ids": [1]},
            "vanished_fp": {"agent_playbook_id": 20, "user_playbook_ids": [2, 3]},
        }

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 not in changed
        assert 1 in changed
        assert 20 in to_archive


# ---------------------------------------------------------------------------
# _build_change_log
# ---------------------------------------------------------------------------


class TestBuildChangeLog:
    def test_full_archive_empty_before_and_saved(self):
        """Full archive with no previous or new playbooks."""
        agg = _make_aggregator()
        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=True,
            before_playbooks_by_id={},
            saved_playbooks=[],
            archived_playbook_ids=[],
            prev_fingerprints={},
        )
        assert log.run_mode == "full_archive"
        assert log.added_agent_playbooks == []
        assert log.removed_agent_playbooks == []
        assert log.updated_agent_playbooks == []

    def test_full_archive_all_new_clusters(self):
        """Full archive: old playbooks are removed, new ones added."""
        agg = _make_aggregator()
        old_fb = _agent_playbook(fid=1, content="old")
        new_fb = _agent_playbook(fid=2, content="new")

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=True,
            before_playbooks_by_id={1: old_fb},
            saved_playbooks=[new_fb],
            archived_playbook_ids=[],
            prev_fingerprints={},
        )

        assert len(log.removed_agent_playbooks) == 1
        assert log.removed_agent_playbooks[0].agent_playbook_id == 1
        assert len(log.added_agent_playbooks) == 1
        assert log.added_agent_playbooks[0].agent_playbook_id == 2

    def test_full_archive_filters_none_saved(self):
        """None entries in saved_playbooks should be filtered out."""
        agg = _make_aggregator()
        fb = _agent_playbook(fid=3)

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=True,
            before_playbooks_by_id={},
            saved_playbooks=[None, fb, None],  # type: ignore[list-item]
            archived_playbook_ids=[],
            prev_fingerprints={},
        )

        assert len(log.added_agent_playbooks) == 1

    def test_incremental_update_pairs_old_and_new(self):
        """Incremental mode pairs archived old playbooks with saved new ones."""
        agg = _make_aggregator()
        old_fb = _agent_playbook(fid=10, content="old")
        new_fb = _agent_playbook(fid=20, content="new")

        prev_fps = {"fp1": {"agent_playbook_id": 10, "user_playbook_ids": [1]}}

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={10: old_fb},
            saved_playbooks=[new_fb],
            archived_playbook_ids=[10],
            prev_fingerprints=prev_fps,
        )

        assert len(log.updated_agent_playbooks) == 1
        assert log.updated_agent_playbooks[0].before.agent_playbook_id == 10
        assert log.updated_agent_playbooks[0].after.agent_playbook_id == 20
        assert log.added_agent_playbooks == []
        assert log.removed_agent_playbooks == []

    def test_incremental_unmatched_archived_becomes_removed(self):
        """Archived playbook not paired with a new one should be a removal."""
        agg = _make_aggregator()
        old_fb = _agent_playbook(fid=10, content="old")

        prev_fps = {"fp1": {"agent_playbook_id": 10, "user_playbook_ids": [1]}}

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={10: old_fb},
            saved_playbooks=[],
            archived_playbook_ids=[10],
            prev_fingerprints=prev_fps,
        )

        assert len(log.removed_agent_playbooks) == 1
        assert log.removed_agent_playbooks[0].agent_playbook_id == 10
        assert log.updated_agent_playbooks == []
        assert log.added_agent_playbooks == []

    def test_incremental_saved_with_no_archived_becomes_added(self):
        """Saved playbook with nothing archived -> addition."""
        agg = _make_aggregator()
        new_fb = _agent_playbook(fid=20, content="new")

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={},
            saved_playbooks=[new_fb],
            archived_playbook_ids=[],
            prev_fingerprints={},
        )

        assert len(log.added_agent_playbooks) == 1
        assert log.added_agent_playbooks[0].agent_playbook_id == 20

    def test_incremental_filters_none_saved_playbooks(self):
        """None entries in saved_playbooks should be skipped in incremental mode."""
        agg = _make_aggregator()
        new_fb = _agent_playbook(fid=20, content="new")

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={},
            saved_playbooks=[None, new_fb],  # type: ignore[list-item]
            archived_playbook_ids=[],
            prev_fingerprints={},
        )

        assert len(log.added_agent_playbooks) == 1

    def test_incremental_paired_old_id_not_in_before_becomes_added(self):
        """If paired old_id exists but not in before_playbooks_by_id, treat as added."""
        agg = _make_aggregator()
        new_fb = _agent_playbook(fid=20, content="new")

        prev_fps = {"fp1": {"agent_playbook_id": 10, "user_playbook_ids": [1]}}

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={},  # 10 not present
            saved_playbooks=[new_fb],
            archived_playbook_ids=[10],
            prev_fingerprints=prev_fps,
        )

        assert len(log.added_agent_playbooks) == 1
        assert log.added_agent_playbooks[0].agent_playbook_id == 20

    def test_incremental_multiple_saved_skip_already_matched(self):
        """Branch 349->348: second saved_fb skips already-matched old_id."""
        agg = _make_aggregator()
        old_fb1 = _agent_playbook(fid=10, content="old1")
        old_fb2 = _agent_playbook(fid=11, content="old2")
        new_fb1 = _agent_playbook(fid=20, content="new1")
        new_fb2 = _agent_playbook(fid=21, content="new2")

        prev_fps = {
            "fp1": {"agent_playbook_id": 10, "user_playbook_ids": [1]},
            "fp2": {"agent_playbook_id": 11, "user_playbook_ids": [2]},
        }

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={10: old_fb1, 11: old_fb2},
            saved_playbooks=[new_fb1, new_fb2],
            archived_playbook_ids=[10, 11],
            prev_fingerprints=prev_fps,
        )

        assert len(log.updated_agent_playbooks) == 2
        assert log.added_agent_playbooks == []
        assert log.removed_agent_playbooks == []

    def test_incremental_archived_not_in_before_ignored(self):
        """Archived id not present in before_playbooks_by_id should be ignored for removals."""
        agg = _make_aggregator()

        log = agg._build_change_log(
            playbook_name="fb",
            full_archive=False,
            before_playbooks_by_id={},
            saved_playbooks=[],
            archived_playbook_ids=[999],
            prev_fingerprints={
                "fp1": {"agent_playbook_id": 999, "user_playbook_ids": [1]}
            },
        )

        assert log.removed_agent_playbooks == []


# ---------------------------------------------------------------------------
# _get_playbook_aggregator_config
# ---------------------------------------------------------------------------


class TestGetPlaybookAggregatorConfig:
    def test_returns_matching_config(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=3, reaggregation_trigger_count=5
        )
        afc = PlaybookConfig(
            extractor_name="my_fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = [afc]

        result = agg._get_playbook_aggregator_config("my_fb")

        assert result is fac

    def test_returns_none_when_no_match(self):
        agg = _make_aggregator()
        afc = PlaybookConfig(
            extractor_name="other",
            extraction_definition_prompt="prompt",
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = [afc]

        assert agg._get_playbook_aggregator_config("missing") is None

    def test_returns_none_when_no_playbook_configs(self):
        agg = _make_aggregator()
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = None

        assert agg._get_playbook_aggregator_config("any") is None


# ---------------------------------------------------------------------------
# run() orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the top-level run() method using mocks."""

    def _make_runnable_aggregator(self):
        """Return an aggregator wired for a successful run()."""
        agg = _make_aggregator()
        # config
        fac = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=2
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = [afc]
        # storage returns
        agg.storage.count_user_playbooks.return_value = 5
        agg.storage.get_agent_playbooks.return_value = []
        agg.storage.get_user_playbooks.return_value = [_raw(rid=1), _raw(rid=2)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=100)]
        return agg

    def test_no_config_returns_early(self):
        agg = _make_aggregator()
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = None

        req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    def test_min_threshold_below_two_returns_early(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=1, reaggregation_trigger_count=2
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = [afc]

        req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    def test_not_enough_new_playbooks_skips(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=10
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_configs = [afc]
        agg.storage.count_user_playbooks.return_value = 1

        req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_rerun_mode_archives_all(self, mock_gen, mock_clust):
        """rerun=True should call archive_agent_playbooks_by_playbook_name."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1)]}
        mock_gen.return_value = [_agent_playbook(fid=100)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=100)]

        req = PlaybookAggregatorRequest(
            agent_version="v1", playbook_name="fb", rerun=True
        )
        agg.run(req)

        agg.storage.archive_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_rerun_deletes_archived_playbooks_after_success(self, mock_gen, mock_clust):
        """After successful rerun, delete_archived_agent_playbooks_by_playbook_name is called."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1)]}
        mock_gen.return_value = [_agent_playbook(fid=100)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=100)]

        req = PlaybookAggregatorRequest(
            agent_version="v1", playbook_name="fb", rerun=True
        )
        agg.run(req)

        agg.storage.delete_archived_agent_playbooks_by_playbook_name.assert_called_once()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_first_run_no_prev_fingerprints_full_archive(self, mock_gen, mock_clust):
        """First run (no previous fingerprints) triggers full archive."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1), _raw(rid=2)]}
        mock_gen.return_value = [_agent_playbook(fid=100)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=100)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        agg.storage.archive_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    def test_incremental_no_changes_updates_bookmark_only(self, mock_clust):
        """When no cluster changes detected, update bookmark and return."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        agg.storage.get_user_playbooks.return_value = raws
        mock_clust.return_value = {0: raws}
        fp = PlaybookAggregator._compute_cluster_fingerprint(raws)

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                fp: {"agent_playbook_id": 10, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        # Should NOT call _generate_playbooks_from_clusters
        agg.storage.save_agent_playbooks.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_incremental_with_changes_archives_selectively(self, mock_gen, mock_clust):
        """Incremental mode with changed clusters archives only affected playbook_ids."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.return_value = [_agent_playbook(fid=200)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=200)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": 50, "user_playbook_ids": [1, 2]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        agg.storage.archive_agent_playbooks_by_ids.assert_called_once_with([50])
        agg.storage.delete_agent_playbooks_by_ids.assert_called_once_with([50])

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_save_exception_restores_full_archive(self, mock_gen, mock_clust):
        """Exception during save_agent_playbooks in full-archive mode restores playbooks."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1)]}
        mock_gen.side_effect = RuntimeError("LLM failed")

        req = PlaybookAggregatorRequest(
            agent_version="v1", playbook_name="fb", rerun=True
        )

        with pytest.raises(RuntimeError, match="LLM failed"):
            agg.run(req)

        agg.storage.restore_archived_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_save_exception_restores_incremental_archive(self, mock_gen, mock_clust):
        """Exception during save_agent_playbooks in incremental mode restores by ids."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.side_effect = RuntimeError("Boom")

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": 50, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")

            with pytest.raises(RuntimeError, match="Boom"):
                agg.run(req)

        agg.storage.restore_archived_agent_playbooks_by_ids.assert_called_once_with(
            [50]
        )

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_change_log_exception_is_caught(self, mock_gen, mock_clust):
        """Exception in add_playbook_aggregation_change_log should be caught, not raised."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1)]}
        mock_gen.return_value = [_agent_playbook(fid=100)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=100)]
        agg.storage.add_playbook_aggregation_change_log.side_effect = RuntimeError(
            "DB down"
        )

        req = PlaybookAggregatorRequest(
            agent_version="v1", playbook_name="fb", rerun=True
        )

        # Should NOT raise
        agg.run(req)

        # Despite the exception, delete should still proceed
        agg.storage.delete_archived_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_run_fingerprint_state_updated(self, mock_gen, mock_clust):
        """Fingerprint state should be updated after a successful run."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1), _raw(rid=2)]
        mock_clust.return_value = {0: raws}
        saved = _agent_playbook(fid=100)
        saved.agent_playbook_id = 100
        mock_gen.return_value = [saved]
        agg.storage.save_agent_playbooks.return_value = [saved]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        fingerprints_arg = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        assert fingerprints_arg is not None
        # The fingerprint for the cluster should have agent_playbook_id=100 assigned
        for fp_data in fingerprints_arg.values():
            if fp_data["agent_playbook_id"] is not None:
                assert fp_data["agent_playbook_id"] == 100

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_incremental_changed_clusters_but_no_archived_ids(
        self, mock_gen, mock_clust
    ):
        """Branch 508->511: changed clusters exist but archived_playbook_ids is empty."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.return_value = [_agent_playbook(fid=200)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=200)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            # prev fingerprints exist but the new cluster fingerprint is different,
            # and the old fingerprint has agent_playbook_id=None so nothing to archive
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": None, "user_playbook_ids": [1, 2]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        # archive_agent_playbooks_by_ids should NOT be called (no ids to archive)
        agg.storage.archive_agent_playbooks_by_ids.assert_not_called()
        # delete_agent_playbooks_by_ids should NOT be called either (branch 627->exit)
        agg.storage.delete_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_saved_fb_without_playbook_id_skipped_in_fingerprint_assignment(
        self, mock_gen, mock_clust
    ):
        """Branch 577->576: saved_fb with falsy playbook_id skipped during fp assignment."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        # AgentPlaybook with agent_playbook_id=0 (falsy)
        fb_no_id = _agent_playbook(fid=0, content="no id")
        fb_no_id.agent_playbook_id = 0
        mock_gen.return_value = [fb_no_id]
        agg.storage.save_agent_playbooks.return_value = [fb_no_id]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # The fingerprint should still have agent_playbook_id=None since fb_no_id.agent_playbook_id was falsy
        for fp_data in new_fps.values():
            assert fp_data["agent_playbook_id"] is None

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_exception_in_incremental_no_archived_ids_still_raises(
        self, mock_gen, mock_clust
    ):
        """Branch 641->644: exception in incremental mode with empty archived_playbook_ids."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.side_effect = RuntimeError("Kaboom")

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            # prev fingerprints with no playbook_id => no archived_playbook_ids
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": None, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")

            with pytest.raises(RuntimeError, match="Kaboom"):
                agg.run(req)

        # Neither restore method should be called since archived_playbook_ids is empty
        # and full_archive is False
        agg.storage.restore_archived_agent_playbooks_by_playbook_name.assert_not_called()
        agg.storage.restore_archived_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_run_with_none_saved_playbooks_in_list(self, mock_gen, mock_clust):
        """saved_playbooks list containing None entries should not cause errors."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        mock_gen.return_value = [None]
        agg.storage.save_agent_playbooks.return_value = [None]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            # Should not raise
            agg.run(req)

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_multiple_saved_playbooks_assigned_to_multiple_fingerprints(
        self, mock_gen, mock_clust
    ):
        """Branch 580->579: second saved_fb skips first fp (already assigned) and finds second."""
        agg = self._make_runnable_aggregator()
        raws_a = [_raw(rid=1)]
        raws_b = [_raw(rid=2)]
        mock_clust.return_value = {0: raws_a, 1: raws_b}
        fb1 = _agent_playbook(fid=100, content="a")
        fb1.agent_playbook_id = 100
        fb2 = _agent_playbook(fid=200, content="b")
        fb2.agent_playbook_id = 200
        mock_gen.return_value = [fb1, fb2]
        agg.storage.save_agent_playbooks.return_value = [fb1, fb2]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # Both fingerprints should have playbook_ids assigned
        assigned_ids = [
            v["agent_playbook_id"]
            for v in new_fps.values()
            if v["agent_playbook_id"] is not None
        ]
        assert len(assigned_ids) == 2
        assert set(assigned_ids) == {100, 200}

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_saved_fb_no_matching_fingerprint_exhausts_loop(self, mock_gen, mock_clust):
        """Branch 579->576: inner loop exhausts without finding a match (all fps have ids)."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        fb1 = _agent_playbook(fid=100, content="a")
        fb1.agent_playbook_id = 100
        # Two saved playbooks but only one cluster fingerprint
        fb2 = _agent_playbook(fid=200, content="b")
        fb2.agent_playbook_id = 200
        mock_gen.return_value = [fb1, fb2]
        agg.storage.save_agent_playbooks.return_value = [fb1, fb2]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # Only one fingerprint exists, should have first fb's id
        assigned_ids = [
            v["agent_playbook_id"]
            for v in new_fps.values()
            if v["agent_playbook_id"] is not None
        ]
        assert len(assigned_ids) == 1
        assert assigned_ids[0] == 100

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_from_clusters")
    def test_incremental_carries_forward_unchanged_fingerprints(
        self, mock_gen, mock_clust
    ):
        """Unchanged cluster fingerprints are carried forward in incremental mode."""
        agg = self._make_runnable_aggregator()
        # Two clusters: one unchanged, one new
        raws_unchanged = [_raw(rid=1)]
        raws_new = [_raw(rid=5), _raw(rid=6)]
        fp_unchanged = PlaybookAggregator._compute_cluster_fingerprint(raws_unchanged)

        all_raws = raws_unchanged + raws_new
        agg.storage.get_user_playbooks.return_value = all_raws
        mock_clust.return_value = {0: raws_unchanged, 1: raws_new}
        mock_gen.return_value = [_agent_playbook(fid=200)]
        agg.storage.save_agent_playbooks.return_value = [_agent_playbook(fid=200)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            prev_fps = {
                fp_unchanged: {"agent_playbook_id": 10, "user_playbook_ids": [1]},
                "vanished_fp": {"agent_playbook_id": 20, "user_playbook_ids": [2]},
            }
            mgr.get_cluster_fingerprints.return_value = prev_fps
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1", playbook_name="fb")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # Unchanged fingerprint should be carried forward
        assert fp_unchanged in new_fps
        assert new_fps[fp_unchanged]["agent_playbook_id"] == 10


# ---------------------------------------------------------------------------
# _format_cluster_input
# ---------------------------------------------------------------------------


class TestFormatClusterInput:
    def test_all_fields_present(self):
        """Each playbook becomes a numbered block with Content and Trigger."""
        agg = _make_aggregator()
        raws = [
            _raw(rid=1, when="cond1"),
            _raw(rid=2, when="cond2"),
        ]

        result = agg._format_cluster_input(raws)

        assert "[1]" in result
        assert "[2]" in result
        assert 'Content: "content-1"' in result
        assert 'Content: "content-2"' in result
        assert 'Trigger: "cond1"' in result
        assert 'Trigger: "cond2"' in result

    def test_no_trigger_omits_trigger_line(self):
        agg = _make_aggregator()
        raws = [_raw(rid=1, when=None)]

        result = agg._format_cluster_input(raws)

        assert "Trigger:" not in result

    def test_empty_list_returns_placeholder(self):
        """Empty input returns a placeholder string."""
        agg = _make_aggregator()
        result = agg._format_cluster_input([])
        assert result == "(No playbook items)"

    def test_content_is_first_field_after_number(self):
        """Content line appears immediately after the numbered header."""
        agg = _make_aggregator()
        raws = [_raw(rid=1, when="cond")]

        result = agg._format_cluster_input(raws)

        lines = result.strip().split("\n")
        assert lines[0] == "[1]"
        assert lines[1].startswith("Content:")

    def test_multiple_playbooks_separated_by_blank_lines(self):
        """Multiple playbooks are separated by blank lines."""
        agg = _make_aggregator()
        raws = [_raw(rid=1, when="cond1"), _raw(rid=2, when="cond2")]

        result = agg._format_cluster_input(raws)

        # Two blocks separated by double newline
        assert "\n\n" in result
        assert "[1]" in result
        assert "[2]" in result


# ---------------------------------------------------------------------------
# get_clusters
# ---------------------------------------------------------------------------


class TestGetClusters:
    def test_no_config_returns_empty(self):
        agg = _make_aggregator()
        result = agg.get_clusters([_raw()], None)  # type: ignore[arg-type]
        assert result == {}

    def test_no_user_playbooks_returns_empty(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=2)
        result = agg.get_clusters([], config)
        assert result == {}

    def test_fewer_than_min_returns_empty(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=5)
        raws = [_raw(rid=i) for i in range(3)]
        # Need real embeddings for len check
        for r in raws:
            r.embedding = [0.0] * 10

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
            result = agg.get_clusters(raws, config)

        assert result == {}

    def test_mock_mode_clusters_by_when_condition(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=2)
        raws = [
            _raw(rid=1, when="cond_a"),
            _raw(rid=2, when="cond_a"),
            _raw(rid=3, when="cond_b"),
        ]

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "true"}):
            result = agg.get_clusters(raws, config)

        # Only cond_a has 2 playbooks (meets threshold)
        assert len(result) == 1
        assert len(list(result.values())[0]) == 2


# ---------------------------------------------------------------------------
# _process_aggregation_response
# ---------------------------------------------------------------------------


class TestProcessAggregationResponse:
    def test_none_response_returns_none(self):
        agg = _make_aggregator()
        assert agg._process_aggregation_response(None, [_raw()]) is None  # type: ignore[arg-type]

    def test_null_playbook_returns_none(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
        )

        agg = _make_aggregator()
        response = PlaybookAggregationOutput(playbook=None)
        assert agg._process_aggregation_response(response, [_raw()]) is None

    def test_valid_response_returns_playbook(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
            StructuredPlaybookContent,
        )

        agg = _make_aggregator()
        structured = StructuredPlaybookContent(
            trigger="when testing",
            content="do something",
        )
        response = PlaybookAggregationOutput(playbook=structured)

        result = agg._process_aggregation_response(response, [_raw()])

        assert result is not None
        assert result.trigger == "when testing"
        assert result.content == "do something"
        assert result.playbook_status == PlaybookStatus.PENDING
