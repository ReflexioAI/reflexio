"""Integration test: aggregation emits op=aggregate set-level lineage events.

Drives one aggregation run (mock the LLM cluster/consolidation decision) that
rolls user playbooks UP-a and UP-b into a new agent playbook AP.

Asserts:
- a lineage event with op="aggregate" and entity_type="agent_playbook" exists,
- entity_id matches the saved agent playbook id (AP),
- prov_relation="wasDerivedFrom",
- source_ids contains str(UP-a.user_playbook_id) and str(UP-b.user_playbook_id).

Also includes a regression test verifying that a failure in the atomic
``save_agent_playbook_with_aggregate_event`` ABORTS the run, restores any
archived playbooks, and re-raises — all-or-nothing semantics (C1).

Mirrors the real-SQLite + mocked-cluster fixture style of
``test_consolidation_lineage_integration.py``.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.lib._agent_playbook import reconstruct_playbook_aggregation_change_log
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_dir():
    """Per-test temp directory for SQLite isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sqlite_storage(temp_storage_dir, worker_id):
    """Real SQLite storage in a per-test temp dir + per-worker org id."""
    return SQLiteStorage(
        org_id=f"test-agg-lineage-{worker_id}",
        db_path=os.path.join(temp_storage_dir, "agg_lineage.db"),
    )


@pytest.fixture
def request_context(sqlite_storage, temp_storage_dir, worker_id):
    """RequestContext wired to real SQLite storage + mocked configurator."""
    context = RequestContext(
        org_id=f"test-agg-lineage-{worker_id}",
        storage_base_dir=temp_storage_dir,
    )
    context.storage = sqlite_storage
    context.prompt_manager = MagicMock()
    context.configurator = MagicMock()
    # Wire a minimal PlaybookConfig so the aggregator doesn't skip early.
    agg_config = PlaybookAggregatorConfig(
        min_cluster_size=2,
        reaggregation_trigger_count=2,
    )
    context.configurator.get_config.return_value.user_playbook_extractor_config = (
        PlaybookConfig(
            extractor_name="default",
            extraction_definition_prompt="stub",
            aggregation_config=agg_config,
        )
    )
    return context


@pytest.fixture
def aggregator(request_context):
    """PlaybookAggregator wired to the real SQLite storage via request_context."""
    return PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=request_context,
        agent_version="v0",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_user_playbook(storage: SQLiteStorage, uid: int, org_id: str) -> UserPlaybook:
    """Save a minimal user playbook and return the persisted row."""
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version="v0",
        request_id=f"req-{uid}",
        playbook_name="default",
        content=f"Do thing {uid}.",
        trigger=f"when cond {uid}",
        rationale="r",
        status=None,
        source="chat",
        source_interaction_ids=[],
    )
    storage.save_user_playbooks([pb])
    saved = storage.get_user_playbooks(user_id="u1")
    # Return the most recently saved one (highest id).
    return max(saved, key=lambda p: p.user_playbook_id)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_aggregation_emits_aggregate_lineage_event(
    sqlite_storage: SQLiteStorage,
    request_context: RequestContext,
    aggregator: PlaybookAggregator,
    worker_id: str,
):
    """One aggregation run rolling UP-a, UP-b -> AP emits op=aggregate event.

    The cluster logic and LLM generation are mocked so the test is deterministic.
    The storage and lineage append are real (SQLite).
    """
    org_id = request_context.org_id

    # Seed two user playbooks that will form the cluster.
    up_a = _seed_user_playbook(sqlite_storage, uid=1, org_id=org_id)
    up_b = _seed_user_playbook(sqlite_storage, uid=2, org_id=org_id)

    cluster_playbooks = [up_a, up_b]

    # The agent playbook the LLM would produce (id=0 before save; storage assigns a real id).
    unsaved_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Aggregated AP.",
        playbook_status=PlaybookStatus.PENDING,
    )

    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_playbooks},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(unsaved_ap, cluster_playbooks)],
        ),
    ):
        aggregator.run(PlaybookAggregatorRequest(agent_version="v0", rerun=True))

    # Retrieve the saved agent playbook to get its real id.
    saved_aps = sqlite_storage.get_agent_playbooks()
    assert saved_aps, "Expected at least one agent playbook to be saved"
    ap = saved_aps[0]
    ap_id = str(ap.agent_playbook_id)

    # Assert the aggregate lineage event was emitted.
    events = sqlite_storage.get_lineage_events(
        entity_type="agent_playbook",
        entity_id=ap_id,
    )
    aggregate_events = [e for e in events if e.op == "aggregate"]
    assert len(aggregate_events) == 1, f"Expected 1 aggregate event, got: {events}"

    evt = aggregate_events[0]
    assert evt.entity_type == "agent_playbook"
    assert evt.prov_relation == "wasDerivedFrom"
    assert str(up_a.user_playbook_id) in evt.source_ids, evt.source_ids
    assert str(up_b.user_playbook_id) in evt.source_ids, evt.source_ids
    assert evt.actor == "aggregator"
    assert evt.org_id == org_id
    assert evt.request_id != "", "request_id must be non-empty"


def test_aggregate_save_failure_aborts_and_restores(
    sqlite_storage: SQLiteStorage,
    request_context: RequestContext,
    aggregator: PlaybookAggregator,
    worker_id: str,
):
    """C1: a failure in save_agent_playbook_with_aggregate_event aborts the run and restores archives.

    The per-playbook save no longer silently skips on failure.  Instead the
    exception propagates to the outer handler which:
    (a) re-raises so the caller knows the run failed,
    (b) calls restore_archived_agent_playbooks_by_* to un-archive the old generation,
    (c) leaves no orphan agent_playbook row (atomic rollback of the INSERT + event).

    Setup: seed one archived agent playbook (the old generation) + two user
    playbooks.  Patch save_agent_playbook_with_aggregate_event to raise.
    The archived playbook must survive (be restorable) and no new row must appear.
    """
    org_id = request_context.org_id

    # Seed an old archived agent playbook that the aggregator would supersede on success
    old_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="old generation",
        playbook_status=PlaybookStatus.PENDING,
    )
    saved_list = sqlite_storage.save_agent_playbooks([old_ap])
    old_ap_id = saved_list[0].agent_playbook_id
    sqlite_storage.archive_agent_playbooks_by_ids([old_ap_id])

    up_a = _seed_user_playbook(sqlite_storage, uid=10, org_id=org_id)
    up_b = _seed_user_playbook(sqlite_storage, uid=11, org_id=org_id)
    cluster_playbooks = [up_a, up_b]

    new_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="New AP that will fail to save.",
        playbook_status=PlaybookStatus.PENDING,
    )

    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_playbooks},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(new_ap, cluster_playbooks)],
        ),
        patch.object(
            sqlite_storage,
            "save_agent_playbook_with_aggregate_event",
            side_effect=RuntimeError("simulated atomic failure"),
        ),
        pytest.raises(RuntimeError, match="simulated atomic failure"),
    ):
        # (a) run must RAISE — failure is no longer swallowed
        aggregator.run(PlaybookAggregatorRequest(agent_version="v0", rerun=True))

    # (b) old archived playbook is restored (status cleared back to normal)
    restored = sqlite_storage.get_agent_playbook_by_id(old_ap_id)
    assert restored is not None, (
        "Old archived playbook must be restored after save failure"
    )

    # (c) no new agent_playbook row was persisted (atomic rollback — no orphan)
    all_aps = sqlite_storage.get_agent_playbooks()
    new_ids = [
        ap.agent_playbook_id for ap in all_aps if ap.agent_playbook_id != old_ap_id
    ]
    assert not new_ids, (
        "No new agent playbook must be saved when save_agent_playbook_with_aggregate_event fails"
    )


def _make_storage_and_ctx(
    temp_dir: str, suffix: str, worker_id: str
) -> tuple[SQLiteStorage, RequestContext]:
    """Build a fresh SQLite storage + RequestContext for an e2e test."""
    storage = SQLiteStorage(
        org_id=f"test-e2e-reconstruct-{worker_id}{suffix}",
        db_path=os.path.join(temp_dir, f"e2e{suffix}.db"),
    )
    ctx = RequestContext(
        org_id=f"test-e2e-reconstruct-{worker_id}{suffix}",
        storage_base_dir=temp_dir,
    )
    ctx.storage = storage
    ctx.prompt_manager = MagicMock()
    ctx.configurator = MagicMock()
    agg_config = PlaybookAggregatorConfig(
        min_cluster_size=2,
        reaggregation_trigger_count=2,
    )
    ctx.configurator.get_config.return_value.user_playbook_extractor_config = (
        PlaybookConfig(
            extractor_name="default",
            extraction_definition_prompt="stub",
            aggregation_config=agg_config,
        )
    )
    return storage, ctx


def test_e2e_reconstruct_added_and_run_mode(
    temp_storage_dir: str,
    worker_id: str,
) -> None:
    """E2E: run aggregation (full_archive) → reconstruct → assert added + run_mode.

    Validates the D1 rewire end-to-end: each saved playbook's aggregate event is
    emitted atomically via ``save_agent_playbook_with_aggregate_event``, and
    ``reconstruct_playbook_aggregation_change_log`` can reconstruct the run with:
    - correct ``added_agent_playbooks`` (from aggregate events),
    - ``run_mode == "full_archive"`` (reason == "aggregate:full_archive"),
    - non-empty ``removed_agent_playbooks`` (ties task B always-soft to D1).
    """
    storage, ctx = _make_storage_and_ctx(temp_storage_dir, "-e2e-full", worker_id)

    # Seed an old archived agent playbook (will be soft-superseded by the run).
    old_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="old content to supersede",
        playbook_status=PlaybookStatus.PENDING,
    )
    saved_old_list = storage.save_agent_playbooks([old_ap])
    old_ap_saved = saved_old_list[0]
    storage.archive_agent_playbooks_by_ids([old_ap_saved.agent_playbook_id])

    # Seed user playbooks that will cluster into the new AP.
    up_a = _seed_user_playbook(storage, uid=20, org_id=ctx.org_id)
    up_b = _seed_user_playbook(storage, uid=21, org_id=ctx.org_id)
    cluster_playbooks = [up_a, up_b]

    new_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="New aggregated content.",
        playbook_status=PlaybookStatus.PENDING,
    )

    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_playbooks},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(new_ap, cluster_playbooks)],
        ),
    ):
        aggregator = PlaybookAggregator(
            llm_client=MagicMock(),
            request_context=ctx,
            agent_version="v0",
        )
        stats = aggregator.run(
            PlaybookAggregatorRequest(agent_version="v0", rerun=True)
        )

    assert stats.get("playbooks_generated", 0) == 1, (
        f"Expected 1 playbook generated, got {stats}"
    )

    # Reconstruct and assert.
    result = reconstruct_playbook_aggregation_change_log(
        storage,
        playbook_name="default",
        agent_version="v0",
    )

    assert result.success
    assert result.change_logs, "Expected at least one reconstructed change log entry"

    log = result.change_logs[0]
    assert log.run_mode == "full_archive", (
        f"Expected run_mode='full_archive', got {log.run_mode!r}"
    )
    assert log.added_agent_playbooks, (
        "Expected non-empty added_agent_playbooks from aggregate events"
    )
    assert log.removed_agent_playbooks, (
        "Expected non-empty removed_agent_playbooks (old archived playbook soft-superseded by D1+task-B)"
    )


def test_e2e_reconstruct_incremental_run_mode(
    temp_storage_dir: str,
    worker_id: str,
) -> None:
    """E2E (incremental): second run with changed cluster reconstructs as 'incremental'.

    Run 1 (rerun=True) seeds cluster fingerprints. Run 2 (rerun=False) sees a
    NEW cluster (different fingerprints) and runs incrementally. The reconstructed
    change log for run 2 has run_mode='incremental'.
    """
    storage, ctx = _make_storage_and_ctx(temp_storage_dir, "-e2e-incr", worker_id)

    # Seed user playbooks for run 1.
    up_a = _seed_user_playbook(storage, uid=30, org_id=ctx.org_id)
    up_b = _seed_user_playbook(storage, uid=31, org_id=ctx.org_id)
    cluster_run1 = [up_a, up_b]

    ap_run1 = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Run 1 content.",
        playbook_status=PlaybookStatus.PENDING,
    )

    # Run 1 — full_archive seeds fingerprints in operation state.
    with (
        patch.object(
            PlaybookAggregator, "get_clusters", return_value={0: cluster_run1}
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(ap_run1, cluster_run1)],
        ),
    ):
        agg1 = PlaybookAggregator(
            llm_client=MagicMock(), request_context=ctx, agent_version="v0"
        )
        agg1.run(PlaybookAggregatorRequest(agent_version="v0", rerun=True))

    # Seed additional user playbooks so the count threshold is met for run 2.
    up_c = _seed_user_playbook(storage, uid=32, org_id=ctx.org_id)
    up_d = _seed_user_playbook(storage, uid=33, org_id=ctx.org_id)
    # New cluster uses different user playbooks so fingerprint is different.
    cluster_run2 = [up_c, up_d]

    ap_run2 = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Run 2 incremental content.",
        playbook_status=PlaybookStatus.PENDING,
    )

    # Run 2 — incremental (rerun=False, prev fingerprints now present).
    with (
        patch.object(
            PlaybookAggregator, "get_clusters", return_value={0: cluster_run2}
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(ap_run2, cluster_run2)],
        ),
    ):
        agg2 = PlaybookAggregator(
            llm_client=MagicMock(), request_context=ctx, agent_version="v0"
        )
        agg2.run(PlaybookAggregatorRequest(agent_version="v0", rerun=False))

    result = reconstruct_playbook_aggregation_change_log(
        storage,
        playbook_name="default",
        agent_version="v0",
    )

    assert result.success
    assert result.change_logs, "Expected at least one reconstructed entry"
    # BOTH runs appear. Run 2 (later timestamp) is the most recent entry. Run 1's ap was
    # superseded by run 2 but is still shown in run 1's 'added' list — reconstruct resolves
    # the added side with include_tombstones, so run 1 is no longer dropped from history.
    assert len(result.change_logs) == 2, (
        f"Expected both runs reconstructed, got {len(result.change_logs)}"
    )
    log_most_recent = result.change_logs[0]
    assert log_most_recent.run_mode == "incremental", (
        f"Expected run_mode='incremental' for the most recent run, got {log_most_recent.run_mode!r}"
    )
    assert log_most_recent.added_agent_playbooks, (
        "Expected added playbooks in incremental run"
    )
    assert log_most_recent.removed_agent_playbooks, (
        "Expected removed playbooks in incremental run (ap1 superseded under run2 request_id)"
    )
    # Run 1 survives supersession: its (now-tombstoned) added playbook is still present.
    log_run1 = result.change_logs[1]
    run1_added_contents = {snap.content for snap in log_run1.added_agent_playbooks}
    assert "Run 1 content." in run1_added_contents, (
        f"Run 1's added playbook must survive supersession, got {run1_added_contents}"
    )
