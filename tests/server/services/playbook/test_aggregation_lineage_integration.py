"""Integration test: aggregation emits op=aggregate set-level lineage events.

Drives one aggregation run (mock the LLM cluster/consolidation decision) that
rolls user playbooks UP-a and UP-b into a new agent playbook AP.

Asserts:
- a lineage event with op="aggregate" and entity_type="agent_playbook" exists,
- entity_id matches the saved agent playbook id (AP),
- prov_relation="wasDerivedFrom",
- source_ids contains str(UP-a.user_playbook_id) and str(UP-b.user_playbook_id).

Also includes a PB-7 best-effort regression test verifying that a transient
``append_lineage_event`` failure does NOT abort the aggregation run and that
``capture_anomaly`` is called with "lineage.aggregate.append_failed".

Mirrors the real-SQLite + mocked-cluster fixture style of
``test_consolidation_lineage_integration.py``.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
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


def test_aggregate_lineage_append_failure_is_best_effort(
    sqlite_storage: SQLiteStorage,
    request_context: RequestContext,
    aggregator: PlaybookAggregator,
    worker_id: str,
):
    """PB-7: a transient append_lineage_event failure must NOT abort aggregation.

    Pins the safety property: the try/except around ``storage.append_lineage_event``
    swallows the exception, saves the agent playbook(s) anyway, and calls
    ``capture_anomaly("lineage.aggregate.append_failed", ...)``.
    """
    org_id = request_context.org_id

    up_a = _seed_user_playbook(sqlite_storage, uid=10, org_id=org_id)
    up_b = _seed_user_playbook(sqlite_storage, uid=11, org_id=org_id)
    cluster_playbooks = [up_a, up_b]

    unsaved_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Best-effort AP.",
        playbook_status=PlaybookStatus.PENDING,
    )

    def _raise_on_append(event):  # noqa: ANN001
        raise RuntimeError("transient lineage failure")

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
        patch.object(
            sqlite_storage, "append_lineage_event", side_effect=_raise_on_append
        ),
        patch(
            "reflexio.server.services.playbook.playbook_aggregator.capture_anomaly"
        ) as mock_capture,
    ):
        # (a) run must complete without raising
        result = aggregator.run(
            PlaybookAggregatorRequest(agent_version="v0", rerun=True)
        )

    # (a) no exception propagated — result is the stats dict
    assert isinstance(result, dict), f"Expected dict, got: {result!r}"
    assert "playbooks_generated" in result

    # (b) the agent playbook was still saved despite the lineage failure
    saved_aps = sqlite_storage.get_agent_playbooks()
    assert saved_aps, "Agent playbook must be saved even when lineage append fails"

    # (c) capture_anomaly was called with the correct anomaly key
    assert mock_capture.called, (
        "capture_anomaly must be called on lineage append failure"
    )
    anomaly_keys = [c.args[0] for c in mock_capture.call_args_list]
    assert "lineage.aggregate.append_failed" in anomaly_keys, anomaly_keys
