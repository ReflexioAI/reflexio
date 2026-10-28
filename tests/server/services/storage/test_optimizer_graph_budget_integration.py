"""One physical-row budget reclaims complete eligible optimizer histories."""

import pytest

from reflexio.server.services.storage.retention import get_row_retention_limits
from tests.server.services.storage.test_optimizer_retention_integration import (
    seed,
    snapshot,
    store,  # noqa: F401
)

pytestmark = pytest.mark.integration
GRAPH = "playbook_optimization_history"
LEGACY_CAPS = (
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_JOBS",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_CANDIDATES",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVALUATIONS",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVENTS",
)


def test_graph_count_and_cleanup_account_for_every_physical_history_row(
    store,  # noqa: F811
):
    first = seed(store, "completed", 1)
    second = seed(store, "completed", 2)
    assert store.count_retention_target_rows(GRAPH) == 8
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4
    assert all(not rows for rows in snapshot(store, first.job_id))
    assert all(len(rows) == 1 for rows in snapshot(store, second.job_id))
    assert store.count_retention_target_rows(GRAPH) == 4


def test_protected_graph_does_not_starve_complete_terminal_eviction(
    store,  # noqa: F811
):
    active = seed(store, "running", 1)
    terminal = seed(store, "completed", 2)
    before = snapshot(store, active.job_id)
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4
    assert snapshot(store, active.job_id) == before
    assert all(not rows for rows in snapshot(store, terminal.job_id))


def test_aggregate_default_preserves_the_previous_combined_nominal_ceiling():
    assert get_row_retention_limits(env={})[GRAPH] == 1_000_000
    assert (
        get_row_retention_limits(env={"REFLEXIO_OPERATIONAL_ROW_LIMIT": "17"})[GRAPH]
        == 68
    )
    assert (
        get_row_retention_limits(env={"REFLEXIO_OPERATIONAL_ROW_LIMIT": "0"})[GRAPH]
        == 0
    )
    assert (
        get_row_retention_limits(
            env={"REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY": "0"}
        )[GRAPH]
        == 0
    )


@pytest.mark.parametrize("legacy", LEGACY_CAPS)
@pytest.mark.parametrize("value", ["", "0", "17", "250000"])
def test_legacy_caps_require_an_explicit_policy_migration(legacy, value):
    with pytest.raises(ValueError, match=legacy):
        get_row_retention_limits(
            env={
                legacy: value,
                "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY": "500",
            }
        )


@pytest.mark.parametrize(("budget", "deleted"), [(0, 0), (5, 4), (8, 4), (9, 0)])
def test_actual_sweep_enforces_one_graph_budget(store, monkeypatch, budget, deleted):  # noqa: F811
    from reflexio.server.services.storage.retention_sweep import sweep_retention_caps

    first = seed(store, "completed", 1)
    second = seed(store, "completed", 2)
    monkeypatch.setenv("REFLEXIO_OPERATIONAL_ROW_LIMIT", "0")
    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY", str(budget))
    result = sweep_retention_caps(store.org_id, store)
    assert not result.failed
    assert result.deleted == deleted
    assert store.count_retention_target_rows(GRAPH) == 8 - deleted
    assert all(not rows for rows in snapshot(store, first.job_id)) == bool(deleted)
    assert all(len(rows) == 1 for rows in snapshot(store, second.job_id))


def test_variable_fanout_never_leaves_a_partial_history(store):  # noqa: F811
    from reflexio.models.api_schema.domain import PlaybookOptimizationEvent

    first = seed(store, "completed", 1)
    second = seed(store, "completed", 2)
    store.insert_playbook_optimization_event(
        PlaybookOptimizationEvent(job_id=first.job_id, event_type="extra", created_at=1)
    )
    assert store.count_retention_target_rows(GRAPH) == 9
    assert store.delete_oldest_retention_target_rows(GRAPH, 4) == 5
    assert all(not rows for rows in snapshot(store, first.job_id))
    assert all(len(rows) == 1 for rows in snapshot(store, second.job_id))


def test_retired_cap_blocks_the_sweep_before_storage_io(store, monkeypatch):  # noqa: F811
    from reflexio.server.services.storage.retention_sweep import sweep_retention_caps

    seed(store, "completed", 1)
    monkeypatch.setenv(LEGACY_CAPS[0], "0")
    queries = []
    store.conn.set_trace_callback(queries.append)
    try:
        result = sweep_retention_caps(store.org_id, store)
    finally:
        store.conn.set_trace_callback(None)
    assert result.failed and result.deleted == 0
    assert queries == []
    assert store.count_retention_target_rows(GRAPH) == 4
