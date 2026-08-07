from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.playbook import aggregation_scheduler
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationBacklog,
    PlaybookAggregationClaim,
)


def _context(storage: Any) -> Any:
    aggregation_config = SimpleNamespace()
    config = SimpleNamespace(
        user_playbook_extractor_config=SimpleNamespace(
            aggregation_config=aggregation_config
        )
    )
    return SimpleNamespace(
        org_id="org-1",
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
    )


def test_scheduler_keeps_invalidation_and_clustering_budgets_separate(
    monkeypatch, caplog
) -> None:
    claim = PlaybookAggregationClaim("v1", "owner", 7, 3, 10_000)
    storage = MagicMock(supports_incremental_playbook_aggregation=True)
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = claim
    after = PlaybookAggregationBacklog(4, 1, 0, residual_retry_after_seconds=60)
    storage.get_playbook_aggregation_backlog.return_value = after
    storage.get_playbook_aggregation_invalidations.return_value = [
        SimpleNamespace(invalidation_id=1),
        SimpleNamespace(invalidation_id=2),
    ]
    storage.apply_playbook_aggregation_invalidations.return_value = True
    storage.finish_playbook_aggregation_claim.return_value = True
    captured: dict[str, object] = {}

    class Aggregator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, _request: object) -> dict[str, int]:
            return {"playbooks_generated": 1}

    monkeypatch.setattr(aggregation_scheduler, "PlaybookAggregator", Aggregator)
    monkeypatch.setattr(aggregation_scheduler, "_aggregation_budget", lambda: 8)
    monkeypatch.setattr(
        "reflexio.lib.generation_client.create_generation_litellm_client",
        lambda _context: MagicMock(),
    )
    monkeypatch.setattr(
        aggregation_scheduler,
        "run_with_operation_limit",
        lambda *, fn, **_kwargs: fn(),
    )
    caplog.set_level(logging.INFO, logger=aggregation_scheduler.logger.name)

    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    scheduler._run_context(_context(storage))

    assert captured["aggregation_claim"] == claim
    assert captured["residual_batch_limit"] == 8
    storage.finish_playbook_aggregation_claim.assert_called_once()
    assert storage.finish_playbook_aggregation_claim.call_args.kwargs["success"] is True
    assert (
        storage.finish_playbook_aggregation_claim.call_args.kwargs[
            "backlog_retry_after_seconds"
        ]
        == 1
    )
    assert (
        storage.finish_playbook_aggregation_claim.call_args.kwargs["backlog"] == after
    )
    storage.get_playbook_aggregation_backlog.assert_called_once_with("v1")
    assert "state=succeeded" in caplog.text


def test_scheduler_drains_invalidation_page_before_llm_work(
    monkeypatch, caplog
) -> None:
    claim = PlaybookAggregationClaim("v1", "owner", 7, 3, 10_000)
    storage = MagicMock(supports_incremental_playbook_aggregation=True)
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = claim
    storage.get_playbook_aggregation_invalidations.return_value = [
        SimpleNamespace(invalidation_id=value)
        for value in range(
            1, aggregation_scheduler.AGGREGATION_INVALIDATION_BATCH_SIZE + 2
        )
    ]
    storage.apply_playbook_aggregation_invalidations.return_value = True
    after = PlaybookAggregationBacklog(0, 0, 1)
    storage.get_playbook_aggregation_backlog.return_value = after
    storage.finish_playbook_aggregation_claim.return_value = True
    run = MagicMock()
    monkeypatch.setattr(aggregation_scheduler, "run_with_operation_limit", run)
    caplog.set_level(logging.INFO, logger=aggregation_scheduler.logger.name)

    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    scheduler._run_context(_context(storage))

    requested = storage.get_playbook_aggregation_invalidations.call_args.kwargs
    assert requested["limit"] == (
        aggregation_scheduler.AGGREGATION_INVALIDATION_BATCH_SIZE + 1
    )
    applied_ids = storage.apply_playbook_aggregation_invalidations.call_args.args[1]
    assert applied_ids == list(
        range(1, aggregation_scheduler.AGGREGATION_INVALIDATION_BATCH_SIZE + 1)
    )
    run.assert_not_called()
    assert storage.finish_playbook_aggregation_claim.call_args.kwargs["success"] is True
    assert "state=draining_invalidations" in caplog.text


def test_scheduler_keeps_pending_on_limiter_deferral(monkeypatch) -> None:
    claim = PlaybookAggregationClaim("v1", "owner", 7, 3, 10_000)
    storage = MagicMock(supports_incremental_playbook_aggregation=True)
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = claim
    storage.get_playbook_aggregation_backlog.return_value = PlaybookAggregationBacklog(
        1, 0, 0
    )
    storage.get_playbook_aggregation_invalidations.return_value = []
    storage.finish_playbook_aggregation_claim.return_value = True
    monkeypatch.setattr(
        "reflexio.lib.generation_client.create_generation_litellm_client",
        lambda _context: MagicMock(),
    )
    monkeypatch.setattr(
        aggregation_scheduler,
        "run_with_operation_limit",
        lambda **_kwargs: (_ for _ in ()).throw(TimeoutError),
    )

    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    scheduler._run_context(_context(storage))

    assert (
        storage.finish_playbook_aggregation_claim.call_args.kwargs["success"] is False
    )


def test_scheduler_throttles_idle_repair_scans(monkeypatch) -> None:
    storage = MagicMock(supports_incremental_playbook_aggregation=True)
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = None
    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    context = _context(storage)

    scheduler._run_context(context)
    scheduler._run_context(context)
    scheduler._last_repair_at["org-1"] -= (
        aggregation_scheduler._REPAIR_INTERVAL_SECONDS + 1
    )
    scheduler._run_context(context)

    assert storage.repair_playbook_aggregation_pending_state.call_count == 2
    assert storage.claim_due_playbook_aggregation.call_count == 3


def test_lease_heartbeat_marks_renewal_exception_as_lost() -> None:
    claim = PlaybookAggregationClaim("v1", "owner", 7, 3, 10_000)
    storage = MagicMock()
    storage.renew_playbook_aggregation_claim.side_effect = RuntimeError("db down")
    heartbeat = aggregation_scheduler.AggregationLeaseHeartbeat(storage, claim)
    heartbeat._stop.wait = MagicMock(return_value=False)

    heartbeat._run()

    with pytest.raises(RuntimeError, match="lease was lost"):
        heartbeat.require_live()


def test_stopped_local_scheduler_drops_captured_context(monkeypatch) -> None:
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    aggregation_scheduler._LOCAL_SCHEDULERS.clear()
    monkeypatch.setattr(
        aggregation_scheduler.PlaybookAggregationScheduler,
        "start",
        MagicMock(),
    )
    monkeypatch.setattr(
        aggregation_scheduler.PlaybookAggregationScheduler,
        "is_running",
        MagicMock(return_value=False),
    )
    first_context = _context(MagicMock())
    first = aggregation_scheduler.ensure_local_playbook_aggregation_scheduler(
        first_context
    )
    assert first is not None

    first._on_stopped()
    second_context = _context(MagicMock())
    second = aggregation_scheduler.ensure_local_playbook_aggregation_scheduler(
        second_context
    )

    assert second is not None
    assert second is not first
    assert list(second._context_provider()) == [second_context]


def test_scheduler_keeps_sqlite_work_pending_when_vector_index_is_unavailable(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
    storage = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "missing-vec.db"))
    storage.schedule_playbook_aggregation("v1")
    caplog.set_level(logging.WARNING, logger=aggregation_scheduler.logger.name)

    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    scheduler._run_context(_context(storage))

    state = storage.conn.execute(
        "SELECT pending FROM playbook_aggregation_state WHERE agent_version='v1'"
    ).fetchone()
    assert state is not None and int(state[0]) == 1
    assert storage.supports_incremental_playbook_aggregation is False
    assert "blocked_missing_vector_index" in caplog.text


@pytest.mark.parametrize(
    ("run_result", "expected"),
    [
        ({"playbooks_generated": 0}, "embedding_pending=0"),
        ({"playbooks_generated": 0, "embedding_pending": 2}, "embedding_pending=2"),
    ],
    ids=["idle_run", "starved_run"],
)
def test_succeeded_log_distinguishes_a_starved_run_from_an_idle_one(
    monkeypatch, caplog, run_result: dict[str, int], expected: str
) -> None:
    """A run that created nothing for want of vectors must not read as idle.

    Candidates without an embedding are dispositioned residual/embedding_pending,
    but `residual` sums three unrelated reasons, so at the log line a run starved
    of embeddings looked exactly like one with no work: `state=succeeded
    creations=0`. That is the shape an embedding outage takes in the logs, so it
    has to be readable there.
    """
    claim = PlaybookAggregationClaim("v1", "owner", 7, 3, 10_000)
    storage = MagicMock(supports_incremental_playbook_aggregation=True)
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = claim
    storage.get_playbook_aggregation_backlog.return_value = PlaybookAggregationBacklog(
        4, 1, 0, residual_retry_after_seconds=60
    )
    storage.get_playbook_aggregation_invalidations.return_value = []
    storage.finish_playbook_aggregation_claim.return_value = True

    class Aggregator:
        def __init__(self, **kwargs: object) -> None: ...

        def run(self, _request: object) -> dict[str, int]:
            return run_result

    monkeypatch.setattr(aggregation_scheduler, "PlaybookAggregator", Aggregator)
    monkeypatch.setattr(aggregation_scheduler, "_aggregation_budget", lambda: 8)
    monkeypatch.setattr(
        "reflexio.lib.generation_client.create_generation_litellm_client",
        lambda _context: MagicMock(),
    )
    monkeypatch.setattr(
        aggregation_scheduler,
        "run_with_operation_limit",
        lambda *, fn, **_kwargs: fn(),
    )
    caplog.set_level(logging.INFO, logger=aggregation_scheduler.logger.name)

    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    scheduler._run_context(_context(storage))

    assert "state=succeeded" in caplog.text
    assert "creations=0" in caplog.text
    assert expected in caplog.text
