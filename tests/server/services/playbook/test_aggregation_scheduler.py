from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from reflexio.server.services.playbook import aggregation_scheduler
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationBacklog,
    PlaybookAggregationClaim,
)


def _context(storage: MagicMock) -> Any:
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


def test_scheduler_passes_claim_and_remaining_shared_budget(
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
    assert captured["work_budget"] == 6
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
    monkeypatch.setattr(aggregation_scheduler.time, "monotonic", lambda: 10.0)
    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], worker_id="worker"
    )
    context = _context(storage)

    scheduler._run_context(context)
    scheduler._run_context(context)

    storage.repair_playbook_aggregation_pending_state.assert_called_once_with()
    assert storage.claim_due_playbook_aggregation.call_count == 2
