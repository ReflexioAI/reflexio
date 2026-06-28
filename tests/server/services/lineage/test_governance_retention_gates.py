from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models import config_schema
from reflexio.models.config_schema import Config, LineageGCConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import (
    _ENTITY_TYPES,
    _HIGH_VOLUME_THRESHOLD,
    LineageGCScheduler,
    maybe_start_lineage_gc,
)


def _make_ctx(
    *,
    lineage_gc_enabled: bool,
    audit_events_retention_enabled: bool = False,
    include_governance_retention: bool = True,
):
    storage = MagicMock()
    storage.gc_expired_tombstones.return_value = 0
    storage.gc_governance_retention.return_value = 0
    config = SimpleNamespace(
        lineage_gc=LineageGCConfig(enabled=lineage_gc_enabled),
    )
    if include_governance_retention:
        config.governance_retention = SimpleNamespace(
            audit_events_retention_enabled=audit_events_retention_enabled,
        )
    return SimpleNamespace(
        org_id="org_1",
        storage=storage,
        configurator=SimpleNamespace(get_config=MagicMock(return_value=config)),
    )


def _scheduler(ctx) -> LineageGCScheduler:
    return LineageGCScheduler(
        request_context_factory=lambda _: ctx,
        bootstrap_org_id="org_1",
    )


def _with_governance_flag(flag_name: str) -> dict[str, bool]:
    return {flag_name: True}


def test_config_exposes_governance_retention_defaults():
    governance_cls = getattr(config_schema, "GovernanceRetentionConfig", None)

    assert governance_cls is not None

    cfg = Config(storage_config=None)

    assert isinstance(cfg.governance_retention, governance_cls)
    assert cfg.governance_retention.audit_events_retention_enabled is False
    assert cfg.governance_retention.audit_events_retention_days == 365
    assert cfg.governance_retention.audit_events_delete_batch_limit == 500


@pytest.mark.parametrize(
    (
        "lineage_gc_enabled",
        "governance_gate_enabled",
        "expect_tombstone_gc",
        "expect_governance_gc",
    ),
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
)
def test_gc_tick_gates_tombstone_and_governance_paths(
    lineage_gc_enabled: bool,
    governance_gate_enabled: bool,
    expect_tombstone_gc: bool,
    expect_governance_gc: bool,
):
    ctx = _make_ctx(
        lineage_gc_enabled=lineage_gc_enabled,
        audit_events_retention_enabled=governance_gate_enabled,
    )

    sched = _scheduler(ctx)
    sched._gc_tick(["org_1"])

    if expect_tombstone_gc:
        assert ctx.storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)
    else:
        ctx.storage.gc_expired_tombstones.assert_not_called()

    if expect_governance_gc:
        ctx.storage.gc_governance_retention.assert_called_once_with(
            config=ctx.configurator.get_config.return_value.governance_retention
        )
    else:
        ctx.storage.gc_governance_retention.assert_not_called()


def test_gc_tick_runs_governance_gc_for_audit_event_retention_gate():
    ctx = _make_ctx(lineage_gc_enabled=False, audit_events_retention_enabled=True)

    sched = _scheduler(ctx)
    sched._gc_tick(["org_1"])

    ctx.storage.gc_expired_tombstones.assert_not_called()
    ctx.storage.gc_governance_retention.assert_called_once_with(
        config=ctx.configurator.get_config.return_value.governance_retention
    )


def test_gc_tick_high_volume_anomaly_ignores_governance_deletions():
    ctx = _make_ctx(lineage_gc_enabled=False, audit_events_retention_enabled=True)
    ctx.storage.gc_governance_retention.return_value = _HIGH_VOLUME_THRESHOLD + 1

    sched = _scheduler(ctx)

    with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
        sched._gc_tick(["org_1"])

    mock_anomaly.assert_not_called()


def test_maybe_start_lineage_gc_starts_for_governance_retention_only():
    ctx = _make_ctx(lineage_gc_enabled=False, audit_events_retention_enabled=True)
    request_context_factory = cast(
        Callable[[str], RequestContext],
        lambda _: ctx,
    )

    with patch.object(LineageGCScheduler, "start") as mock_start:
        sched = maybe_start_lineage_gc(
            request_context_factory, bootstrap_org_id="org_1"
        )

    assert sched is not None
    mock_start.assert_called_once_with()


def test_gc_tick_legacy_config_without_governance_retention_runs_tombstone_only():
    ctx = _make_ctx(
        lineage_gc_enabled=True,
        include_governance_retention=False,
    )

    sched = _scheduler(ctx)
    sched._gc_tick(["org_1"])

    assert ctx.storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)
    ctx.storage.gc_governance_retention.assert_not_called()


def test_maybe_start_lineage_gc_legacy_config_without_governance_retention():
    ctx = _make_ctx(
        lineage_gc_enabled=True,
        include_governance_retention=False,
    )
    request_context_factory = cast(
        Callable[[str], RequestContext],
        lambda _: ctx,
    )

    with patch.object(LineageGCScheduler, "start") as mock_start:
        sched = maybe_start_lineage_gc(
            request_context_factory, bootstrap_org_id="org_1"
        )

    assert sched is not None
    mock_start.assert_called_once_with()
