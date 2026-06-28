from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models import config_schema
from reflexio.models.config_schema import Config, LineageGCConfig
from reflexio.server.services.lineage.gc_scheduler import (
    _ENTITY_TYPES,
    LineageGCScheduler,
)


def _make_ctx(*, lineage_gc_enabled: bool, governance_gate_enabled: bool):
    storage = MagicMock()
    storage.gc_expired_tombstones.return_value = 0
    storage.gc_governance_retention.return_value = 0
    config = SimpleNamespace(
        lineage_gc=LineageGCConfig(enabled=lineage_gc_enabled),
        governance_retention=SimpleNamespace(
            purge_expired_profiles_enabled=governance_gate_enabled,
            row_count_retention_enabled=False,
            audit_events_retention_enabled=False,
        ),
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


def test_config_exposes_governance_retention_defaults():
    governance_cls = getattr(config_schema, "GovernanceRetentionConfig", None)

    assert governance_cls is not None

    cfg = Config(storage_config=None)

    assert isinstance(cfg.governance_retention, governance_cls)
    assert cfg.governance_retention.purge_expired_profiles_enabled is False
    assert cfg.governance_retention.row_count_retention_enabled is False
    assert cfg.governance_retention.audit_events_retention_enabled is False
    assert cfg.governance_retention.audit_events_retention_days == 365
    assert cfg.governance_retention.audit_events_delete_batch_limit == 500


@pytest.mark.parametrize(
    ("lineage_gc_enabled", "governance_gate_enabled", "expect_tombstone_gc", "expect_governance_gc"),
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
        governance_gate_enabled=governance_gate_enabled,
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
