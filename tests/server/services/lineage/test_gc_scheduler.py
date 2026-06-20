"""Unit tests for LineageGCScheduler._gc_tick (no real threads needed)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import (
    _ENTITY_TYPES,
    _HIGH_VOLUME_THRESHOLD,
    LineageGCScheduler,
    maybe_start_lineage_gc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(*, gc_return: int = 0) -> MagicMock:
    """Return a mock storage whose gc_expired_tombstones returns ``gc_return``."""
    storage = MagicMock()
    storage.gc_expired_tombstones.return_value = gc_return
    return storage


def _make_ctx(org_id: str, *, lineage_gc: LineageGCConfig, storage=None):
    """Build a minimal request-context stand-in."""
    if storage is None:
        storage = _make_storage()
    config = SimpleNamespace(lineage_gc=lineage_gc)
    return SimpleNamespace(
        org_id=org_id,
        storage=storage,
        configurator=SimpleNamespace(get_config=MagicMock(return_value=config)),
    )


def _scheduler(*, bootstrap_org_id: str = "org_bootstrap", factory=None):
    """Return a LineageGCScheduler backed by ``factory`` (not started)."""
    if factory is None:
        factory = lambda org_id: _make_ctx(
            org_id, lineage_gc=LineageGCConfig(enabled=True)
        )  # noqa: E731
    return LineageGCScheduler(
        request_context_factory=factory,
        bootstrap_org_id=bootstrap_org_id,
    )


# ---------------------------------------------------------------------------
# (a) Enabled org — all three entity types get a gc call with correct cutoff
# ---------------------------------------------------------------------------


def test_gc_tick_calls_all_entity_types_with_correct_cutoff():
    grace_days = 30
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=grace_days)
    storage = _make_storage(gc_return=0)
    ctx = _make_ctx("org_1", lineage_gc=cfg, storage=storage)

    sched = _scheduler(factory=lambda _: ctx)

    before = int(time.time())
    sched._gc_tick(["org_1"])
    after = int(time.time())

    assert storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)
    for et in _ENTITY_TYPES:
        # Find the call for this entity type
        matching = [
            c
            for c in storage.gc_expired_tombstones.call_args_list
            if c.kwargs.get("entity_type") == et
        ]
        assert len(matching) == 1, f"Expected one call for {et}"
        epoch = matching[0].kwargs["older_than_epoch"]
        expected_low = before - grace_days * 86400
        expected_high = after - grace_days * 86400
        assert expected_low <= epoch <= expected_high, (
            f"older_than_epoch={epoch} not in [{expected_low}, {expected_high}]"
        )


# ---------------------------------------------------------------------------
# (b) Disabled org — no gc calls
# ---------------------------------------------------------------------------


def test_gc_tick_skips_disabled_org():
    cfg = LineageGCConfig(enabled=False)
    storage = _make_storage()
    ctx = _make_ctx("org_disabled", lineage_gc=cfg, storage=storage)

    sched = _scheduler(factory=lambda _: ctx)
    sched._gc_tick(["org_disabled"])

    storage.gc_expired_tombstones.assert_not_called()


# ---------------------------------------------------------------------------
# (c) Resilience — one org failure triggers capture_anomaly, next org proceeds
# ---------------------------------------------------------------------------


def test_gc_tick_continues_after_org_failure():
    good_storage = _make_storage(gc_return=1)
    good_cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)

    def factory(org_id: str):
        if org_id == "org_bad":
            raise RuntimeError("simulated storage failure")
        return _make_ctx(org_id, lineage_gc=good_cfg, storage=good_storage)

    sched = _scheduler(factory=factory)

    with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
        sched._gc_tick(["org_bad", "org_good"])

    # anomaly fired for the failing org
    mock_anomaly.assert_called_once_with("lineage.gc.run_failed", org_id="org_bad")
    # good org was still processed
    assert good_storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)


# ---------------------------------------------------------------------------
# (d) High-volume tripwire — capture_anomaly fires when count exceeds threshold
# ---------------------------------------------------------------------------


def test_gc_tick_fires_high_volume_anomaly():
    # Each of the 3 entity types returns enough to exceed the threshold in total
    per_entity = (_HIGH_VOLUME_THRESHOLD // len(_ENTITY_TYPES)) + 1
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)
    storage = _make_storage(gc_return=per_entity)
    ctx = _make_ctx("org_bigdel", lineage_gc=cfg, storage=storage)

    sched = _scheduler(factory=lambda _: ctx)

    with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
        sched._gc_tick(["org_bigdel"])

    total = per_entity * len(_ENTITY_TYPES)
    assert total > _HIGH_VOLUME_THRESHOLD
    mock_anomaly.assert_called_once_with(
        "lineage.gc.high_volume", org_id="org_bigdel", count=total
    )


def test_gc_tick_no_high_volume_anomaly_below_threshold():
    # Exactly at threshold — should NOT fire
    per_entity = _HIGH_VOLUME_THRESHOLD // len(_ENTITY_TYPES)
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)
    storage = _make_storage(gc_return=per_entity)
    ctx = _make_ctx("org_small", lineage_gc=cfg, storage=storage)

    sched = _scheduler(factory=lambda _: ctx)

    with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
        sched._gc_tick(["org_small"])

    mock_anomaly.assert_not_called()


# ---------------------------------------------------------------------------
# maybe_start_lineage_gc — off-by-default factory
# ---------------------------------------------------------------------------


def test_maybe_start_lineage_gc_returns_none_when_disabled():
    cfg = LineageGCConfig(enabled=False)
    ctx = _make_ctx("org_1", lineage_gc=cfg)
    result = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_1")
    assert result is None


def test_maybe_start_lineage_gc_returns_scheduler_when_enabled():
    cfg = LineageGCConfig(enabled=True)
    ctx = _make_ctx("org_1", lineage_gc=cfg)
    sched = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_1")
    assert sched is not None
    sched.stop(timeout_seconds=1.0)


def test_maybe_start_lineage_gc_returns_none_on_factory_error():
    def bad_factory(org_id: str):
        raise RuntimeError("can't build context")

    result = maybe_start_lineage_gc(bad_factory, bootstrap_org_id="org_1")
    assert result is None
