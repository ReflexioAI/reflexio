"""Unit tests for LineageGCScheduler._gc_tick (no real threads needed)."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.scheduling import LeaderGate
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage import gc_scheduler as gc_mod
from reflexio.server.services.lineage.gc_scheduler import (
    _DEFAULT_ORG_FANOUT_WORKERS,
    _DEFAULT_POLL_INTERVAL_SECONDS,
    _ENTITY_TYPES,
    _HIGH_VOLUME_THRESHOLD,
    _MIN_POLL_SECONDS,
    LineageGCScheduler,
    clear_global_sweeps,
    clear_per_org_sweeps,
    maybe_start_lineage_gc,
    register_per_org_sweep,
    set_leader_gate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(*, gc_return: int = 0) -> MagicMock:
    """Return a mock storage whose gc_expired_tombstones returns ``gc_return``."""
    storage = MagicMock()
    storage.gc_expired_tombstones.return_value = gc_return
    # expire_active_profiles is now called at the start of each tick; default to 0
    # so existing tests that don't exercise the sweep see a clean integer return.
    storage.expire_active_profiles.return_value = 0
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


def _default_factory(org_id: str):
    return _make_ctx(org_id, lineage_gc=LineageGCConfig(enabled=True))


def _scheduler(*, bootstrap_org_id: str = "org_bootstrap", factory=None):
    """Return a LineageGCScheduler backed by ``factory`` (not started)."""
    if factory is None:
        factory = _default_factory
    return LineageGCScheduler(
        request_context_factory=factory,  # type: ignore[arg-type]
        bootstrap_org_id=bootstrap_org_id,
    )


@pytest.fixture
def gc_scheduler_factory():
    """Return a factory building a ``LineageGCScheduler`` with a stubbed sweep.

    The returned callable takes ``org_ids`` (used to build per-org contexts
    and seed the bootstrap org), an optional ``storage_class_name`` (a
    stand-in storage type name so ``_org_fanout_workers``'s SQLite check can
    be exercised without a real ``SQLiteStorage``), and an optional
    ``leader_gate``. ``_sweep_org`` is replaced with a stub that records the
    org id into a shared list (thread-safely) instead of doing real storage
    work, so fan-out tests can assert on which orgs were swept without
    depending on ``lineage_gc``/``expiry_reclamation`` config gating.

    Returns:
        Callable[..., tuple[LineageGCScheduler, list[str]]]: Factory
        returning ``(scheduler, swept_org_ids)``.
    """

    def _factory(
        *,
        org_ids: list[str],
        storage_class_name: str | None = None,
        leader_gate: LeaderGate | None = None,
    ) -> tuple[LineageGCScheduler, list[str]]:
        swept: list[str] = []
        lock = threading.Lock()

        def factory(org_id: str):
            storage = (
                type(storage_class_name, (), {})()
                if storage_class_name is not None
                else None
            )
            return _make_ctx(
                org_id, lineage_gc=LineageGCConfig(enabled=False), storage=storage
            )

        sched = LineageGCScheduler(
            request_context_factory=factory,  # type: ignore[arg-type]
            bootstrap_org_id=org_ids[0] if org_ids else "org_bootstrap",
            leader_gate=leader_gate,
        )

        def _stub_sweep_org(org_id: str) -> None:
            with lock:
                swept.append(org_id)

        sched._sweep_org = _stub_sweep_org  # type: ignore[method-assign]
        return sched, swept

    return _factory


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
    # Precondition: total must genuinely be below the threshold for this test
    # to be meaningful. Integer-division truncation could silently trip this.
    total = per_entity * len(_ENTITY_TYPES)
    assert total < _HIGH_VOLUME_THRESHOLD, (
        f"Test setup error: per_entity={per_entity} yields total={total} "
        f">= _HIGH_VOLUME_THRESHOLD={_HIGH_VOLUME_THRESHOLD}; adjust the calculation"
    )

    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)
    storage = _make_storage(gc_return=per_entity)
    ctx = _make_ctx("org_small", lineage_gc=cfg, storage=storage)

    sched = _scheduler(factory=lambda _: ctx)

    with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
        sched._gc_tick(["org_small"])

    mock_anomaly.assert_not_called()


# ---------------------------------------------------------------------------
# Shed invariant — the reduced OSS scheduler NEVER runs governance retention
# ---------------------------------------------------------------------------


def test_gc_tick_never_runs_governance_retention():
    """Invariant: the reduced OSS scheduler sheds the premium concern. Even when
    a governance_retention config with audit_events_retention_enabled=True is
    present, _gc_tick must NEVER call gc_governance_retention directly — that
    behavior is only invoked if an enterprise per-org sweep is registered via
    register_per_org_sweep (Task 1 seam)."""
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)
    storage = _make_storage(gc_return=0)
    ctx = _make_ctx("org_gov", lineage_gc=cfg, storage=storage)
    # Attach a governance_retention config that *would* have enabled the
    # premium path under the old combined scheduler.
    ctx.configurator.get_config.return_value = SimpleNamespace(
        lineage_gc=cfg,
        governance_retention=SimpleNamespace(audit_events_retention_enabled=True),
    )

    sched = _scheduler(factory=lambda _: ctx)
    sched._gc_tick(["org_gov"])

    storage.gc_governance_retention.assert_not_called()
    # tombstone GC still ran (behavior preserved):
    assert storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)


def test_gc_tick_works_for_legacy_config_without_governance_retention():
    """Legacy-config path: a config object lacking ``governance_retention`` still
    ticks fine — the reduced gate only reads ``lineage_gc.enabled``."""
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=10)
    storage = _make_storage(gc_return=0)
    # _make_ctx builds SimpleNamespace(lineage_gc=...) with NO governance_retention.
    ctx = _make_ctx("org_legacy", lineage_gc=cfg, storage=storage)
    assert not hasattr(ctx.configurator.get_config(), "governance_retention")

    sched = _scheduler(factory=lambda _: ctx)
    sched._gc_tick(["org_legacy"])

    assert storage.gc_expired_tombstones.call_count == len(_ENTITY_TYPES)
    storage.gc_governance_retention.assert_not_called()


def test_lineage_gc_enabled_default_is_true():
    """Default-flip tripwire: OSS must always reclaim tombstones by default."""
    assert LineageGCConfig().enabled is True


# ---------------------------------------------------------------------------
# maybe_start_lineage_gc — off-by-default factory
# ---------------------------------------------------------------------------


def test_maybe_start_lineage_gc_returns_none_when_disabled():
    cfg = LineageGCConfig(enabled=False)
    ctx = _make_ctx("org_1", lineage_gc=cfg)
    result = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_1")  # type: ignore[arg-type]
    assert result is None


def test_maybe_start_lineage_gc_returns_scheduler_when_enabled():
    cfg = LineageGCConfig(enabled=True)
    ctx = _make_ctx("org_1", lineage_gc=cfg)
    sched = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_1")  # type: ignore[arg-type]
    assert sched is not None
    sched.stop(timeout_seconds=1.0)


def test_gc_scheduler_recovers_when_first_org_appears():
    org_ids = [DEFAULT_ORG_ID]
    factory_calls: list[str] = []

    def factory(org_id: str):
        factory_calls.append(org_id)
        return _make_ctx(org_id, lineage_gc=LineageGCConfig(enabled=True))

    scheduler = LineageGCScheduler(
        request_context_factory=factory,  # type: ignore[arg-type]
        bootstrap_org_id=DEFAULT_ORG_ID,
        org_id_provider=lambda: list(org_ids),
    )
    scheduler._gc_tick = MagicMock()  # type: ignore[method-assign]
    scheduler._run_global_sweeps = MagicMock()  # type: ignore[method-assign]

    assert scheduler._run_once() == 5
    assert factory_calls == []

    org_ids[:] = ["org_1"]
    assert scheduler._run_once() == _DEFAULT_POLL_INTERVAL_SECONDS
    assert scheduler.bootstrap_org_id == "org_1"
    assert factory_calls == ["org_1"]
    scheduler._gc_tick.assert_called_once()  # type: ignore[attr-defined]


def test_maybe_start_lineage_gc_returns_none_on_factory_error():
    def bad_factory(org_id: str):
        raise RuntimeError("can't build context")

    result = maybe_start_lineage_gc(bad_factory, bootstrap_org_id="org_1")
    assert result is None


def test_maybe_start_lineage_gc_starts_when_config_unreadable_but_sweeps_registered(
    caplog,
):
    """A config-read failure must not silence explicitly registered sweeps.

    Registered hooks carry their own per-org gates, so the documented
    "start unconditionally, gate per-org" invariant still applies. Production
    ran with 8 registered sweeps never firing because this read raised on a
    bootstrap org id that matched no organization row.
    """

    def bad_factory(org_id: str):
        raise RuntimeError("Organization self-host-org not found")

    register_per_org_sweep(lambda _org_id, _budget: 0)
    try:
        with caplog.at_level(logging.WARNING):
            sched = maybe_start_lineage_gc(bad_factory, bootstrap_org_id="org_1")
        assert sched is not None
        sched.stop(timeout_seconds=1.0)
    finally:
        clear_per_org_sweeps()

    assert "lineage_gc_config_read_failed" in caplog.text
    assert "lineage_gc_scheduler_start_skipped" not in caplog.text


def test_maybe_start_lineage_gc_defers_empty_enterprise_fleet(caplog):
    def bad_factory(_org_id: str):
        raise RuntimeError("Organization self-host-org not found")

    register_per_org_sweep(lambda _org_id, _budget: 0)
    gc_scheduler.set_org_id_provider(lambda: [DEFAULT_ORG_ID])
    try:
        with (
            patch.object(LineageGCScheduler, "start"),
            caplog.at_level(logging.INFO),
        ):
            scheduler = maybe_start_lineage_gc(
                bad_factory,
                bootstrap_org_id=DEFAULT_ORG_ID,
            )
        assert scheduler is not None
    finally:
        gc_scheduler.set_org_id_provider(None)
        clear_per_org_sweeps()

    assert "lineage_gc_scheduler_start_deferred" in caplog.text
    assert "lineage_gc_config_read_failed" not in caplog.text


# ---------------------------------------------------------------------------
# list_org_ids — degraded-mode fallback is visible (not silent)
# ---------------------------------------------------------------------------


def test_discover_org_ids_not_implemented_warns_and_falls_back_to_bootstrap(
    caplog,
):
    """When list_org_ids raises NotImplementedError the bootstrap org is still
    processed and a warning is logged (degraded mode is VISIBLE)."""
    storage = MagicMock()
    storage.list_org_ids.side_effect = NotImplementedError("not impl")
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=7)
    bootstrap_ctx = _make_ctx("org_bootstrap", lineage_gc=cfg, storage=storage)

    sched = _scheduler(bootstrap_org_id="org_bootstrap")

    with caplog.at_level(
        logging.WARNING, logger="reflexio.server.services.lineage.gc_scheduler"
    ):
        org_ids = sched._discover_org_ids(bootstrap_ctx)  # type: ignore[arg-type]

    assert org_ids == ["org_bootstrap"]
    assert any(
        "lineage_gc_list_org_ids_not_implemented" in record.message
        for record in caplog.records
    ), "Expected a warning with event=lineage_gc_list_org_ids_not_implemented"


def test_discover_org_ids_provider_raises_falls_back_to_bootstrap():
    """When org_id_provider raises, _discover_org_ids falls back to bootstrap-only.

    Regression guard for Task 3.5 fix 1: a provider exception must NOT propagate
    to _run_loop (which would skip the entire tick). The old NotImplementedError
    fallback behaviour is restored — bootstrap is always swept.
    """
    bootstrap_ctx = _make_ctx("org_bootstrap", lineage_gc=LineageGCConfig(enabled=True))

    def exploding_provider() -> list[str]:
        raise RuntimeError("provider blown up")

    sched = LineageGCScheduler(
        request_context_factory=lambda _: bootstrap_ctx,  # type: ignore[arg-type]
        bootstrap_org_id="org_bootstrap",
        org_id_provider=exploding_provider,
    )

    # Must not raise — provider failure is isolated; bootstrap still included.
    org_ids = sched._discover_org_ids(bootstrap_ctx)  # type: ignore[arg-type]
    assert org_ids == ["org_bootstrap"], (
        "bootstrap org must be present even when provider raises"
    )


# ---------------------------------------------------------------------------
# list_org_ids — SQLite single-tenant implementation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sqlite_list_org_ids_returns_own_org(tmp_path):
    """SQLiteStorage.list_org_ids() returns [self.org_id] for a fresh DB."""
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(org_id="test_org_abc", db_path=db_path)
    assert storage.list_org_ids() == ["test_org_abc"]


# ---------------------------------------------------------------------------
# mid-tick stop check — _gc_tick honours _stop_event between orgs
# ---------------------------------------------------------------------------


def test_gc_tick_stops_mid_tick_when_stop_event_set():
    """_gc_tick must break out of the per-org loop when _stop_event fires."""
    calls: list[str] = []
    cfg = LineageGCConfig(enabled=True, tombstone_grace_window_days=7)

    def factory(org_id: str):
        calls.append(org_id)
        return _make_ctx(org_id, lineage_gc=cfg)

    sched = _scheduler(factory=factory)
    # Set stop after the scheduler is created but before the tick runs.
    sched._stop_event.set()

    sched._gc_tick(["org_a", "org_b", "org_c"])

    # With the stop event already set, the very first iteration should break.
    assert calls == [], f"Expected no orgs processed after stop, got {calls}"


# ---------------------------------------------------------------------------
# Poll interval clamp — non-positive values are floored to _MIN_POLL_SECONDS
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers for dead-knob warning tests
# ---------------------------------------------------------------------------


def _config(
    *, lineage_gc_enabled: bool = True, audit_events_retention_enabled: bool = False
):
    """Build a minimal config-like object for dead-knob warning tests."""
    return SimpleNamespace(
        lineage_gc=SimpleNamespace(enabled=lineage_gc_enabled),
        governance_retention=SimpleNamespace(
            audit_events_retention_enabled=audit_events_retention_enabled
        ),
    )


def _ctx(org_id: str, *, storage, cfg):
    """Build a minimal request-context stand-in backed by an explicit config."""
    return SimpleNamespace(
        org_id=org_id,
        storage=storage,
        configurator=SimpleNamespace(get_config=MagicMock(return_value=cfg)),
    )


def _mock_storage() -> MagicMock:
    return _make_storage()


# ---------------------------------------------------------------------------
# Dead-knob warning — audit-event retention is enterprise-only
# ---------------------------------------------------------------------------


def test_oss_dead_knob_warns_when_retention_enabled_without_enterprise(
    monkeypatch, caplog
):
    """OSS-only deployment with audit_events_retention_enabled=True must warn that
    the knob is enterprise-only (the OSS scheduler does not reclaim audit events)."""
    import reflexio.server.services.configurator.configurator as _conf_mod
    from reflexio.server.services.configurator.configurator import DefaultConfigurator

    # Pin the configurator class to DefaultConfigurator (OSS context).
    monkeypatch.setattr(_conf_mod, "_configurator_class", DefaultConfigurator)

    # Use lineage_gc_enabled=False so maybe_start_lineage_gc returns None without
    # spawning a daemon thread — the dead-knob warning fires before the enabled gate.
    cfg = _config(lineage_gc_enabled=False, audit_events_retention_enabled=True)
    ctx = _ctx(org_id="org1", storage=_mock_storage(), cfg=cfg)

    with caplog.at_level(logging.WARNING):
        maybe_start_lineage_gc(lambda _org: ctx, bootstrap_org_id="org1")  # type: ignore[arg-type]

    assert any(
        "audit_events_retention_enabled" in r.message
        and "enterprise" in r.message.lower()
        for r in caplog.records
    ), (
        f"Expected a dead-knob warning; got records: {[r.message for r in caplog.records]}"
    )


def test_oss_dead_knob_does_not_warn_when_not_oss(monkeypatch, caplog):
    """When a non-DefaultConfigurator class is active (enterprise mode), no warning fires."""
    import reflexio.server.services.configurator.configurator as _conf_mod
    from reflexio.server.services.configurator.configurator import DefaultConfigurator

    class _FakeEnterpriseConfigurator(DefaultConfigurator):
        """Stand-in for the enterprise configurator (not DefaultConfigurator)."""

    monkeypatch.setattr(_conf_mod, "_configurator_class", _FakeEnterpriseConfigurator)

    # Use lineage_gc_enabled=False so no daemon thread is spawned — the warning
    # check runs before the enabled gate.
    cfg = _config(lineage_gc_enabled=False, audit_events_retention_enabled=True)
    ctx = _ctx(org_id="org1", storage=_mock_storage(), cfg=cfg)

    with caplog.at_level(logging.WARNING):
        maybe_start_lineage_gc(lambda _org: ctx, bootstrap_org_id="org1")  # type: ignore[arg-type]

    dead_knob_records = [
        r for r in caplog.records if "audit_events_retention_enabled" in r.message
    ]
    assert not dead_knob_records, (
        f"Unexpected dead-knob warning in enterprise mode: {[r.message for r in dead_knob_records]}"
    )


def test_oss_dead_knob_does_not_warn_when_retention_disabled(monkeypatch, caplog):
    """No warning when audit_events_retention_enabled=False (even in OSS mode)."""
    import reflexio.server.services.configurator.configurator as _conf_mod
    from reflexio.server.services.configurator.configurator import DefaultConfigurator

    monkeypatch.setattr(_conf_mod, "_configurator_class", DefaultConfigurator)

    # Use lineage_gc_enabled=False so no daemon thread is spawned — the warning
    # check runs before the enabled gate.
    cfg = _config(lineage_gc_enabled=False, audit_events_retention_enabled=False)
    ctx = _ctx(org_id="org1", storage=_mock_storage(), cfg=cfg)

    with caplog.at_level(logging.WARNING):
        maybe_start_lineage_gc(lambda _org: ctx, bootstrap_org_id="org1")  # type: ignore[arg-type]

    dead_knob_records = [
        r for r in caplog.records if "audit_events_retention_enabled" in r.message
    ]
    assert not dead_knob_records, (
        f"Unexpected dead-knob warning when retention disabled: {[r.message for r in dead_knob_records]}"
    )


def test_oss_dead_knob_warns_even_when_lineage_gc_disabled(monkeypatch, caplog):
    """Dead-knob warning must fire even when lineage_gc.enabled=False."""
    import reflexio.server.services.configurator.configurator as _conf_mod
    from reflexio.server.services.configurator.configurator import DefaultConfigurator

    monkeypatch.setattr(_conf_mod, "_configurator_class", DefaultConfigurator)

    cfg = _config(lineage_gc_enabled=False, audit_events_retention_enabled=True)
    ctx = _ctx(org_id="org1", storage=_mock_storage(), cfg=cfg)

    with caplog.at_level(logging.WARNING):
        maybe_start_lineage_gc(lambda _org: ctx, bootstrap_org_id="org1")  # type: ignore[arg-type]

    assert any(
        "audit_events_retention_enabled" in r.message
        and "enterprise" in r.message.lower()
        for r in caplog.records
    ), (
        f"Expected dead-knob warning even with GC disabled; got: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Poll interval clamp — non-positive values are floored to _MIN_POLL_SECONDS
# ---------------------------------------------------------------------------


def test_run_loop_clamps_non_positive_poll_interval():
    """_run_loop must pass at least _MIN_POLL_SECONDS to _stop_event.wait.

    Even though config validation now rejects non-positive values, the scheduler
    defends itself at runtime: if a zero or negative interval somehow reaches
    _run_loop, it is clamped to _MIN_POLL_SECONDS to prevent a hot-loop.
    """
    wait_calls: list[float] = []

    # Patch _stop_event.wait to capture the timeout and then immediately stop
    # the loop on the first call.
    sched = _scheduler()

    original_wait = sched._stop_event.wait

    def capturing_wait(timeout: float) -> bool:
        wait_calls.append(timeout)
        sched._stop_event.set()  # stop after one iteration
        return original_wait(0)  # return immediately

    sched._stop_event.wait = capturing_wait  # type: ignore[method-assign]

    # Directly monkey-patch _run_loop's poll_interval by making the config
    # return 0 for poll_interval_seconds.
    zero_cfg = SimpleNamespace(
        lineage_gc=SimpleNamespace(
            poll_interval_seconds=0,
            tombstone_grace_window_days=90,
        )
    )
    zero_ctx = SimpleNamespace(
        org_id="org_bootstrap",
        storage=MagicMock(),
        configurator=SimpleNamespace(get_config=MagicMock(return_value=zero_cfg)),
    )
    sched.request_context_factory = lambda _: zero_ctx  # type: ignore[assignment]

    sched._run_loop()

    assert wait_calls, "wait was never called"
    assert all(t >= _MIN_POLL_SECONDS for t in wait_calls), (
        f"Non-positive poll interval was not clamped: got {wait_calls}"
    )


# ---------------------------------------------------------------------------
# Per-org sweep seam — register_per_org_sweep / clear_per_org_sweeps
# ---------------------------------------------------------------------------


def test_gc_tick_runs_registered_per_org_sweep_gated_live():
    """Per-org sweeps registered via register_per_org_sweep are invoked once per
    org per tick, unconditionally (no expiry_reclamation or lineage_gc gate).
    Each sweep receives (org_id: str, now: int) exactly once per org.
    """
    calls: list[tuple[str, int]] = []

    def fake_sweep(org_id: str, now: int) -> int:
        calls.append((org_id, now))
        return 0

    register_per_org_sweep(fake_sweep)
    try:
        # Disable both Class-A and Class-B to confirm the per-org sweep runs
        # unconditionally regardless of those gates.
        cfg = LineageGCConfig(enabled=False)
        sched = _scheduler(
            factory=lambda org_id: _make_ctx(
                org_id, lineage_gc=cfg, storage=_make_storage()
            )
        )

        before = int(time.time())
        sched._gc_tick(["orgA", "orgB"])
        after = int(time.time())
    finally:
        clear_per_org_sweeps()

    org_ids_called = [c[0] for c in calls]
    assert org_ids_called == ["orgA", "orgB"], (
        f"Expected exactly one call per org in order; got {org_ids_called}"
    )
    for _, now_val in calls:
        assert isinstance(now_val, int)
        assert before <= now_val <= after, (
            f"now_val={now_val} not in [{before}, {after}]"
        )


def test_gc_tick_isolates_per_org_sweep_failure():
    """A raising per-org sweep emits capture_anomaly and does NOT skip sibling
    sweeps registered after it (per-sweep failure isolation).
    """
    sibling_calls: list[tuple[str, int]] = []

    def raising_sweep(org_id: str, now: int) -> int:
        raise RuntimeError("boom")

    def sibling_sweep(org_id: str, now: int) -> int:
        sibling_calls.append((org_id, now))
        return 0

    register_per_org_sweep(raising_sweep)
    register_per_org_sweep(sibling_sweep)
    try:
        cfg = LineageGCConfig(enabled=False)
        sched = _scheduler(
            factory=lambda org_id: _make_ctx(
                org_id, lineage_gc=cfg, storage=_make_storage()
            )
        )

        with patch.object(gc_scheduler, "capture_anomaly") as mock_anomaly:
            sched._gc_tick(["orgA"])
    finally:
        clear_per_org_sweeps()

    mock_anomaly.assert_called_once_with(
        "lineage.per_org_sweep.failed",
        org_id="orgA",
        sweep=raising_sweep.__qualname__,
    )
    assert len(sibling_calls) == 1
    assert sibling_calls[0][0] == "orgA"


# ---------------------------------------------------------------------------
# maybe_start_lineage_gc — registered hooks widen the start gate
# ---------------------------------------------------------------------------


def test_maybe_start_starts_when_a_per_org_sweep_is_registered_even_with_flags_off():
    """Scheduler must start when any per-org sweep is registered, even if all
    config flags (lineage_gc.enabled, expiry_reclamation.enabled,
    governance_retention.audit_events_retention_enabled) are False.

    This preserves the invariant of the deleted GovernanceRetentionScheduler,
    which started unconditionally so that a non-bootstrap tenant with governance
    retention ON would still be swept.
    """

    def fake_sweep(org_id: str, now: int) -> int:
        return 0

    register_per_org_sweep(fake_sweep)
    sched = None
    try:
        cfg = SimpleNamespace(
            lineage_gc=SimpleNamespace(enabled=False, poll_interval_seconds=86400),
            expiry_reclamation=SimpleNamespace(enabled=False),
            governance_retention=SimpleNamespace(audit_events_retention_enabled=False),
        )
        ctx = SimpleNamespace(
            org_id="org_bootstrap",
            storage=_make_storage(),
            configurator=SimpleNamespace(get_config=MagicMock(return_value=cfg)),
        )
        sched = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_bootstrap")  # type: ignore[arg-type]
        assert sched is not None, (
            "Expected scheduler to start when a per-org sweep is registered, "
            "regardless of config flags"
        )
    finally:
        clear_per_org_sweeps()
        clear_global_sweeps()
        if sched is not None:
            sched.stop(timeout_seconds=1.0)


# ---------------------------------------------------------------------------
# Bounded org fan-out — _gc_tick(max_workers=...) + leader-gate forwarding
# ---------------------------------------------------------------------------


def test_gc_tick_parallel_processes_all_orgs(gc_scheduler_factory) -> None:
    """max_workers>1 sweeps every org exactly once (same set as serial)."""
    sched, swept = gc_scheduler_factory(org_ids=[f"o{i}" for i in range(10)])
    sched._gc_tick([f"o{i}" for i in range(10)], max_workers=4)
    assert sorted(swept) == sorted(f"o{i}" for i in range(10))


def test_gc_tick_default_is_serial(gc_scheduler_factory) -> None:
    """Default max_workers=1 preserves strict serial order (byte-compat)."""
    sched, swept = gc_scheduler_factory(org_ids=["a", "b", "c"])
    sched._gc_tick(["a", "b", "c"])
    assert swept == ["a", "b", "c"]


def test_repeat_timeout_escalates_anomaly(monkeypatch, gc_scheduler_factory) -> None:
    """The same org timing out in two consecutive ticks fires capture_anomaly."""
    anomalies: list[tuple] = []
    monkeypatch.setattr(
        gc_mod, "capture_anomaly", lambda name, **kw: anomalies.append((name, kw))
    )
    sched, _ = gc_scheduler_factory(org_ids=["x"])
    monkeypatch.setattr(
        gc_mod, "iterate_orgs_bounded", lambda *_args, **_kwargs: ["x"]
    )  # every tick: org x times out
    sched._gc_tick(["x"], max_workers=2)
    assert not [a for a in anomalies if a[0] == "lineage.gc.org_sweep_timeout_repeat"]
    sched._gc_tick(["x"], max_workers=2)
    assert [a for a in anomalies if a[0] == "lineage.gc.org_sweep_timeout_repeat"]


def test_alternating_timeout_does_not_escalate_anomaly(
    monkeypatch, gc_scheduler_factory
) -> None:
    """An org timing out, then ticking clean, then timing out again is NOT a
    repeat — only STRICTLY CONSECUTIVE timeouts escalate. Guards against a
    naive "seen it time out before" set that never gets cleared on a clean
    tick.
    """
    anomalies: list[tuple] = []
    monkeypatch.setattr(
        gc_mod, "capture_anomaly", lambda name, **kw: anomalies.append((name, kw))
    )
    sched, _ = gc_scheduler_factory(org_ids=["x"])

    # tick1: times out. tick2: clean. tick3: times out again (not consecutive).
    results = iter([["x"], [], ["x"]])
    monkeypatch.setattr(
        gc_mod, "iterate_orgs_bounded", lambda *_args, **_kwargs: next(results)
    )

    sched._gc_tick(["x"], max_workers=2)  # tick1: first-ever timeout
    sched._gc_tick(["x"], max_workers=2)  # tick2: clean tick resets the streak
    sched._gc_tick(["x"], max_workers=2)  # tick3: timed out again, but not consecutive

    assert not [
        a for a in anomalies if a[0] == "lineage.gc.org_sweep_timeout_repeat"
    ], "alternating timeouts must not escalate — only strictly consecutive ticks do"


def test_sqlite_forces_serial(monkeypatch, gc_scheduler_factory) -> None:
    """SQLite storage pins the fan-out to max_workers=1 (spec 6.1)."""
    sched, _ = gc_scheduler_factory(org_ids=["a"], storage_class_name="SQLiteStorage")
    assert sched._org_fanout_workers(sched.request_context_factory("a")) == 1


def test_org_fanout_workers_env_unset_defaults(
    monkeypatch, gc_scheduler_factory
) -> None:
    """Non-SQLite storage + no env override resolves to the module default."""
    monkeypatch.delenv("REFLEXIO_SCHEDULER_ORG_WORKERS", raising=False)
    sched, _ = gc_scheduler_factory(org_ids=["a"])
    assert (
        sched._org_fanout_workers(sched.request_context_factory("a"))
        == _DEFAULT_ORG_FANOUT_WORKERS
    )


def test_org_fanout_workers_env_valid_custom(monkeypatch, gc_scheduler_factory) -> None:
    """A valid positive integer override is honored."""
    monkeypatch.setenv("REFLEXIO_SCHEDULER_ORG_WORKERS", "3")
    sched, _ = gc_scheduler_factory(org_ids=["a"])
    assert sched._org_fanout_workers(sched.request_context_factory("a")) == 3


def test_org_fanout_workers_env_non_numeric_falls_back(
    monkeypatch, gc_scheduler_factory
) -> None:
    """A non-numeric override falls back to the module default."""
    monkeypatch.setenv("REFLEXIO_SCHEDULER_ORG_WORKERS", "abc")
    sched, _ = gc_scheduler_factory(org_ids=["a"])
    assert (
        sched._org_fanout_workers(sched.request_context_factory("a"))
        == _DEFAULT_ORG_FANOUT_WORKERS
    )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_org_fanout_workers_env_non_positive_falls_back(
    monkeypatch, gc_scheduler_factory, value
) -> None:
    """Zero or negative overrides fall back to the module default."""
    monkeypatch.setenv("REFLEXIO_SCHEDULER_ORG_WORKERS", value)
    sched, _ = gc_scheduler_factory(org_ids=["a"])
    assert (
        sched._org_fanout_workers(sched.request_context_factory("a"))
        == _DEFAULT_ORG_FANOUT_WORKERS
    )


def test_leader_gate_forwarded(gc_scheduler_factory) -> None:
    class _Gate:
        def should_run(self) -> bool:
            return False

    gate = _Gate()
    sched, _ = gc_scheduler_factory(org_ids=["a"], leader_gate=gate)
    assert sched._leader_gate is gate


def test_maybe_start_lineage_gc_reads_module_leader_gate_hook():
    """maybe_start_lineage_gc picks up a leader gate set via set_leader_gate,
    mirroring the module-hook fallback for set_org_id_provider (see
    test_maybe_start_lineage_gc_reads_module_provider_hook in
    test_gc_scheduler_multitenant_integration.py)."""

    class _Gate:
        def should_run(self) -> bool:
            return True

    gate = _Gate()
    cfg = LineageGCConfig(enabled=True)
    ctx = _make_ctx("org_1", lineage_gc=cfg)

    set_leader_gate(gate)
    sched = None
    try:
        sched = maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org_1")  # type: ignore[arg-type]
        assert sched is not None
        assert sched._leader_gate is gate
    finally:
        set_leader_gate(None)
        if sched is not None:
            sched.stop(timeout_seconds=1.0)
    assert gc_mod._leader_gate_hook is None
