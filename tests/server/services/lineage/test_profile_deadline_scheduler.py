"""Explicit-deadline scheduling shares native maintenance capacity and cadence."""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.services.lineage import gc_scheduler as gc
from reflexio.server.services.profile.learning.explicit_retention import (
    ProfileExplicitRetentionMixin,
)

_PROJECT = ContextVar("test_deadline_project", default=None)


class MemoryRetention(ProfileExplicitRetentionMixin):
    """Explicitly owns the retention contract; no incidental duck-typed methods."""

    def __init__(self, due=None):
        self.due = due or {}
        self.calls = []
        self.fail = set()

    def next_profile_deadline(self, *, user_id=None):
        project = _PROJECT.get()
        self.calls.append(("discover", project))
        if project in self.fail:
            raise RuntimeError("retention read failed")
        return self.due.get(project)

    def purge_profile_deadlines(self, now, **kwargs):
        project = _PROJECT.get()
        self.calls.append(("purge", project, now, kwargs))
        due = self.due.get(project)
        if due is not None and due <= now:
            self.due.pop(project)
            return [{"status": "hard_deleted"}]
        return []


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(gc, "_project_id_provider_hook", None)
    monkeypatch.setattr(gc, "_per_org_sweep_hooks", [])
    monkeypatch.setattr(gc, "_global_sweep_hooks", [])
    monkeypatch.setattr(gc, "_always_global_sweep_hooks", [])
    monkeypatch.setattr(gc, "_profile_deadline_schedulers", weakref.WeakSet())
    monkeypatch.setattr(gc, "_maintenance_schedule_hook", gc.MaintenanceSchedule())
    monkeypatch.setattr(
        gc, "sweep_retention_caps", lambda *_: SimpleNamespace(failed=False)
    )

    @contextmanager
    def bound(scope):
        token = _PROJECT.set(scope.project_id if scope is not None else None)
        try:
            yield
        finally:
            _PROJECT.reset(token)

    monkeypatch.setattr(gc, "bind_work_scope", bound)


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(wall=1000.0, monotonic=100.0)
    monkeypatch.setattr(
        gc,
        "time",
        SimpleNamespace(
            time=lambda: state.wall,
            monotonic=lambda: state.monotonic,
        ),
    )

    def advance(seconds):
        state.wall += seconds
        state.monotonic += seconds

    state.advance = advance
    return state


def config():
    return SimpleNamespace(
        lineage_gc=LineageGCConfig(tombstone_grace_window_days=0),
        expiry_reclamation=SimpleNamespace(enabled=False),
        governance_retention=SimpleNamespace(audit_events_max_age_days=0),
    )


def scheduler(storage, *, projects=(None,), schedule=None):
    ctx = Mock(
        org_id="org",
        storage=storage,
        configurator=SimpleNamespace(get_config=config),
    )
    gc.set_project_id_provider(lambda _: list(projects))
    result = gc.LineageGCScheduler(
        request_context_factory=lambda _: ctx,
        bootstrap_org_id="org",
        org_id_provider=lambda: ["org"],
        schedule=schedule or gc.MaintenanceSchedule(),
    )
    result._org_fanout_workers = Mock(return_value=1)
    maintenance = []
    result._sweep_project_data = Mock(
        side_effect=lambda *_args, **_kwargs: maintenance.append("project")
    )
    result._run_per_org_sweeps = Mock(
        side_effect=lambda *_args, **_kwargs: maintenance.append("org")
    )
    result._run_global_sweeps = Mock(
        side_effect=lambda *_args, **_kwargs: maintenance.append("global")
    )
    result._run_always_global_sweeps = lambda: maintenance.append("always-global")
    return result, maintenance


def test_due_dispatch_skips_unrelated_scans_and_daily_maintenance(clock):
    storage = MemoryRetention({"soon": 1005, "later": 2000})
    sched, maintenance = scheduler(storage, projects=("soon", "later"))
    assert sched._run_once() == 5
    assert maintenance == ["always-global", "project", "project", "org", "global"]
    storage.calls.clear()
    clock.advance(5)
    assert sched._run_once() == 55
    assert [call[1] for call in storage.calls] == ["soon", "soon", "soon"]
    assert storage.calls[1] == ("purge", "soon", 1005, {"limit": 1000})
    assert maintenance == ["always-global", "project", "project", "org", "global"]
    clock.advance(55)
    assert sched._run_once() == 60
    assert maintenance.count("always-global") == 1


def test_cross_process_earlier_writes_are_discovered_without_local_wake(clock):
    storage = MemoryRetention()
    sched, maintenance = scheduler(storage)
    assert sched._run_once() == 60
    storage.due[None] = 1001  # Another process commits after initial discovery.
    clock.advance(60)
    assert sched._run_once() == 60
    assert None not in storage.due
    assert any(call[0] == "purge" for call in storage.calls)
    assert maintenance.count("always-global") == 1


def test_local_postcommit_notification_requests_fresh_discovery(clock):
    storage = MemoryRetention()
    sched, maintenance = scheduler(storage)
    sched._on_started()
    try:
        assert sched._run_once() == 60
        storage.due[None] = 1004
        gc.notify_profile_deadline_change(org_id="org")
        assert sched._wake_event.is_set()
        assert sched._run_once() == 4
        assert maintenance.count("always-global") == 1
    finally:
        sched._on_stopped()


def test_notification_during_discovery_is_not_lost(clock):
    storage = MemoryRetention()
    sched, _ = scheduler(storage)
    native = storage.next_profile_deadline
    notified = []

    def discover(**kwargs):
        if not notified:
            notified.append(True)
            sched._notify_profile_deadline_change(org_id="org")
        return native(**kwargs)

    storage.next_profile_deadline = discover
    assert sched._run_once() == 60
    storage.calls.clear()
    assert sched._run_once() == 60
    assert storage.calls == [("discover", None)]


def test_full_maintenance_runs_only_when_its_original_due_time_arrives(clock):
    storage = MemoryRetention()
    sched, maintenance = scheduler(storage)
    assert sched._run_once() == 60
    clock.advance(60)
    sched._run_once()
    assert maintenance.count("always-global") == 1
    clock.advance(86400 - 60)
    sched._run_once()
    assert maintenance.count("always-global") == 2
    assert maintenance.count("project") == 2


def test_initial_ordinary_slot_hold_never_holds_deadline_discovery(clock):
    storage = MemoryRetention({None: 1005})
    initial_intervals = []
    sched, maintenance = scheduler(
        storage,
        schedule=gc.MaintenanceSchedule(
            initial_maintenance_delay_provider=lambda interval: (
                initial_intervals.append(interval) or 1000
            ),
            healthy_interval_provider=lambda _: 5000,
        ),
    )
    assert sched._run_once() == 5
    assert initial_intervals == [86400]
    assert maintenance == []
    clock.advance(5)
    assert sched._run_once() == 55
    assert storage.due == {} and maintenance == []
    clock.advance(995)
    assert sched._run_once() == 60
    assert maintenance == ["always-global", "project", "org", "global"]
    assert sched._maintenance_due == clock.monotonic + 5000


def test_deadline_ticks_do_not_spend_or_reset_failed_ordinary_retry_budget(clock):
    storage = MemoryRetention()
    sched, maintenance = scheduler(storage)
    attempts = []

    def ordinary(*_args):
        attempts.append(True)
        if len(attempts) == 1:
            sched._record_tick_failure()

    sched._sweep_project_data = Mock(side_effect=ordinary)
    assert sched._run_once() == 60
    assert sched._maintenance_due == clock.monotonic + 300
    assert sched._consecutive_failed_ticks == 1
    clock.advance(60)
    sched._run_once()
    assert len(attempts) == 1 and sched._consecutive_failed_ticks == 1
    assert maintenance.count("always-global") == 1
    clock.advance(240)
    sched._run_once()
    assert len(attempts) == 2 and sched._consecutive_failed_ticks == 0


def test_due_backlog_retries_only_its_cached_scope_with_native_limit(clock):
    class Backlog(MemoryRetention):
        def purge_profile_deadlines(self, now, **kwargs):
            result = super().purge_profile_deadlines(now, **kwargs)
            if not getattr(self, "once", False):
                self.once = True
                self.due["busy"] = 900
            return result

    storage = Backlog({"busy": 900, "quiet": 2000})
    sched, _ = scheduler(storage, projects=("busy", "quiet"))
    assert sched._run_once() == 1
    storage.calls.clear()
    clock.advance(1)
    assert sched._run_once() == 59
    assert all(call[1] == "busy" for call in storage.calls)
    assert storage.calls[1][-1] == {"limit": 1000}


def test_failed_project_is_visible_without_clearing_healthy_deadline(
    clock, monkeypatch
):
    storage = MemoryRetention({"healthy": 1005, "broken": 1001})
    storage.fail.add("broken")
    sched, _ = scheduler(storage, projects=("broken", "healthy"))
    failures, successes = [], []
    monkeypatch.setattr(
        gc,
        "report_background_failure",
        lambda *_args, **kwargs: failures.append(kwargs["scope"]),
    )
    monkeypatch.setattr(gc, "report_background_success", successes.append)
    assert sched._run_once() == 5
    assert "profile-explicit-retention:org:broken" in failures
    assert "profile-explicit-retention:org:broken" not in successes
    assert "profile-explicit-retention:org:healthy" in successes
    assert sched._profile_deadlines == {("org", "healthy"): 1005}


def test_old_worker_generation_cannot_replace_new_earliest_deadline(clock):
    sched, _ = scheduler(MemoryRetention())
    sched._tick_generation = 2
    sched._record_profile_deadline("org", "current", 1010)
    sched._worker_generation.value = 1
    try:
        sched._record_profile_deadline("org", "old", 1001)
        sched._forget_profile_deadline_org("org")
    finally:
        del sched._worker_generation.value
    assert sched._profile_deadlines == {("org", "current"): 1010}


def test_authoritative_discovery_prunes_removed_scopes(clock):
    storage = MemoryRetention({"removed": 1005, "current": 2000})
    sched, _ = scheduler(storage, projects=("removed", "current"))
    assert sched._run_once() == 5
    gc.set_project_id_provider(lambda _: ["current"])
    sched._notify_profile_deadline_change(org_id="org", project_id="removed")
    assert sched._run_once() == 60
    assert sched._profile_deadlines == {("org", "current"): 2000}


def test_scoped_notification_never_rescans_unrelated_project_metadata(clock):
    storage = MemoryRetention({"quiet": 2000})
    sched, _ = scheduler(storage, projects=("quiet", "changed"))
    assert sched._run_once() == 60
    storage.calls.clear()
    storage.due["changed"] = 1005
    sched._notify_profile_deadline_change(org_id="org", project_id="changed")
    assert sched._run_once() == 5
    assert storage.calls == [("discover", "changed")]


def test_no_argument_notification_does_not_force_a_fleet_metadata_scan(clock):
    storage = MemoryRetention()
    sched, _ = scheduler(storage)
    assert sched._run_once() == 60
    storage.calls.clear()
    sched._notify_profile_deadline_change()
    assert sched._run_once() == 60
    assert storage.calls == []


def test_startup_does_not_require_unrelated_retention_knobs(monkeypatch):
    storage = MemoryRetention()
    ctx = Mock(storage=storage, configurator=SimpleNamespace(get_config=config))
    monkeypatch.setattr(gc, "get_row_retention_limits", dict)
    started = []
    monkeypatch.setattr(
        gc.LineageGCScheduler, "start", lambda self: started.append(self)
    )
    result = gc.maybe_start_lineage_gc(lambda _: ctx, bootstrap_org_id="org")
    assert result is not None and started == [result]
    assert result._deadline_discovery_active


def test_config_failure_does_not_hide_first_explicit_catchup(clock):
    storage = MemoryRetention({None: 900})
    sched, maintenance = scheduler(storage)

    def broken_config():
        raise RuntimeError("ordinary configuration unavailable")

    ctx = sched.request_context_factory("org")
    ctx.configurator.get_config = broken_config
    assert sched._run_once() == 60
    assert storage.due == {}
    assert "project" not in maintenance
    assert sched._tick_had_failure


def test_retention_uses_existing_leader_gate(clock):
    storage = MemoryRetention({None: 900})
    sched, maintenance = scheduler(storage)
    sched._leader_gate = Mock(should_run=Mock(return_value=False))
    sched._elected_interval()
    assert storage.calls == [] and maintenance == []


def test_real_native_scheduler_idle_restart_deletes_without_reader_trigger(tmp_path):
    import time

    from reflexio.models.api_schema.service_schemas import UserProfile
    from reflexio.models.config_schema import LLMConfig
    from reflexio.server.background_work import is_background_work
    from reflexio.server.services.profile.learning.explicit_retention import KEYS
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
    from reflexio.server.services.storage.sqlite_storage.profiles._explicit_retention import (
        ExplicitRetentionMixin,
    )

    discovered, deleted = threading.Event(), threading.Event()

    class OwnedSQLite(SQLiteStorage, ExplicitRetentionMixin):
        def next_profile_deadline(self, **kwargs):
            assert is_background_work()
            result = super().next_profile_deadline(**kwargs)
            discovered.set()
            return result

        def purge_profile_deadlines(self, *args, **kwargs):
            result = super().purge_profile_deadlines(*args, **kwargs)
            if result:
                deleted.set()
            return result

    store = OwnedSQLite(
        org_id="idle-deadline",
        db_path=str(tmp_path / "idle.db"),
        llm_config=LLMConfig(generation_model_name="openai/gpt-4o-mini"),
    )
    deadline = int(time.time()) + 2
    item = UserProfile(
        profile_id="idle",
        user_id="owner",
        content="User paints.",
        last_modified_timestamp=10,
        generated_from_request_id="idle-publish",
        expiration_timestamp=deadline - 1,
        custom_features={
            KEYS[0]: {
                "basis": "explicit_retention",
                "logical_deadline": deadline,
                "boundary": "stop_at",
            }
        },
        embedding=[0.0] * 512,
    )
    store.add_user_profile("owner", [item], skip_embedding=True)
    first, _ = scheduler(store)
    second, _ = scheduler(store)
    try:
        first.start()
        assert discovered.wait(3)
        first.stop()
        second.start()
        # No profile queries, publication notifications or replay clocks drive deletion.
        assert deleted.wait(5)
        second.stop()
        assert store.conn.execute("SELECT count(*) FROM profiles").fetchone()[0] == 0
        assert (
            store.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 0
        )
        events = [
            event
            for event in store.get_lineage_events(entity_type="profile")
            if event.op == "hard_delete"
        ]
        assert len(events) == 1 and events[0].reason == "explicit_retention_deadline"
    finally:
        first.stop()
        second.stop()
        store.conn.close()
