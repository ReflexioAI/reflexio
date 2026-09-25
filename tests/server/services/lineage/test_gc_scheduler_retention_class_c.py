"""Class C runs once per project, with that project bound.

THE INVARIANT: a retention delete only ever runs with a project bound.

Why this is the thing to guard rather than a deleted-row count: under the
enterprise row-level policies an unbound pass reads ZERO ROWS and reports
success, and zero deletions is also the correct answer whenever nothing is over
its cap -- which is the answer for every org in production today. A count-based
assertion therefore cannot tell the bug from the healthy case.

Scope of this file: it proves the SCHEDULER passes the right scope and wraps the
sweep in it, using an in-test work-scope provider. That the policies actually
confine a bound statement is a different claim, proven against a real
``NOBYPASSRLS`` login in
``reflexio_ext/tests/.../test_project_rls_roles_and_policies_integration.py``.

MUTATION CHECK (do this, do not assume it):
  (a) change ``bind_work_scope(scope)`` in ``_sweep_org`` to
      ``bind_work_scope(None)`` -> must fail
      ``test_each_retention_pass_runs_under_its_own_project``
  (b) move the ``sweep_retention_caps`` call from ``_sweep_project_data``
      out to ``_sweep_org``, after the project loop -> must fail the same test
Both were verified before this file was committed.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import create_autospec, patch

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.extensions import register_service
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    set_project_id_provider,
)
from reflexio.server.work_scope import (
    WORK_SCOPE_PROVIDER,
    WorkScope,
    current_project_id,
)

_ORG = "org-1"


class _StubProvider:
    """Minimal ``WorkScopeProvider``: binds a project id on this thread.

    OSS registers no provider, so ``bind_work_scope`` is a no-op there and a
    test relying on the OSS default would pass against an unbound sweep. That is
    exactly the check-that-cannot-fail shape this file exists to avoid, so the
    provider is supplied here.
    """

    def __init__(self) -> None:
        self.scope: WorkScope | None = None

    def current(self) -> WorkScope | None:
        return self.scope

    @contextmanager
    def bind(self, scope: WorkScope) -> Iterator[None]:
        previous = self.scope
        self.scope = scope
        try:
            yield
        finally:
            self.scope = previous


@pytest.fixture(autouse=True)
def _provider() -> Iterator[None]:
    register_service(WORK_SCOPE_PROVIDER, _StubProvider(), override=True)
    set_project_id_provider(None)
    yield
    register_service(WORK_SCOPE_PROVIDER, None, override=True)  # type: ignore[arg-type]
    set_project_id_provider(None)


def _ctx():
    """A context reaching the project loop with every gated sweep off.

    ``lineage_gc`` and ``expiry_reclamation`` are disabled on purpose: Class C is
    ungated, so this also proves it does not depend on either flag.
    """
    return types.SimpleNamespace(
        storage=types.SimpleNamespace(),
        configurator=types.SimpleNamespace(
            get_config=lambda: types.SimpleNamespace(
                lineage_gc=types.SimpleNamespace(enabled=False),
                expiry_reclamation=types.SimpleNamespace(enabled=False),
            )
        ),
    )


def _scheduler() -> LineageGCScheduler:
    return LineageGCScheduler(
        request_context_factory=lambda _org_id: _ctx(),  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )


def test_each_retention_pass_runs_under_its_own_project():
    """THE assertion. The mutations in this file's docstring must kill it."""
    set_project_id_provider(lambda _org: ["prj-a", "prj-b"])
    bound: list[str | None] = []

    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _org, _storage: (
            bound.append(current_project_id()) or gc_scheduler.RetentionSweepResult(0)
        ),
    ):
        _scheduler()._sweep_org(_ORG)

    assert bound == ["prj-a", "prj-b"], (
        f"retention did not run once per bound project: {bound}"
    )


def test_class_c_runs_with_every_gated_sweep_disabled():
    """Class C is ungated -- it must not inherit Class A/B's config gates."""
    set_project_id_provider(lambda _org: ["prj-a"])
    calls: list[str] = []

    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda org, _storage: calls.append(org) or gc_scheduler.RetentionSweepResult(0),
    ):
        _scheduler()._sweep_org(_ORG)

    assert calls == [_ORG]


def test_no_project_provider_still_sweeps_once_unscoped():
    """OSS/SQLite: one unscoped pass, which is correct with no row-level policies."""
    bound: list[str | None] = []

    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _org, _storage: (
            bound.append(current_project_id()) or gc_scheduler.RetentionSweepResult(0)
        ),
    ):
        _scheduler()._sweep_org(_ORG)

    assert bound == [None]


def test_a_project_whose_enumeration_failed_gets_no_pass():
    """Spec §7.1: enumeration failure suspends retention rather than going unscoped."""

    def _boom(_org: str) -> list[str]:
        raise RuntimeError("control plane down")

    set_project_id_provider(_boom)
    calls: list[str] = []

    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda org, _storage: calls.append(org) or gc_scheduler.RetentionSweepResult(0),
    ):
        _scheduler()._sweep_org(_ORG)

    assert calls == []


# ---------------------------------------------------------------------------
# The scheduler must START for retention, not only run Class C once started
# ---------------------------------------------------------------------------
#
# The tests above call `_sweep_org` directly, which bypasses the start gate
# entirely -- so `test_class_c_runs_with_every_gated_sweep_disabled` is named for
# a claim it cannot check on its own. Before this block existed, turning
# `lineage_gc` off in an OSS deployment stopped row-count retention completely:
# the caps had been enforced on every publish, `maybe_start_lineage_gc` returned
# None, and nothing swept and nothing said so.


def _bootstrap_ctx(
    *, lineage_gc: bool, expiry: bool, governance: bool
) -> RequestContext:
    context = create_autospec(RequestContext, instance=True)
    context.storage = types.SimpleNamespace()
    context.configurator = types.SimpleNamespace(
        get_config=lambda: types.SimpleNamespace(
            lineage_gc=types.SimpleNamespace(
                enabled=lineage_gc, poll_interval_seconds=86400
            ),
            expiry_reclamation=types.SimpleNamespace(enabled=expiry),
            governance_retention=types.SimpleNamespace(
                audit_events_retention_enabled=governance
            ),
        )
    )
    return context


def test_the_scheduler_starts_for_retention_with_every_other_gate_off(monkeypatch):
    """Row caps are unconditional, so they must not need another feature's flag.

    The production change this must catch: dropping retention from
    `maybe_start_lineage_gc`'s start conditions. With `lineage_gc`,
    `expiry_reclamation` and `governance_retention` all off and no sweep hook
    registered, the scheduler previously returned None and the caps went
    unenforced with no signal.
    """
    started: list[object] = []
    monkeypatch.setattr(LineageGCScheduler, "start", lambda self: started.append(self))

    scheduler = gc_scheduler.maybe_start_lineage_gc(
        lambda _org: _bootstrap_ctx(lineage_gc=False, expiry=False, governance=False),
        bootstrap_org_id="org-boot",
    )

    assert scheduler is not None, (
        "the scheduler did not start, so row-count retention never runs: a "
        "deployment with lineage_gc disabled silently loses its caps"
    )
    assert started == [scheduler]


def test_the_scheduler_still_declines_when_every_retention_cap_is_disabled(
    monkeypatch,
):
    """The converse, so the new start condition is not a rubber stamp.

    With every row limit set to 0 there is no retention work either, and the
    pre-existing "nothing to do" answer must survive.
    """
    monkeypatch.setattr(
        gc_scheduler, "get_row_retention_limits", lambda: {"interactions": 0}
    )
    monkeypatch.setattr(
        LineageGCScheduler,
        "start",
        lambda _self: pytest.fail("scheduler started with no work to do"),
    )

    assert (
        gc_scheduler.maybe_start_lineage_gc(
            lambda _org: _bootstrap_ctx(
                lineage_gc=False, expiry=False, governance=False
            ),
            bootstrap_org_id="org-boot",
        )
        is None
    )


# ---------------------------------------------------------------------------
# A tick that FAILED must not wait a full poll interval to try again
# ---------------------------------------------------------------------------
#
# Measured on staging, 2026-09-24: 18 of 18 orgs failed one tick with
# `PGRST002: Could not query the database for the schema cache. Retrying.`
# raised from `acquire_simple_lock`. The scheduler's first tick fires seconds
# after boot, which is exactly when PostgREST's schema cache is cold after a
# deploy -- so the boot tick is the MOST likely to hit it, not the least.
#
# On the old publish path a transient cost one publish and retried ~300s later.
# Here `_run_once` returned `poll_interval_seconds` (86400s by default) whether
# the tick worked or not, so one boot transient cost a DAY of retention -- and a
# service redeployed more often than daily would never sweep at all.
#
# The retry is bounded: after `_MAX_CONSECUTIVE_FAST_RETRIES` failed ticks the
# cadence falls back to the poll interval. A persistently broken dependency is
# already firing anomalies on every tick, and hammering it every 5 minutes
# forever would reintroduce exactly the repeated-probe cost this whole change
# removed from the publish path.


def _failing_ctx():
    """A bootstrap context whose per-org sweep raises inside Class C."""
    return types.SimpleNamespace(
        storage=types.SimpleNamespace(),
        configurator=types.SimpleNamespace(
            get_config=lambda: types.SimpleNamespace(
                lineage_gc=types.SimpleNamespace(
                    enabled=False, poll_interval_seconds=86400
                ),
                expiry_reclamation=types.SimpleNamespace(enabled=False),
            )
        ),
    )


def _tick_scheduler() -> LineageGCScheduler:
    sched = LineageGCScheduler(
        request_context_factory=lambda _org_id: _failing_ctx(),  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )
    sched._discover_org_ids = lambda _ctx: [_ORG]  # type: ignore[method-assign]
    return sched


def test_a_clean_tick_waits_the_full_poll_interval():
    """The baseline the retry must not disturb."""
    sched = _tick_scheduler()
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _org, _storage: gc_scheduler.RetentionSweepResult(0, failed=False),
    ):
        assert sched._run_once() == 86400


def test_a_failed_tick_retries_soon_instead_of_in_a_day():
    """THE assertion. Without it, one boot transient costs a full day."""
    sched = _tick_scheduler()
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _org, _storage: gc_scheduler.RetentionSweepResult(0, failed=True),
    ):
        interval = sched._run_once()

    assert interval == gc_scheduler._FAILED_TICK_RETRY_SECONDS, (
        f"a failed tick waits {interval}s; retention would be down that long"
    )


def test_the_fast_retry_is_bounded():
    """A persistently broken dependency must not be hammered forever."""
    sched = _tick_scheduler()
    intervals: list[float] = []
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _org, _storage: gc_scheduler.RetentionSweepResult(0, failed=True),
    ):
        intervals.extend(
            sched._run_once()
            for _ in range(gc_scheduler._MAX_CONSECUTIVE_FAST_RETRIES + 2)
        )

    fast = gc_scheduler._FAILED_TICK_RETRY_SECONDS
    assert (
        intervals[: gc_scheduler._MAX_CONSECUTIVE_FAST_RETRIES]
        == [fast] * gc_scheduler._MAX_CONSECUTIVE_FAST_RETRIES
    ), intervals
    assert intervals[gc_scheduler._MAX_CONSECUTIVE_FAST_RETRIES :] == [86400, 86400], (
        f"the fast retry never gave up: {intervals}"
    )


def test_a_success_after_failures_resets_the_retry_budget():
    """Otherwise one bad day permanently spends the fast retries."""
    sched = _tick_scheduler()
    failing = gc_scheduler.RetentionSweepResult(0, failed=True)
    clean = gc_scheduler.RetentionSweepResult(0, failed=False)

    with patch.object(gc_scheduler, "sweep_retention_caps", lambda _o, _s: failing):
        assert sched._run_once() == gc_scheduler._FAILED_TICK_RETRY_SECONDS
    with patch.object(gc_scheduler, "sweep_retention_caps", lambda _o, _s: clean):
        assert sched._run_once() == 86400
    with patch.object(gc_scheduler, "sweep_retention_caps", lambda _o, _s: failing):
        assert sched._run_once() == gc_scheduler._FAILED_TICK_RETRY_SECONDS


def test_a_configured_interval_shorter_than_the_retry_is_not_lengthened():
    """`poll_interval_seconds` may be as low as 1; "retry sooner" must not slow it.

    Returning the flat 300s constant here would delay lineage GC and every
    registered global sweep relative to the operator's chosen cadence.
    """
    sched = _tick_scheduler()
    sched.request_context_factory = lambda _org: types.SimpleNamespace(  # type: ignore[assignment]
        storage=types.SimpleNamespace(),
        configurator=types.SimpleNamespace(
            get_config=lambda: types.SimpleNamespace(
                lineage_gc=types.SimpleNamespace(
                    enabled=False, poll_interval_seconds=60
                ),
                expiry_reclamation=types.SimpleNamespace(enabled=False),
            )
        ),
    )
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _o, _s: gc_scheduler.RetentionSweepResult(0, failed=True),
    ):
        assert sched._run_once() == 60


def test_a_project_enumeration_failure_fails_the_tick():
    """No project ids means no pass ran at all — including Class C."""
    sched = _tick_scheduler()

    def _boom(_org: str) -> list[str]:
        raise RuntimeError("control plane down")

    set_project_id_provider(_boom)
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _o, _s: gc_scheduler.RetentionSweepResult(0),
    ):
        assert sched._run_once() == gc_scheduler._FAILED_TICK_RETRY_SECONDS


def test_a_timed_out_org_fails_the_tick(monkeypatch):
    """A straggler did not finish, so the tick is not clean.

    Recorded from `_gc_tick` rather than the worker thread, which may still be
    running and would otherwise land its failure in a LATER tick's state.
    """
    sched = _tick_scheduler()
    monkeypatch.setattr(gc_scheduler, "iterate_orgs_bounded", lambda *_a, **_k: [_ORG])
    with patch.object(
        gc_scheduler,
        "sweep_retention_caps",
        lambda _o, _s: gc_scheduler.RetentionSweepResult(0),
    ):
        assert sched._run_once() == gc_scheduler._FAILED_TICK_RETRY_SECONDS


def test_retention_still_starts_when_the_bootstrap_config_read_fails():
    """A config failure must not decide an env-driven cap's fate.

    `retention_enabled` is resolved before the config read for this reason; the
    handler for that read returns early, and retention would be lost with it.
    """
    started: list[object] = []

    def _boom(_org: str):
        raise RuntimeError("bootstrap config unreadable")

    with patch.object(LineageGCScheduler, "start", lambda self: started.append(self)):
        sched = gc_scheduler.maybe_start_lineage_gc(_boom, bootstrap_org_id="org-boot")

    assert sched is not None, (
        "a bootstrap config failure dropped row-count retention, which does not "
        "depend on that config at all"
    )
