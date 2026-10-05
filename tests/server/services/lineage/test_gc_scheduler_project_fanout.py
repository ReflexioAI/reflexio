"""``_sweep_org`` fans its data sweeps out over projects, not over orgs.

The sweeps in ``_sweep_project_data`` write project-scoped tables. Run with no
project bound, a deployment with row-level project policies matches nothing, so
the sweep deletes nothing and reports success -- which is how lineage GC ran
every tick for a week while doing no work at all.

What these tests can and cannot show
------------------------------------
They pin the FAN-OUT: one pass per enumerated project, one unscoped pass when no
provider is registered (the OSS default), the per-org hooks still firing exactly
once, and an enumeration failure not degrading into a silent unscoped pass.

They deliberately do NOT show that the project is actually *bound* to the
writes. ``bind_work_scope`` is inert in OSS -- no provider is registered, by
design -- so a version of ``_sweep_org`` with the ``bind_work_scope`` call
deleted would still pass every test in this file. The binding is asserted where
a provider exists, in the enterprise two-project integration test. Saying so
here is the point: a fan-out test read as a binding test is exactly the kind of
check that cannot fail.
"""

import types

import pytest

from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    set_project_id_provider,
)

_ORG = "org-1"


def _scheduler(**kwargs) -> LineageGCScheduler:
    return LineageGCScheduler(
        request_context_factory=lambda _org_id: types.SimpleNamespace(),  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _isolate_provider():
    set_project_id_provider(None)
    yield
    set_project_id_provider(None)


def _record_passes(scheduler: LineageGCScheduler) -> list[object]:
    """Replace the data sweep with a recorder, returning the recorded calls."""
    seen: list[object] = []
    scheduler._sweep_project_data = lambda *args: seen.append(args)  # type: ignore[method-assign]
    return seen


def _ctx():
    """A context whose storage is present and whose config disables every sweep.

    The sweep body is stubbed out in these tests, so the config only has to be
    shaped well enough for ``_sweep_org`` to reach the loop.
    """
    return types.SimpleNamespace(
        storage=types.SimpleNamespace(),
        configurator=types.SimpleNamespace(get_config=lambda: types.SimpleNamespace()),
    )


def test_no_provider_keeps_the_single_unscoped_pass():
    """The OSS default must be byte-for-byte the old behaviour: exactly one pass."""
    scheduler = _scheduler()
    scheduler.request_context_factory = lambda _org: _ctx()  # type: ignore[assignment]
    seen = _record_passes(scheduler)

    scheduler._sweep_org(_ORG)

    assert len(seen) == 1


def test_one_pass_per_enumerated_project():
    set_project_id_provider(lambda _org: ["prj_a", "prj_b", "prj_c"])
    scheduler = _scheduler()
    scheduler.request_context_factory = lambda _org: _ctx()  # type: ignore[assignment]
    seen = _record_passes(scheduler)

    scheduler._sweep_org(_ORG)

    assert len(seen) == 3


def test_the_per_org_hooks_still_run_exactly_once():
    """Not once per project.

    The registered per-org sweeps reach ``gc_governance_retention``, which is
    org-level compliance GC and already privileged. Folding it into the
    per-project loop would multiply it by the project count -- which is why the
    hook call sits after the loop rather than inside it.
    """
    set_project_id_provider(lambda _org: ["prj_a", "prj_b", "prj_c"])
    scheduler = _scheduler()
    scheduler.request_context_factory = lambda _org: _ctx()  # type: ignore[assignment]
    _record_passes(scheduler)
    hook_calls: list[str] = []
    scheduler._run_per_org_sweeps = hook_calls.append  # type: ignore[method-assign]

    scheduler._sweep_org(_ORG)

    assert hook_calls == [_ORG]


def test_an_org_with_no_projects_is_skipped_not_swept_unscoped():
    """Falling back to one unscoped pass here would be theatre.

    Under project row-level policies an unscoped sweep matches nothing, so the
    fallback does no work AND re-emits the unbound-credential warning this
    change exists to clear. Skipping is the same outcome, said honestly.
    """
    set_project_id_provider(lambda _org: [])
    scheduler = _scheduler()
    scheduler.request_context_factory = lambda _org: _ctx()  # type: ignore[assignment]
    seen = _record_passes(scheduler)

    scheduler._sweep_org(_ORG)

    assert seen == []


def test_a_failing_enumerator_sweeps_nothing_rather_than_sweeping_unscoped():
    """The dangerous failure is the quiet one.

    Falling back to an unscoped pass here would look like work and do none,
    reproducing the exact defect this change exists to fix. Zero passes is the
    honest answer; the anomaly is what carries the signal.
    """

    def _boom(_org: str) -> list[str]:
        raise RuntimeError("control plane unreachable")

    set_project_id_provider(_boom)
    scheduler = _scheduler()
    scheduler.request_context_factory = lambda _org: _ctx()  # type: ignore[assignment]
    seen = _record_passes(scheduler)

    scheduler._sweep_org(_ORG)

    assert seen == []


def test_project_sweep_failures_are_scoped_per_project(transient_failure_classifier):
    """One failing project must not share a streak with its healthy siblings.

    With an org-wide scope the failing project would count once per project per
    tick, and the healthy project's success would clear it every time.
    """
    from reflexio.server import background_work
    from reflexio.server.services.lineage import gc_scheduler

    def _expire(*, now: int) -> int:
        if failing[0]:
            raise transient_failure_classifier("server closed the connection")
        return 0

    failing = [False]
    storage = types.SimpleNamespace(
        expire_active_profiles=_expire,
        gc_expired_tombstones=lambda **_kwargs: 0,
    )
    cfg = types.SimpleNamespace(
        lineage_gc=types.SimpleNamespace(enabled=True, tombstone_grace_window_days=30),
        expiry_reclamation=None,
    )
    scheduler = _scheduler()
    original = scheduler._sweep_project_data

    def _sweep(org_id, storage_, cfg_, project_id=None):
        failing[0] = project_id == "prj_a"
        original(org_id, storage_, cfg_, project_id)

    scheduler._sweep_project_data = _sweep  # type: ignore[method-assign]
    scheduler.request_context_factory = lambda _org: types.SimpleNamespace(  # type: ignore[assignment]
        storage=storage, configurator=types.SimpleNamespace(get_config=lambda: cfg)
    )
    set_project_id_provider(lambda _org: ["prj_a", "prj_b"])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            gc_scheduler,
            "sweep_retention_caps",
            lambda *_args: types.SimpleNamespace(failed=False),
        )
        scheduler._sweep_org(_ORG)
        scheduler._sweep_org(_ORG)

    assert background_work._streaks["lineage-expiry-sweep:org-1:prj_a"][0] == 2
    assert "lineage-expiry-sweep:org-1:prj_b" not in background_work._streaks


def test_hooks_sharing_a_qualname_keep_separate_streaks(transient_failure_classifier):
    """Closures from one factory share a qualname; one's success must not
    clear the other's streak."""
    from reflexio.server import background_work
    from reflexio.server.services.lineage import gc_scheduler

    def make(fails: bool):
        def sweep(_org_id: str, _now: int) -> int:
            if fails:
                raise transient_failure_classifier("server closed the connection")
            return 0

        return sweep

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gc_scheduler, "_per_org_sweep_hooks", [make(True), make(False)])
        _scheduler()._run_per_org_sweeps(_ORG)

    (scope,) = background_work._streaks
    assert scope.startswith("lineage-per-org-sweep:org-1:0:")
