from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.services.extraction import resume_scheduler


def _request_context(org_id: str = "org_1", *, storage=None):
    config = SimpleNamespace(
        pending_tool_call_config=PendingToolCallConfig(
            enabled=True,
        )
    )
    return SimpleNamespace(
        org_id=org_id,
        storage=storage,
        configurator=SimpleNamespace(get_config=MagicMock(return_value=config)),
    )


def test_resume_scheduler_observes_feature_enabled_after_startup(monkeypatch):
    context = _request_context(
        storage=SimpleNamespace(
            list_resumable_work_org_ids=MagicMock(return_value=[]),
            expire_pending_tool_calls=MagicMock(return_value=0),
        )
    )
    context.configurator.get_config().pending_tool_call_config.enabled = False
    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda ctx: ctx.configurator.get_config().pending_tool_call_config.enabled,
    )
    monkeypatch.setattr(
        resume_scheduler.ExtractionResumeScheduler, "start", lambda _self: None
    )
    drain = MagicMock(return_value=0)
    monkeypatch.setattr(
        resume_scheduler,
        "ExtractionResumeWorker",
        lambda **_kwargs: SimpleNamespace(drain=drain),
    )
    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], lambda _org_id: context),
        bootstrap_org_id="org_1",
    )
    scheduler._run_once()
    drain.assert_not_called()
    context.configurator.get_config().pending_tool_call_config.enabled = True
    scheduler._run_once()
    drain.assert_called_once_with(max_runs=10)


def test_resume_scheduler_recovers_when_first_org_appears(monkeypatch):
    org_ids = [DEFAULT_ORG_ID]
    factory_calls: list[str] = []

    def factory(org_id: str):
        factory_calls.append(org_id)
        if org_id == DEFAULT_ORG_ID:
            raise RuntimeError("organization not found")
        return _request_context(org_id)

    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda _ctx: True,
    )
    monkeypatch.setattr(
        resume_scheduler.ExtractionResumeScheduler,
        "start",
        lambda _self: None,
    )

    def drain(_self, *, max_runs: int) -> int:
        assert max_runs == 10
        return 0

    monkeypatch.setattr(
        resume_scheduler.ExtractionResumeWorker,
        "drain",
        drain,
    )

    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], factory),
        bootstrap_org_id=DEFAULT_ORG_ID,
        org_id_provider=lambda: list(org_ids),
    )

    assert scheduler is not None
    factory_calls.clear()
    assert scheduler._run_once() == 5.0
    assert factory_calls == []

    org_ids[:] = ["org_1"]
    assert scheduler._run_once() == 5.0
    assert scheduler.bootstrap_org_id == "org_1"
    assert factory_calls == ["org_1"]


def test_resume_scheduler_drains_all_orgs_and_stops_cleanly(monkeypatch):
    ticked = threading.Event()
    drained_orgs: list[str] = []
    storage = SimpleNamespace(
        expire_pending_tool_calls=MagicMock(return_value=1),
        # Cross-org discovery surfaces two tenants with work; the bootstrap org
        # is not among them, so the scheduler must sweep all three.
        list_resumable_work_org_ids=MagicMock(return_value=["org_2", "org_3"]),
    )

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            assert max_runs == 10
            drained_orgs.append(self.request_context.org_id)
            if {"org_1", "org_2", "org_3"} <= set(drained_orgs):
                ticked.set()
            return 1

    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda _ctx: True,
    )
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)

    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(org_id, storage=storage),
        ),
        bootstrap_org_id="org_1",
    )
    assert scheduler is not None
    try:
        assert ticked.wait(timeout=1.0)
    finally:
        scheduler.stop(timeout_seconds=1.0)
    assert storage.expire_pending_tool_calls.called
    # The bootstrap org plus every discovered org are all drained.
    assert {"org_1", "org_2", "org_3"} <= set(drained_orgs)


def test_resume_scheduler_stale_bootstrap_cannot_block_future_discovery(monkeypatch):
    provider = MagicMock(side_effect=[["org_2"], ["org_3"]])
    drained: list[str] = []
    live_org_ids = {"org_2"}
    factory_calls: list[str] = []

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            drained.append(self.request_context.org_id)
            return 0

    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)

    def factory(org_id: str):
        factory_calls.append(org_id)
        if org_id not in live_org_ids:
            raise RuntimeError(f"stale org: {org_id}")
        return _request_context(
            org_id,
            storage=SimpleNamespace(
                expire_pending_tool_calls=MagicMock(return_value=0),
                list_resumable_work_org_ids=MagicMock(return_value=["wrong-org"]),
            ),
        )

    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(Callable[[str], RequestContext], factory),
        bootstrap_org_id="org_1",
        org_id_provider=provider,
    )

    scheduler._run_once()
    live_org_ids.clear()
    live_org_ids.add("org_3")
    scheduler._run_once()

    assert provider.call_count == 2
    assert drained == ["org_2", "org_3"]
    assert factory_calls == ["org_2", "org_3"]
    assert scheduler.bootstrap_org_id == "org_3"


def test_resume_scheduler_provider_failure_still_drains_bootstrap(monkeypatch):
    drained: list[str] = []
    storage = SimpleNamespace(expire_pending_tool_calls=MagicMock(return_value=0))

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            drained.append(self.request_context.org_id)
            return 0

    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(org_id, storage=storage),
        ),
        bootstrap_org_id="org_1",
        org_id_provider=MagicMock(side_effect=RuntimeError("registry unavailable")),
    )

    scheduler._run_once()

    assert drained == ["org_1"]


def test_resume_scheduler_expires_pending_calls_on_each_discovered_ref(monkeypatch):
    storages = {
        org_id: SimpleNamespace(expire_pending_tool_calls=MagicMock(return_value=0))
        for org_id in ("org_1", "org_2")
    }

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            return 0

    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(org_id, storage=storages[org_id]),
        ),
        bootstrap_org_id="org_1",
        org_id_provider=lambda: ["org_1", "org_2"],
    )

    scheduler._run_once()

    storages["org_1"].expire_pending_tool_calls.assert_called_once()
    storages["org_2"].expire_pending_tool_calls.assert_called_once()


def test_a_disabled_org_with_work_says_so_instead_of_skipping_silently(
    monkeypatch, caplog
):
    """The skip is specified; its silence was not.

    `test_resume_scheduler_observes_feature_enabled_after_startup` pins that a
    disabled org is NOT drained, and that stays true. This pins that the skip
    is legible: the provider has just named the org as having resumable work,
    so an operator reading the logs must be able to tell "disabled on purpose"
    from "broken".

    Prod 2026-09-20 is why: 238 runs across three orgs sat behind this branch,
    every one a finalization retry already holding `committed_output`, and the
    sweep reported a nonzero `resumable_orgs` every two minutes with nothing
    anywhere explaining why none of it moved.
    """
    context = _request_context(
        storage=SimpleNamespace(
            # NON-empty on purpose. An earlier cut stubbed `[]` and still
            # asserted the line fired -- which passed only because the code
            # logged for the unconditionally-prepended bootstrap org, i.e.
            # the test encoded the very defect Codex caught (>17k
            # misleading lines/day on an idle install). Discovery has to
            # actually name the org for "has resumable work" to be true.
            list_resumable_work_org_ids=MagicMock(return_value=["org_1"]),
            expire_pending_tool_calls=MagicMock(return_value=0),
        )
    )
    context.configurator.get_config().pending_tool_call_config.enabled = False
    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda ctx: ctx.configurator.get_config().pending_tool_call_config.enabled,
    )
    monkeypatch.setattr(
        resume_scheduler.ExtractionResumeScheduler, "start", lambda _self: None
    )
    drain = MagicMock(return_value=0)
    monkeypatch.setattr(
        resume_scheduler,
        "ExtractionResumeWorker",
        lambda **_kwargs: SimpleNamespace(drain=drain),
    )
    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], lambda _org_id: context),
        bootstrap_org_id="org_1",
    )

    with caplog.at_level(logging.INFO):
        scheduler._run_once()

    drain.assert_not_called()
    skips = [
        record.getMessage()
        for record in caplog.records
        if "extraction_resume_drain_skipped_gate_disabled" in record.getMessage()
    ]
    assert skips, f"the skip must be logged; saw {[r.message for r in caplog.records]}"
    # It must name WHICH gate. `pending_tool_calls_enabled` is false for three
    # different reasons, and an operator told to enable a setting that is
    # already on has been sent to fix the wrong thing by the diagnostic.
    assert "reason=pending_tool_call_config.enabled is false" in skips[0], skips[0]


def test_an_idle_install_does_not_claim_the_bootstrap_org_has_work(monkeypatch, caplog):
    """No provider, no discovered work: say nothing.

    `_discover_local_org_ids` prepends the bootstrap org unconditionally so a
    single-tenant install still drains. That org has NOT been shown to hold
    resumable work, and the default OSS path has no provider to narrow it --
    so an unconditional skip line fires on every tick. At the 5s default poll
    interval that is >17,000 entries a day, each asserting work exists that
    does not, on an install where nothing is wrong.

    Caught by Codex on #522 against a first cut whose PR description claimed
    the path was "already narrowed to orgs that actually have work". It was
    not, on this path.
    """
    context = _request_context(
        storage=SimpleNamespace(
            list_resumable_work_org_ids=MagicMock(return_value=[]),
            expire_pending_tool_calls=MagicMock(return_value=0),
        )
    )
    context.configurator.get_config().pending_tool_call_config.enabled = False
    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda ctx: ctx.configurator.get_config().pending_tool_call_config.enabled,
    )
    monkeypatch.setattr(
        resume_scheduler.ExtractionResumeScheduler, "start", lambda _self: None
    )
    drain = MagicMock(return_value=0)
    monkeypatch.setattr(
        resume_scheduler,
        "ExtractionResumeWorker",
        lambda **_kwargs: SimpleNamespace(drain=drain),
    )
    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], lambda _org_id: context),
        bootstrap_org_id="org_1",
    )

    with caplog.at_level(logging.INFO):
        scheduler._run_once()

    drain.assert_not_called()
    noise = [
        record.getMessage()
        for record in caplog.records
        if "extraction_resume_drain_skipped" in record.getMessage()
    ]
    assert not noise, (
        "an idle install must not claim the bootstrap org has resumable "
        f"work; saw {noise}"
    )


def test_transient_org_drain_failure_is_a_warning(
    caplog, transient_failure_classifier
) -> None:
    def factory(_org_id: str):
        raise transient_failure_classifier("terminating connection")

    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(Callable[[str], RequestContext], factory),
        bootstrap_org_id="org_1",
    )
    caplog.set_level(logging.WARNING, logger=resume_scheduler.logger.name)

    scheduler._drain_org("org_1")

    failures = [r for r in caplog.records if "org_failed" in r.getMessage()]
    assert [r.levelno for r in failures] == [logging.WARNING]
    assert not failures[0].exc_info

    from reflexio.server import background_work

    assert set(background_work._streaks) == {"extraction-resume:org_1"}
    scheduler.request_context_factory = lambda _org_id: SimpleNamespace(  # type: ignore[assignment]
        storage=None
    )
    with (
        patch.object(resume_scheduler, "pending_tool_calls_enabled", lambda _c: True),
        patch.object(resume_scheduler, "ExtractionResumeWorker") as worker,
    ):
        worker.return_value.drain.return_value = 0
        scheduler._drain_org("org_1")
    assert background_work._streaks == {}


def test_an_empty_provider_tick_does_not_clear_the_tick_streak(
    transient_failure_classifier,
) -> None:
    """The tick's failing work (context and config) never ran."""
    from reflexio.server import background_work

    background_work.report_background_failure(
        resume_scheduler.logger,
        "seeded_failure",
        transient_failure_classifier("dropped"),
        scope="extraction-resume-tick",
    )
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(Callable[[str], RequestContext], MagicMock()),
        bootstrap_org_id="org_1",
        org_id_provider=lambda: [],
    )

    scheduler._run_once()

    assert "extraction-resume-tick" in background_work._streaks


def test_start_gate_holds_the_first_sweep_with_no_storage_call(monkeypatch):
    """A shut start gate defers the whole first tick: no context, no discovery."""
    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _c: True)
    from reflexio.server import scheduling

    monkeypatch.setattr(scheduling, "_START_GATE_POLL_SECONDS", 0.005)
    drained = threading.Event()
    monkeypatch.setattr(
        resume_scheduler,
        "ExtractionResumeWorker",
        lambda **_kw: SimpleNamespace(drain=lambda **_k: drained.set() or 0),
    )
    factory = MagicMock(
        side_effect=lambda org_id: _request_context(
            org_id,
            storage=SimpleNamespace(
                list_resumable_work_org_ids=MagicMock(return_value=[]),
                expire_pending_tool_calls=MagicMock(return_value=0),
            ),
        )
    )
    is_open = threading.Event()
    polled = threading.Event()

    def gate() -> bool:
        polled.set()
        return is_open.is_set()

    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], factory),
        bootstrap_org_id="org_1",
        start_gate=gate,
    )
    try:
        assert polled.wait(timeout=1.0)
        assert not drained.wait(timeout=0.1)
        factory.assert_not_called()
        is_open.set()
        assert drained.wait(timeout=1.0), "first sweep never ran once open"
    finally:
        scheduler.stop(timeout_seconds=1.0)


def test_fleet_polling_does_not_read_one_orgs_timing():
    context = _request_context()
    context.configurator.get_config.side_effect = AssertionError(
        "no bootstrap policy read"
    )
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext], lambda _org: context
        ),
        bootstrap_org_id="org_1",
        org_id_provider=lambda: ["org_1", "org_2"],
    )
    with patch.object(scheduler, "_drain_org") as drain:
        assert scheduler._run_once() == 5.0
    assert drain.call_count == 2
    context.configurator.get_config.assert_not_called()


def test_first_org_context_failure_cannot_block_other_discovered_orgs(monkeypatch):
    drained: list[str] = []

    def factory(org_id):
        if org_id == "broken":
            raise RuntimeError("organization configuration unavailable")
        return _request_context(org_id)

    class Worker:
        def __init__(self, *, request_context):
            self.context = request_context

        def drain(self, *, max_runs):
            drained.append(self.context.org_id)
            return 0

    monkeypatch.setattr(
        resume_scheduler, "pending_tool_calls_enabled", lambda _ctx: True
    )
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", Worker)
    monkeypatch.setenv("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "0.25")
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(Callable[[str], RequestContext], factory),
        bootstrap_org_id="broken",
        org_id_provider=lambda: ["broken", "healthy"],
    )
    assert scheduler._run_once() == 0.25
    assert drained == ["healthy"]


# --- Idle backoff -----------------------------------------------------------
#
# Invariant: every tick is still a complete scan; only the gap between ticks
# that proved there was no work grows. Work, a provider failure, a tick
# failure, and stop/start all return the wait to ``poll_seconds``, and
# ``wake()`` cuts a backed-off wait short.


def _idle_provider_scheduler(provider, monkeypatch, *, poll: str = "5"):
    monkeypatch.setenv("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", poll)
    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)

    class Worker:
        def __init__(self, *, request_context):
            self.context = request_context

        def drain(self, *, max_runs: int) -> int:
            return 0

    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", Worker)
    return resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext], lambda org_id: _request_context(org_id)
        ),
        bootstrap_org_id="org_1",
        org_id_provider=provider,
    )


def test_idle_provider_ticks_double_the_wait_up_to_the_cap(monkeypatch):
    scheduler = _idle_provider_scheduler(lambda: [], monkeypatch)

    waits = [scheduler._run_once() for _ in range(7)]

    assert waits == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]
    assert resume_scheduler.IDLE_BACKOFF_CAP_SECONDS == 60.0


def test_poll_interval_above_the_cap_is_never_shortened(monkeypatch):
    scheduler = _idle_provider_scheduler(lambda: [], monkeypatch, poll="90")

    assert [scheduler._run_once() for _ in range(3)] == [90.0, 90.0, 90.0]


def test_work_found_resets_the_backoff(monkeypatch):
    answers = [[], [], [], ["org_2"], []]
    scheduler = _idle_provider_scheduler(lambda: answers.pop(0), monkeypatch)

    waits = [scheduler._run_once() for _ in range(5)]

    # Third idle tick waited 20s; the tick that found work returns to the
    # poll, and the next idle streak starts over from it.
    assert waits == [5.0, 10.0, 20.0, 5.0, 5.0]


def test_provider_failure_returns_to_the_poll_interval(monkeypatch):
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("registry unavailable")
        return []

    scheduler = _idle_provider_scheduler(provider, monkeypatch)

    waits = [scheduler._run_once() for _ in range(5)]

    # A failed discovery is not evidence of idleness: poll, then restart.
    assert waits == [5.0, 10.0, 20.0, 5.0, 5.0]


def test_incomplete_discovery_returns_to_the_poll_interval(monkeypatch):
    """A provider that swallowed its outage must not read as an idle fleet."""
    answers: list[list[str] | Exception] = [
        [],
        [],
        [],
        resume_scheduler.ResumeDiscoveryIncomplete(),
        resume_scheduler.ResumeDiscoveryIncomplete(),
        [],
        [],
    ]

    def provider() -> list[str]:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    scheduler = _idle_provider_scheduler(provider, monkeypatch)

    waits = [scheduler._run_once() for _ in range(7)]

    # Outage ticks poll, and the idle streak restarts after recovery.
    assert waits == [5.0, 10.0, 20.0, 5.0, 5.0, 5.0, 10.0]


def test_incomplete_discovery_is_not_reported_again(monkeypatch):
    """The provider already reported its failure: no second report, no success."""
    failures = MagicMock()
    successes = MagicMock()
    monkeypatch.setattr(resume_scheduler, "report_background_failure", failures)
    monkeypatch.setattr(resume_scheduler, "report_background_success", successes)

    def provider() -> list[str]:
        raise resume_scheduler.ResumeDiscoveryIncomplete()

    scheduler = _idle_provider_scheduler(provider, monkeypatch)
    scheduler._run_once()

    failures.assert_not_called()
    assert "extraction-resume-org-discovery" not in [
        call.args[0] for call in successes.call_args_list
    ]


def test_empty_incomplete_discovery_still_drains_bootstrap(monkeypatch):
    drained: list[str] = []

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            drained.append(self.request_context.org_id)
            return 0

    def provider() -> list[str]:
        raise resume_scheduler.ResumeDiscoveryIncomplete()

    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(
                org_id,
                storage=SimpleNamespace(
                    expire_pending_tool_calls=MagicMock(return_value=0)
                ),
            ),
        ),
        bootstrap_org_id="org_1",
        org_id_provider=provider,
    )

    scheduler._run_once()

    assert drained == ["org_1"]


def test_partial_discovery_drains_what_it_found_without_backing_off(monkeypatch):
    drained: list[str] = []

    class FakeWorker:
        def __init__(self, *, request_context):
            self.request_context = request_context

        def drain(self, *, max_runs: int) -> int:
            drained.append(self.request_context.org_id)
            return 0

    def provider() -> list[str]:
        raise resume_scheduler.ResumeDiscoveryIncomplete(
            ["org_2", DEFAULT_ORG_ID, "org_2"]
        )

    monkeypatch.setenv("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "5")
    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", FakeWorker)
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(
                org_id,
                storage=SimpleNamespace(
                    expire_pending_tool_calls=MagicMock(return_value=0)
                ),
            ),
        ),
        bootstrap_org_id="org_1",
        org_id_provider=provider,
    )

    assert [scheduler._run_once() for _ in range(3)] == [5.0, 5.0, 5.0]
    assert drained == ["org_2", "org_2", "org_2"]
    assert scheduler.bootstrap_org_id == "org_2"


def test_tick_failure_returns_to_the_poll_interval(monkeypatch):
    scheduler = _idle_provider_scheduler(lambda: [], monkeypatch)
    scheduler.org_id_provider = None
    failing = {"on": False}

    def factory(org_id: str):
        if failing["on"]:
            raise RuntimeError("bootstrap context unavailable")
        return _request_context(
            org_id,
            storage=SimpleNamespace(
                list_resumable_work_org_ids=MagicMock(return_value=[]),
                expire_pending_tool_calls=MagicMock(return_value=0),
            ),
        )

    scheduler.request_context_factory = cast(Callable[[str], RequestContext], factory)
    assert [scheduler._run_once() for _ in range(3)] == [5.0, 10.0, 20.0]
    failing["on"] = True
    assert scheduler._run_once() == 5.0
    failing["on"] = False
    assert scheduler._run_once() == 5.0


def test_bootstrap_org_alone_is_not_counted_as_work(monkeypatch):
    """No provider: the always-prepended bootstrap org is visited, not work."""
    drained: list[str] = []
    inspected = {"n": 0}
    discovered: list[str] = []
    storage = SimpleNamespace(
        list_resumable_work_org_ids=MagicMock(side_effect=lambda **_: list(discovered)),
        expire_pending_tool_calls=MagicMock(return_value=0),
    )

    class Worker:
        def __init__(self, *, request_context):
            self.context = request_context

        def drain(self, *, max_runs: int) -> int:
            drained.append(self.context.org_id)
            return inspected["n"]

    monkeypatch.setenv("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "5")
    monkeypatch.setattr(resume_scheduler, "pending_tool_calls_enabled", lambda _: True)
    monkeypatch.setattr(resume_scheduler, "ExtractionResumeWorker", Worker)
    scheduler = resume_scheduler.ExtractionResumeScheduler(
        request_context_factory=cast(
            Callable[[str], RequestContext],
            lambda org_id: _request_context(org_id, storage=storage),
        ),
        bootstrap_org_id="org_1",
    )

    assert [scheduler._run_once() for _ in range(3)] == [5.0, 10.0, 20.0]
    # Every idle tick still drained the bootstrap org: a complete scan.
    assert drained == ["org_1", "org_1", "org_1"]

    # Discovery naming an org is work, and so is a drain that inspected a
    # claim on a backend that cannot discover.
    discovered[:] = ["org_2"]
    assert scheduler._run_once() == 5.0
    discovered.clear()
    assert scheduler._run_once() == 5.0
    assert scheduler._run_once() == 10.0
    inspected["n"] = 1
    assert scheduler._run_once() == 5.0


def test_stop_and_start_reset_the_backoff(monkeypatch):
    scheduler = _idle_provider_scheduler(lambda: [], monkeypatch)
    assert [scheduler._run_once() for _ in range(3)] == [5.0, 10.0, 20.0]

    scheduler.stop(timeout_seconds=0)
    assert scheduler._run_once() == 5.0

    scheduler._run_once()
    monkeypatch.setattr(resume_scheduler.ThreadedScheduler, "start", lambda _s: None)
    scheduler.start()
    assert scheduler._run_once() == 5.0


def test_wake_cuts_a_backed_off_wait_short(monkeypatch):
    """A tool resolution must not wait out the idle backoff."""
    ticks: list[float] = []
    first = threading.Event()
    second = threading.Event()
    scheduler = _idle_provider_scheduler(lambda: [], monkeypatch, poll="30")
    real_run_once = scheduler._run_once

    def counting_run_once() -> float:
        interval = real_run_once()
        ticks.append(interval)
        if len(ticks) == 1:
            first.set()
        if len(ticks) == 2:
            second.set()
        return interval

    scheduler._run_once = counting_run_once  # type: ignore[method-assign]
    scheduler.start()
    try:
        # The first idle tick returns a 30s wait; only wake() can bring the
        # second tick inside the 2s bound below.
        assert first.wait(2)
        assert not second.is_set()
        assert ticks == [30.0]
        resume_scheduler.wake_resume_schedulers()
        assert second.wait(2), "wake() did not cut the backed-off wait short"
    finally:
        scheduler.stop(timeout_seconds=2)
    assert ticks[:2] == [30.0, 60.0]
