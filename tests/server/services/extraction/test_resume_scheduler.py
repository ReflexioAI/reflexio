from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.services.extraction import resume_scheduler


def _request_context(
    org_id: str = "org_1", *, poll_interval: float = 0.01, storage=None
):
    config = SimpleNamespace(
        pending_tool_call_config=PendingToolCallConfig(
            enabled=True,
            resume_poll_interval_seconds=poll_interval,
        )
    )
    return SimpleNamespace(
        org_id=org_id,
        storage=storage,
        configurator=SimpleNamespace(get_config=MagicMock(return_value=config)),
    )


def test_maybe_start_resume_scheduler_skips_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(
        resume_scheduler,
        "pending_tool_calls_enabled",
        lambda _ctx: False,
    )

    scheduler = resume_scheduler.maybe_start_resume_scheduler(
        cast(Callable[[str], RequestContext], lambda org_id: _request_context(org_id)),
        bootstrap_org_id="org_1",
    )

    assert scheduler is None


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
    assert scheduler._run_once() == 0.01
    assert scheduler.bootstrap_org_id == "org_1"
    assert factory_calls == ["org_1", "org_1"]


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
    assert factory_calls == ["org_2", "org_2", "org_3", "org_3"]
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
