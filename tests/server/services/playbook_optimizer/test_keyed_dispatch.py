"""Optimizer dispatch crosses the actual shared executor and scope seam."""

import threading

from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.callback_executor import BoundedCallbackExecutor
from reflexio.server.services.playbook_optimizer import scheduler as module
from reflexio.server.services.playbook_optimizer.optimizer import (
    PlaybookOptimizationTarget,
)


def test_optimizer_serializes_same_target_and_coalesces_pending_work(monkeypatch):
    executor = BoundedCallbackExecutor(workers=2, queue_size=3)
    monkeypatch.setattr(module, "submit_callback", executor.submit)
    scheduler = module.PlaybookOptimizationScheduler()
    target = PlaybookOptimizationTarget(kind="agent_playbook", target_id=1)
    started, release, other = (threading.Event(), threading.Event(), threading.Event())
    ran = []

    def first():
        started.set()
        assert release.wait(5)
        ran.append("first")
        return "completed"

    scheduler.enqueue(org_id="org", target=target, callback=first)
    assert started.wait(2)
    try:
        scheduler.enqueue(
            org_id="org", target=target, callback=lambda: ran.append("old")
        )
        scheduler.enqueue(
            org_id="org", target=target, callback=lambda: ran.append("new")
        )
        scheduler.enqueue(org_id="other-org", target=target, callback=other.set)
        assert other.wait(2)
        assert ran == []
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    assert ran == ["first", "new"]


def test_queued_optimizer_retains_captured_custom_failure_policy(monkeypatch):
    executor = BoundedCallbackExecutor(workers=1, queue_size=3)
    monkeypatch.setattr(module, "submit_callback", executor.submit)
    scheduler = module.PlaybookOptimizationScheduler()
    target = PlaybookOptimizationTarget(kind="user_playbook", target_id=1)
    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(5)

    executor.submit("blocker", blocker)
    assert started.wait(2)
    config = PlaybookOptimizerConfig(
        abort_cooldown_threshold=1, cooldown_after_aborts_seconds=300
    )
    try:
        scheduler.enqueue(
            org_id="org",
            target=target,
            callback=lambda: "aborted",
            policy=config.scheduling_policy,
        )
        config.abort_cooldown_threshold = 99
        config.cooldown_after_aborts_seconds = 0
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    scheduler.enqueue(
        org_id="org", target=target, callback=lambda: ran.append("blocked")
    )
    scheduler.enqueue(
        org_id="other-org", target=target, callback=lambda: ran.append("other")
    )
    assert executor.drain(timeout_seconds=3)
    assert ran == ["other"]
