import threading
import time

from reflexio.server import callback_executor as ce_mod
from reflexio.server.callback_executor import BoundedCallbackExecutor


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_callbacks_run() -> None:
    ex = BoundedCallbackExecutor(workers=2, queue_size=8)
    ran = threading.Event()
    ex.submit("t", ran.set)
    assert ran.wait(timeout=2.0)


def test_drain_waits_for_active_and_queued_callbacks() -> None:
    ex = BoundedCallbackExecutor(workers=1, queue_size=8)
    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    def blocker() -> None:
        started.set()
        release.wait(timeout=5)
        ran.append("blocker")

    ex.submit("blocker", blocker)
    assert started.wait(timeout=2.0), "blocker never started running"
    ex.submit("after", lambda: ran.append("after"))

    drained = threading.Event()

    def drain() -> None:
        if ex.drain(timeout_seconds=2.0):
            drained.set()

    waiter = threading.Thread(target=drain)
    waiter.start()
    time.sleep(0.05)
    assert not drained.is_set(), "drain returned before active work completed"

    release.set()
    waiter.join(timeout=2.0)
    assert drained.is_set(), f"drain timed out before callbacks finished: {ran}"
    assert ran == ["blocker", "after"]


def test_queue_bound_drops_oldest(monkeypatch) -> None:
    monkeypatch.setattr(
        ce_mod,
        "capture_anomaly",
        lambda _, **__: None,  # noqa: ARG005
    )
    ex = BoundedCallbackExecutor(workers=1, queue_size=2)
    gate = threading.Event()
    started = threading.Event()

    def blocker() -> None:
        started.set()
        gate.wait(timeout=5)

    ex.submit("blocker", blocker)  # occupies the 1 worker
    assert started.wait(timeout=2.0), "blocker never started running"
    ran: list[str] = []
    ex.submit("old", lambda: ran.append("old"))
    ex.submit("mid", lambda: ran.append("mid"))
    ex.submit("new", lambda: ran.append("new"))  # queue full -> "old" dropped
    gate.set()
    assert _wait_until(lambda: ran == ["mid", "new"]), f"unexpected ran={ran}"
    assert "old" not in ran


def test_drop_rate_escalates_anomaly(monkeypatch) -> None:
    anomalies: list[str] = []
    monkeypatch.setattr(
        ce_mod,
        "capture_anomaly",
        lambda name, **__: anomalies.append(name),  # noqa: ARG005
    )
    ex = BoundedCallbackExecutor(workers=1, queue_size=1)
    gate = threading.Event()
    started = threading.Event()

    def blocker() -> None:
        started.set()
        gate.wait(timeout=5)

    ex.submit("blocker", blocker)
    assert started.wait(timeout=2.0), "blocker never started running"
    for i in range(ce_mod._DROP_RATE_PER_MINUTE + 3):
        ex.submit(f"cb{i}", lambda: None)  # each overflow drops the prior one
    gate.set()
    assert _wait_until(lambda: "callback_executor.drop_rate_exceeded" in anomalies)
    assert anomalies.count("callback_executor.drop_rate_exceeded") == 1  # throttled


def test_callback_exception_does_not_kill_worker() -> None:
    ex = BoundedCallbackExecutor(workers=1, queue_size=8)

    def boom() -> None:
        raise RuntimeError("boom")

    ran = threading.Event()
    ex.submit("boom", boom)
    ex.submit("after", ran.set)
    assert ran.wait(timeout=2.0)  # worker survived the exception


def test_module_singleton_constants() -> None:
    assert ce_mod._WORKERS == 16
    assert ce_mod._QUEUE_SIZE == 256
    assert ce_mod._DROP_RATE_PER_MINUTE == 10
