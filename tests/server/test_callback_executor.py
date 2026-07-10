import threading
import time

from reflexio.server import callback_executor as ce_mod
from reflexio.server.callback_executor import BoundedCallbackExecutor


def _drain(executor: BoundedCallbackExecutor, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with executor._cond:
            if not executor._queue:
                return
        time.sleep(0.01)


def test_callbacks_run() -> None:
    ex = BoundedCallbackExecutor(workers=2, queue_size=8)
    ran = threading.Event()
    ex.submit("t", ran.set)
    assert ran.wait(timeout=2.0)


def test_queue_bound_drops_oldest(monkeypatch) -> None:
    monkeypatch.setattr(
        ce_mod,
        "capture_anomaly",
        lambda _, **__: None,  # noqa: ARG005
    )
    ex = BoundedCallbackExecutor(workers=1, queue_size=2)
    gate = threading.Event()
    ex.submit("blocker", lambda: gate.wait(timeout=5))  # occupies the 1 worker
    time.sleep(0.05)
    ran: list[str] = []
    ex.submit("old", lambda: ran.append("old"))
    ex.submit("mid", lambda: ran.append("mid"))
    ex.submit("new", lambda: ran.append("new"))  # queue full -> "old" dropped
    gate.set()
    _drain(ex)
    time.sleep(0.05)
    assert "old" not in ran
    assert ran == ["mid", "new"]


def test_drop_rate_escalates_anomaly(monkeypatch) -> None:
    anomalies: list[str] = []
    monkeypatch.setattr(
        ce_mod,
        "capture_anomaly",
        lambda name, **__: anomalies.append(name),  # noqa: ARG005
    )
    ex = BoundedCallbackExecutor(workers=1, queue_size=1)
    gate = threading.Event()
    ex.submit("blocker", lambda: gate.wait(timeout=5))
    time.sleep(0.05)
    for i in range(ce_mod._DROP_RATE_PER_MINUTE + 3):
        ex.submit(f"cb{i}", lambda: None)  # each overflow drops the prior one
    gate.set()
    assert "callback_executor.drop_rate_exceeded" in anomalies
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
