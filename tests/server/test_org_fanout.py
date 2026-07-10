import threading
import time

from reflexio.server.org_fanout import iterate_orgs_bounded


def test_all_orgs_processed_and_no_timeouts() -> None:
    seen: list[str] = []
    lock = threading.Lock()

    def fn(org_id: str) -> None:
        with lock:
            seen.append(org_id)

    timed_out = iterate_orgs_bounded(
        [f"org{i}" for i in range(20)], fn, max_workers=4, per_org_timeout_seconds=5.0
    )
    assert sorted(seen) == sorted(f"org{i}" for i in range(20))
    assert timed_out == []


def test_concurrency_is_bounded() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def fn(org_id: str) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    iterate_orgs_bounded(
        [str(i) for i in range(12)], fn, max_workers=3, per_org_timeout_seconds=5.0
    )
    assert peak <= 3


def test_stuck_org_times_out_and_others_complete() -> None:
    done: list[str] = []
    release = threading.Event()

    def fn(org_id: str) -> None:
        if org_id == "stuck":
            release.wait(timeout=10)  # far longer than the per-org timeout
            return
        done.append(org_id)

    timed_out = iterate_orgs_bounded(
        ["stuck", "a", "b"], fn, max_workers=3, per_org_timeout_seconds=0.2
    )
    release.set()  # unblock the straggler thread
    assert timed_out == ["stuck"]
    assert sorted(done) == ["a", "b"]


def test_serial_when_max_workers_is_one() -> None:
    order: list[str] = []
    iterate_orgs_bounded(
        ["a", "b", "c"],
        order.append,
        max_workers=1,
        per_org_timeout_seconds=1.0,
    )
    assert order == ["a", "b", "c"]  # strict submission order == serial


def test_stop_event_halts_submission() -> None:
    stop = threading.Event()
    seen: list[str] = []

    def fn(org_id: str) -> None:
        seen.append(org_id)
        stop.set()  # first org requests shutdown

    iterate_orgs_bounded(
        ["a", "b", "c"], fn, max_workers=1, per_org_timeout_seconds=1.0, stop_event=stop
    )
    assert seen == ["a"]  # in-flight finishes; no new submissions
