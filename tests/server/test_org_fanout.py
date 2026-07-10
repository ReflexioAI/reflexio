import logging
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
    # Lower bound too: with 12 orgs and max_workers=3, at least 2 must have
    # been active concurrently at some point — a serial regression (workers
    # never overlapping) must fail this test.
    assert peak >= 2


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


def test_stop_event_halts_submission_with_pool() -> None:
    """The pooled path (max_workers > 1) has its own submission loop with its
    own stop_event check — a separate code path from the serial loop that
    test_stop_event_halts_submission exercises (max_workers <= 1 skips the
    pool entirely). Uses a generator that sets ``stop`` deterministically on
    the submitting thread (between yielding "b" and "c") so the assertion
    does not depend on background-thread scheduling timing.
    """
    stop = threading.Event()
    submitted: list[str] = []
    lock = threading.Lock()

    def fn(org_id: str) -> None:
        with lock:
            submitted.append(org_id)

    def org_ids():
        yield "a"
        yield "b"
        stop.set()  # deterministic: runs on the submitting thread before "c"
        yield "c"
        yield "d"

    iterate_orgs_bounded(
        org_ids(), fn, max_workers=2, per_org_timeout_seconds=1.0, stop_event=stop
    )
    assert sorted(submitted) == ["a", "b"]  # c, d never submitted once stop fired


def test_starved_org_not_falsely_reported_as_timed_out() -> None:
    """Regression for the queue-wait-counted-as-run-time bug (F001): with
    max_workers < len(org_ids), an org that is merely queued (never started)
    when its own wait window elapses must not be reported as timed out. Once
    a worker frees up, the queued org still gets to run and finish normally.
    """
    budget = 0.2
    stuck_duration = 0.5  # sits inside "fast"'s [2*budget, 3*budget) wait window
    done: list[str] = []
    lock = threading.Lock()

    def fn(org_id: str) -> None:
        if org_id in ("stuck1", "stuck2"):
            time.sleep(stuck_duration)
            return
        with lock:
            done.append(org_id)

    timed_out = iterate_orgs_bounded(
        ["stuck1", "stuck2", "fast"],
        fn,
        max_workers=2,
        per_org_timeout_seconds=budget,
    )
    assert timed_out == ["stuck1", "stuck2"]
    assert "fast" not in timed_out
    assert done == ["fast"]  # "fast" got a freed worker and completed within the call


def test_starved_org_when_never_scheduled(caplog) -> None:
    """Starved classification (F001): when both workers stay occupied for the
    entire call, the queued org never starts and must be logged/classified
    as starved — excluded from timed_out, not merely "timed out but silent".
    """
    budget = 0.1
    release = threading.Event()
    done: list[str] = []
    lock = threading.Lock()

    def fn(org_id: str) -> None:
        if org_id in ("stuck1", "stuck2"):
            release.wait(timeout=10)  # far longer than the per-org timeout
            return
        with lock:
            done.append(org_id)

    with caplog.at_level(logging.WARNING, logger="reflexio.server.org_fanout"):
        timed_out = iterate_orgs_bounded(
            ["stuck1", "stuck2", "fast"],
            fn,
            max_workers=2,
            per_org_timeout_seconds=budget,
        )
    release.set()  # unblock the straggler threads

    assert timed_out == ["stuck1", "stuck2"]
    assert "fast" not in timed_out
    assert done == []  # "fast" never got a worker during the call — truly starved
    assert any(
        "org_sweep_starved" in record.message and "fast" in record.message
        for record in caplog.records
    ), f"Expected an org_sweep_starved warning for 'fast'; got {caplog.records}"
