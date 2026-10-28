"""Real bounded workers own keyed coalescing, serialization and discard cleanup."""

import threading

from reflexio.server.callback_executor import BoundedCallbackExecutor


def test_queued_replacements_use_one_slot_and_keep_latest_callback():
    executor = BoundedCallbackExecutor(workers=1, queue_size=2)
    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(5)

    executor.submit("blocker", blocker)
    assert started.wait(2)
    try:
        for value in range(10):
            executor.submit(
                "same", lambda value=value: ran.append(value), coalesce_key="a"
            )
        executor.submit("other", lambda: ran.append("other"), coalesce_key="b")
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    assert ran == [9, "other"]


def test_active_key_serializes_successor_without_blocking_other_keys():
    executor = BoundedCallbackExecutor(workers=2, queue_size=3)
    started, release, other = (threading.Event(), threading.Event(), threading.Event())
    ran = []

    def first():
        started.set()
        assert release.wait(5)
        ran.append("first")

    executor.submit("first", first, coalesce_key="a")
    assert started.wait(2)
    try:
        executor.submit("old", lambda: ran.append("old"), coalesce_key="a")
        executor.submit("latest", lambda: ran.append("latest"), coalesce_key="a")
        executor.submit("other", other.set, coalesce_key="b")
        assert other.wait(2), "same-key successor blocked independent ready work"
        assert ran == [], "a successor overlapped its active key"
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    assert ran == ["first", "latest"]


def test_dropped_key_can_be_enqueued_again():
    executor = BoundedCallbackExecutor(workers=1, queue_size=1)
    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(5)

    executor.submit("blocker", blocker)
    assert started.wait(2)
    try:
        executor.submit("dropped", lambda: ran.append("old"), coalesce_key="a")
        executor.submit("replacement", lambda: ran.append("b"), coalesce_key="b")
        executor.submit("reenqueued", lambda: ran.append("new"), coalesce_key="a")
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    assert ran == ["new"]


def test_callback_can_submit_same_key_and_worker_survives_base_exception():
    executor = BoundedCallbackExecutor(workers=2, queue_size=2)
    ran = []

    def first():
        executor.submit("next", lambda: ran.append("next"), coalesce_key="a")
        assert ran == [], "reentrant same-key work ran before its owner returned"
        raise SystemExit("synthetic callback failure")

    executor.submit("first", first, coalesce_key="a")
    assert executor.drain(timeout_seconds=3)
    assert ran == ["next"]


def test_identical_labels_in_distinct_project_keys_do_not_coalesce():
    executor = BoundedCallbackExecutor(workers=1, queue_size=2)
    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(5)

    executor.submit("blocker", blocker)
    assert started.wait(2)
    try:
        for project in ("p1", "p2"):
            executor.submit(
                "same-label",
                lambda project=project: ran.append(project),
                coalesce_key=("optimizer", "org", project, "user_playbook", 1),
            )
    finally:
        release.set()
    assert executor.drain(timeout_seconds=3)
    assert ran == ["p1", "p2"]
