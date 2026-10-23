"""Durable window coverage and lease invariants against real storage."""

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
)

pytestmark = pytest.mark.integration


def publish(
    storage,
    count,
    *,
    request_id="r",
    user="u",
    width=10,
    stride=5,
    force=False,
    playbook=False,
    eligible=True,
    session="s",
    finished=False,
    idle_timeout=0,
    playbook_width=None,
):
    items = [
        Interaction(
            user_id=user,
            request_id=request_id,
            content=f"Message {i}",
            created_at=100 - i,
        )
        for i in range(count)
    ]
    policy = {
        "eligible": eligible,
        "window_size": width,
        "stride_size": stride,
        "extractor": {},
    }
    with storage.commit_scope():
        storage.lock_extraction_stream(user)
        storage.add_request(
            Request(
                user_id=user, request_id=request_id, session_id=session, source="test"
            )
        )
        storage.add_user_interactions_bulk(user, items, embeddings_prepared=True)
        storage.admit_extraction(
            user,
            request_id,
            [i.interaction_id for i in items],
            {
                "profile": policy,
                "playbook": {
                    **policy,
                    "eligible": playbook,
                    "window_size": playbook_width or width,
                },
                "force": force,
                "session_finished": finished,
                "idle_timeout_seconds": idle_timeout,
            },
        )
    return items


def drain(storage):
    result = []
    while claim := storage.claim_extraction("test", 300):
        user, token = claim
        window = storage.prepare_extraction(user, token)
        if window is None:
            continue
        result.append(window)
        with storage.commit_scope():
            storage.complete_extraction(window, token, {})
            storage.ack_extraction_effects(window.window_id)
    return result


@pytest.mark.parametrize(
    ("width", "stride", "count", "expected", "end"),
    [
        (10, 5, 1000, 199, 1000),
        (10, 8, 1000, 124, 994),
        (10, 10, 1000, 100, 1000),
        (10, 1, 30, 21, 30),
    ],
)
def test_large_request_visits_all_windows(storage, width, stride, count, expected, end):
    publish(storage, count, width=width, stride=stride)
    windows = drain(storage)
    assert len(windows) == expected
    assert windows[-1].end_seq == end
    assert [m["seq"] for m in windows[0].manifest] == list(range(1, width + 1))
    if len(windows) > 1:
        assert [m["seq"] for m in windows[1].manifest] == list(
            range(stride + 1, stride + width + 1)
        )
    assert storage.extraction_status("u", "r")["status"] == (
        "done" if end == count else "pending"
    )


def test_small_requests_equal_one_batch_and_tail_wakes(storage):
    for i in range(24):
        publish(storage, 1, request_id=f"r{i}")
    windows = drain(storage)
    assert [w.end_seq for w in windows] == [10, 15, 20]
    assert storage.extraction_status("u", "r19")["status"] == "done"
    assert storage.extraction_status("u", "r23")["status"] == "pending"
    publish(storage, 1, request_id="last")
    assert [w.end_seq for w in drain(storage)] == [25]


def test_sibling_cursor_survives_retries_and_other_users_claim(storage):
    publish(storage, 10, playbook=True)
    user, token = storage.claim_extraction("worker1", 300)
    assert storage.claim_extraction("worker2", 300) is None
    window = storage.prepare_extraction(user, token)
    storage.retry_extraction(window, token, "ProviderUnavailable")
    user, token = storage.claim_extraction("worker2", 300)
    sibling = storage.prepare_extraction(user, token)
    assert sibling.kind != window.kind
    storage.complete_extraction(sibling, token, {})
    storage.ack_extraction_effects(sibling.window_id)
    assert storage.extraction_status("u", "r") == {
        "status": "pending",
        "reason": "retrying",
    }
    publish(storage, 10, request_id="other", user="v")
    windows = drain(storage)
    assert [w.user_id for w in windows] == ["v"]


def test_prepared_window_carries_its_cursor_attempt_count(storage):
    publish(storage, 10)
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    assert window.attempts == 0
    for expected in (1, 2):
        storage.retry_extraction(window, token, "ProviderUnavailable")
        with storage._stream_sql() as db:
            db.query("UPDATE extraction_cursors SET retry_at=0")
        user, token = storage.claim_extraction("worker", 300)
        window = storage.prepare_extraction(user, token)
        assert window.attempts == expected
    storage.complete_extraction(window, token, {})


@pytest.mark.parametrize(
    "reason", ["input_budget_exceeded", "input_budget_configuration_error"]
)
def test_terminal_input_budget_failure_holds_coverage_and_keeps_siblings_runnable(
    storage, reason
):
    publish(storage, 10, playbook=True)
    user, token = storage.claim_extraction("worker", 300)
    failed = storage.prepare_extraction(user, token)
    storage.fail_extraction_input_budget(failed, token, reason)
    assert storage.extraction_status(user, "r") == {
        "status": "failed",
        "reason": reason,
    }
    assert len(storage.get_user_interaction(user)) == 10
    with pytest.raises(LeaseLostError):
        storage.fail_extraction_input_budget(failed, token, reason)
    user, token = storage.claim_extraction("sibling", 300)
    sibling = storage.prepare_extraction(user, token)
    assert sibling.kind != failed.kind
    storage.complete_extraction(sibling, token, {})
    storage.ack_extraction_effects(sibling.window_id)
    assert drain(storage) == []
    assert storage.extraction_status(user, "r") == {
        "status": "failed",
        "reason": reason,
    }
    publish(storage, 10, user="v", request_id="other")
    assert [window.user_id for window in drain(storage)] == ["v"]


def _backdate_backlog(storage, seconds):
    with storage._stream_sql() as db:
        db.query(
            f"UPDATE {db.table('learning_work')} SET pending_since=pending_since-? "  # noqa: S608 -- registered identifier; bound values.
            "WHERE pending_since>0",
            (seconds,),
        )


def test_completed_window_restarts_a_still_pending_backlog_age(storage):
    # A user who publishes faster than the queue drains is never empty; the
    # age must measure stalled work, not time since the queue was last empty.
    publish(storage, 30)
    _backdate_backlog(storage, 7200)
    assert storage.oldest_extraction_backlog_age() >= 7200
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    storage.complete_extraction(window, token, {})
    age = storage.oldest_extraction_backlog_age()
    assert age is not None and age < 60


def test_failing_sibling_cursor_keeps_the_backlog_aging(storage):
    publish(storage, 10, playbook=True)
    _backdate_backlog(storage, 7200)
    user, token = storage.claim_extraction("worker1", 300)
    failed = storage.prepare_extraction(user, token)
    storage.retry_extraction(failed, token, "ProviderUnavailable")
    user, token = storage.claim_extraction("worker2", 300)
    sibling = storage.prepare_extraction(user, token)
    assert sibling.kind != failed.kind
    storage.complete_extraction(sibling, token, {})
    assert storage.oldest_extraction_backlog_age() >= 7200


def _clock(monkeypatch, start):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    clock = [float(start)]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: clock[0])
    return clock


def _due_at(storage, user="u"):
    with storage._stream_sql(read_only=True) as db:
        return db.query(
            f"SELECT due_at FROM {db.table('learning_work')} WHERE user_id=?",  # noqa: S608 -- registered identifier; bound values.
            (user,),
        )[0]["due_at"]


def _fail_repeatedly(storage, clock, *, attempts, gap):
    """Fail the runnable window ``attempts`` times, ``gap`` seconds apart.

    After each failure the stream is re-inspected, which parks ``due_at`` on
    the retry backoff AHEAD of now -- the shape of a stuck item whose
    ``due_at`` keeps moving forward (prod org 133, 2026-10-22).
    """
    for _ in range(attempts):
        clock[0] += gap
        user, token = storage.claim_extraction("worker", 300)
        window = storage.prepare_extraction(user, token)
        assert window is not None
        storage.retry_extraction(window, token, "ProviderUnavailable")
        user, token = storage.claim_extraction("worker", 300)
        assert storage.prepare_extraction(user, token) is None
        assert _due_at(storage) > clock[0]


def test_idle_held_work_is_not_late_until_its_deadline(storage, monkeypatch):
    # Prod org 48, 2026-10-22: sessions below one window, held on the 6h idle
    # deadline, read as hours "old" and alarmed although nothing was late.
    clock = _clock(monkeypatch, 1000)
    publish(storage, 2, idle_timeout=21600)
    assert drain(storage) == []
    assert _due_at(storage) == 1000 + 21600
    clock[0] = 1000 + 21599
    assert storage.oldest_extraction_backlog_age() == 0
    # Due and not picked up: late from the deadline, not from admission.
    clock[0] = 1000 + 21600 + 7200
    assert storage.oldest_extraction_backlog_age() == 7200


def test_due_work_never_attempted_ages_from_admission(storage, monkeypatch):
    clock = _clock(monkeypatch, 1000)
    publish(storage, 3)
    clock[0] = 1000 + 7200
    assert storage.oldest_extraction_backlog_age() == 7200


def test_retry_pushed_due_at_never_hides_a_stuck_item(storage, monkeypatch):
    clock = _clock(monkeypatch, 1000)
    publish(storage, 10)
    _fail_repeatedly(storage, clock, attempts=11, gap=330)
    # ``now - due_at`` would read 0 here; the stuck item ages from admission.
    assert storage.oldest_extraction_backlog_age() == 11 * 330 > 3600


def test_failing_cursor_beside_an_idle_held_one_keeps_aging(storage, monkeypatch):
    # The profile window fails while the playbook cursor waits on its idle
    # deadline: a failure dominates policy, so the clock must not park.
    clock = _clock(monkeypatch, 1000)
    publish(storage, 10, playbook=True, playbook_width=20, idle_timeout=21600)
    _fail_repeatedly(storage, clock, attempts=11, gap=330)
    assert _due_at(storage) < 1000 + 21600
    assert storage.oldest_extraction_backlog_age() == 11 * 330


def test_owed_effects_keep_the_clock_although_the_rest_is_idle_held(
    storage, monkeypatch
):
    clock = _clock(monkeypatch, 1000)
    publish(storage, 12, idle_timeout=21600)
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None and window.end_seq == 10
    storage.complete_extraction(window, token, {})
    # Effects never acknowledged; seqs 11-12 wait on the idle deadline.
    user, token = storage.claim_extraction("worker", 300)
    assert storage.prepare_extraction(user, token) is None
    clock[0] = 1000 + 7200
    assert storage.oldest_extraction_backlog_age() == 7200


def test_new_input_or_finish_pulls_a_parked_clock_back(storage, monkeypatch):
    clock = _clock(monkeypatch, 1000)
    publish(storage, 2, idle_timeout=21600)
    assert drain(storage) == []
    # New input pulls the parked clock back: unserved, it ages from admission.
    clock[0] = 2000
    publish(storage, 1, request_id="r2", idle_timeout=21600)
    clock[0] = 2000 + 7200
    assert storage.oldest_extraction_backlog_age() == 7200
    assert drain(storage) == []
    assert storage.oldest_extraction_backlog_age() == 0
    # An explicit finish is runnable now as well.
    clock[0] = 20000
    assert storage.finalize_session("u", "s")
    clock[0] = 20000 + 3600
    assert storage.oldest_extraction_backlog_age() == 3600
    assert [w.end_seq for w in drain(storage)] == [3]


def test_negative_receipt_and_stale_fence(storage):
    publish(storage, 10)
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    with pytest.raises(LeaseLostError):
        storage.complete_extraction(window, "wrong", {})
    storage.save_extraction_outcome(window, token, {"skipped": True})
    assert storage.prepare_extraction(user, token).outcome == {"skipped": True}
    storage.complete_extraction(window, token, {})
    assert storage.extraction_status("u", "r")["status"] == "done"


def test_force_barrier_does_not_absorb_later_tail(storage):
    publish(storage, 3, force=True)
    publish(storage, 4, request_id="later")
    assert [w.end_seq for w in drain(storage)] == [3]
    assert storage.extraction_status("u", "r")["status"] == "done"
    assert storage.extraction_status("u", "later")["status"] == "pending"


def test_rollback_allocates_no_positions_and_legacy_is_not_tracked(storage):
    with pytest.raises(RuntimeError), storage.commit_scope():
        publish(storage, 10, request_id="rolledback")
        raise RuntimeError("rollback")
    publish(storage, 10)
    assert [w.end_seq for w in drain(storage)] == [10]
    storage.add_request(
        Request(user_id="u", request_id="legacy", session_id="s", created_at=1)
    )
    assert storage.extraction_status("u", "legacy")["status"] == "not_tracked"


def test_many_concurrent_publishes_have_one_contiguous_stream(storage):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: publish(storage, 5, request_id=f"r{n}"), range(200)))
    windows = drain(storage)
    assert len(windows) == 199
    assert [m["seq"] for m in windows[-1].manifest] == list(range(991, 1001))
    assert len({w.window_id for w in windows}) == 199
    assert all(
        storage.extraction_status("u", f"r{n}")["status"] == "done" for n in range(200)
    )


def test_hot_user_cannot_starve_waiting_user(storage):
    publish(storage, 100, request_id="hot")
    first = storage.claim_extraction("worker", 300)
    publish(storage, 10, user="v", request_id="waiting")
    window = storage.prepare_extraction(*first)
    storage.complete_extraction(window, first[1], {})
    assert storage.claim_extraction("worker", 300)[0] == "v"


def test_manual_lease_shares_user_exclusion_without_advancing_cursor(storage):
    publish(storage, 10)
    manual = storage.claim_user_extraction("u", "manual", 300)
    assert manual is not None
    assert storage.claim_extraction("automatic", 300) is None
    storage.release_user_extraction("u", "wrong")
    assert storage.claim_extraction("automatic", 300) is None
    storage.release_user_extraction("u", manual)
    assert storage.extraction_status("u", "r")["status"] == "pending"
    assert [w.end_seq for w in drain(storage)] == [10]


def test_missing_cursor_is_never_reported_done(storage):
    publish(storage, 10)
    with storage._stream_sql() as db:
        db.query("DELETE FROM extraction_cursors WHERE kind='profile'")
    assert storage.extraction_status("u", "r") == {
        "status": "pending",
        "reason": "cursor_unavailable",
    }


def test_retention_preserves_backlog_and_overlap_but_explicit_delete_wins(storage):
    items = publish(storage, 25)
    assert storage.delete_oldest_interactions(25) == 0
    assert [w.end_seq for w in drain(storage)] == [10, 15, 20, 25]
    storage.delete_oldest_interactions(25)
    assert {i.interaction_id for i in storage.get_user_interaction("u")} == {
        i.interaction_id for i in items[15:]
    }
    publish(storage, 5, request_id="next")
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    storage.save_extraction_outcome(window, token, {"secret": "cached compute"})
    storage.delete_all_interactions_for_user("u")
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
    )

    with pytest.raises(WindowInputsDeletedError):
        storage.validate_extraction_inputs(window)
    assert storage.prepare_extraction(user, token).outcome is None
    storage.invalidate_extraction(window, token)
    assert drain(storage) == []


def test_counts_only_include_windows_that_cover_new_request_inputs(storage):
    publish(storage, 10, request_id="first")
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    storage.complete_extraction(window, token, {"billing": {"ids": ["a"]}})
    storage.ack_extraction_effects(window.window_id)
    publish(storage, 5, request_id="second")
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    storage.complete_extraction(window, token, {"billing": {"ids": ["b", "c"]}})
    storage.ack_extraction_effects(window.window_id)
    assert storage.extraction_counts("u", "first")["profile"] == 1
    assert storage.extraction_counts("u", "second")["profile"] == 2


def test_partial_force_and_policy_change_keep_new_input_boundary(storage):
    publish(storage, 10, width=10, stride=5)
    assert [w.end_seq for w in drain(storage)] == [10]
    publish(storage, 1, request_id="newpolicy", width=6, stride=2)
    publish(storage, 1, request_id="laterpolicy", width=20, stride=10)
    windows = drain(storage)
    assert [w.end_seq for w in windows] == [12]
    assert [m["seq"] for m in windows[0].manifest] == list(range(7, 13))
    assert windows[0].policy["window_size"] == 6


def test_known_agent_output_reused_without_another_model_call(storage, monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock

    from pydantic import BaseModel

    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
        encode_committed_output,
    )
    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    class Output(BaseModel):
        text: str

    run = AgentRunRecord(
        id="window:stable",
        binding=AgentBinding(
            org_id=storage.org_id,
            extractor_kind="profile",
            user_id="u",
            request_id="r",
            agent_version=None,
            source=None,
        ),
        status=AgentRunStatus.AGENT_COMPLETED,
        generation_request_snapshot={},
        committed_output=encode_committed_output(Output(text="known"), None),
    )
    storage.create_agent_run(run)
    agent = ResumableExtractionAgent(client=Mock(), storage=storage)
    monkeypatch.setattr(
        agent, "_run", lambda **_: pytest.fail("must reuse known output")
    )
    recovered = agent.start(
        run=replace(run, status=AgentRunStatus.RUNNING, committed_output=None),
        messages=[],
        output_schema=Output,
    )
    assert recovered.output == Output(text="known")
    assert recovered.run_id == run.id


def test_erased_force_boundary_cannot_block_following_input(storage):
    items = publish(storage, 3, force=True)
    with storage._stream_sql() as db:
        db.query(
            "DELETE FROM interactions WHERE interaction_id=?",
            (items[-1].interaction_id,),
        )
    assert [w.end_seq for w in drain(storage)] == [2]
    assert storage.extraction_status("u", "r")["status"] == "done"
    publish(storage, 10, request_id="next")
    windows = drain(storage)
    assert windows and windows[0].end_seq > 3
    assert not windows[0].force


def test_erased_request_is_satisfied_without_advancing_other_input(storage):
    publish(storage, 10)
    storage.delete_all_interactions_for_user("u")
    assert storage.extraction_status("u", "r") == {
        "status": "done",
        "reason": "inputs_erased",
    }
    assert drain(storage) == []


def test_receipts_do_not_attribute_disabled_or_erased_gaps(storage):
    publish(storage, 5, request_id="disabled", eligible=False)
    erased = publish(storage, 5, request_id="erased")
    with storage._stream_sql() as db:
        for item in erased:
            db.query(
                "DELETE FROM interactions WHERE interaction_id=?",
                (item.interaction_id,),
            )
    publish(storage, 10, request_id="included")
    user, token = storage.claim_extraction("worker", 300)
    window = storage.prepare_extraction(user, token)
    assert [m["seq"] for m in window.manifest] == list(range(11, 21))
    storage.complete_extraction(window, token, {"billing": {"ids": ["created"]}})
    storage.ack_extraction_effects(window.window_id)
    assert storage.extraction_counts("u", "disabled")["profile"] == 0
    assert storage.extraction_counts("u", "erased")["profile"] == 0
    assert storage.extraction_counts("u", "included")["profile"] == 1


@pytest.mark.parametrize(
    "reason",
    [
        "before_cutover",
        "not_applicable",
        "inputs_erased",
        "cursor_unavailable",
        "waiting_for_window",
        "queued",
        "extracting",
        "retrying",
        "covered",
    ],
)
def test_combined_report_preserves_status_and_shares_admission(
    storage, monkeypatch, reason
):
    from contextlib import contextmanager

    if reason != "before_cutover":
        publish(
            storage,
            0
            if reason == "not_applicable"
            else 1
            if reason == "waiting_for_window"
            else 10,
        )
    if reason == "inputs_erased":
        with storage._stream_sql() as db:
            db.query("DELETE FROM interactions WHERE user_id=?", ("u",))
    if reason == "cursor_unavailable":
        with storage._stream_sql() as db:
            db.query("DELETE FROM extraction_cursors WHERE user_id=?", ("u",))
    if reason in {"extracting", "retrying", "covered"}:
        user, token = storage.claim_extraction("worker", 300)
        window = storage.prepare_extraction(user, token)
        if reason == "retrying":
            storage.retry_extraction(window, token, "test retry")
        elif reason == "covered":
            storage.complete_extraction(
                window, token, {"billing": {"ids": ["profile"]}}
            )
    expected = (
        storage.extraction_status("u", "r"),
        storage.extraction_counts("u", "r"),
    )
    assert expected[0]["reason"] == reason
    original = storage._stream_sql
    queries, scopes, clocks = [], [], []

    @contextmanager
    def observed(**kwargs):
        scopes.append(kwargs)
        with original(**kwargs) as db:
            query, now = db.query, db.now

            def read(sql, params=()):
                queries.append(sql)
                return query(sql, params)

            def clock():
                clocks.append(1)
                return now()

            db.query, db.now = read, clock
            yield db

    monkeypatch.setattr(storage, "_stream_sql", observed)
    assert storage.extraction_report("u", "r") == expected
    assert len(scopes) == 1
    assert sum(q.startswith("SELECT learning_admission FROM") for q in queries) == 1
    needs_clock = reason in {"waiting_for_window", "queued", "extracting", "retrying"}
    assert len(clocks) == int(needs_clock)
    assert sum(q.startswith("SELECT lease_until FROM") for q in queries) == int(
        needs_clock and reason != "retrying"
    )


def test_combined_report_counts_more_than_one_receipt_page(storage):
    publish(storage, 201, width=1, stride=1)
    for _ in range(201):
        user, token = storage.claim_extraction("worker", 300)
        window = storage.prepare_extraction(user, token)
        storage.complete_extraction(
            window, token, {"billing": {"ids": [window.window_id]}}
        )
        storage.ack_extraction_effects(window.window_id)
    status, counts = storage.extraction_report("u", "r")
    assert status == {"status": "done", "reason": "covered"}
    assert counts == {"profile": 201, "playbook": 0}


@pytest.mark.parametrize("playbook", [False, True])
def test_judged_finish_survives_late_full_window_and_is_idempotent(storage, playbook):
    publish(storage, 2, width=5, stride=3, playbook=playbook)
    assert drain(storage) == []
    assert storage.finalize_session("u", "s")
    assert storage.finalize_session("u", "s")
    publish(storage, 3, request_id="late", width=5, stride=3, playbook=playbook)
    windows = drain(storage)
    assert [w.end_seq for w in windows if w.kind == "profile"] == [2, 5]
    if playbook:
        assert [w.end_seq for w in windows if w.kind == "playbook"] == [2, 5]
    assert not any(w.force for w in windows)
    assert storage.finalize_session("u", "s")
    assert drain(storage) == []


def test_ineligible_completion_marker_cannot_stall_later_work(storage):
    publish(storage, 2, width=5, stride=3)
    publish(storage, 1, request_id="excluded", eligible=False, finished=True)
    assert [w.end_seq for w in drain(storage)] == [2]
    publish(storage, 1, request_id="late", width=5, stride=3, finished=True)
    assert [w.end_seq for w in drain(storage)] == [4]


def test_idle_is_session_scoped_and_deadline_survives_reopen(storage, monkeypatch):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    now = 1000.0
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: now)
    publish(storage, 2, width=10, stride=5, session="A", idle_timeout=21600)
    assert drain(storage) == []
    with storage._stream_sql(read_only=True) as db:
        assert db.query("SELECT due_at FROM learning_work")[0]["due_at"] == 22600
    now = 22599
    publish(storage, 1, request_id="B", session="B", idle_timeout=21600)
    assert drain(storage) == []
    now = 22600
    windows = drain(storage)
    assert [w.end_seq for w in windows] == [2]
    assert not windows[0].force
    assert storage.extraction_status("u", "B")["status"] == "pending"


@pytest.mark.parametrize("legacy", [False, True])
def test_disabled_and_legacy_idle_require_explicit_finish(storage, monkeypatch, legacy):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    now = 1.0
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: now)
    publish(storage, 2, idle_timeout=21600 if legacy else 0)
    if legacy:
        # Upgrade fixture: old rows predate the server admission clock.
        with storage._stream_sql() as db:
            db.query(
                "UPDATE requests SET learning_admission=json_remove(learning_admission, '$.admitted_at') WHERE request_id=?",
                ("r",),
            )
    now = 1000000.0
    assert drain(storage) == []
    assert not storage.finalize_session("other", "s")
    assert not storage.finalize_session("u", "unknown")
    assert storage.finalize_session("u", "s")
    assert [w.end_seq for w in drain(storage)] == [2]


def test_session_activity_and_marker_queries_use_indexes(storage):
    with storage._stream_sql(read_only=True) as db:
        activity = db.query(
            "EXPLAIN QUERY PLAN SELECT learning_admission FROM requests r WHERE user_id=? AND session_id=? AND json_extract(r.learning_admission, '$.max_seq') IS NOT NULL ORDER BY json_extract(r.learning_admission, '$.max_seq') DESC LIMIT 1",
            ("u", "s"),
        )
        marker = db.query(
            "EXPLAIN QUERY PLAN SELECT json_extract(r.learning_admission, '$.max_seq') FROM requests r WHERE user_id=? AND json_extract(r.learning_admission, '$.session_finished')=1 AND json_extract(r.learning_admission, '$.max_seq')>? ORDER BY json_extract(r.learning_admission, '$.max_seq') LIMIT 1",
            ("u", 0),
        )
    assert "extraction_session_activity" in str(activity)
    assert "extraction_completion_boundary" in str(marker)


def test_completion_marker_and_wakeup_rollback_together(storage):
    publish(storage, 2)
    assert drain(storage) == []
    with pytest.raises(RuntimeError, match="crash"), storage.commit_scope():
        assert storage.finalize_session("u", "s")
        raise RuntimeError("crash before finalize commit")
    assert drain(storage) == []
    assert storage.finalize_session("u", "s")
    assert [w.end_seq for w in drain(storage)] == [2]


def test_erased_completion_prefix_does_not_block_later_inputs(storage):
    publish(storage, 2, finished=True)
    with storage._stream_sql() as db:
        db.query("DELETE FROM interactions WHERE request_id=?", ("r",))
    publish(storage, 1, request_id="late-after-erasure", finished=True)
    windows = drain(storage)
    assert [w.end_seq for w in windows] == [3]
    assert [m["seq"] for m in windows[0].manifest] == [3]
    assert not windows[0].force
