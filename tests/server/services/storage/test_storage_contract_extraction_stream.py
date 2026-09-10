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
            Request(user_id=user, request_id=request_id, session_id="s", source="test")
        )
        storage.add_user_interactions_bulk(user, items, embeddings_prepared=True)
        storage.admit_extraction(
            user,
            request_id,
            [i.interaction_id for i in items],
            {
                "profile": policy,
                "playbook": {**policy, "eligible": playbook},
                "force": force,
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
