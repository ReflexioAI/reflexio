"""Resume metadata inspection is bounded, owner-aware and never a logical claim."""

from datetime import UTC, datetime, timedelta

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.server.services.storage.storage_base._user_work_position import (
    ResumeCursor,
)
from tests.server.services.extraction import test_resume_worker as resume

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path):
    storage = SQLiteStorage(org_id="org_1", db_path=str(tmp_path / "runs.db"))
    try:
        yield storage
    finally:
        storage.conn.close()


def _run(
    store,
    identity,
    *,
    user: str | None = "one",
    kind="profile",
    status=AgentRunStatus.FINALIZATION_FAILED,
    **kwargs,
):
    return store.create_agent_run(
        AgentRunRecord(
            id=identity,
            binding=AgentBinding(
                org_id=store.org_id,
                extractor_kind=kind,
                user_id=user,
                request_id="request",
                agent_version=None,
                source=None,
                source_interaction_ids=[],
            ),
            status=status,
            generation_request_snapshot={},
            committed_output={"profiles": []},
            **kwargs,
        )
    )


def test_fixed_ceiling_and_rejected_heads_advance_without_run_mutation(store):
    before = [_run(store, key) for key in ("a", "b", "c")]
    cursor, first = store._resume_candidate_page("finalization")
    assert first is not None and first == {
        "id": "a",
        "org_id": store.org_id,
        "project_id": "",
        "extractor_kind": "profile",
        "user_id": "one",
    }
    assert cursor == ResumeCursor("a", "c")
    _run(store, "z")
    seen = [first["id"]]
    for _ in range(2):
        cursor, item = store._resume_candidate_page("finalization", cursor=cursor)
        assert item is not None
        seen.append(item["id"])
    assert seen == ["a", "b", "c"]
    cursor, item = store._resume_candidate_page("finalization", cursor=cursor)
    assert cursor == ResumeCursor() and item is None
    assert [store.get_agent_run(run.id) for run in before] == before
    cursor, item = store._resume_candidate_page("finalization", cursor=cursor)
    assert item is not None and cursor.through_id == "z"


def test_owner_local_inspection_cannot_hide_a_healthy_parent_or_claim_other_user(store):
    run = _run(store, "old", claimed_at=datetime(2000, 1, 1, tzinfo=UTC))
    _run(store, "foreign", user="two")
    token = store.claim_user_extraction("one", "live-owner", 300)
    assert token is not None
    with store.commit_scope():
        store.conn.execute(
            "UPDATE _agent_runs SET user_work_token=? WHERE id=?", (token, run.id)
        )
    assert (
        store._resume_candidate_page("finalization", user_id="one", token=token)[1][
            "id"
        ]
        == "old"
    )
    assert store._resume_candidate_page("finalization")[1]["id"] == "foreign"
    other = store.claim_user_extraction("two", "other", 300)
    assert other is not None
    assert (
        store._resume_candidate_page("finalization", user_id="two", token=other)[1][
            "id"
        ]
        == "foreign"
    )
    assert store.get_agent_run(run.id) == run
    store.release_user_extraction("one", token)
    _, result = store._resume_candidate_page(
        "finalization", cursor=ResumeCursor("foreign", "old")
    )
    assert result is not None and result["id"] == "old"


def test_nullable_old_parent_survives_replacement_but_still_is_only_metadata(store):
    run = _run(store, "legacy", user=None, kind="playbook")
    old = store.claim_user_extraction("one", "old", 300)
    assert old is not None
    with store.commit_scope():
        store.conn.execute(
            "UPDATE _agent_runs SET user_work_token=? WHERE id=?", (old, run.id)
        )
    store.release_user_extraction("one", old)
    current = store.claim_user_extraction("one", "new", 300)
    assert current is not None and current != old
    _, result = store._resume_candidate_page(
        "finalization", user_id="one", token=current
    )
    assert result is not None and result["id"] == "legacy" and result["user_id"] is None
    # Descriptor selection does not manufacture missing source proof.
    with store.commit_scope(), store._stream_sql() as db:
        assert (
            store._adopt_owned_user_work_run(
                db, expected=run, user_id="one", token=current, worker_id="test"
            )
            is None
        )
    assert store.get_agent_run(run.id) == run


def test_business_backoff_overflow_and_unfinished_window_remain_ineligible(store):
    _run(store, "future", next_resume_at=datetime.now(UTC) + timedelta(hours=1))
    old = _run(store, "overflow")
    with store.commit_scope():
        store.conn.execute(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            ('{"overflow":null}', old.id),
        )
    _run(store, "window:unfinished")
    assert store._resume_candidate_page("finalization") == (ResumeCursor(), None)
    assert store._resume_candidate_page("continuation") == (ResumeCursor(), None)


def test_continuation_uses_actual_resolved_dependencies_and_preserves_class_position(
    store,
):
    resume._seed_interactions(store)
    resume._seed_ready_run(store)
    original = store.get_agent_run("run_1")
    assert original is not None
    cursor, item = store._resume_candidate_page("continuation")
    assert item is not None and item["id"] == "run_1"
    assert store._resume_candidate_page("finalization") == (ResumeCursor(), None)
    assert store.get_agent_run("run_1") == original
    with store.commit_scope():
        store.conn.execute(
            "UPDATE _pending_tool_calls SET result=? WHERE id='ptc_1'",
            ('{"not_applicable":true}',),
        )
    assert store._resume_candidate_page("continuation", cursor=cursor) == (
        ResumeCursor(),
        None,
    )
    assert store._resume_candidate_page("continuation") == (ResumeCursor(), None)


def test_inspection_refuses_caller_transactions_and_propagates_unknown_reads(
    store, monkeypatch
):
    _run(store, "one")
    with store.commit_scope(), pytest.raises(ValueError, match="independent writer"):
        store._resume_candidate_page("finalization")
    store.conn.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="independent writer"):
            store._resume_candidate_page("finalization")
    finally:
        store.conn.rollback()
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    def unavailable(*_a, **_kw):
        raise RuntimeError("synthetic unknown writer")

    monkeypatch.setattr(SQLiteStreamSQL, "query", unavailable)
    with pytest.raises(RuntimeError, match="unknown writer"):
        store._resume_candidate_page("finalization", cursor=ResumeCursor("a", "z"))


@pytest.mark.parametrize(
    "kwargs",
    [{"user_id": "one"}, {"token": "token"}, {"user_id": "", "token": "token"}],
)
def test_partial_owner_identity_is_not_general_fleet_inspection(store, kwargs):
    with pytest.raises(ValueError, match="user and token"):
        store._resume_candidate_page("finalization", **kwargs)


@pytest.mark.parametrize("work_class", ["finalization", "continuation"])
@pytest.mark.parametrize(
    "window_org,eligible", [(None, True), ("matching", True), ("foreign", False)]
)
def test_completed_historical_window_inspection_preserves_unknown_org_without_guessing(
    store, work_class, window_org, eligible
):
    storage = store
    window_org = storage.org_id if window_org == "matching" else window_org
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(
            "INSERT INTO extraction_windows (window_id,user_id,kind,predecessor,end_seq,manifest,policy,force,skip_aggregation,project_id,org_id,completed,invalidated) VALUES (?,'one','profile',0,0,?, ?,0,0,?, ?,1,0)",
            ("historical", "[]", "{}", "", window_org),
        )
    status = (
        AgentRunStatus.FINALIZATION_FAILED
        if work_class == "finalization"
        else AgentRunStatus.RESUME_READY
    )
    run = _run(storage, "window:historical", status=status)
    if work_class == "continuation":
        scope = human_feedback_scope(storage.org_id)
        now = datetime.now(UTC)
        storage.create_pending_tool_call(
            PendingToolCallRecord(
                id="window-question",
                org_id=storage.org_id,
                user_id="one",
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key="window-question",
                status=PendingToolCallStatus.RESOLVED,
                question_text="Synthetic window question",
                result={"answer": "Synthetic answer"},
                resolved_at=now,
                expires_at=now + timedelta(hours=1),
                cache_until=now + timedelta(minutes=5),
            )
        )
        storage.attach_run_tool_dependency(
            RunToolDependencyRecord(
                run_id=run.id, pending_tool_call_id="window-question", resolved_at=now
            )
        )
    _, candidate = storage._resume_candidate_page(work_class)
    assert (candidate is not None) is eligible
    assert storage.get_agent_run(run.id) == run
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            "SELECT org_id FROM extraction_windows WHERE window_id=?",
            ("historical",),
        )
    assert rows == [{"org_id": window_org}], (
        "Inspection must never manufacture historical window scope"
    )
