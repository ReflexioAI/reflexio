"""Real user leases fence atomic class progress, adoption and rollback."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.resume_selection import ResumeSelection
from reflexio.server.services.durable_learning.user_lease import owned_user_work_context
from reflexio.server.services.storage.error import ExecutionInputBusyError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    LeaseLostError,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.server.services.storage.storage_base._user_work_position import (
    ResumeCursor,
    UserWorkPosition,
)
from tests.server.services.storage import test_resume_candidate_selection as candidates

store = candidates.store
pytestmark = pytest.mark.integration


def _position(store, user="one"):
    raw = store.conn.execute(
        "SELECT position_json FROM learning_work WHERE org_id=? AND user_id=?",
        (store.org_id, user),
    ).fetchone()
    return UserWorkPosition.from_json(raw[0])


@contextmanager
def _owned(store, user="one"):
    token = store.claim_user_extraction(user, "selection-test", 300)
    assert token is not None
    try:
        with owned_user_work_context(store, user, token, 300) as owner:
            yield token, owner
    finally:
        store.release_user_extraction(user, token)


def _context(store):
    config = Config(
        storage_config=StorageConfigSQLite(db_path=store.db_path),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
    )
    context = RequestContext.__new__(RequestContext)
    context.org_id = store.org_id
    context.storage = store
    context.configurator = MagicMock()
    context.configurator.get_config.return_value = config
    return context


def test_selection_progress_and_logical_adoption_survive_new_owner(store):
    run = candidates._run(store, "selected")
    with _owned(store) as (token, _):
        old = store._read_owned_work_position("one", token)
        cursor, descriptor = store._resume_candidate_page(
            "finalization", user_id="one", token=token
        )
        assert descriptor is not None and descriptor["id"] == run.id
        selected, adopted = store._commit_owned_work_selection(
            user_id="one",
            token=token,
            expected_position=old,
            work_class="finalization",
            cursor=cursor,
            expected_run=run,
            worker_id="test",
        )
        assert (
            selected
            and adopted is not None
            and adopted.status == AgentRunStatus.FINALIZING
        )
        assert adopted.committed_output == run.committed_output
        assert (
            adopted.resume_attempts == run.resume_attempts
            and adopted.finalization_attempts == run.finalization_attempts
        )
        assert token not in adopted.claimed_by
        assert _position(store).next_work_class == "continuation"
    with _owned(store) as (new_token, _):
        assert new_token != token
        assert store._read_owned_work_position("one", new_token).finalization == cursor
        assert (
            store._read_owned_work_position("one", new_token).next_work_class
            == "continuation"
        )


def test_stale_run_and_stale_position_cannot_acquire_execution(store):
    run = candidates._run(store, "current")
    assert run.updated_at is not None
    with _owned(store) as (token, _):
        old = store._read_owned_work_position("one", token)
        cursor, _ = store._resume_candidate_page(
            "finalization", user_id="one", token=token
        )
        stale = replace(run, updated_at=run.updated_at - timedelta(seconds=1))
        selected, adopted = store._commit_owned_work_selection(
            user_id="one",
            token=token,
            expected_position=old,
            work_class="finalization",
            cursor=cursor,
            expected_run=stale,
            worker_id="test",
        )
        assert selected and adopted is None and store.get_agent_run(run.id) == run
        assert _position(store).next_work_class == "continuation"
        assert store._commit_owned_work_selection(
            user_id="one",
            token=token,
            expected_position=old,
            work_class="finalization",
            cursor=cursor,
            expected_run=run,
            worker_id="test",
        ) == (False, None)
        assert store.get_agent_run(run.id) == run


def test_failure_after_adoption_rolls_back_both_progress_and_run(store, monkeypatch):
    run = candidates._run(store, "rollback")
    adopt = store._adopt_owned_user_work_run

    def fail(*args, **kwargs):
        assert adopt(*args, **kwargs) is not None
        raise RuntimeError("synthetic error after adoption")

    with _owned(store) as (token, _):
        old = store._read_owned_work_position("one", token)
        cursor, _ = store._resume_candidate_page(
            "finalization", user_id="one", token=token
        )
        monkeypatch.setattr(store, "_adopt_owned_user_work_run", fail)
        with pytest.raises(RuntimeError, match="after adoption"):
            store._commit_owned_work_selection(
                user_id="one",
                token=token,
                expected_position=old,
                work_class="finalization",
                cursor=cursor,
                expected_run=run,
                worker_id="test",
            )
        assert _position(store) == old
        assert store.get_agent_run(run.id) == run
        store.fence_user_extraction("one", token)


def test_cancelled_handle_rolls_back_even_while_database_lease_is_live(
    store, monkeypatch
):
    run = candidates._run(store, "cancelled")
    get = store.get_agent_run
    with _owned(store) as (token, owner):
        old = store._read_owned_work_position("one", token)
        cursor, _ = store._resume_candidate_page(
            "finalization", user_id="one", token=token
        )

        def cancel(identity):
            current = get(identity)
            owner.lost.set()
            return current

        with monkeypatch.context() as cancel_after_claim:
            cancel_after_claim.setattr(store, "get_agent_run", cancel)
            with pytest.raises(LeaseLostError):
                store._commit_owned_work_selection(
                    user_id="one",
                    token=token,
                    expected_position=old,
                    work_class="finalization",
                    cursor=cursor,
                    expected_run=run,
                    worker_id="test",
                )
        assert _position(store) == old and store.get_agent_run(run.id) == run
        store.fence_user_extraction("one", token)


def test_busy_adoption_rolls_back_then_consumes_only_private_progress(
    store, monkeypatch
):
    run = candidates._run(store, "busy")
    adopt = store._adopt_owned_user_work_run

    def busy(*args, **kwargs):
        assert adopt(*args, **kwargs) is not None
        raise ExecutionInputBusyError("synthetic busy input after tentative adoption")

    monkeypatch.setattr(store, "_adopt_owned_user_work_run", busy)
    with _owned(store) as (token, _):
        kind, selected = ResumeSelection().select_owned(
            _context(store), user_id="one", token=token, worker_id="test"
        )
        assert kind == "finalization" and selected is None
        assert _position(store).next_work_class == "continuation"
        assert _position(store).finalization == ResumeCursor("busy", "busy")
        assert store.get_agent_run(run.id) == run
        store.fence_user_extraction("one", token)


def test_learning_rotation_commits_before_idle_preparation_releases_owner(store):
    with _owned(store) as (token, _):
        prior = UserWorkPosition(last_work_class="continuation")
        with store.commit_scope(), store._stream_sql() as db:
            store._store_owned_user_work_position(db, "one", token, prior)
        kind, adopted = ResumeSelection().select_owned(
            _context(store), user_id="one", token=token, worker_id="test"
        )
        assert kind == "learning" and adopted is None
        assert store.prepare_extraction("one", token) is None
        assert _position(store).last_work_class == "learning"
        with pytest.raises(LeaseLostError):
            store._commit_owned_work_selection(
                user_id="one",
                token=token,
                expected_position=_position(store),
                work_class="finalization",
                cursor=ResumeCursor(),
                worker_id="late",
            )
    with _owned(store) as (new_token, _):
        assert (
            store._read_owned_work_position("one", new_token).next_work_class
            == "finalization"
        )


def test_per_user_class_cursors_survive_new_process_local_selectors_and_poisoned_head(
    store,
):
    invalid = candidates._run(store, "a_invalid_legacy", user=None, kind="playbook")
    for user in ("one", "two"):
        for identity in (f"b_{user}", f"c_{user}"):
            candidates._run(store, identity, user=user)
        run = candidates._run(
            store, f"continuation_{user}", user=user, status=AgentRunStatus.RESUME_READY
        )
        now = datetime.now(UTC)
        scope = human_feedback_scope(store.org_id)
        question = f"question_{user}"
        store.create_pending_tool_call(
            PendingToolCallRecord(
                id=question,
                org_id=store.org_id,
                user_id=user,
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key=question,
                status=PendingToolCallStatus.RESOLVED,
                question_text="Synthetic fairness answer",
                result={"answer": "Synthetic"},
                resolved_at=now,
                expires_at=now + timedelta(hours=1),
                cache_until=now + timedelta(minutes=5),
            )
        )
        store.attach_run_tool_dependency(
            RunToolDependencyRecord(
                run_id=run.id, pending_tool_call_id=question, resolved_at=now
            )
        )
    finalization = {"one": [], "two": []}
    continuation = {"one": [], "two": []}
    # Three fixed finalization descriptors per user, including the invalid
    # shared head, must be inspected within at most 3N=9 completed owner turns.
    # Recreate both storage client and selector every time: no local user map
    # can explain progress, and the other user's cursor cannot reset this one.
    for _ in range(9):
        for user in ("one", "two"):
            peer = SQLiteStorage(org_id=store.org_id, db_path=store.db_path)
            try:
                with _owned(peer, user) as (token, _):
                    kind, run = ResumeSelection().select_owned(
                        _context(peer),
                        user_id=user,
                        token=token,
                        worker_id="new-instance",
                    )
                    if kind == "finalization":
                        finalization[user].append(
                            _position(peer, user).finalization.after_id
                        )
                    elif kind == "continuation" and run is not None:
                        continuation[user].append(run.id)
                    elif kind == "learning":
                        assert peer.prepare_extraction(user, token) is None
            finally:
                peer.conn.close()
    for user in ("one", "two"):
        assert finalization[user] == ["a_invalid_legacy", f"b_{user}", f"c_{user}"]
        assert set(continuation[user]) == {f"continuation_{user}"}
    assert store.get_agent_run(invalid.id) == invalid


def test_wrong_factory_org_is_refused_before_claim_or_source_inspection(
    store, monkeypatch
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    def forbidden(*_a, **_kw):
        raise AssertionError("Wrong-org context cannot inspect or claim data")

    monkeypatch.setattr(store, "claim_extraction", forbidden)
    monkeypatch.setattr(store, "_durable_work_route", forbidden)
    worker = DurableLearningWorker(lambda _org: _context(store))
    assert worker.drain_org("unrelated-org", batch_size=1, lease_seconds=300) == 0
    assert store.conn.execute("SELECT COUNT(*) FROM learning_work").fetchone()[0] == 0


def test_unknown_descriptor_read_advances_without_claim_or_customer_text(
    store, monkeypatch, caplog
):
    run = candidates._run(store, "unreadable")

    def unavailable(_identity):
        raise ValueError("SENTINEL_PRIVATE_OUTPUT_AND_CREDENTIALS")

    with _owned(store) as (token, _):
        with monkeypatch.context() as failure:
            failure.setattr(store, "get_agent_run", unavailable)
            kind, adopted = ResumeSelection().select_owned(
                _context(store), user_id="one", token=token, worker_id="test"
            )
        assert kind == "finalization" and adopted is None
        assert _position(store).next_work_class == "continuation"
        assert _position(store).finalization.after_id == run.id
        assert store.get_agent_run(run.id) == run
    assert "owned_resume_selection_failed" in caplog.text
    assert "error=ValueError" in caplog.text
    assert "SENTINEL_PRIVATE_OUTPUT_AND_CREDENTIALS" not in caplog.text
    failures = [
        record
        for record in caplog.records
        if "owned_resume_selection_failed" in record.getMessage()
    ]
    assert len(failures) == 1 and failures[0].exc_info is None


def test_lost_adoption_commit_reply_does_not_rewind_progress_or_execute(
    store, monkeypatch
):
    run = candidates._run(store, "committed-unacknowledged")
    commit = store._commit_owned_work_selection
    replies = []

    def lost_reply(**kwargs):
        result = commit(**kwargs)
        replies.append(result)
        if len(replies) == 1:
            # The real writer committed both records. Only its response was lost.
            raise OSError("Synthetic adoption commit acknowledgement lost")
        return result

    monkeypatch.setattr(store, "_commit_owned_work_selection", lost_reply)
    with _owned(store) as (token, _):
        kind, adopted = ResumeSelection().select_owned(
            _context(store), user_id="one", token=token, worker_id="test"
        )
        assert kind == "finalization" and adopted is None
        assert len(replies) == 2 and replies[0][0] and replies[0][1] is not None
        assert replies[1] == (False, None), "An ambiguous commit must not rotate twice"
        current = store.get_agent_run(run.id)
        assert current is not None and current.status == AgentRunStatus.FINALIZING
        assert current.claimed_by == replies[0][1].claimed_by
        assert current.resume_attempts == run.resume_attempts
        assert current.finalization_attempts == run.finalization_attempts
        assert _position(store).next_work_class == "continuation"
        assert _position(store).finalization.after_id == run.id


@pytest.mark.parametrize("transition", ["release", "expiry", "replacement"])
def test_busy_fleet_inspection_observes_only_live_parent_without_logical_revision(
    store, transition
):
    run = candidates._run(store, "busy-current-output")
    live = store.claim_user_extraction("one", "actual-busy-owner", 300)
    assert live is not None
    selector = ResumeSelection()
    assert (
        selector.claim_user(_context(store), worker_id="inspector", lease_seconds=300)
        is None
    )
    token = store.conn.execute(
        "SELECT user_work_token FROM _agent_runs WHERE id=?", (run.id,)
    ).fetchone()[0]
    assert token == live
    assert store.get_agent_run(run.id) == run
    assert store._resume_candidate_page("finalization")[1] is None
    if transition == "expiry":
        store._execute(
            "UPDATE learning_work SET lease_until=0 WHERE user_id=?", ("one",)
        )
    else:
        store.release_user_extraction("one", live)
    if transition == "replacement":
        next_token = store.claim_user_extraction("one", "new-actual-owner", 300)
        assert next_token is not None and next_token != live
    assert store._resume_candidate_page("finalization")[1]["id"] == run.id
    assert store.get_agent_run(run.id) == run


@pytest.mark.parametrize("mismatch", ["revision", "user", "org", "no_live_parent"])
def test_parent_observation_refuses_unproven_or_changed_target(store, mismatch):
    run = candidates._run(store, "exact-observation")
    live = store.claim_user_extraction("one", "actual-busy-owner", 300)
    assert live is not None
    expected = run
    if mismatch == "revision":
        store.update_agent_run_status(
            run.id, run.status, last_error="new business revision"
        )
    elif mismatch == "org":
        expected = replace(run, binding=replace(run.binding, org_id="different-org"))
    elif mismatch == "no_live_parent":
        store.release_user_extraction("one", live)
    before = store.get_agent_run(run.id)
    assert not store._observe_live_user_work_parent(
        expected=expected, user_id="two" if mismatch == "user" else "one"
    )
    assert store.get_agent_run(run.id) == before
    assert (
        store.conn.execute(
            "SELECT user_work_token FROM _agent_runs WHERE id=?", (run.id,)
        ).fetchone()[0]
        is None
    )


@pytest.mark.parametrize("case", ["transaction", "foreign_live_parent", "rollback"])
def test_private_parent_observation_cannot_join_or_replace_live_foreign_work(
    store, monkeypatch, case
):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    run = candidates._run(store, "observation-refusal")
    live = store.claim_user_extraction("one", "actual-live-owner", 300)
    assert live is not None
    if case == "transaction":
        with (
            store.commit_scope(),
            pytest.raises(ValueError, match="independent writer"),
        ):
            store._observe_live_user_work_parent(expected=run, user_id="one")
    elif case == "foreign_live_parent":
        other = store.claim_user_extraction("two", "foreign-live-owner", 300)
        assert other is not None
        store._execute(
            "UPDATE _agent_runs SET user_work_token=? WHERE id=?", (other, run.id)
        )
        assert not store._observe_live_user_work_parent(expected=run, user_id="one")
        assert (
            store.conn.execute(
                "SELECT user_work_token FROM _agent_runs WHERE id=?", (run.id,)
            ).fetchone()[0]
            == other
        )
    else:
        query = SQLiteStreamSQL.query

        def fail_after_update(db, statement, params=()):
            result = query(db, statement, params)
            if statement.startswith("UPDATE ") and "SET user_work_token=" in statement:
                raise RuntimeError("Synthetic post-update rollback")
            return result

        with monkeypatch.context() as failed:
            failed.setattr(SQLiteStreamSQL, "query", fail_after_update)
            with pytest.raises(RuntimeError, match="Synthetic post-update rollback"):
                store._observe_live_user_work_parent(expected=run, user_id="one")
        assert (
            store.conn.execute(
                "SELECT user_work_token FROM _agent_runs WHERE id=?", (run.id,)
            ).fetchone()[0]
            is None
        )
    assert store.get_agent_run(run.id) == run


def test_deferred_presence_distinguishes_expiry_and_uses_one_clock(store):
    scope = human_feedback_scope(store.org_id)
    expired = PendingToolCallRecord(
        id="expiry-only",
        org_id=store.org_id,
        user_id="one",
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key="expiry-only",
        status=PendingToolCallStatus.PENDING,
        question_text="Synthetic expiry question",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        cache_until=datetime.now(UTC) - timedelta(seconds=1),
    )
    store.create_pending_tool_call(expired)
    statements = []
    store.conn.set_trace_callback(statements.append)
    try:
        presence = store._deferred_work_presence()
    finally:
        store.conn.set_trace_callback(None)
    assert presence.expiry and not presence.resume and not presence.financial
    assert not presence.needs_turn
    assert len(statements) == 2
    assert all(statement.lstrip().startswith("SELECT ") for statement in statements)
    assert (
        store.get_pending_tool_call(expired.id).status == PendingToolCallStatus.PENDING
    )
    with store.commit_scope(), pytest.raises(ValueError, match="independent writer"):
        store._deferred_work_presence()


@pytest.mark.parametrize("failed_source", ["learning", "deferred", "both"])
def test_discovery_preserves_independent_proof_and_never_calls_partial_idle(
    store, monkeypatch, failed_source
):
    from reflexio.server.services.durable_learning.discovery import (
        IncompleteWorkDiscoveryError,
        discover_work,
    )

    candidates._run(store, "discovery-proof")
    token = store.claim_user_extraction("one", "create-canonical-work", 300)
    assert token is not None
    store.release_user_extraction("one", token)
    assert store.list_extraction_orgs() == [store.org_id]
    assert store._deferred_work_presence().needs_turn
    fault = RuntimeError("Synthetic unavailable discovery source")

    def fail():
        raise fault

    if failed_source in ("learning", "both"):
        monkeypatch.setattr(store, "list_extraction_orgs", fail)
    if failed_source in ("deferred", "both"):
        monkeypatch.setattr(store, "_deferred_work_presence", fail)
    with pytest.raises(IncompleteWorkDiscoveryError) as caught:
        list(discover_work(store))
    assert caught.value.org_ids == (() if failed_source == "both" else (store.org_id,))
    assert caught.value.__cause__ is fault


@pytest.mark.parametrize("mismatch", ["context", "storage"])
def test_default_discovery_rejects_wrong_org_before_any_source_read(
    store, monkeypatch, mismatch
):
    from reflexio.server.services.durable_learning import local
    from reflexio.server.services.durable_learning import scheduler as scheduling

    context = _context(store)
    target = store.org_id if mismatch == "context" else "different-org"
    context.org_id = "different-org" if mismatch == "context" else target

    def forbidden():
        raise AssertionError("Wrong-org factory must not inspect source data")

    monkeypatch.setattr(store, "list_extraction_orgs", forbidden)
    monkeypatch.setattr(store, "_deferred_work_presence", forbidden)
    monkeypatch.setattr(
        scheduling.DurableLearningScheduler, "start", lambda _self: None
    )
    monkeypatch.setattr(local, "adopt_server_scheduler", lambda _scheduler: None)
    scheduler = scheduling.maybe_start_durable_learning(
        lambda _org: context, bootstrap_org_id=target
    )
    with pytest.raises(ValueError, match="source org mismatch"):
        scheduler._org_ids_provider()


def _accepted_run(store, work_class):
    now = datetime.now(UTC)
    run = candidates._run(
        store,
        "accepted",
        status=AgentRunStatus.FINALIZATION_FAILED
        if work_class == "finalization"
        else AgentRunStatus.RESUME_READY,
    )
    if work_class == "continuation":
        scope = human_feedback_scope(store.org_id)
        store.create_pending_tool_call(
            PendingToolCallRecord(
                id="accepted-answer",
                org_id=store.org_id,
                user_id="one",
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key="accepted",
                status=PendingToolCallStatus.RESOLVED,
                question_text="Synthetic decision?",
                result={"answer": "yes"},
                resolved_at=now,
                expires_at=now + timedelta(hours=1),
                cache_until=now + timedelta(minutes=5),
                valid_until=now + timedelta(hours=1),
            )
        )
        store.attach_run_tool_dependency(
            RunToolDependencyRecord(
                run_id=run.id, pending_tool_call_id="accepted-answer", resolved_at=now
            )
        )
    return run


@pytest.mark.parametrize("creation_policy", [False, True, "unavailable", "missing"])
@pytest.mark.parametrize("work_class", ["finalization", "continuation"])
def test_accepted_work_validates_configuration_without_creation_veto(
    store, creation_policy, work_class
):
    context = _context(store)
    config = context.configurator.get_config()
    assert config is not None
    config.pending_tool_call_config.enabled = creation_policy is True
    configurator = MagicMock()
    configurator.get_config.return_value = config
    if creation_policy == "unavailable":
        configurator.get_config.side_effect = ValueError(
            "model configuration unavailable"
        )
    elif creation_policy == "missing":
        configurator.get_config.return_value = None
    context.configurator = configurator
    run = _accepted_run(store, work_class)
    selection = ResumeSelection()
    claimed = selection.claim_user(
        context, worker_id="accepted-owner", lease_seconds=300
    )
    if creation_policy in ("unavailable", "missing"):
        assert claimed is None and store.get_agent_run(run.id) == run
        # A real second claimant proves the failed validation held no owner.
        token = store.claim_user_extraction("one", "validation-control", 300)
        assert token is not None
        store.release_user_extraction("one", token)
        configurator.get_config.assert_called()
        return
    assert claimed is not None and claimed[0] == "one"
    user, token = claimed
    try:
        with owned_user_work_context(store, user, token, 300):
            kind, adopted = selection.select_owned(
                context, user_id=user, token=token, worker_id="accepted-owner"
            )
        assert kind == work_class and adopted is not None and adopted.id == run.id
        assert adopted.status == (
            AgentRunStatus.FINALIZING
            if work_class == "finalization"
            else AgentRunStatus.RESUMING
        )
        assert (
            adopted.resume_attempts == run.resume_attempts
            and adopted.finalization_attempts == run.finalization_attempts
        )
        configurator.get_config.assert_called()
    finally:
        store.release_user_extraction(user, token)


@pytest.mark.parametrize("work_class", ["finalization", "continuation"])
def test_configuration_failure_after_user_claim_cannot_adopt_run(store, work_class):
    context = _context(store)
    configurator = MagicMock()
    configurator.get_config.return_value = context.configurator.get_config()
    context.configurator = configurator
    run = _accepted_run(store, work_class)
    selection = ResumeSelection()
    claimed = selection.claim_user(
        context, worker_id="accepted-owner", lease_seconds=300
    )
    assert claimed is not None
    user, token = claimed
    configurator.get_config.side_effect = ValueError(
        "configuration changed while waiting"
    )
    try:
        with owned_user_work_context(store, user, token, 300):
            kind, adopted = selection.select_owned(
                context, user_id=user, token=token, worker_id="accepted-owner"
            )
        assert kind == work_class and adopted is None
        assert store.get_agent_run(run.id) == run
    finally:
        store.release_user_extraction(user, token)
