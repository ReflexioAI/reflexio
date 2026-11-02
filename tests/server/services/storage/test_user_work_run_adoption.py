"""One real owner and one exact logical run revision govern execution adoption."""

from copy import copy
from dataclasses import asdict, replace

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    LeaseLostError,
)


@pytest.fixture
def store(tmp_path):
    storage = SQLiteStorage(org_id="owned_run", db_path=str(tmp_path / "runs.db"))
    try:
        yield storage
    finally:
        storage.conn.close()


@pytest.mark.unit
def test_native_transaction_token_is_context_bound_and_lost_owner_is_not_unbound(
    store, tmp_path
):
    from reflexio.server.services.durable_learning.user_lease import (
        current_user_work_token,
        owned_user_work_context,
    )

    other = SQLiteStorage(org_id="other_owner", db_path=str(tmp_path / "other.db"))
    token = store.claim_user_extraction("one", "native", 300)
    assert token is not None
    try:
        assert current_user_work_token(store) is None
        with owned_user_work_context(store, "one", token, 300) as owner:
            assert current_user_work_token(store) == token
            assert current_user_work_token(other) is None
            owner.lost.set()
            with pytest.raises(LeaseLostError, match="cancelled or lost"):
                current_user_work_token(store)
        assert current_user_work_token(store) is None
    finally:
        store.release_user_extraction("one", token)
        other.conn.close()


def _run(store, *, user_id: str | None = "one", kind="profile", source_ids=None):
    return store.create_agent_run(
        AgentRunRecord(
            id="synthetic_run",
            binding=AgentBinding(
                org_id=store.org_id,
                extractor_kind=kind,
                user_id=user_id,
                request_id="synthetic_request",
                agent_version=None,
                source=None,
                source_interaction_ids=source_ids or [],
            ),
            status=AgentRunStatus.FINALIZATION_FAILED,
            generation_request_snapshot={},
            committed_output={"profiles": []},
            finalization_attempts=3,
        )
    )


def _adopt(store, run, user, token):
    with store.commit_scope(), store._stream_sql() as db:
        return store._adopt_owned_user_work_run(
            db, expected=run, user_id=user, token=token, worker_id="synthetic_worker"
        )


def test_adoption_keeps_one_owner_and_refuses_stale_logical_revision(store):
    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    claim = _adopt(store, run, "one", token)
    assert claim is not None
    adopted = store.get_agent_run(run.id)
    assert adopted is not None
    assert (adopted.claimed_by, adopted.claimed_at) == claim
    assert token not in adopted.claimed_by
    assert adopted.status == run.status and adopted.finalization_attempts == 3
    assert "user_work_token" not in asdict(adopted)
    parent = store.conn.execute(
        "SELECT user_work_token FROM _agent_runs WHERE id=?", (run.id,)
    ).fetchone()[0]
    assert parent == token
    assert _adopt(store, run, "one", token) is None
    store.fence_user_extraction("one", token)


def test_other_users_live_token_is_never_authority_for_the_run(store):
    run = _run(store)
    token = store.claim_user_extraction("two", "other", 300)
    assert token is not None
    assert _adopt(store, run, "two", token) is None
    assert store.get_agent_run(run.id) == run


def test_foreign_live_parent_must_finish_before_replacement_can_adopt(store):
    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    foreign = store.claim_user_extraction("two", "other", 300)
    assert token is not None and foreign is not None
    store.conn.execute(
        "UPDATE _agent_runs SET user_work_token=? WHERE id=?", (foreign, run.id)
    )
    store.conn.commit()
    assert _adopt(store, run, "one", token) is None
    store.release_user_extraction("two", foreign)
    assert _adopt(store, run, "one", token) is not None


def test_released_user_token_cannot_adopt_an_unclaimed_revision(store):
    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    store.release_user_extraction("one", token)
    with pytest.raises(LeaseLostError):
        _adopt(store, run, "one", token)
    assert store.get_agent_run(run.id) == run


def test_nullable_legacy_owner_without_complete_sources_is_refused(store):
    run = _run(store, user_id=None, kind="playbook", source_ids=[100])
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    assert _adopt(store, run, "one", token) is None
    assert store.get_agent_run(run.id) == run


@pytest.mark.parametrize(
    "interaction_user,request_user,request_exists,accepted",
    [
        (" one ", "one", True, True),
        ("two", "one", True, False),
        ("one", "two", True, False),
        ("one", "one", False, False),
    ],
)
def test_legacy_adoption_requires_every_interaction_and_request_owner(
    store, interaction_user, request_user, request_exists, accepted
):
    store.conn.execute(
        "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES('first','one','2026-01-01','session')"
    )
    if request_exists:
        store.conn.execute(
            "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES('second',?,'2026-01-01','session')",
            (request_user,),
        )
    store.conn.execute(
        "INSERT INTO interactions(interaction_id,user_id,request_id,created_at) VALUES(1,'one','first','2026-01-01')"
    )
    store.conn.execute(
        "INSERT INTO interactions(interaction_id,user_id,request_id,created_at) VALUES(2,?,'second','2026-01-01')",
        (interaction_user,),
    )
    store.conn.commit()
    run = _run(store, user_id=None, kind="playbook", source_ids=[2, 1, 2])
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    assert (_adopt(store, run, "one", token) is not None) is accepted


def test_offsetless_historical_revision_is_compared_as_utc(store):
    run = _run(store)
    store.conn.execute(
        "UPDATE _agent_runs SET updated_at='2026-01-01T00:00:00' WHERE id=?", (run.id,)
    )
    store.conn.commit()
    historical = store.get_agent_run(run.id)
    assert historical is not None
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    assert _adopt(store, historical, "one", token) is not None


def test_owner_is_rechecked_after_run_lock_wait(store):
    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with (
        pytest.raises(LeaseLostError, match="expired during run adoption"),
        store.commit_scope(),
        store._stream_sql() as db,
    ):
        original_query = db.query

        def after_wait(sql, params=()):
            rows = original_query(sql, params)
            if sql.startswith("SELECT r.user_id,r.extractor_kind"):
                # Deterministic elapsed-lock witness; the actual DB fence still
                # reads and rejects the real expired row, not a mocked outcome.
                store.conn.execute(
                    "UPDATE learning_work SET lease_until=0 WHERE user_id='one'"
                )
            return rows

        db.query = after_wait
        store._adopt_owned_user_work_run(
            db, expected=run, user_id="one", token=token, worker_id="worker"
        )
    assert store.get_agent_run(run.id) == run


@pytest.mark.parametrize(
    "count,complete", [(500, True), (501, True), (1001, True), (1001, False)]
)
def test_complete_legacy_proof_pages_under_real_sqlite_parameter_limit(
    store, count, complete
):
    import sqlite3

    store.conn.execute(
        "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES('source','one','2026-01-01','session')"
    )
    store.conn.executemany(
        "INSERT INTO interactions(interaction_id,user_id,request_id,created_at) VALUES(?,'one',?,'2026-01-01')",
        (
            (value, "missing" if not complete and value == count else "source")
            for value in range(1, count + 1)
        ),
    )
    store.conn.commit()
    run = _run(
        store, user_id=None, kind="playbook", source_ids=list(range(1, count + 1))
    )
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    previous = store.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        assert (_adopt(store, run, "one", token) is not None) is complete
    finally:
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)


def test_proof_cannot_renew_its_budget_or_accept_partial_sources(store):
    store.conn.execute(
        "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES('source','one','2026-01-01','session')"
    )
    store.conn.execute(
        "INSERT INTO interactions(interaction_id,user_id,request_id,created_at) VALUES(1,'one','source','2026-01-01')"
    )
    store.conn.commit()
    run = _run(store, user_id=None, kind="playbook", source_ids=[1])
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with pytest.raises(LeaseLostError), store.commit_scope(), store._stream_sql() as db:
        original_query = db.query

        def expire_after_proof(sql, params=()):
            rows = original_query(sql, params)
            if sql.startswith(
                "SELECT i.interaction_id,i.user_id,r.user_id AS request_user"
            ):
                store.conn.execute(
                    "UPDATE learning_work SET lease_until=0 WHERE user_id='one'"
                )
            return rows

        db.query = expire_after_proof
        store._adopt_owned_user_work_run(
            db, expected=run, user_id="one", token=token, worker_id="worker"
        )
    assert store.get_agent_run(run.id) == run


def _adopted(store):
    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None and _adopt(store, run, "one", token) is not None
    current = store.get_agent_run(run.id)
    assert current is not None
    return current, token


def test_execution_writer_requires_both_bound_context_and_adopted_claim(store):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    with (
        pytest.raises(LeaseLostError, match="active user-work"),
        owned_agent_run_write(store, run),
    ):
        pytest.fail("unowned writer entered")
    with (
        owned_user_work_context(store, "one", token, 300),
        owned_agent_run_write(store, run),
    ):
        store.update_agent_run_status(run.id, AgentRunStatus.FINALIZED)
    assert store.get_agent_run(run.id).status == AgentRunStatus.FINALIZED


def test_new_claim_on_same_user_token_withdraws_old_execution_writer(store):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )

    stale, token = _adopted(store)
    assert _adopt(store, stale, "one", token) is not None
    current = store.get_agent_run(stale.id)
    assert current is not None
    with owned_user_work_context(store, "one", token, 300):
        with (
            pytest.raises(LeaseLostError, match="revision is no longer current"),
            owned_agent_run_write(store, stale),
        ):
            pytest.fail("stale logical revision entered writer")
        with owned_agent_run_write(store, current):
            pass


def test_authorized_status_withdrawal_still_invalidates_execution(store):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    # A source/governance callback owns independent authorization; it does not
    # obtain generation authority or need to adopt the execution claim.
    store.update_agent_run_status(run.id, AgentRunStatus.CANCELLED)
    with (
        owned_user_work_context(store, "one", token, 300),
        pytest.raises(LeaseLostError, match="revision is no longer current"),
        owned_agent_run_write(store, run),
    ):
        pytest.fail("withdrawn run entered writer")
    assert store.get_agent_run(run.id).status == AgentRunStatus.CANCELLED


def test_execution_effect_rolls_back_when_parent_expires_before_commit(store):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    with (
        owned_user_work_context(store, "one", token, 300),
        pytest.raises(LeaseLostError),
        owned_agent_run_write(store, run),
    ):
        store.update_agent_run_status(run.id, AgentRunStatus.FINALIZED)
        store.conn.execute("UPDATE learning_work SET lease_until=0 WHERE user_id='one'")
    assert store.get_agent_run(run.id) == run


def test_execution_context_cannot_be_reused_for_another_storage_instance(
    store, tmp_path
):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    other = SQLiteStorage(org_id=store.org_id, db_path=str(tmp_path / "other.db"))
    try:
        with (
            owned_user_work_context(store, "one", token, 300),
            pytest.raises(LeaseLostError, match="different storage target"),
            owned_agent_run_write(other, run),
        ):
            pytest.fail("owner context crossed storage targets")
    finally:
        other.conn.close()


def test_owned_creation_is_atomic_and_reuse_keeps_its_logical_claim(store):
    import sqlite3

    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        create_owned_agent_run,
        owned_user_work_context,
    )

    template = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with owned_user_work_context(store, "one", token, 300):
        fresh = create_owned_agent_run(
            store, replace(template, id="fresh", status=AgentRunStatus.RUNNING)
        )
        assert adopt_agent_run_for_execution(store, fresh) == fresh

        def deny_parent_write(action, table, column, _db, _trigger):
            if (
                action == sqlite3.SQLITE_UPDATE
                and table == "_agent_runs"
                and column == "user_work_token"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.conn.set_authorizer(deny_parent_write)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                create_owned_agent_run(store, replace(template, id="failed_adoption"))
        finally:
            store.conn.set_authorizer(None)
        assert store.get_agent_run("failed_adoption") is None
        assert store.get_agent_run("fresh") == fresh


@pytest.mark.parametrize("user_id", [None, "two"])
def test_new_execution_requires_the_established_user_before_insertion(store, user_id):
    from reflexio.server.services.durable_learning.user_lease import (
        create_owned_agent_run,
        owned_user_work_context,
    )

    template = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with (
        owned_user_work_context(store, "one", token, 300),
        pytest.raises(LeaseLostError, match="established user"),
    ):
        create_owned_agent_run(
            store,
            replace(
                template,
                id="wrong_owner",
                binding=replace(template.binding, user_id=user_id),
            ),
        )
    assert store.get_agent_run("wrong_owner") is None


def test_nested_agent_entry_reuses_the_existing_process_admission(store, monkeypatch):
    from reflexio.server.services.durable_learning import user_lease

    run = _run(store)
    reservations = []
    original = user_lease._reserve

    def reserve():
        reservations.append(1)
        return original()

    monkeypatch.setattr(user_lease, "_reserve", reserve)
    with (
        user_lease.user_extraction_lease(store, "one"),
        user_lease.agent_run_user_work(store, run),
    ):
        assert reservations == [1]
    assert reservations == [1]


def test_caught_owned_write_rejection_poisoned_outer_transaction_cannot_commit(store):
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        owned_user_work_context,
    )
    from reflexio.server.services.storage.error import StorageError

    run, token = _adopted(store)
    with owned_user_work_context(store, "one", token, 300):
        with pytest.raises(StorageError, match="rollback"), store.commit_scope():
            try:
                with owned_agent_run_write(store, run):
                    store.update_agent_run_status(
                        run.id, AgentRunStatus.AGENT_COMPLETED
                    )
                    store.conn.execute(
                        "UPDATE learning_work SET lease_until=0 WHERE user_id='one'"
                    )
            except LeaseLostError:
                pass  # A caller swallowing loss must not commit its effect.
        assert store.get_agent_run(run.id) == run


def test_same_claimed_owner_context_reuses_one_heartbeat(store, monkeypatch):
    from reflexio.server.services.durable_learning import user_lease

    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    real_thread = user_lease.threading.Thread
    threads = []

    def track_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(user_lease.threading, "Thread", track_thread)
    with user_lease.owned_user_work_context(store, "one", token, 300) as owner:
        with user_lease.owned_user_work_context(store, "one", token, 300) as nested:
            assert nested is owner
            assert len(threads) == 1
        owner.fence(store)
        assert threads[0].is_alive()
    assert not threads[0].is_alive()
    # Binding a claimed lease never transfers admission/release from its caller.
    store.fence_user_extraction("one", token)


@pytest.mark.parametrize("mismatch", ["storage", "user", "token", "released"])
def test_claimed_owner_context_refuses_nested_owner_change(store, mismatch):
    from reflexio.server.services.durable_learning import user_lease

    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with user_lease.owned_user_work_context(store, "one", token, 300) as owner:
        target = store
        if mismatch == "storage":
            target = copy(
                store
            )  # Same DB/org does not confer this instance's authority.
        if mismatch == "released":
            store.release_user_extraction("one", token)
        with (
            pytest.raises(LeaseLostError),
            user_lease.owned_user_work_context(
                target,
                "two" if mismatch == "user" else "one",
                "foreign" if mismatch == "token" else token,
                300,
            ),
        ):
            pytest.fail("nested execution replaced the actual owner")
        assert user_lease._active_lease.get() is owner


def test_cancelled_turn_rolls_back_inflight_write_and_cannot_readopt(store):
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        cancel_extraction_turn,
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    with owned_user_work_context(store, "one", token, 300):
        with (
            pytest.raises(LeaseLostError, match="cancelled"),
            owned_agent_run_write(store, run),
        ):
            store.update_agent_run_status(run.id, AgentRunStatus.AGENT_COMPLETED)
            cancel_extraction_turn(store)
        assert store.get_agent_run(run.id) == run
        cancel_extraction_turn(store)  # Cancellation is idempotent.
        with pytest.raises(LeaseLostError, match="cancelled"):
            adopt_agent_run_for_execution(store, run)
        with (
            pytest.raises(LeaseLostError, match="cancelled"),
            owned_user_work_context(store, "one", token, 300),
        ):
            pytest.fail("cancelled handle reentered execution")
        store.fence_user_extraction("one", token)


def test_turn_cancellation_refuses_another_storage_handle(store):
    from reflexio.server.services.durable_learning.user_lease import (
        cancel_extraction_turn,
        owned_user_work_context,
    )

    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    with owned_user_work_context(store, "one", token, 300) as owner:
        with pytest.raises(LeaseLostError, match="different storage"):
            cancel_extraction_turn(copy(store))
        owner.fence(store)
        assert not owner.lost.is_set()


def test_cancellation_during_final_database_fence_rolls_back(store, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event

    from reflexio.server.services.durable_learning.user_lease import (
        cancel_extraction_turn,
        owned_agent_run_write,
        owned_user_work_context,
    )

    run, token = _adopted(store)
    in_final_fence, resume, body_finished = Event(), Event(), Event()
    original = store.fence_user_extraction

    def pause_final_fence(user_id, claimed_token):
        original(user_id, claimed_token)
        if body_finished.is_set():
            in_final_fence.set()
            assert resume.wait(5), "Final fence was not released by the test"

    monkeypatch.setattr(store, "fence_user_extraction", pause_final_fence)

    def write():
        try:
            with owned_agent_run_write(store, run):
                store.update_agent_run_status(run.id, AgentRunStatus.AGENT_COMPLETED)
                body_finished.set()
        except LeaseLostError:
            return "rejected"
        return "committed"

    with owned_user_work_context(store, "one", token, 300):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, write)
            try:
                assert in_final_fence.wait(5), "Writer did not reach its final fence"
                cancel_extraction_turn(store)
            finally:
                resume.set()
            assert future.result(timeout=5) == "rejected"
        assert store.get_agent_run(run.id) == run


def test_cancellation_during_adoption_rolls_back_the_new_claim(store, monkeypatch):
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        cancel_extraction_turn,
        owned_user_work_context,
    )

    run = _run(store)
    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    original = store._adopt_owned_user_work_run

    def cancelled_after_adoption(*args, **kwargs):
        claim = original(*args, **kwargs)
        assert claim is not None
        cancel_extraction_turn(store)
        return claim

    monkeypatch.setattr(store, "_adopt_owned_user_work_run", cancelled_after_adoption)
    with owned_user_work_context(store, "one", token, 300):
        with pytest.raises(LeaseLostError, match="cancelled"):
            adopt_agent_run_for_execution(store, run)
        assert store.get_agent_run(run.id) == run


def test_cancelled_turn_stops_heartbeat_before_another_renewal(store, monkeypatch):
    from threading import Event

    from reflexio.server.services.durable_learning import user_lease

    token = store.claim_user_extraction("one", "owner", 300)
    assert token is not None
    interval_entered, continue_interval = Event(), Event()
    threads, renewals = [], []
    real_thread = user_lease.threading.Thread

    def track_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    def controlled_interval(_):
        interval_entered.set()
        assert continue_interval.wait(5)
        return 0  # Controlled wakeup, not a runtime renewal policy.

    def renew(*args):
        renewals.append(args)
        return False

    monkeypatch.setattr(user_lease.threading, "Thread", track_thread)
    monkeypatch.setattr(
        user_lease.durable_learning_policy,
        "renewal_interval_seconds",
        controlled_interval,
    )
    monkeypatch.setattr(store, "renew_extraction", renew)
    with user_lease.owned_user_work_context(store, "one", token, 300):
        try:
            assert interval_entered.wait(5)
            user_lease.cancel_extraction_turn(store)
        finally:
            continue_interval.set()
        # Wait before the context's finally sets stop, so stop cannot make a
        # missing cancellation check falsely pass this test.
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        assert renewals == []
    store.fence_user_extraction("one", token)
