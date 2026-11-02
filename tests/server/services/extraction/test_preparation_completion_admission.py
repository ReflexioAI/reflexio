"""Physical SQLite admission and post-write evidence expiry keep cache atomic."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Event

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.services.extraction.history_inputs import (
    record_historical_profiles,
)
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.services.storage.storage_base._publication_preparations import (
    PublicationPreparationStore,
)
from tests.server.services.extraction import (
    test_initial_publication_integration as initial,
)
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
actual_embedding_dispatch = initial.actual_embedding_dispatch
pytestmark = pytest.mark.integration


def test_sqlite_admission_busy_retains_completed_result_at_cap_one(
    publication, monkeypatch, actual_embedding_dispatch
):
    storage, worker, kind, _, events = publication
    service, request, calls = initial._initial(publication, monkeypatch)
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    storage.conn.execute("PRAGMA busy_timeout=50")
    busy = Event()
    held = Event()
    refusals = []
    computes = []
    scope = storage.commit_scope
    resolve = type(service)._resolve_write_plan

    @contextmanager
    def observed_scope():
        try:
            with scope():
                yield
        except Exception as error:
            cause = error
            while cause is not None:
                if (
                    isinstance(cause, sqlite3.OperationalError)
                    and getattr(cause, "sqlite_errorcode", 0) & 255
                    == sqlite3.SQLITE_BUSY
                ):
                    refusals.append(cause)
                    busy.set()
                    break
                cause = cause.__cause__
            raise

    def hold_writer():
        with sqlite3.connect(storage.db_path) as other:
            other.execute("BEGIN IMMEDIATE")
            other.execute(
                "UPDATE interactions SET created_at=created_at WHERE interaction_id=1"
            )
            held.set()
            assert busy.wait(5), (
                "Expected an actual SQLite transaction-admission timeout"
            )
            other.rollback()

    monkeypatch.setattr(storage, "commit_scope", observed_scope)
    with ThreadPoolExecutor(max_workers=1) as pool:
        holds = []

        def compute_then_block(self, groups):
            result = resolve(self, groups)
            computes.append(1)
            holds.append(pool.submit(hold_writer))
            assert held.wait(5)
            return result

        monkeypatch.setattr(type(service), "_resolve_write_plan", compute_then_block)
        initial._run_initial(service, request)
        for hold in holds:
            hold.result(timeout=5)
    run = service._last_extraction_run
    assert refusals and computes == [1] and calls == [1]
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert run.finalization_attempts == 0 and run.resume_attempts == 0
    assert len(base._learnings(storage, kind)) == 1
    assert len([e for e in events if e.event_name == "learnings_generated"]) == 1


@pytest.mark.parametrize("evidence", ["answer", "history"])
def test_expiry_after_cache_write_rolls_back_completion(
    shared_worker, publication, monkeypatch, actual_embedding_dispatch, evidence
):
    storage, worker, kind, _, events = publication
    now = datetime.now(UTC)
    deadline = int(now.timestamp()) + 60
    profile = None
    if evidence == "answer":
        used = storage.get_pending_tool_call("ptc_1")
        assert used is not None
        used = storage.update_resolved_pending_tool_call_result(
            used.id, result=used.result, resolved_at=now, valid_for_seconds=60
        )
        assert used is not None and used.valid_until is not None
        deadline = used.valid_until.timestamp()
    else:
        storage.add_user_profile(
            "user_1",
            [
                UserProfile(
                    profile_id="expiring-preparation-input",
                    user_id="user_1",
                    content="Prior deployment policy.",
                    last_modified_timestamp=int(now.timestamp()),
                    generated_from_request_id="request_1",
                    expiration_timestamp=deadline,
                )
            ],
            skip_embedding=True,
        )
        profile = storage.get_user_profile("user_1")[0]
        deadline = profile.expiration_timestamp
    service_type = (
        initial.ProfileGenerationService
        if kind == "profile"
        else initial.PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan

    def prepare(self, groups):
        if profile is not None:
            record_historical_profiles(storage, [profile])
        return resolve(self, groups)

    monkeypatch.setattr(service_type, "_resolve_write_plan", prepare)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    write = PublicationPreparationStore._write_preparation_row
    written = []

    def write_then_expire(storage, run, db, rows, preparation, active):
        write(storage, run, db, rows, preparation, active)
        if preparation is not None:
            written.append(1)
            monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _db: deadline + 1)

    monkeypatch.setattr(
        PublicationPreparationStore,
        "_write_preparation_row",
        staticmethod(write_then_expire),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    run = storage.get_agent_run("run_1")
    assert written == [1]
    assert run is not None and run.status == AgentRunStatus.FINALIZATION_FAILED
    row = storage.conn.execute(
        "SELECT preparation FROM _agent_output_preparations WHERE run_id=?", (run.id,)
    ).fetchone()
    assert row is None or row[0] is None, (
        "Expired raw completion must roll back with its compute-entry clear"
    )
    assert not [e for e in events if e.event_name == "learnings_generated"]


@pytest.mark.parametrize("refusal", ["input", "readiness"])
def test_completed_resume_preparation_is_bounded_before_primary_model_entry(
    shared_worker, publication, monkeypatch, actual_embedding_dispatch, refusal
):
    import litellm

    from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
    from reflexio.server.services.storage.error import (
        ExecutionInputBusyError,
        ReadinessUnavailableError,
    )

    storage, worker, kind, _, _ = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_resume_attempts = 1
    api = litellm.embedding
    responses = []

    def embedding(**kwargs):
        value = api(**kwargs)
        responses.append(1)
        return value

    def prepare_then_refuse(_self, **_kwargs):
        assert storage._get_embedding("Synthetic prior context")
        if refusal == "input":
            raise ExecutionInputBusyError("Input refused after preparation")
        raise ReadinessUnavailableError("Readiness refused after preparation")

    monkeypatch.setattr(litellm, "embedding", embedding)
    monkeypatch.setattr(
        ExtractionResumeWorker, "_messages_with_prior_knowledge", prepare_then_refuse
    )
    for _ in range(6):
        storage._execute(
            "UPDATE _agent_runs SET next_resume_at=NULL WHERE id='run_1'", ()
        )
        shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    run = storage.get_agent_run("run_1")
    assert responses == [1]
    assert run is not None and run.status == AgentRunStatus.FAILED
    assert run.resume_attempts == 1 and run.committed_output is None


@pytest.mark.parametrize("blocked", [False, True])
def test_writer_blocks_both_result_and_charge_without_refunding_preparation(
    shared_worker, publication, monkeypatch, actual_embedding_dispatch, blocked
):
    import litellm

    from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker

    storage, worker, kind, _, _ = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_resume_attempts = 1
    storage.conn.execute("PRAGMA busy_timeout=50")
    busy, held = Event(), Event()
    responses, refusals, holders = [], [], []
    scope = storage.commit_scope
    api = litellm.embedding

    def embedding(**kwargs):
        response = api(**kwargs)
        responses.append(1)
        return response

    @contextmanager
    def observed_writer():
        try:
            with scope():
                yield
        except Exception as error:
            cause = error
            while cause is not None:
                if (
                    isinstance(cause, sqlite3.OperationalError)
                    and getattr(cause, "sqlite_errorcode", 0) & 255
                    == sqlite3.SQLITE_BUSY
                ):
                    refusals.append(1)
                    if len(refusals) % 2 == 0:
                        busy.set()
                    break
                cause = cause.__cause__
            raise

    def hold_writer():
        with sqlite3.connect(storage.db_path) as other:
            other.execute("BEGIN IMMEDIATE")
            other.execute("UPDATE interactions SET created_at=created_at")
            held.set()
            assert busy.wait(5), (
                "Hold through both refused writers, not only the raw-save refusal"
            )
            other.rollback()

    monkeypatch.setattr(storage, "commit_scope", observed_writer)
    monkeypatch.setattr(litellm, "embedding", embedding)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kwargs: base._loop_result(kwargs, kind, storage, nonempty=False),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:

        def prepare(_self, *, messages, **_kwargs):
            assert storage._get_embedding("Synthetic prior context")
            if blocked:
                busy.clear()
                held.clear()
                holders.append(pool.submit(hold_writer))
                assert held.wait(5)
            return messages

        monkeypatch.setattr(
            ExtractionResumeWorker, "_messages_with_prior_knowledge", prepare
        )
        for _ in range(6):
            storage._execute(
                "UPDATE _agent_runs SET next_resume_at=NULL WHERE id='run_1'", ()
            )
            shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
        for holder in holders:
            holder.result(timeout=5)
    run = storage.get_agent_run("run_1")
    assert responses == [1], "One admitted resume cannot repeat paid preparation"
    assert run is not None and run.resume_attempts == 1
    if blocked:
        assert len(refusals) == 2 and run.status == AgentRunStatus.FAILED
        assert run.committed_output is None
    else:
        assert not refusals and run.status == AgentRunStatus.FINALIZED
        assert run.committed_output is not None
