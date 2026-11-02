"""Dependency publication consumes precisely the retained output's inputs."""

import sqlite3
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.durable_learning.user_lease import owned_agent_run_write
from reflexio.server.services.extraction.overflow_answers import answer_witness
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    LeaseLostError,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)
from tests.server.services.extraction import (
    test_owned_resume_transitions as transitions,
)

pipeline = transitions.pipeline
execution = transitions.execution
pytestmark = pytest.mark.integration


def _answer(storage, run, identifier):
    now = datetime.now(UTC)
    question = f"What is the correct answer for {identifier}?"
    scope = human_feedback_scope(storage.org_id)
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=identifier,
            org_id=storage.org_id,
            user_id=run.binding.user_id,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=question
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=run.id, pending_tool_call_id=identifier)
    )
    resolved = storage.resolve_pending_tool_call(
        identifier,
        result={"answer": f"Answer {identifier}"},
        resolved_at=now,
        valid_for_seconds=30,
    )
    assert resolved is not None
    return resolved


def _output(storage, run, answers, *, prior_answers=()):
    payload = {
        "_reflexio_envelope_version": 1,
        "output": {"profiles": []},
        "publication": {
            "id": uuid4().hex,
            "dependency_inputs": {
                a.id: build_scope_hash(answer_witness(asdict(a))) for a in answers
            },
            "answer_inputs": {
                a.id: build_scope_hash(answer_witness(asdict(a)))
                for a in (*answers, *prior_answers)
            },
        },
    }
    with owned_agent_run_write(storage, run):
        saved = storage.update_agent_run_status(
            run.id,
            AgentRunStatus.AGENT_COMPLETED,
            committed_output=payload,
            expected_claimed_by=run.claimed_by,
            expected_claimed_at=run.claimed_at,
        )
    assert saved is not None
    return saved


@pytest.fixture
def output(execution):
    storage, run, token, worker = execution
    used = _answer(storage, run, "answer_used")
    completed = _output(storage, run, [used])
    # This real answer arrives after the output's input snapshot was committed.
    later = _answer(storage, completed, "answer_later")
    return storage, completed, used, later, token


def _consumption(storage, run):
    return {
        d.pending_tool_call_id: d.consumed_at is not None
        for d in storage.list_run_tool_dependencies(run.id)
    }


def test_consumes_only_answers_supplied_to_retained_output(output):
    storage, run, used, later, _ = output
    storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == {used.id: True, later.id: False}
    assert storage.get_agent_run(run.id) == run  # This primitive never terminalizes.


def test_dependency_publication_joins_caller_rollback(output):
    storage, run, used, later, _ = output
    with (
        pytest.raises(RuntimeError, match="after consumption"),
        storage.commit_scope(),
    ):
        storage.consume_agent_output_dependencies(run)
        assert _consumption(storage, run)[used.id]
        raise RuntimeError("after consumption")
    assert _consumption(storage, run) == {used.id: False, later.id: False}


def test_explicit_empty_inputs_do_not_consume_new_answers(output):
    storage, run, used, later, _ = output
    empty = _output(storage, run, [])
    storage.consume_agent_output_dependencies(empty)
    assert _consumption(storage, run) == {used.id: False, later.id: False}


@pytest.mark.parametrize(
    "withdrawal",
    [
        "answer",
        "missing_answer",
        "dependency",
        "output",
        "parent",
        "status",
        "snapshot",
    ],
)
def test_changed_evidence_cannot_consume_answers(output, withdrawal):
    storage, run, used, later, _ = output
    if withdrawal == "answer":
        storage._execute(
            "UPDATE _pending_tool_calls SET result=? WHERE id=?",
            ('{"answer":"replacement"}', used.id),
        )
    elif withdrawal == "missing_answer":
        storage._execute("DELETE FROM _pending_tool_calls WHERE id=?", (used.id,))
    elif withdrawal == "dependency":
        storage._execute(
            "DELETE FROM _run_tool_dependencies WHERE run_id=? AND pending_tool_call_id=?",
            (run.id, used.id),
        )
    elif withdrawal == "output":
        payload = {
            **run.committed_output,
            "output": {"profiles": [{"content": "changed"}]},
        }
        storage.update_agent_run_status(run.id, run.status, committed_output=payload)
    elif withdrawal == "parent":
        storage._execute(
            "UPDATE _agent_runs SET user_work_token='withdrawn' WHERE id=?", (run.id,)
        )
    elif withdrawal == "status":
        storage.update_agent_run_status(run.id, AgentRunStatus.FINALIZED)
    else:
        run = replace(run, committed_output={"profiles": []})
    before = _consumption(storage, run)
    with pytest.raises((InputBudgetConfigurationError, LeaseLostError)):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


def test_expired_answer_is_refused_even_with_matching_digest(output, monkeypatch):
    storage, run, _, _, _ = output
    now = SQLiteStreamSQL.now
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda db: now(db) + 60)
    before = _consumption(storage, run)
    with pytest.raises(InputBudgetConfigurationError, match="answer.*unavailable"):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


def test_lease_expiry_after_consumption_rolls_back(output, monkeypatch):
    storage, run, _, _, _ = output
    consume = storage._consume_answer_dependencies

    def expire_after_consumption(*args, **kwargs):
        result = consume(*args, **kwargs)
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET lease_until=0 WHERE org_id=? AND user_id=?",
                (storage.org_id, run.binding.user_id),
            )
        return result

    monkeypatch.setattr(
        storage, "_consume_answer_dependencies", expire_after_consumption
    )
    before = _consumption(storage, run)
    with pytest.raises(LeaseLostError):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


@pytest.mark.parametrize("missing_last", [False, True])
def test_owned_input_pages_do_not_impose_overflow_answer_limit(execution, missing_last):
    storage, run, _, _ = execution
    answers = [_answer(storage, run, f"page_answer_{i:04}") for i in range(1001)]
    completed = _output(storage, run, answers)
    if missing_last:
        storage._execute(
            "DELETE FROM _run_tool_dependencies WHERE run_id=? AND pending_tool_call_id=?",
            (run.id, answers[-1].id),
        )
    # Exercise the historical SQLite bind-variable floor, not a mocked query.
    prior = storage.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        if missing_last:
            with pytest.raises(
                InputBudgetConfigurationError, match="dependency changed"
            ):
                storage.consume_agent_output_dependencies(completed)
        else:
            storage.consume_agent_output_dependencies(completed)
    finally:
        storage.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, prior)
    observed = _consumption(storage, completed)
    if missing_last:
        assert len(observed) == len(answers) - 1
        assert not any(observed.values())  # Earlier consumed pages roll back too.
    else:
        assert len(observed) == len(answers)
        assert all(observed.values())


def test_answer_expiring_during_dependency_write_rolls_back(output, monkeypatch):
    storage, run, _, _, _ = output
    consume = storage._consume_answer_dependencies
    now = SQLiteStreamSQL.now

    def consume_then_advance(*args, **kwargs):
        result = consume(*args, **kwargs)
        monkeypatch.setattr(SQLiteStreamSQL, "now", lambda db: now(db) + 60)
        return result

    monkeypatch.setattr(storage, "_consume_answer_dependencies", consume_then_advance)
    before = _consumption(storage, run)
    with pytest.raises(InputBudgetConfigurationError, match="answer.*unavailable"):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


def test_caller_mutation_does_not_redefine_captured_publication(output, monkeypatch):
    storage, run, used, later, _ = output
    lock = storage._lock_extraction_answers

    def mutate_caller_then_lock(db, identifiers):
        run.committed_output["publication"]["dependency_inputs"][later.id] = (
            build_scope_hash(answer_witness(asdict(later)))
        )
        return lock(db, identifiers)

    monkeypatch.setattr(storage, "_lock_extraction_answers", mutate_caller_then_lock)
    storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == {used.id: True, later.id: False}


@pytest.fixture
def output_with_prior(output):
    storage, run, used, later, token = output
    prior = replace(
        used,
        id="org_shared_prior_answer",
        user_id="another_user",
        dedup_key="org_shared_prior_guidance",
        question_text="Which org-wide guidance applies?",
        args={"question": "Which org-wide guidance applies?"},
        result={"answer": "Require approval before deployment."},
        valid_until=datetime.now(UTC) + timedelta(hours=1),
    )
    storage.create_pending_tool_call(prior)
    prior = storage.get_pending_tool_call(prior.id)
    assert prior is not None
    completed = _output(storage, run, [used], prior_answers=[prior])
    return storage, completed, used, later, prior, token


def test_valid_prior_context_is_checked_but_never_attached_or_consumed(
    output_with_prior,
):
    storage, run, used, later, prior, _ = output_with_prior
    storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == {used.id: True, later.id: False}
    assert prior.id not in _consumption(storage, run)
    assert storage.get_agent_run(run.id) == run


@pytest.mark.parametrize("withdrawal", ["changed", "missing", "expired"])
def test_prior_context_change_refuses_entire_dependency_write(
    output_with_prior, monkeypatch, withdrawal
):
    storage, run, used, _, prior, _ = output_with_prior
    if withdrawal == "changed":
        storage.update_resolved_pending_tool_call_result(
            prior.id,
            result={"answer": "Different guidance."},
            resolved_at=datetime.now(UTC),
            valid_for_seconds=3600,
        )
    elif withdrawal == "missing":
        storage._execute("DELETE FROM _pending_tool_calls WHERE id=?", (prior.id,))
    else:
        short_lived = storage.update_resolved_pending_tool_call_result(
            prior.id,
            result=prior.result,
            resolved_at=used.resolved_at,
            valid_for_seconds=10,
        )
        assert short_lived is not None and short_lived.valid_until is not None
        run = _output(storage, run, [used], prior_answers=[short_lived])
        expired_at = short_lived.valid_until.timestamp() + 1
        monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _db: expired_at)
    before = _consumption(storage, run)
    with pytest.raises(InputBudgetConfigurationError, match="answer"):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


def test_context_only_output_still_validates_its_prior_answer(output_with_prior):
    storage, run, _, _, prior, _ = output_with_prior
    run = _output(storage, run, [], prior_answers=[prior])
    storage.update_resolved_pending_tool_call_result(
        prior.id,
        result={"answer": "Different guidance."},
        resolved_at=datetime.now(UTC),
        valid_for_seconds=3600,
    )
    before = _consumption(storage, run)
    with pytest.raises(InputBudgetConfigurationError, match="answer"):
        storage.consume_agent_output_dependencies(run)
    assert _consumption(storage, run) == before


@pytest.mark.parametrize(
    "invalid",
    ["missing", "dependency_absent", "mismatched", "bad_digest", "unknown_answer"],
)
def test_complete_answer_proof_and_exact_dependency_subset_are_required(
    output, invalid
):
    storage, run, used, _, _ = output
    payload = deepcopy(run.committed_output)
    proof = payload["publication"]
    if invalid == "missing":
        del proof["answer_inputs"]
    elif invalid == "dependency_absent":
        proof["answer_inputs"] = {}
    elif invalid == "mismatched":
        proof["answer_inputs"][used.id] = "0" * 64
    elif invalid == "bad_digest":
        proof["answer_inputs"][used.id] = "not-a-digest"
    else:
        proof["answer_inputs"]["missing_answer"] = "0" * 64
    with owned_agent_run_write(storage, run):
        retained = storage.update_agent_run_status(
            run.id,
            run.status,
            committed_output=payload,
            expected_claimed_by=run.claimed_by,
            expected_claimed_at=run.claimed_at,
        )
    assert retained is not None
    before = _consumption(storage, retained)
    with pytest.raises(InputBudgetConfigurationError):
        storage.consume_agent_output_dependencies(retained)
    assert _consumption(storage, retained) == before


def test_prior_context_expiring_after_dependency_write_rolls_back_all_consumption(
    output_with_prior, monkeypatch
):
    storage, run, used, _, prior, _ = output_with_prior
    short_lived = storage.update_resolved_pending_tool_call_result(
        prior.id,
        result=prior.result,
        resolved_at=used.resolved_at,
        valid_for_seconds=10,
    )
    assert short_lived is not None and short_lived.valid_until is not None
    expired_at = short_lived.valid_until.timestamp() + 1
    assert used.valid_until is not None and expired_at < used.valid_until.timestamp()
    run = _output(storage, run, [used], prior_answers=[short_lived])
    consume = storage._consume_answer_dependencies
    writes = []

    def consume_then_expire_prior(*args, **kwargs):
        result = consume(*args, **kwargs)
        writes.append(1)
        monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _db: expired_at)
        return result

    monkeypatch.setattr(
        storage, "_consume_answer_dependencies", consume_then_expire_prior
    )
    before = _consumption(storage, run)
    with pytest.raises(InputBudgetConfigurationError, match="answer.*unavailable"):
        storage.consume_agent_output_dependencies(run)
    assert writes == [1]
    assert _consumption(storage, run) == before
