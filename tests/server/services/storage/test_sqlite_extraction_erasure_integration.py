"""Erasure of actual owned extraction source preserves only pending finances."""

import sqlite3
from dataclasses import replace

import pytest

from reflexio.server.services.extraction.publication import _deliver
from reflexio.server.services.extraction.publication_receipts import PublicationReceipt
from reflexio.server.services.storage.storage_base._publication_receipts import (
    _RECEIPT_COLUMNS,
)
from reflexio.server.usage_metrics import (
    UsageEventDeliveryStatus,
    configure_usage_event_recorder,
)
from tests.server.services.extraction import (
    test_initial_publication_integration as initial,
)

storage = initial.storage
request_context = initial.request_context
publication = initial.publication
explicit_test_billing_exemption = initial.explicit_test_billing_exemption
no_background_tagging = initial.no_background_tagging
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("delivery_state", ["pending", "accepted"])
def test_clear_erases_actual_owned_run_and_preparation_but_retains_debt(
    publication, monkeypatch, abort, delivery_state
):
    storage, _, kind, _, events = publication
    service, request, calls = initial._initial(publication, monkeypatch)

    def reject_learning(event):
        if event.event_name == "learnings_generated" and delivery_state == "pending":
            return UsageEventDeliveryStatus.FAILED
        events.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(reject_learning)
    initial._run_initial(service, request)
    run = service._last_extraction_run
    assert run is not None and run.binding.user_id is not None
    assert run.committed_output is not None and calls == [1]
    output_id = run.committed_output["publication"]["id"]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(db, run.id, output_id)
        assert db.query(
            "SELECT run_id FROM _agent_output_preparations WHERE run_id=?", (run.id,)
        ), "The real writer must supply a private preparation for this regression"
    assert receipt is not None and receipt.delivery_state == delivery_state
    assert initial.base._learnings(storage, kind)

    if abort:
        tables = (
            "_agent_runs",
            "_agent_output_preparations",
            "_agent_output_receipts",
            "_pending_tool_calls",
            "_run_tool_dependencies",
            "learning_work",
            "requests",
            "interactions",
            "profiles",
            "user_playbooks",
            "interactions_fts",
            "profiles_fts",
            "user_playbooks_fts",
        )

        def snapshot():
            return {
                table: [
                    tuple(row)
                    for row in storage.conn.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"  # noqa: S608 -- Fixed test tables above.
                    )
                ]
                for table in tables
            }

        before = snapshot()
        storage.conn.execute(
            "CREATE TRIGGER forced_source_erase_failure BEFORE DELETE ON requests "
            "BEGIN SELECT RAISE(ABORT,'forced source erasure failure'); END"
        )
        with pytest.raises(
            sqlite3.IntegrityError, match="forced source erasure failure"
        ):
            storage.clear_user_data(run.binding.user_id)
        assert snapshot() == before
        return

    storage.clear_user_data(run.binding.user_id)
    assert storage.get_agent_run(run.id) is None
    assert initial.base._learnings(storage, kind) == []
    with storage._stream_sql(read_only=True) as db:
        assert (
            db.query(
                "SELECT run_id FROM _agent_output_preparations WHERE run_id=?",
                (run.id,),
            )
            == []
        )
        rows = db.query(
            f"SELECT {_RECEIPT_COLUMNS} FROM _agent_output_receipts WHERE receipt_id=?",  # noqa: S608 -- Fixed receipt-schema columns, never user input.
            (receipt.receipt_id,),
        )
    if delivery_state == "accepted":
        assert rows == []
        return
    assert len(rows) == 1
    assert PublicationReceipt.from_row(rows[0]) == replace(
        receipt, run_id=None, output_id=None
    )
    configure_usage_event_recorder(lambda _event: UsageEventDeliveryStatus.APPENDED)
    _deliver(storage, receipt)
    with storage._stream_sql(read_only=True) as db:
        assert (
            db.query(
                "SELECT receipt_id FROM _agent_output_receipts WHERE receipt_id=?",
                (receipt.receipt_id,),
            )
            == []
        )


def _empty_owned_run(storage, run_id, user_id):
    """Actual user lease and owned empty output; no model or financial charge."""
    import uuid

    from reflexio.server.services.durable_learning.user_lease import (
        create_owned_agent_run,
    )
    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    run = create_owned_agent_run(
        storage,
        AgentRunRecord(
            id=run_id,
            binding=AgentBinding(
                org_id=storage.org_id,
                extractor_kind="profile",
                user_id=user_id,
                request_id=f"request-{run_id}",
                agent_version="v1",
                source="api",
            ),
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={"request_id": f"request-{run_id}"},
            committed_output={
                "_reflexio_envelope_version": 1,
                "output": {"profiles": []},
                "publication": {"id": uuid.uuid4().hex},
            },
        ),
    )
    receipt, created = storage._create_agent_output_receipt(
        run, learning_ids=[], origin="resume"
    )
    assert created and receipt.delivery_state == "accepted"
    return run


@pytest.mark.parametrize(
    "phase,remaining,expected,attached",
    [
        ("resuming", "none", "finalized", True),
        ("resuming", "pending", "finalized_pending_tool", True),
        ("resuming", "resolved", "resume_ready", True),
        ("resume_ready", "none", "finalized", True),
        ("resume_ready", "pending", "finalized_pending_tool", True),
        ("resume_ready", "resolved", "resume_ready", True),
        ("finalized_pending_tool", "none", "finalized", True),
        ("finalized_pending_tool", "pending", "finalized_pending_tool", True),
        ("finalized_pending_tool", "resolved", "resume_ready", True),
        ("finalized_pending_tool", "mixed", "resume_ready", True),
        ("resuming", "none", "finalized", False),
        ("resuming", "boolean_marker", "finalized", True),
        ("resuming", "false_marker", "resume_ready", True),
        ("resuming", "integer_marker", "resume_ready", True),
        ("resuming", "float_marker", "resume_ready", True),
        ("resuming", "text_marker", "resume_ready", True),
    ],
)
def test_erased_answer_settles_shared_survivor_and_fences_old_execution(
    storage, phase, remaining, expected, attached
):
    from contextvars import Context
    from datetime import UTC, datetime, timedelta

    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
        user_extraction_lease,
    )
    from reflexio.server.services.storage.storage_base import (
        AgentRunStatus,
        LeaseLostError,
        PendingToolCallStatus,
        RunToolDependencyRecord,
        build_pending_tool_call_dedup_key,
    )
    from tests.server.services.storage.sqlite_storage.test_agent_run_storage import (
        _pending_call,
    )

    markers = {
        "boolean_marker": True,
        "false_marker": False,
        "integer_marker": 1,
        "float_marker": 1.0,
        "text_marker": "true",
    }
    now = datetime.now(UTC)
    with user_extraction_lease(storage, "erased-user"):
        _empty_owned_run(storage, "erased-run", "erased-user")
    erased = storage.create_pending_tool_call(
        replace(
            _pending_call("erased-answer", now=now, org_id=storage.org_id),
            user_id="erased-user",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "private answer from erased user"},
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    surviving = None
    if remaining != "none":
        surviving_status = (
            "pending"
            if remaining == "mixed"
            else "resolved"
            if remaining in markers
            else remaining
        )
        question = "Independent surviving question"
        surviving = storage.create_pending_tool_call(
            replace(
                _pending_call("surviving-answer", now=now, org_id=storage.org_id),
                user_id="surviving-user",
                question_text=question,
                args={"question": question},
                dedup_key=build_pending_tool_call_dedup_key(
                    tool_name="ask_human", question_text=question
                ),
                status=PendingToolCallStatus(surviving_status),
                result={"answer": "independent surviving answer"}
                if surviving_status == "resolved"
                else None,
                resolved_at=now if surviving_status == "resolved" else None,
                valid_until=now + timedelta(hours=1)
                if surviving_status == "resolved"
                else None,
            )
        )
    if remaining in markers:
        assert surviving is not None
        surviving = storage.update_resolved_pending_tool_call_result(
            surviving.id,
            result={
                "answer": "independent answer",
                "not_applicable": markers[remaining],
            },
            resolved_at=now,
            valid_for_seconds=3600,
        )
        assert surviving is not None and surviving.result is not None
        assert type(surviving.result["not_applicable"]) is type(markers[remaining])
        assert surviving.result["not_applicable"] == markers[remaining]
    additional = None
    if remaining == "mixed":
        assert surviving is not None
        text = "Another independent surviving question"
        additional = storage.create_pending_tool_call(
            replace(
                surviving,
                id="another-surviving-answer",
                question_text=text,
                args={"question": text},
                dedup_key=build_pending_tool_call_dedup_key(
                    tool_name="ask_human", question_text=text
                ),
                status=PendingToolCallStatus.RESOLVED,
                result={"answer": "another independent answer"},
                resolved_at=now,
                valid_until=now + timedelta(hours=1),
            )
        )
    survivors = [call for call in (surviving, additional) if call is not None]
    with user_extraction_lease(storage, "surviving-user"):
        run = _empty_owned_run(storage, "surviving-run", "surviving-user")
        for call in [erased, *survivors]:
            if call.id == erased.id and not attached:
                continue
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(
                    run_id=run.id,
                    pending_tool_call_id=call.id,
                    resolved_at=call.resolved_at,
                )
            )
        with owned_agent_run_write(storage, run):
            claimed = storage.update_agent_run_status(
                run.id,
                AgentRunStatus(phase),
                pending_tool_call_ids=[
                    erased.id,
                    *[call.id for call in survivors],
                ],
                expected_claimed_by=run.claimed_by,
                expected_claimed_at=run.claimed_at,
            )
        assert claimed is not None
        if not attached:
            assert storage.list_run_tool_dependencies(run.id) == []
        Context().run(storage.clear_user_data, "erased-user")
        assert storage.get_agent_run("erased-run") is None
        assert storage.get_pending_tool_call(erased.id) is None
        current = storage.get_agent_run(run.id)
        assert current is not None and current.status == AgentRunStatus(expected)
        assert current.pending_tool_call_ids == [call.id for call in survivors]
        assert {
            d.pending_tool_call_id for d in storage.list_run_tool_dependencies(run.id)
        } == {call.id for call in survivors}
        for call in survivors:
            assert storage.get_pending_tool_call(call.id) == call
        if phase == "resuming":
            with pytest.raises(LeaseLostError), owned_agent_run_write(storage, claimed):
                pass


@pytest.mark.parametrize(
    "marker,should_claim",
    [
        (True, False),
        (False, True),
        (1, True),
        (1.0, True),
        ("true", True),
    ],
)
def test_sqlite_ready_selection_uses_boolean_only_not_applicable_marker(
    storage, marker, should_claim
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.services.storage.storage_base import (
        AgentRunStatus,
        PendingToolCallStatus,
        RunToolDependencyRecord,
        is_not_applicable_tool_result,
    )
    from tests.server.services.storage.sqlite_storage.test_agent_run_storage import (
        _agent_run,
        _pending_call,
    )

    now = datetime.now(UTC)
    result = {"answer": "valid independent answer", "not_applicable": marker}
    assert (not is_not_applicable_tool_result(result)) == should_claim
    run = storage.create_agent_run(
        _agent_run("marker-run", AgentRunStatus.RESUME_READY)
    )
    call = storage.create_pending_tool_call(
        replace(
            _pending_call("marker-answer", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result=result,
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id=run.id,
            pending_tool_call_id=call.id,
            resolved_at=now,
        )
    )
    from reflexio.server.services.durable_learning.user_lease import (
        owned_user_work_context,
    )

    ready = storage.get_agent_run(run.id)
    assert ready is not None
    _, descriptor = storage._resume_candidate_page("continuation")
    assert (descriptor is not None) == should_claim
    assert storage.get_agent_run(run.id) == ready
    token = storage.claim_user_extraction("user_1", "marker-selector", 300)
    assert token is not None
    try:
        with owned_user_work_context(storage, "user_1", token, 300):
            position = storage._read_owned_work_position("user_1", token)
            assert position.next_work_class == "finalization"
            final_cursor, final_descriptor = storage._resume_candidate_page(
                "finalization", user_id="user_1", token=token
            )
            assert final_descriptor is None
            selected, empty = storage._commit_owned_work_selection(
                user_id="user_1",
                token=token,
                expected_position=position,
                work_class="finalization",
                cursor=final_cursor,
                worker_id="marker-selector",
            )
            assert selected and empty is None
            position = storage._read_owned_work_position("user_1", token)
            assert position.next_work_class == "continuation"
            cursor, descriptor = storage._resume_candidate_page(
                "continuation", user_id="user_1", token=token
            )
            assert (descriptor is not None) == should_claim
            assert storage.get_agent_run(run.id) == ready
            selected, adopted = storage._commit_owned_work_selection(
                user_id="user_1",
                token=token,
                expected_position=position,
                work_class="continuation",
                cursor=cursor,
                expected_run=ready,
                worker_id="marker-selector",
            )
            assert selected and (adopted is not None) == should_claim
            current = storage.get_agent_run(run.id)
            assert current is not None
            assert current.resume_attempts == ready.resume_attempts
            if should_claim:
                assert adopted is not None and adopted.id == ready.id
                assert current.status == AgentRunStatus.RESUMING
                assert (
                    storage.claim_user_extraction("user_1", "competitor", 300) is None
                )
            else:
                assert current == ready
    finally:
        storage.release_user_extraction("user_1", token)
