"""Financial retry must outlive erased raw inputs and execution ownership."""

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
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


@pytest.mark.parametrize("pending_tools_enabled", [False, True])
def test_admitted_turn_delivers_erased_output_debt_without_user_work(
    publication, monkeypatch, pending_tools_enabled
):
    storage, resume_worker, _, _, _ = publication
    service, request, model_calls = initial._initial(publication, monkeypatch)
    configure_usage_event_recorder(
        lambda event: (
            UsageEventDeliveryStatus.FAILED
            if event.event_name == "learnings_generated"
            else UsageEventDeliveryStatus.APPENDED
        )
    )
    initial._run_initial(service, request)
    run = service._last_extraction_run
    assert run is not None and run.binding.user_id is not None
    assert model_calls == [1] and run.committed_output is not None
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(
            db, run.id, run.committed_output["publication"]["id"]
        )
    assert receipt is not None and receipt.delivery_state == "pending"
    storage.clear_user_data(run.binding.user_id)
    assert storage.get_agent_run(run.id) is None
    assert storage.conn.execute("SELECT COUNT(*) FROM learning_work").fetchone()[0] == 0
    context = resume_worker.request_context
    context.configurator.get_config().pending_tool_call_config.enabled = (
        pending_tools_enabled
    )
    presence = storage._deferred_work_presence()
    assert presence.financial and not presence.resume and not presence.expiry
    from reflexio.server.services.durable_learning import local
    from reflexio.server.services.durable_learning import scheduler as scheduling

    monkeypatch.setattr(
        scheduling.DurableLearningScheduler, "start", lambda _self: None
    )
    monkeypatch.setattr(local, "adopt_server_scheduler", lambda _scheduler: None)
    scheduler = scheduling.maybe_start_durable_learning(
        lambda _org: context, bootstrap_org_id=storage.org_id
    )
    assert list(scheduler._org_ids_provider()) == [storage.org_id]
    accepted = []

    def accept(event):
        if event.event_name == "learnings_generated":
            accepted.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(accept)
    worker = DurableLearningWorker(lambda _org_id: context)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert not storage._deferred_work_presence().financial
    assert len(accepted) == 1, (
        "Admitted turns must deliver debt without erased user work"
    )
    assert model_calls == [1]
    assert storage.get_agent_run(run.id) is None
    assert storage.conn.execute("SELECT COUNT(*) FROM learning_work").fetchone()[0] == 0
    with storage._stream_sql(read_only=True) as db:
        assert (
            db.query(
                "SELECT receipt_id FROM _agent_output_receipts WHERE receipt_id=?",
                (receipt.receipt_id,),
            )
            == []
        )


def test_failed_head_advances_across_new_clients_and_wraps_finite_ceiling(
    storage, monkeypatch
):
    import json
    from unittest.mock import patch

    from reflexio.server import usage_metrics
    from reflexio.server.services.durable_learning.financial_delivery import (
        FinancialDeliverySweep,
    )
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    sweep = FinancialDeliverySweep()
    facts = {
        "pipeline": "profile",
        "created_at": 1700000000,
        "platform_llm": False,
        "input_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }

    def insert(identity, position):
        with storage.commit_scope():
            storage.conn.execute(
                "INSERT INTO _agent_output_receipts(rowid,receipt_id,org_id,row_kind,delivery_key,entity_type,learning_ids,delivery_state,delivery_facts) VALUES (?,?,?,'window_handoff',?,'profile',?,'pending',?)",
                (
                    position,
                    identity,
                    storage.org_id,
                    identity,
                    json.dumps([identity]),
                    json.dumps(facts),
                ),
            )

    head, later, arrival = "f" * 32, "8" * 32, "1" * 32
    insert(head, -1)
    insert(later, 0)
    attempts = []

    def record(event):
        if event.event_name == "learnings_generated":
            attempts.append(event)
            return (
                UsageEventDeliveryStatus.FAILED
                if event.entity_id == head
                else UsageEventDeliveryStatus.APPENDED
            )
        return UsageEventDeliveryStatus.EXEMPT

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    assert not sweep.deliver_one(storage, storage.org_id)
    insert(arrival, 1)  # Newer debt does not extend the current ceiling=0.
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        peer = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    try:
        assert sweep.deliver_one(peer, storage.org_id)
        assert not sweep.deliver_one(peer, storage.org_id)  # Ends finite sweep.
        assert not sweep.deliver_one(
            peer, storage.org_id
        )  # Failed head retried next sweep.
        assert sweep.deliver_one(peer, storage.org_id)
    finally:
        peer.conn.close()
    assert [event.entity_id for event in attempts] == [head, later, head, arrival]
    assert (
        storage.conn.execute(
            "SELECT receipt_id FROM _agent_output_receipts"
        ).fetchall()[0][0]
        == head
    )


def test_concurrent_inspection_cas_does_not_hold_state_lock_over_io(
    storage, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from reflexio.server.services.durable_learning.financial_delivery import (
        FinancialDeliverySweep,
    )

    sweep = FinancialDeliverySweep()
    gate = Barrier(2)
    calls = []

    def inspect(**kwargs):
        # Two inspectors must both enter I/O; holding the state lock deadlocks.
        gate.wait(timeout=5)
        calls.append(kwargs)
        return None, None

    monkeypatch.setattr(storage, "_publication_delivery_page", inspect)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _n: sweep.deliver_one(storage, storage.org_id), range(2)
            )
        )
    assert results == [False, False] and len(calls) == 2
    assert sweep._positions[storage.org_id].generation == 1


@pytest.mark.parametrize("failure_stage", ["selection", "delivery", "recorder"])
def test_financial_retry_keeps_debt_and_sanitizes_exception_causes(
    publication, monkeypatch, caplog, failure_stage
):
    from reflexio.server.services.durable_learning import financial_delivery

    storage, _, _, _, _ = publication
    service, request, calls = initial._initial(publication, monkeypatch)
    configure_usage_event_recorder(lambda _event: UsageEventDeliveryStatus.FAILED)
    initial._run_initial(service, request)
    run = service._last_extraction_run
    assert run is not None and run.committed_output is not None
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(
            db, run.id, run.committed_output["publication"]["id"]
        )
    assert receipt is not None and receipt.delivery_state == "pending"
    private = "SENTINEL_PRIVATE_FINANCIAL_VALUE_AND_CREDENTIAL"

    def unavailable(*_args, **_kwargs):
        try:
            raise ValueError(private)
        except ValueError as cause:
            raise RuntimeError(private) from cause

    target = storage if failure_stage == "selection" else financial_delivery
    attribute = (
        "_publication_delivery_page" if failure_stage == "selection" else "_deliver"
    )
    sweep = financial_delivery.FinancialDeliverySweep()
    with monkeypatch.context() as patcher:
        if failure_stage == "recorder":
            configure_usage_event_recorder(unavailable)
        else:
            patcher.setattr(target, attribute, unavailable)
        assert not sweep.deliver_one(storage, storage.org_id)
    assert (
        storage._get_publication_receipt(
            receipt.receipt_id, project_id=receipt.project_id
        ).delivery_state
        == "pending"
    )
    records = [
        record
        for record in caplog.records
        if record.name == financial_delivery.__name__
    ]
    assert records and all(record.exc_info is None for record in records)
    expected_type = (
        "ReceiptDeliveryError" if failure_stage == "recorder" else "RuntimeError"
    )
    assert any(f"error={expected_type} at=" in record.message for record in records)
    assert private not in caplog.text
    accepted = []

    def accept(event):
        accepted.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(accept)
    # A delivery failure advances the bounded cursor; reset at the end of its sweep.
    assert any(sweep.deliver_one(storage, storage.org_id) for _ in range(2))
    assert (
        storage._get_publication_receipt(
            receipt.receipt_id, project_id=receipt.project_id
        ).delivery_state
        == "accepted"
    )
    assert len(accepted) == 1 and calls == [1]
