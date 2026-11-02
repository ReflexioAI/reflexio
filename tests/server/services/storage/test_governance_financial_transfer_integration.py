"""Authorized erasure must preserve only immutable, pending financial facts."""

import sqlite3

import pytest

from tests.server.services import test_durable_window_pipeline as window_tests

pipeline = window_tests.pipeline
payload = window_tests.payload

pytestmark = pytest.mark.integration


def _published_pending_window(pipeline):
    engine, worker, client = pipeline
    response = client.post(
        "/api/publish_interaction", json=payload("erase-pending-billing", force=True)
    )
    assert response.json()["success"]
    assert worker.drain_org("stream-test", 1, 300) == 1
    storage = engine.get_storage()
    [(window, effects)] = storage.pending_extraction_effects()
    assert effects["billing"]["ids"]
    assert storage.get_user_profile(window.user_id)
    return storage, window, effects["billing"]


def test_authorized_clear_hands_off_actual_pending_window_finances(pipeline):
    storage, window, billing = _published_pending_window(pipeline)
    _, _, client = pipeline
    response = client.post("/api/clear_user_data", json={"user_id": window.user_id})
    assert response.json()["success"]
    assert storage.get_request("erase-pending-billing") is None
    assert storage.get_user_profile(window.user_id) == []
    with storage._stream_sql(read_only=True) as db:
        assert (
            db.query(
                "SELECT window_id FROM extraction_windows WHERE window_id=?",
                (window.window_id,),
            )
            == []
        )
        transferred = db.query(
            "SELECT receipt_id,org_id,project_id,run_id,output_id,row_kind,"
            "entity_type,learning_ids,delivery_owner,delivery_state,delivery_key,"
            "delivery_facts FROM _agent_output_receipts WHERE org_id=? "
            "AND row_kind='window_handoff' AND delivery_key=? LIMIT 2",
            (storage.org_id, window.window_id),
        )
    assert len(transferred) == 1, "Erasure discarded a committed financial obligation"
    from reflexio.server.services.extraction.publication_receipts import (
        PublicationReceipt,
    )

    receipt = PublicationReceipt.from_row(transferred[0])
    assert receipt.run_id is receipt.output_id is None
    assert receipt.delivery_state == "pending"
    assert receipt.entity_type == "profile"
    assert receipt.learning_ids == tuple(billing["ids"])
    assert receipt.financial_facts == {
        "pipeline": "profile",
        "created_at": billing["created_at"],
        "platform_llm": billing["platform_llm"],
        **{
            name: billing.get(name, 0)
            for name in (
                "input_tokens",
                "prompt_tokens",
                "completion_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
            )
        },
    }


def test_direct_delete_cannot_discard_actual_pending_window_finances(pipeline):
    storage, window, _ = _published_pending_window(pipeline)
    with (
        pytest.raises(sqlite3.IntegrityError, match="pending"),
        storage._stream_sql() as db,
    ):
        db.query(
            "DELETE FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )
    assert storage.pending_extraction_effects()


def test_failed_erasure_rolls_back_handoff_and_original_data(pipeline):
    storage, window, _ = _published_pending_window(pipeline)
    storage.conn.execute(
        "CREATE TRIGGER forced_erase_failure BEFORE DELETE ON extraction_windows "
        "BEGIN SELECT RAISE(ABORT,'forced erase rollback'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="forced erase rollback"):
        storage.clear_user_data(window.user_id)
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)
    assert storage.pending_extraction_effects()
    with storage._stream_sql(read_only=True) as db:
        assert (
            db.query(
                "SELECT receipt_id FROM _agent_output_receipts WHERE row_kind='window_handoff' LIMIT 1"
            )
            == []
        )
        assert db.query("SELECT profile_id FROM profiles_fts LIMIT 1")


@pytest.mark.parametrize("changed_field", ["created_at", "input_tokens", "ids"])
def test_malformed_financial_snapshot_refuses_erasure(pipeline, changed_field):
    import json

    storage, window, billing = _published_pending_window(pipeline)
    broken = dict(billing)
    broken[changed_field] = {
        "created_at": None,
        "input_tokens": -1,
        "ids": billing["ids"] * 2,
    }[changed_field]
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_windows SET effects=json_set(effects,'$.billing',json(?)) "
            "WHERE window_id=?",
            (json.dumps(broken), window.window_id),
        )
    with pytest.raises(ValueError, match="Invalid|Duplicate"):
        storage.clear_user_data(window.user_id)
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)
    with storage._stream_sql(read_only=True) as db:
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []
        assert db.query(
            "SELECT window_id FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )


@pytest.mark.parametrize("changed_identity", [False, True])
def test_replace_cannot_bypass_pending_window_guard(pipeline, changed_identity):
    storage, window, _ = _published_pending_window(pipeline)
    with (
        pytest.raises(sqlite3.IntegrityError, match="pending"),
        storage._stream_sql() as db,
    ):
        db.query(
            "INSERT OR REPLACE INTO extraction_windows "
            "(window_id,user_id,kind,project_id,predecessor,end_seq,manifest,policy,completed,effects) "
            "SELECT ?,user_id,kind,project_id,predecessor,end_seq,manifest,policy,completed,effects "
            "FROM extraction_windows WHERE window_id=?",
            (
                "replacement-window" if changed_identity else window.window_id,
                window.window_id,
            ),
        )
    assert storage.pending_extraction_effects()


def test_acknowledged_window_has_no_new_obligation_on_clear(pipeline, monkeypatch):
    from reflexio.server.usage_metrics import exempt_usage_event_recorder

    storage, window, _ = _published_pending_window(pipeline)
    monkeypatch.setattr(
        "reflexio.server.usage_metrics._recorder", exempt_usage_event_recorder
    )
    _, worker, client = pipeline
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert storage.pending_extraction_effects() == []
    assert client.post("/api/clear_user_data", json={"user_id": window.user_id}).json()[
        "success"
    ]
    with storage._stream_sql(read_only=True) as db:
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []
        assert db.query("SELECT window_id FROM extraction_windows LIMIT 1") == []


def test_partial_live_delivery_then_erasure_replays_exact_finances(
    pipeline, monkeypatch
):
    from dataclasses import asdict

    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.services.extraction.publication_receipts import (
        PublicationReceipt,
    )
    from reflexio.server.usage_metrics import (
        UsageEventDeliveryError,
        UsageEventDeliveryStatus,
    )

    storage, window, billing = _published_pending_window(pipeline)
    engine, _, client = pipeline
    accepted, attempts = {}, []
    available = False

    def recorder(event):
        descriptor = asdict(event)
        # These are the erased subject correlation fields, not financial data.
        descriptor.pop("user_id")
        descriptor.pop("request_id")
        attempts.append((event.event_key, descriptor))
        if event.event_key in accepted:
            assert descriptor == accepted[event.event_key]
            return UsageEventDeliveryStatus.DUPLICATE
        if event.event_name == "extraction_tokens" and not available:
            return UsageEventDeliveryStatus.FAILED
        accepted[event.event_key] = descriptor
        return UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr("reflexio.server.usage_metrics._recorder", recorder)
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    with pytest.raises(UsageEventDeliveryError):
        executor._bill(window, billing)
    assert set(accepted) == {
        f"learn:profile:{identifier}" for identifier in billing["ids"]
    }
    assert client.post("/api/clear_user_data", json={"user_id": window.user_id}).json()[
        "success"
    ]
    with storage._stream_sql(read_only=True) as db:
        [row] = db.query(
            "SELECT receipt_id,org_id,project_id,run_id,output_id,row_kind,entity_type,"
            "learning_ids,delivery_owner,delivery_state,delivery_key,delivery_facts "
            "FROM _agent_output_receipts WHERE delivery_key=? LIMIT 2",
            (window.window_id,),
        )
    receipt = PublicationReceipt.from_row(row)
    with pytest.raises(UsageEventDeliveryError):
        _deliver(storage, receipt)
    with storage._stream_sql(read_only=True) as db:
        assert db.query(
            "SELECT delivery_state FROM _agent_output_receipts WHERE receipt_id=?",
            (receipt.receipt_id,),
        ) == [{"delivery_state": "pending"}]
    available = True
    _deliver(storage, receipt)
    assert set(accepted) == {
        *[f"learn:profile:{identifier}" for identifier in billing["ids"]],
        f"tok:{window.window_id}",
    }
    assert all(
        descriptor["created_at"] == billing["created_at"]
        for descriptor in accepted.values()
    )
    assert len(attempts) > len(accepted)
    with storage._stream_sql(read_only=True) as db:
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []
    assert storage.get_request("erase-pending-billing") is None
    assert storage.get_user_profile(window.user_id) == []


@pytest.mark.parametrize("wrong_field", ["org", "ids", "facts"])
def test_handoff_key_without_exact_scoped_finances_cannot_authorize_delete(
    pipeline, wrong_field
):
    import json
    import uuid

    from reflexio.server.services.extraction.publication_receipts import (
        window_handoff_facts,
    )

    storage, window, billing = _published_pending_window(pipeline)
    ids, facts = window_handoff_facts(window.kind, billing)
    org_id = storage.org_id
    if wrong_field == "org":
        org_id = "foreign-org"
    elif wrong_field == "ids":
        ids = ["unrelated-learning"]
    else:
        facts["created_at"] += 1
    with storage._stream_sql() as db:
        db.query(
            "INSERT INTO _agent_output_receipts "
            "(receipt_id,org_id,project_id,row_kind,entity_type,learning_ids,delivery_state,delivery_key,delivery_facts) "
            "VALUES(?,?,'','window_handoff','profile',?,'pending',?,?)",
            (
                uuid.uuid4().hex,
                org_id,
                json.dumps(ids),
                window.window_id,
                json.dumps(facts),
            ),
        )
    with (
        pytest.raises(sqlite3.IntegrityError, match="pending"),
        storage._stream_sql() as db,
    ):
        db.query(
            "DELETE FROM extraction_windows WHERE window_id=?", (window.window_id,)
        )
    assert storage.pending_extraction_effects()


def test_handoff_replay_is_idempotent_and_changed_facts_refuse(pipeline):
    storage, window, _ = _published_pending_window(pipeline)
    with storage._stream_sql() as db:
        storage._handoff_erased_window_finances(db, window.user_id)
        storage._handoff_erased_window_finances(db, window.user_id)
        assert (
            len(
                db.query(
                    "SELECT receipt_id FROM _agent_output_receipts WHERE delivery_key=? LIMIT 2",
                    (window.window_id,),
                )
            )
            == 1
        )
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_windows SET effects=json_set(effects,'$.billing.created_at',1) WHERE window_id=?",
            (window.window_id,),
        )
    with pytest.raises(ValueError, match="handoff changed"):
        storage.clear_user_data(window.user_id)
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)


def test_clear_refuses_borrowed_writer_before_financial_or_source_changes(pipeline):
    from reflexio.server.services.storage.error import StorageError

    storage, window, _ = _published_pending_window(pipeline)
    with storage.commit_scope():
        with pytest.raises(StorageError, match="own transaction"):
            storage.clear_user_data(window.user_id)
        with storage._stream_sql(read_only=True) as db:
            assert (
                db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []
            )
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)
    assert storage.pending_extraction_effects()


@pytest.mark.parametrize("completion_state", [0, 2])
def test_inconsistent_window_state_cannot_drop_existing_billing(
    pipeline, completion_state
):
    storage, window, _ = _published_pending_window(pipeline)
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_windows SET completed=? WHERE window_id=?",
            (completion_state, window.window_id),
        )
    # A stale/malformed completion marker is not evidence of financial absence.
    # The transfer must refuse unknown publication, not invent a new owner.
    with pytest.raises(sqlite3.IntegrityError, match="pending"):
        storage.clear_user_data(window.user_id)
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)
    with storage._stream_sql(read_only=True) as db:
        assert db.query(
            "SELECT completed FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"completed": completion_state}]
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []


def test_actual_unpublished_window_without_finances_can_be_removed(pipeline):
    engine, _, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("unfinished", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("incomplete-control", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    with storage._stream_sql() as db:
        assert db.query(
            "SELECT completed,effects FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"completed": 0, "effects": None}]
        db.query(
            "DELETE FROM extraction_windows WHERE window_id=?", (window.window_id,)
        )
        assert (
            db.query(
                "SELECT window_id FROM extraction_windows WHERE window_id=?",
                (window.window_id,),
            )
            == []
        )
