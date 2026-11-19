"""Compatibility window closure is proved by the leased effects ACK."""

import json

import pytest

from tests.server.services.storage.test_transition_inventory_integration import (
    seed_run,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "case",
    [
        "owned",
        "unfenced",
        "historical_compacted",
        "no_derived_claim",
        "active_cursor",
        "foreign_active_cursor",
        "corrupt_ack",
        "corrupt_manifest",
        "fractional_predecessor",
        "fractional_end",
        "nul_digest",
        "nul_stage",
        "nul_ack",
        "adapter_without_proof",
        "wrong_effect_run",
        "foreign_window",
        "invalidated",
        "retired",
        "fresh_extracted",
        "unknown_version",
        "mismatched_output",
    ],
)
def test_compatibility_window_ack_proves_only_its_fenced_legacy_parent(
    storage, case, monkeypatch
):
    if case == "adapter_without_proof":
        from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
            SQLiteStreamSQL,
        )

        monkeypatch.delattr(SQLiteStreamSQL, "transition_window_publication_proof")
    window_id = "compatibility-window"
    run_id = f"window:{window_id}"
    output = {"_reflexio_envelope_version": 1, "output": {}}
    if case == "mismatched_output":
        output["publication"] = {"id": "wrong-output"}
    stage = {"retired": "retired", "fresh_extracted": "extracted"}.get(
        case, "recovered"
    )
    if case == "nul_stage":
        stage += "\x00extra"
    seed_run(
        storage,
        run_id,
        status="finalized",
        snapshot={
            "overflow": {
                "version": 2 if case == "unknown_version" else 1,
                "checkpoint": {"stage": stage},
                "source_digest": "a" * 64
                + ("\x00extra" if case == "nul_digest" else ""),
            },
        },
        output=output,
    )
    effects = {
        "extraction_run_ids": ["wrong-run" if case == "wrong_effect_run" else run_id]
    }
    if case == "historical_compacted":
        effects = {"version": 1, "skipped": True}
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO learning_work(org_id,user_id,highwater,pending_since) VALUES(?, 'user', 2, 0)",
            (storage.org_id,),
        )
        storage.conn.execute(
            "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,"
            "manifest,policy,completed,completed_at,effects,effects_done,derived_claimed,invalidated) "
            "VALUES(?, 'user', 'profile', ?, 0, 2, '[]', '{}', 1, 1, ?, ?, ?, ?)",
            (
                window_id,
                "foreign" if case == "foreign_window" else storage.org_id,
                json.dumps(effects),
                int(case == "historical_compacted"),
                int(case != "no_derived_claim"),
                int(case == "invalidated"),
            ),
        )
        if case in {"active_cursor", "foreign_active_cursor"}:
            storage.conn.execute(
                "INSERT INTO extraction_cursors(user_id,kind,window_id) VALUES(?,'profile',?)",
                ("user" if case == "active_cursor" else "another-user", window_id),
            )
    if case == "unfenced":
        storage.ack_extraction_effects(window_id)
    else:
        token = storage.claim_user_extraction("user", "ack-proof-test", 30)
        assert token
        storage.ack_extraction_effects(window_id, user_id="user", token=token)
        storage.release_user_extraction("user", token)
    if case == "nul_ack":
        with storage.commit_scope():
            storage.conn.execute(
                "UPDATE extraction_windows SET effects=json_set(effects, "
                "'$.acknowledged_window_run_id', ?) WHERE window_id=?",
                (run_id + "\x00extra", window_id),
            )
    if case in {"corrupt_ack", "corrupt_manifest"}:
        query = (
            "UPDATE extraction_windows SET effects=? WHERE window_id=?"
            if case == "corrupt_ack"
            else "UPDATE extraction_windows SET manifest=? WHERE window_id=?"
        )
        with storage.commit_scope():
            storage.conn.execute(query, ("invalid JSON", window_id))
    if case in {"fractional_predecessor", "fractional_end"}:
        query = (
            "UPDATE extraction_windows SET predecessor=0.5 WHERE window_id=?"
            if case == "fractional_predecessor"
            else "UPDATE extraction_windows SET end_seq=2.5 WHERE window_id=?"
        )
        with storage.commit_scope():
            storage.conn.execute(query, (window_id,))
    run = storage.transition_inventory_page("runs").items[0]
    assert ("unreceipted_output" not in run.reasons) == (case == "owned")
    if case == "owned":
        assert run.reasons == ()
