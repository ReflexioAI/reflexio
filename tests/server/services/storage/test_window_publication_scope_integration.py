"""An original window org is evidence; current context cannot reconstruct it."""

import pytest

from tests.server.services.storage import (
    test_governance_financial_transfer_integration as handoff,
)

pipeline = handoff.pipeline
pytestmark = pytest.mark.integration


def _pre_scope_window(storage, window, org_id):
    # Simulate a predecessor/corrupt persisted row, before the immutable guard.
    # This fixture writes no new financial proof; runtime must refuse it.
    columns = {
        row["name"]
        for row in storage.conn.execute("PRAGMA table_info(extraction_windows)")
    }
    if "org_id" not in columns:
        storage.conn.execute("ALTER TABLE extraction_windows ADD COLUMN org_id TEXT")
    with storage.commit_scope():
        storage.conn.execute("DROP TRIGGER IF EXISTS publication_window_scope_guard")
        storage.conn.execute(
            "UPDATE extraction_windows SET org_id=? WHERE window_id=?",
            (org_id, window.window_id),
        )
    from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
        migrate_publication_receipts,
    )

    migrate_publication_receipts(storage.conn, org_id=storage.org_id)


def test_current_window_persists_actual_storage_org(pipeline):
    storage, window, _ = handoff._published_pending_window(pipeline)
    columns = {
        row["name"]
        for row in storage.conn.execute("PRAGMA table_info(extraction_windows)")
    }
    assert "org_id" in columns, (
        "No durable window org exists for native financial guards"
    )
    with storage._stream_sql(read_only=True) as db:
        assert db.query(
            "SELECT org_id FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"org_id": storage.org_id}]


@pytest.mark.parametrize("original_org", [None, "foreign-org"])
def test_handoff_refuses_missing_or_foreign_original_window_org(pipeline, original_org):
    storage, window, _ = handoff._published_pending_window(pipeline)
    _pre_scope_window(storage, window, original_org)
    with pytest.raises(ValueError, match="scope"):
        storage.clear_user_data(window.user_id)
    assert storage.get_request("erase-pending-billing") is not None
    assert storage.get_user_profile(window.user_id)
    with storage._stream_sql(read_only=True) as db:
        assert db.query(
            "SELECT org_id FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"org_id": original_org}]
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []


def test_old_window_without_org_still_delivers_original_finances(pipeline, monkeypatch):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
    from reflexio.server.usage_metrics import exempt_usage_event_recorder

    storage, window, _ = handoff._published_pending_window(pipeline)
    _pre_scope_window(storage, window, None)
    peer = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    try:
        with peer._stream_sql(read_only=True) as db:
            assert db.query(
                "SELECT org_id FROM extraction_windows WHERE window_id=?",
                (window.window_id,),
            ) == [{"org_id": None}]
    finally:
        peer.conn.close()
    monkeypatch.setattr(
        "reflexio.server.usage_metrics._recorder", exempt_usage_event_recorder
    )
    _, worker, client = pipeline
    assert worker.drain_org(storage.org_id, 1, 300) == 1
    assert storage.pending_extraction_effects() == []
    assert client.post("/api/clear_user_data", json={"user_id": window.user_id}).json()[
        "success"
    ]
    with storage._stream_sql(read_only=True) as db:
        assert db.query("SELECT receipt_id FROM _agent_output_receipts LIMIT 1") == []


@pytest.mark.parametrize("changed_org", [None, "foreign-org"])
def test_current_window_scope_is_immutable(pipeline, changed_org):
    import sqlite3

    storage, window, _ = handoff._published_pending_window(pipeline)
    with (
        pytest.raises(sqlite3.IntegrityError, match="scope is immutable"),
        storage._stream_sql() as db,
    ):
        db.query(
            "UPDATE extraction_windows SET org_id=? WHERE window_id=?",
            (changed_org, window.window_id),
        )
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_windows SET org_id=? WHERE window_id=?",
            (storage.org_id, window.window_id),
        )
        assert db.query(
            "SELECT org_id FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"org_id": storage.org_id}]


def test_legacy_scope_cannot_be_enriched_from_current_context(pipeline):
    import sqlite3

    storage, window, _ = handoff._published_pending_window(pipeline)
    _pre_scope_window(storage, window, None)
    with (
        pytest.raises(sqlite3.IntegrityError, match="scope is immutable"),
        storage._stream_sql() as db,
    ):
        db.query(
            "UPDATE extraction_windows SET org_id=? WHERE window_id=?",
            (storage.org_id, window.window_id),
        )
    with storage._stream_sql(read_only=True) as db:
        assert db.query(
            "SELECT org_id FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        ) == [{"org_id": None}]
