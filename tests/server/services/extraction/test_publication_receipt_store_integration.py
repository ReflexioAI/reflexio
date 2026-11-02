"""Real SQLite ownership, receipt persistence and financial state transitions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import replace
from unittest.mock import patch

import pytest

from reflexio.server.services.durable_learning.user_lease import (
    create_owned_agent_run,
    owned_agent_run_write,
    user_extraction_lease,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    LeaseLostError,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def execution(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="receipt_org", db_path=str(tmp_path / "receipts.db")
        )
        with user_extraction_lease(storage, "receipt_user"):
            run = create_owned_agent_run(
                storage,
                AgentRunRecord(
                    id="receipt_run",
                    binding=AgentBinding(
                        org_id=storage.org_id,
                        extractor_kind="profile",
                        user_id="receipt_user",
                        request_id="request",
                        agent_version="v1",
                        source="api",
                    ),
                    status=AgentRunStatus.AGENT_COMPLETED,
                    generation_request_snapshot={},
                    committed_output={
                        "_reflexio_envelope_version": 1,
                        "output": {"profiles": []},
                        "publication": {"id": uuid.uuid4().hex},
                    },
                ),
            )
            yield storage, run
        storage.conn.close()


def _rows(storage):
    return [
        dict(row)
        for row in storage.conn.execute(
            "SELECT rowid AS delivery_order,receipt_id,run_id,output_id,learning_ids,delivery_state,delivery_facts FROM _agent_output_receipts"
        )
    ]


@pytest.mark.parametrize("ids", [[], ["profile_one"]])
def test_exact_output_receipt_and_idempotent_replay(execution, ids):
    storage, run = execution
    receipt, created = storage._create_agent_output_receipt(
        run, learning_ids=ids, origin="resume"
    )
    assert created and list(receipt.learning_ids) == ids
    assert receipt.delivery_state == ("pending" if ids else "accepted")
    assert (
        storage.get_agent_output_finalization_receipt(
            run_id=run.id, output_id=receipt.output_id, entity_type="profile"
        )
        == ids
    )
    assert (
        storage.get_agent_run_finalization_receipt(run_id=run.id, entity_type="profile")
        == ids
    )
    replay, created = storage._create_agent_output_receipt(
        run, learning_ids=ids, origin="resume"
    )
    assert not created and replay == receipt and len(_rows(storage)) == 1
    if ids:
        assert run.created_at is not None
        assert receipt.financial_facts == {
            "pipeline": "profile",
            "source": "resumable_extraction",
            "created_at": run.created_at.timestamp(),
        }
        copy = receipt.financial_facts
        copy["created_at"] = 0
        assert receipt.financial_facts["created_at"] != 0


@pytest.mark.parametrize("prior_ids", [[], ["prior_profile"]])
def test_second_output_does_not_reuse_first_output_receipt(execution, prior_ids):
    storage, run = execution
    first, _ = storage._create_agent_output_receipt(
        run, learning_ids=prior_ids, origin="resume"
    )
    payload = {**run.committed_output, "publication": {"id": uuid.uuid4().hex}}
    with owned_agent_run_write(storage, run):
        second_run = storage.update_agent_run_status(
            run.id,
            AgentRunStatus.AGENT_COMPLETED,
            committed_output=payload,
            expected_claimed_by=run.claimed_by,
            expected_claimed_at=run.claimed_at,
        )
    assert second_run is not None
    assert (
        storage.get_agent_run_finalization_receipt(run_id=run.id, entity_type="profile")
        is None
    )
    between = storage.get_agent_run_recovery_facts(run.id)
    assert not between["receipt_exists"]
    assert between["receipt_count"] is None
    assert between["receipt_missing_results"] == 0
    second, created = storage._create_agent_output_receipt(
        second_run, learning_ids=["profile_two"], origin="resume"
    )
    assert created and second.receipt_id != first.receipt_id
    assert (
        storage.get_agent_output_finalization_receipt(
            run_id=run.id, output_id=first.output_id, entity_type="profile"
        )
        == prior_ids
    )
    assert storage.get_agent_run_finalization_receipt(
        run_id=run.id, entity_type="profile"
    ) == ["profile_two"]
    assert len(_rows(storage)) == 2
    current = storage.get_agent_run_recovery_facts(run.id)
    assert current["receipt_exists"]
    assert current["receipt_count"] == 1
    assert current["receipt_missing_results"] == 1


def test_modern_output_never_falls_back_to_historical_run_receipt(execution):
    storage, run = execution
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_run_finalization_receipts(run_id,entity_type,learning_ids) VALUES(?,?,?)",
            (run.id, "profile", '["historical_profile"]'),
        )
    assert (
        storage.get_agent_run_finalization_receipt(run_id=run.id, entity_type="profile")
        is None
    )
    assert not storage.get_agent_run_recovery_facts(run.id)["receipt_exists"]
    receipt, created = storage._create_agent_output_receipt(
        run, learning_ids=[], origin="resume"
    )
    assert created and receipt.delivery_state == "accepted"
    assert (
        storage.get_agent_run_finalization_receipt(run_id=run.id, entity_type="profile")
        == []
    )
    assert (
        storage.conn.execute(
            "SELECT learning_ids FROM _agent_run_finalization_receipts WHERE run_id=?",
            (run.id,),
        ).fetchone()[0]
        == '["historical_profile"]'
    )


def test_receipt_and_progress_join_outer_rollback(execution):
    storage, run = execution
    before = storage.get_agent_run(run.id)
    with (
        pytest.raises(RuntimeError, match="publication rollback"),
        owned_agent_run_write(storage, run),
    ):
        storage._create_agent_output_receipt(
            run, learning_ids=["profile_one"], origin="resume"
        )
        raise RuntimeError("publication rollback")
    assert _rows(storage) == [] and storage.get_agent_run(run.id) == before


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE _agent_output_receipts SET org_id='foreign'",
        "UPDATE _agent_output_receipts SET rowid=rowid+100",
        "UPDATE _agent_output_receipts SET learning_ids='[]'",
        "UPDATE _agent_output_receipts SET delivery_facts=NULL",
        "UPDATE _agent_output_receipts SET output_id='00000000000000000000000000000000'",
    ],
)
def test_pending_receipt_financial_identity_is_immutable(execution, statement):
    storage, run = execution
    storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    before = _rows(storage)
    with pytest.raises(sqlite3.IntegrityError), storage.commit_scope():
        storage.conn.execute(statement)
    assert _rows(storage) == before


def test_pending_deletion_refuses_whole_parent_delete(execution):
    storage, run = execution
    storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    before = storage.get_agent_run(run.id)
    with (
        pytest.raises(sqlite3.IntegrityError, match="pending financial"),
        storage.commit_scope(),
    ):
        storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (run.id,))
    assert storage.get_agent_run(run.id) == before and len(_rows(storage)) == 1


def test_strict_acceptance_cleanup_and_parent_cascade(execution):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    assert storage._ack_publication_receipt(receipt)
    row = _rows(storage)[0]
    assert row["delivery_state"] == "accepted" and row["delivery_facts"] is None
    assert json.loads(row["learning_ids"]) == ["profile_one"]
    assert storage._ack_publication_receipt(receipt)
    with storage.commit_scope():
        storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (run.id,))
    assert _rows(storage) == []


def test_erase_unlink_preserves_debt_until_exact_ack(execution):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    with storage.commit_scope(), storage._stream_sql() as db:
        storage._handoff_erased_output_finances(db, run.binding.user_id)
        storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (run.id,))
    row = _rows(storage)[0]
    assert (
        row["run_id"] is None
        and row["delivery_state"] == "pending"
        and row["delivery_facts"] is not None
    )
    wrong = replace(
        receipt,
        _facts_json=json.dumps(
            {"pipeline": "profile", "source": "wrong", "created_at": 0}
        ),
    )
    assert not storage._ack_publication_receipt(wrong)
    assert len(_rows(storage)) == 1
    assert storage._ack_publication_receipt(receipt)
    assert _rows(storage) == []


@pytest.mark.parametrize("field", ["request", "clock"])
def test_copied_execution_cannot_rewrite_publication_provenance(execution, field):
    from datetime import UTC, datetime

    storage, run = execution
    fake = (
        replace(run, binding=replace(run.binding, request_id="forged"))
        if field == "request"
        else replace(run, created_at=datetime(2000, 1, 1, tzinfo=UTC))
    )
    with pytest.raises(LeaseLostError):
        storage._create_agent_output_receipt(
            fake, learning_ids=["profile_one"], origin="resume"
        )
    assert _rows(storage) == []


_OLD_TABLE = """CREATE TABLE _agent_run_finalization_receipts (
 run_id TEXT PRIMARY KEY,entity_type TEXT NOT NULL,learning_ids TEXT NOT NULL,
 created_at TEXT NOT NULL,FOREIGN KEY(run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE)"""


def test_upgrade_preserves_historical_marker_without_creating_ack_or_debt(tmp_path):
    filename = str(tmp_path / "historical.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        original = SQLiteStorage(org_id="old_org", db_path=filename)
        original.create_agent_run(
            AgentRunRecord(
                id="old_run",
                binding=AgentBinding(
                    org_id="old_org",
                    extractor_kind="profile",
                    user_id="old_user",
                    request_id="old_request",
                    agent_version=None,
                    source=None,
                ),
                status=AgentRunStatus.FINALIZED,
                generation_request_snapshot={},
                committed_output={"profiles": []},
            )
        )
        original.conn.close()
        with sqlite3.connect(filename) as conn:
            conn.execute("DROP TABLE _agent_run_finalization_receipts")
            conn.execute(_OLD_TABLE)
            conn.execute(
                "INSERT INTO _agent_run_finalization_receipts VALUES(?,?,?,?)",
                ("old_run", "profile", '["historical_id"]', "2020-01-02T03:04:05Z"),
            )
        upgraded = SQLiteStorage(org_id="old_org", db_path=filename)
        row = dict(
            upgraded.conn.execute(
                "SELECT * FROM _agent_run_finalization_receipts"
            ).fetchone()
        )
        assert set(row) == {"run_id", "entity_type", "learning_ids", "created_at"}
        assert _rows(upgraded) == []  # History was neither copied nor made debt.
        assert (
            row["created_at"] == "2020-01-02T03:04:05Z"
            and row["learning_ids"] == '["historical_id"]'
        )
        assert upgraded.get_agent_run_finalization_receipt(
            run_id="old_run", entity_type="profile"
        ) == ["historical_id"]
        upgraded.conn.close()
        reopened = SQLiteStorage(org_id="old_org", db_path=filename)
        assert (
            dict(
                reopened.conn.execute(
                    "SELECT * FROM _agent_run_finalization_receipts"
                ).fetchone()
            )
            == row
        )
        with reopened.commit_scope():
            reopened.conn.execute("DELETE FROM _agent_runs WHERE id='old_run'")
        assert _rows(reopened) == []
        reopened.conn.close()


def test_upgrade_refuses_unknown_owner_without_dropping_old_receipt(tmp_path):
    from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
        migrate_publication_receipts,
    )

    with sqlite3.connect(tmp_path / "unknown.db") as conn:
        conn.execute("CREATE TABLE _agent_runs(id TEXT PRIMARY KEY,org_id TEXT)")
        conn.execute(_OLD_TABLE)
        conn.execute(
            "INSERT INTO _agent_run_finalization_receipts VALUES('missing','profile','[]','2020-01-01')"
        )
        conn.commit()
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='_agent_run_finalization_receipts'"
        ).fetchone()[0]
        with pytest.raises(ValueError, match="owner is unavailable"):
            migrate_publication_receipts(conn, org_id="org")
        assert (
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='_agent_run_finalization_receipts'"
            ).fetchone()[0]
            == before
        )
        assert conn.execute(
            "SELECT * FROM _agent_run_finalization_receipts"
        ).fetchall() == [("missing", "profile", "[]", "2020-01-01")]


@pytest.mark.parametrize(
    "bad",
    [
        "accepted_without_delivery",
        "duplicate_ids",
        "extra_source_body",
        "nan_clock",
        "boolean_clock",
        "huge_clock",
    ],
)
def test_direct_insert_cannot_forge_financial_state_or_payload(execution, bad):
    storage, run = execution
    ids = ["profile_one"]
    state = "pending"
    facts = {"pipeline": "profile", "source": "resumable_extraction", "created_at": 1}
    if bad == "accepted_without_delivery":
        state = "accepted"
        facts = None
    elif bad == "duplicate_ids":
        ids *= 2
    elif bad == "extra_source_body":
        facts["request_body"] = "must not be retained"
    elif bad == "nan_clock":
        facts["created_at"] = float("nan")
    elif bad == "huge_clock":
        facts["created_at"] = 10**400
    else:
        facts["created_at"] = True
    with pytest.raises(sqlite3.IntegrityError), storage.commit_scope():
        storage.conn.execute(
            """INSERT INTO _agent_output_receipts
          (org_id,run_id,output_id,entity_type,learning_ids,delivery_state,delivery_facts)
          VALUES(?,?,?,?,?,?,?)""",
            (
                storage.org_id,
                run.id,
                run.committed_output["publication"]["id"],
                "profile",
                json.dumps(ids),
                state,
                json.dumps(facts) if facts is not None else None,
            ),
        )
    assert _rows(storage) == []


@pytest.mark.parametrize("collision", ["receipt_id", "output_id"])
def test_replace_cannot_destroy_pending_financial_obligation(execution, collision):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    replacement_id = uuid.uuid4().hex
    output_id = receipt.output_id
    if collision == "receipt_id":
        replacement_id = receipt.receipt_id
        output_id = uuid.uuid4().hex
        with owned_agent_run_write(storage, run):
            storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output={
                    **run.committed_output,
                    "publication": {"id": output_id},
                },
                expected_claimed_by=run.claimed_by,
                expected_claimed_at=run.claimed_at,
            )
    before = _rows(storage)
    with pytest.raises(sqlite3.IntegrityError), storage.commit_scope():
        storage.conn.execute(
            """INSERT OR REPLACE INTO _agent_output_receipts
         (receipt_id,org_id,run_id,output_id,entity_type,learning_ids,delivery_state,delivery_facts)
         VALUES(?,?,?,?,?,?,?,?)""",
            (
                replacement_id,
                storage.org_id,
                run.id,
                output_id,
                "profile",
                json.dumps(["replacement"]),
                "pending",
                json.dumps(receipt.financial_facts),
            ),
        )
    assert _rows(storage) == before


def test_concurrent_receipt_install_preserves_new_pending_output(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
        migrate_publication_receipts,
    )

    path = str(tmp_path / "concurrent-receipts.db")
    output_id = uuid.uuid4().hex
    receipt_id = uuid.uuid4().hex
    facts = json.dumps({"pipeline": "profile", "source": "api", "created_at": 1})
    with sqlite3.connect(path) as setup:
        setup.execute(
            "CREATE TABLE _agent_runs(id TEXT PRIMARY KEY,org_id TEXT,status TEXT,extractor_kind TEXT,committed_output TEXT)"
        )
        setup.execute(
            "CREATE TABLE _agent_run_finalization_receipts(run_id TEXT PRIMARY KEY,entity_type TEXT,learning_ids TEXT,created_at TEXT)"
        )
    barrier = Barrier(2)
    lock = Lock()
    writers = []

    class SynchronizedWriter(sqlite3.Connection):
        first_writer = False

        def execute(self, sql, parameters=()):
            if sql == "BEGIN IMMEDIATE":
                barrier.wait(timeout=10)
                result = super().execute(sql, parameters)
                with lock:
                    self.first_writer = not writers
                    writers.append(True)
                return result
            return super().execute(sql, parameters)

        def commit(self):
            if self.first_writer:
                self.first_writer = False
                # The new writer starts only after empty legacy preflight/DDL.
                self.execute(
                    "INSERT INTO _agent_runs VALUES(?,?,?,?,?)",
                    (
                        "run",
                        "org",
                        "agent_completed",
                        "profile",
                        json.dumps({"publication": {"id": output_id}}),
                    ),
                )
                self.execute(
                    """INSERT INTO _agent_output_receipts
                  (receipt_id,org_id,run_id,output_id,entity_type,learning_ids,delivery_state,delivery_facts)
                  VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        receipt_id,
                        "org",
                        "run",
                        output_id,
                        "profile",
                        json.dumps(["profile_one"]),
                        "pending",
                        facts,
                    ),
                )
            return super().commit()

    def install():
        conn = sqlite3.connect(path, timeout=10, factory=SynchronizedWriter)
        try:
            migrate_publication_receipts(conn, org_id="org")
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(install) for _ in range(2)]
        for result in results:
            result.result(timeout=15)
    with sqlite3.connect(path) as observed:
        assert observed.execute(
            "SELECT receipt_id,output_id,row_kind,delivery_state,delivery_facts FROM _agent_output_receipts"
        ).fetchall() == [(receipt_id, output_id, "output", "pending", facts)]


@pytest.mark.parametrize("mode", ["missing", "changed_definition"])
@pytest.mark.parametrize(
    ("name", "drop", "weaken"),
    [
        (
            "publication_receipt_delete_guard",
            "DROP TRIGGER publication_receipt_delete_guard",
            "CREATE TRIGGER publication_receipt_delete_guard BEFORE DELETE ON _agent_output_receipts BEGIN SELECT 1; END",
        ),
        (
            "publication_receipt_insert_guard",
            "DROP TRIGGER publication_receipt_insert_guard",
            "CREATE TRIGGER publication_receipt_insert_guard BEFORE INSERT ON _agent_output_receipts BEGIN SELECT 1; END",
        ),
        (
            "publication_window_scope_guard",
            "DROP TRIGGER publication_window_scope_guard",
            "CREATE TRIGGER publication_window_scope_guard BEFORE UPDATE OF org_id ON extraction_windows BEGIN SELECT 1; END",
        ),
        (
            "publication_window_delete_guard",
            "DROP TRIGGER publication_window_delete_guard",
            "CREATE TRIGGER publication_window_delete_guard BEFORE DELETE ON extraction_windows BEGIN SELECT 1; END",
        ),
        (
            "publication_window_replace_guard",
            "DROP TRIGGER publication_window_replace_guard",
            "CREATE TRIGGER publication_window_replace_guard BEFORE INSERT ON extraction_windows BEGIN SELECT 1; END",
        ),
        (
            "publication_output_identity",
            "DROP INDEX publication_output_identity",
            "CREATE INDEX publication_output_identity ON _agent_output_receipts(output_id)",
        ),
        (
            "publication_pending_delivery",
            "DROP INDEX publication_pending_delivery",
            "CREATE INDEX publication_pending_delivery ON _agent_output_receipts(org_id,project_id,receipt_id) WHERE delivery_state='pending'",
        ),
    ],
)
def test_install_repairs_missing_or_weakened_guard_without_rewriting_debt(
    execution, mode, name, drop, weaken
):
    from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
        migrate_publication_receipts,
    )

    storage, run = execution
    storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    before = _rows(storage)
    expected = storage.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name=?", (name,)
    ).fetchone()[0]
    with storage.commit_scope():
        storage.conn.execute(drop)
        if mode == "changed_definition":
            storage.conn.execute(weaken)
    migrate_publication_receipts(storage.conn, org_id=storage.org_id)
    assert (
        storage.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()[0]
        == expected
    )
    assert _rows(storage) == before


@pytest.mark.parametrize("abort", [False, True])
def test_output_financial_handoff_joins_the_erasure_writer(execution, abort):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["profile_one"], origin="resume"
    )
    before = _rows(storage)
    with storage.commit_scope(), storage._stream_sql() as db:
        storage._handoff_erased_output_finances(db, "another-user")
    assert _rows(storage) == before

    class AbortErasureError(Exception):
        pass

    def erase():
        with storage.commit_scope(), storage._stream_sql() as db:
            storage._handoff_erased_output_finances(db, run.binding.user_id)
            # Repeat transfer before deletion: it must not add or change debt.
            storage._handoff_erased_output_finances(db, run.binding.user_id)
            storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (run.id,))
            if abort:
                raise AbortErasureError

    if abort:
        with pytest.raises(AbortErasureError):
            erase()
        assert _rows(storage) == before
        assert storage.get_agent_run(run.id) is not None
    else:
        erase()
        assert storage.get_agent_run(run.id) is None
        row = _rows(storage)[0]
        assert row == {**before[0], "run_id": None, "output_id": None}
        assert storage._ack_publication_receipt(receipt)
        assert _rows(storage) == []


def test_receipt_metadata_page_reads_actual_owned_output_without_facts(execution):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["metadata-learning"], origin="resume"
    )
    ceiling, item = storage._publication_delivery_page()
    assert item is not None and item == {
        "delivery_order": ceiling,
        "receipt_id": receipt.receipt_id,
        "org_id": storage.org_id,
        "project_id": "",
    }
    assert storage._ack_publication_receipt(receipt)
    assert storage._publication_delivery_page() == (None, None)


def test_receipt_metadata_page_bounds_arrivals_and_preserves_failed_position(execution):
    """Real guarded SQL queue; this is selection, not publisher/ledger evidence."""
    storage, _ = execution
    facts = json.dumps(
        {
            "pipeline": "profile",
            "created_at": 1700000000,
            "platform_llm": False,
            "input_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        }
    )

    def insert(identity):
        with storage.commit_scope():
            storage.conn.execute(
                "INSERT INTO _agent_output_receipts(receipt_id,org_id,row_kind,delivery_key,entity_type,learning_ids,delivery_state,delivery_facts) VALUES (?,?,'window_handoff',?,'profile',?,'pending',?)",
                (
                    identity,
                    storage.org_id,
                    identity,
                    json.dumps(["learning-" + identity]),
                    facts,
                ),
            )

    insert("8" * 32)
    insert("f" * 32)
    ceiling, first = storage._publication_delivery_page()
    assert first is not None and first["receipt_id"] == "8" * 32
    insert("1" * 32)  # Lower UUID, newer position.
    same, second = storage._publication_delivery_page(
        after=first["delivery_order"], through=ceiling
    )
    assert same == ceiling and second is not None and second["receipt_id"] == "f" * 32
    assert storage._publication_delivery_page(
        after=second["delivery_order"], through=ceiling
    ) == (ceiling, None)
    _, new = storage._publication_delivery_page(after=ceiling)
    assert new is not None and new["receipt_id"] == "1" * 32
    assert (
        storage.conn.execute(
            "SELECT count(*) FROM _agent_output_receipts WHERE delivery_state='pending'"
        ).fetchone()[0]
        == 3
    )


def test_receipt_metadata_page_refuses_active_commit_scope(execution):
    storage, _ = execution
    with storage.commit_scope():
        with pytest.raises(ValueError, match="independent writer"):
            storage._publication_delivery_page()
        assert storage.in_commit_scope
        assert storage.conn.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize("position", [-1, 0])
def test_receipt_metadata_does_not_confuse_sqlite_position_with_empty(
    execution, position
):
    storage, _ = execution
    identity = uuid.uuid4().hex
    facts = json.dumps(
        {
            "pipeline": "profile",
            "created_at": 1700000000,
            "platform_llm": False,
            "input_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        }
    )
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_output_receipts(rowid,receipt_id,org_id,row_kind,delivery_key,entity_type,learning_ids,delivery_state,delivery_facts) VALUES (?,?,?,'window_handoff',?,'profile',?,'pending',?)",
            (
                position,
                identity,
                storage.org_id,
                identity,
                json.dumps(["learning"]),
                facts,
            ),
        )
    ceiling, item = storage._publication_delivery_page()
    assert ceiling == position and item is not None
    assert item["delivery_order"] == position and item["receipt_id"] == identity


def test_receipt_metadata_refuses_unmanaged_sqlite_transaction(execution):
    storage, _ = execution
    storage.conn.execute("BEGIN IMMEDIATE")
    try:
        identity = uuid.uuid4().hex
        facts = json.dumps(
            {
                "pipeline": "profile",
                "created_at": 1700000000,
                "platform_llm": False,
                "input_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
            }
        )
        storage.conn.execute(
            "INSERT INTO _agent_output_receipts(receipt_id,org_id,row_kind,delivery_key,entity_type,learning_ids,delivery_state,delivery_facts) VALUES (?,?,'window_handoff',?,'profile',?,'pending',?)",
            (
                identity,
                storage.org_id,
                identity,
                json.dumps(["uncommitted-learning"]),
                facts,
            ),
        )
        assert storage.conn.in_transaction and not storage.in_commit_scope
        with pytest.raises(ValueError, match="independent writer"):
            storage._publication_delivery_page()
        assert storage.conn.in_transaction
    finally:
        storage.conn.rollback()
    assert storage._publication_delivery_page() == (None, None)


@pytest.mark.parametrize("field", ["org_id", "project_id"])
def test_strict_delivery_rejects_wrong_scope_before_recording(
    execution, monkeypatch, field
):
    from reflexio.server import usage_metrics
    from reflexio.server.services.extraction.publication import _deliver

    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["scope-learning"], origin="resume"
    )
    events = []

    def record(event):
        if event.event_name == "learnings_generated":
            events.append(event)
        return usage_metrics.UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    wrong = replace(receipt, **{field: "wrong-scope"})
    with pytest.raises(ValueError, match="scope"):
        _deliver(storage, wrong)
    assert events == []
    assert _rows(storage)[0]["delivery_state"] == "pending"


@pytest.mark.parametrize("transaction", ["managed", "unmanaged"])
def test_strict_delivery_rejects_caller_transaction_before_recording(
    execution, monkeypatch, transaction
):
    from reflexio.server import usage_metrics
    from reflexio.server.services.extraction.publication import _deliver

    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["transaction-learning"], origin="resume"
    )
    events = []

    def record(event):
        if event.event_name == "learnings_generated":
            events.append(event)
        return usage_metrics.UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    if transaction == "managed":
        with storage.commit_scope(), pytest.raises(ValueError, match="independent"):
            _deliver(storage, receipt)
    else:
        storage.conn.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(ValueError, match="independent"):
                _deliver(storage, receipt)
        finally:
            storage.conn.rollback()
    assert events == []
    assert _rows(storage)[0]["delivery_state"] == "pending"


@pytest.fixture
def unlinked_financial_receipt(execution):
    storage, run = execution
    receipt, _ = storage._create_agent_output_receipt(
        run, learning_ids=["concurrent-learning"], origin="resume"
    )
    with storage.commit_scope(), storage._stream_sql() as db:
        storage._handoff_erased_output_finances(db, run.binding.user_id)
        storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (run.id,))
    return storage, receipt


def test_delivery_owner_accepts_concurrent_cleanup_only_after_strict_acceptance(
    unlinked_financial_receipt, monkeypatch
):
    from reflexio.server import usage_metrics
    from reflexio.server.services.extraction.publication import _deliver

    storage, receipt = unlinked_financial_receipt
    accepted = []

    def record(event):
        if event.event_name == "learnings_generated":
            accepted.append(event)  # Durable acceptance precedes peer ACK.
            assert storage._ack_publication_receipt(receipt)
        return usage_metrics.UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    _deliver(storage, receipt)
    assert (
        len(accepted) == 1
        and accepted[0].created_at == receipt.financial_facts["created_at"]
    )
    assert _rows(storage) == []
    assert not storage._ack_publication_receipt(
        receipt
    )  # Primitive contract unchanged.


@pytest.mark.parametrize("status", ["FAILED", "UNKNOWN"])
def test_failed_delivery_never_infers_acceptance_from_absent_receipt(
    unlinked_financial_receipt, monkeypatch, status
):
    from reflexio.server import usage_metrics
    from reflexio.server.metering_events import ReceiptDeliveryError
    from reflexio.server.services.extraction.publication import _deliver

    storage, receipt = unlinked_financial_receipt

    def record(event):
        if event.event_name == "learnings_generated":
            # A peer can finish the same debt without this caller observing
            # acceptance. Missing data must not override this caller's failure.
            assert storage._ack_publication_receipt(receipt)
            return getattr(usage_metrics.UsageEventDeliveryStatus, status)
        return usage_metrics.UsageEventDeliveryStatus.EXEMPT

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    with pytest.raises(ReceiptDeliveryError) as error:
        _deliver(storage, receipt)
    assert error.value.status == getattr(usage_metrics.UsageEventDeliveryStatus, status)
    assert _rows(storage) == [] and not storage._ack_publication_receipt(receipt)


@pytest.mark.parametrize("drift", ["unbound", "org", "project"])
def test_recorder_scope_drift_cannot_ack_or_classify_absence(
    unlinked_financial_receipt, monkeypatch, drift
):
    from types import SimpleNamespace

    from reflexio.server import usage_metrics
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.services.storage.storage_base import _publication_receipts
    from reflexio.server.work_scope import WorkScope

    storage, receipt = unlinked_financial_receipt
    provider = SimpleNamespace(scope=WorkScope(storage.org_id))
    provider.current = lambda: provider.scope
    monkeypatch.setattr(_publication_receipts, "get_service", lambda _key: provider)
    reloaded = storage._get_publication_receipt(receipt.receipt_id, project_id="")
    assert reloaded is not None and reloaded.run_id is None
    assert reloaded.financial_facts == receipt.financial_facts

    def record(event):
        if event.event_name == "learnings_generated":
            provider.scope = {
                "unbound": None,
                "org": WorkScope("another-org"),
                "project": WorkScope(storage.org_id, "another-project"),
            }[drift]
        return usage_metrics.UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    with pytest.raises(ValueError, match="scope"):
        _deliver(storage, receipt)
    assert _rows(storage)[0]["delivery_state"] == "pending"
    with pytest.raises(ValueError, match="scope"):
        storage._get_publication_receipt(receipt.receipt_id, project_id="")


def test_partial_window_finances_retry_original_keys_before_cleanup(
    execution, monkeypatch
):
    from reflexio.server import usage_metrics
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.usage_metrics import UsageEventDeliveryError

    storage, _ = execution
    identity = uuid.uuid4().hex
    facts = {
        "pipeline": "profile",
        "created_at": 1700000000,
        "platform_llm": True,
        "input_tokens": 42,
        "prompt_tokens": 40,
        "completion_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }
    with storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO _agent_output_receipts(receipt_id,org_id,row_kind,delivery_key,entity_type,learning_ids,delivery_state,delivery_facts) VALUES (?,?,'window_handoff',?,'profile',?,'pending',?)",
            (
                identity,
                storage.org_id,
                identity,
                json.dumps(["partial-learning"]),
                json.dumps(facts),
            ),
        )
    receipt = storage._get_publication_receipt(identity, project_id="")
    assert receipt is not None
    attempts, accepted = [], {}
    failed_token = False

    def record(event):
        nonlocal failed_token
        attempts.append(event)
        if event.event_key == "tok:" + identity and not failed_token:
            failed_token = True
            return usage_metrics.UsageEventDeliveryStatus.FAILED
        if event.event_key in accepted:
            return usage_metrics.UsageEventDeliveryStatus.DUPLICATE
        accepted[event.event_key] = event
        return usage_metrics.UsageEventDeliveryStatus.APPENDED

    monkeypatch.setattr(usage_metrics, "_recorder", record)
    with pytest.raises(UsageEventDeliveryError):
        _deliver(storage, receipt)
    assert len(accepted) == 1 and _rows(storage)[0]["delivery_state"] == "pending"
    _deliver(storage, receipt)
    assert len(attempts) == 4 and len(accepted) == 2 and _rows(storage) == []
    assert [event.event_key for event in attempts[:2]] == [
        event.event_key for event in attempts[2:]
    ]
    assert all(event.created_at == facts["created_at"] for event in attempts)


@pytest.mark.parametrize(
    "status,has_output,has_receipt,refused",
    [
        ("finalization_failed", True, True, True),
        ("finalization_failed", True, False, True),
        ("finalizing", True, False, True),
        ("agent_completed", True, False, True),
        ("running", False, False, True),
        ("resuming", False, False, True),
        ("failed", True, False, True),
        ("failed", False, True, True),
        ("expired", False, True, True),
        ("cancelled", True, False, True),
        ("finalized", True, True, False),
        ("finalized_pending_tool", True, False, False),
        ("failed", False, False, False),
    ],
)
def test_upgrade_classifies_legacy_publication_before_admitting_new_writer(
    tmp_path, status, has_output, has_receipt, refused
):
    from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
        migrate_publication_receipts,
    )

    storage = SQLiteStorage(org_id="legacy-org", db_path=str(tmp_path / "legacy.db"))
    try:
        storage.create_agent_run(
            AgentRunRecord(
                id="legacy-run",
                binding=AgentBinding(
                    org_id=storage.org_id,
                    extractor_kind="profile",
                    user_id="user",
                    request_id="request",
                    agent_version=None,
                    source="old-writer",
                ),
                status=AgentRunStatus(status),
                generation_request_snapshot={},
                committed_output={"profiles": []} if has_output else None,
            )
        )
        # Recreate the actual predecessor relations; no new receipt, ID or ACK.
        with storage.conn:
            storage.conn.execute("DROP TABLE _agent_output_receipts")
            storage.conn.execute("DROP TABLE _agent_run_finalization_receipts")
            storage.conn.execute(_OLD_TABLE)
            if has_receipt:
                storage.conn.execute(
                    "INSERT INTO _agent_run_finalization_receipts VALUES(?,?,?,?)",
                    (
                        "legacy-run",
                        "profile",
                        '["old-learning"]',
                        "2020-01-02T03:04:05Z",
                    ),
                )
        before_run = tuple(
            storage.conn.execute(
                "SELECT id,status,committed_output FROM _agent_runs WHERE id='legacy-run'"
            ).fetchone()
        )
        before_receipts = [
            tuple(row)
            for row in storage.conn.execute(
                "SELECT run_id,entity_type,learning_ids,created_at FROM _agent_run_finalization_receipts"
            )
        ]
        if refused:
            with pytest.raises(ValueError, match="Legacy publication cutover refused"):
                migrate_publication_receipts(storage.conn, org_id=storage.org_id)
            assert (
                storage.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='_agent_output_receipts'"
                ).fetchone()
                is None
            )
        else:
            migrate_publication_receipts(storage.conn, org_id=storage.org_id)
            assert _rows(storage) == []
        assert not storage.conn.in_transaction
        assert (
            tuple(
                storage.conn.execute(
                    "SELECT id,status,committed_output FROM _agent_runs WHERE id='legacy-run'"
                ).fetchone()
            )
            == before_run
        )
        assert [
            tuple(row)
            for row in storage.conn.execute(
                "SELECT run_id,entity_type,learning_ids,created_at FROM _agent_run_finalization_receipts"
            )
        ] == before_receipts
    finally:
        storage.conn.close()


@pytest.mark.parametrize("answer_order", ["before", "after", "expired"])
def test_window_terminalization_reconciles_live_dependency_state(
    execution, answer_order
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.services.base_generation._extraction_lifecycle import (
        ExtractionRunLifecycleMixin,
    )
    from reflexio.server.services.storage.storage_base import (
        PendingToolCallRecord,
        PendingToolCallStatus,
        RunToolDependencyRecord,
        build_scope_hash,
        human_feedback_scope,
    )

    storage, run = execution
    now = datetime.now(UTC)
    scope = human_feedback_scope(storage.org_id)
    question = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="window-question",
            org_id=storage.org_id,
            user_id=run.binding.user_id,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key="window-question",
            status=PendingToolCallStatus.PENDING,
            question_text="Synthetic compatibility-window question",
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=run.id, pending_tool_call_id=question.id)
    )
    with owned_agent_run_write(storage, run):
        current = storage.update_agent_run_status(
            run.id,
            AgentRunStatus.AGENT_COMPLETED,
            pending_tool_call_ids=[question.id],
            expected_statuses=(run.status,),
            expected_claimed_by=run.claimed_by,
            expected_claimed_at=run.claimed_at,
        )
    assert current is not None

    def resolve():
        storage.resolve_pending_tool_call(
            question.id,
            result={"answer": "Synthetic answer"},
            resolved_at=datetime.now(UTC),
            valid_for_seconds=3600,
        )

    if answer_order == "before":
        resolve()
    elif answer_order == "expired":
        with storage.commit_scope():
            storage.conn.execute(
                "UPDATE _pending_tool_calls SET expires_at=? WHERE id=?",
                ((now - timedelta(seconds=1)).isoformat(), question.id),
            )
        storage._expire_pending_question(question.id)
    lifecycle = ExtractionRunLifecycleMixin()
    lifecycle.storage = storage
    lifecycle._last_extraction_run_ids = [current.id]
    lifecycle._last_extraction_run = current
    terminal = lifecycle._finalize_extraction_runs()
    assert terminal is not None
    if answer_order == "expired":
        assert (
            terminal.status == AgentRunStatus.FINALIZED
            and not terminal.pending_tool_call_ids
        )
    elif answer_order == "before":
        assert terminal.status == AgentRunStatus.RESUME_READY
        assert (
            terminal.pending_tool_call_ids == [question.id]
            and terminal.next_resume_at is not None
        )
    else:
        assert terminal.status == AgentRunStatus.FINALIZED_PENDING_TOOL
        resolve()
        assert storage.get_agent_run(run.id).status == AgentRunStatus.RESUME_READY
    assert terminal.resume_attempts == run.resume_attempts
    assert terminal.finalization_attempts == run.finalization_attempts
