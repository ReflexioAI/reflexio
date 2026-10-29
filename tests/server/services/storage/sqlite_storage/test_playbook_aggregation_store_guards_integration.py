"""SQLite aggregation store: argument guards, fences, legacy adoption and scopes.

Every test runs against a real ``SQLiteStorage`` file in ``tmp_path``. The
assertions read persisted rows back, so a guard that silently wrote partial
state would fail here as well as one that raised the wrong error.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage.playbook._aggregation import (
    AGGREGATION_DDL,
    AGGREGATION_TRIGGER_DDL,
    _ddl_is_current,
    init_playbook_aggregation_tables,
)
from reflexio.server.services.storage.sqlite_storage.playbook._aggregation_operations import (
    AGGREGATION_OPERATIONS_DDL,
)
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationClaim,
)

VERSION = "agg-guards-v1"
DIM = 512  # the SQLite cluster vector index dimension


def _vec(axis: int = 0, dim: int = DIM) -> list[float]:
    return [1.0 if i == axis else 0.0 for i in range(dim)]


@pytest.fixture
def store(tmp_path, worker_id) -> SQLiteStorage:
    return SQLiteStorage(
        org_id=f"agg-guards-{worker_id}", db_path=str(tmp_path / "agg.db")
    )


@pytest.fixture
def vec_store(store) -> SQLiteStorage:
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    return store


def _insert_user_playbook(store: SQLiteStorage, item_id: int, version=VERSION):
    store.conn.execute(
        "INSERT INTO user_playbooks (user_playbook_id, user_id, playbook_name, "
        "created_at, request_id, agent_version, content) VALUES "
        "(?, ?, 'playbook', '2026-07-31T00:00:00+00:00', ?, ?, ?)",
        (item_id, f"user-{item_id}", f"req-{item_id}", version, f"rule-{item_id}"),
    )
    store.conn.commit()


def _claim(store: SQLiteStorage, owner="owner-a") -> PlaybookAggregationClaim:
    store.schedule_playbook_aggregation(VERSION)
    claim = store.claim_due_playbook_aggregation(
        owner=owner, lease_seconds=60, agent_version=VERSION
    )
    assert claim is not None
    return claim


def _dead_claim(claim: PlaybookAggregationClaim) -> PlaybookAggregationClaim:
    return PlaybookAggregationClaim(
        claim.agent_version, claim.owner, claim.fence + 99, claim.state_version, 0
    )


def _cluster_row(store: SQLiteStorage, cluster_id: str):
    return store.conn.execute(
        "SELECT state, member_count, agent_playbook_id, centroid, rebuild_cursor, "
        "index_rowid FROM playbook_aggregation_cluster WHERE cluster_id=?",
        (cluster_id,),
    ).fetchone()


def _vec_rowids(store: SQLiteStorage) -> set[int]:
    return {
        int(row[0])
        for row in store.conn.execute(
            "SELECT rowid FROM playbook_aggregation_clusters_vec"
        ).fetchall()
    }


def _adopt(store, cluster_id="legacy-c1", *, members, complete, cursor=10, dim=DIM):
    store.adopt_legacy_playbook_aggregation_cluster_page(
        cluster_id=cluster_id,
        agent_version=VERSION,
        agent_playbook_id=501,
        centroid_embedding=_vec(0, dim),
        member_embeddings=[(item, _vec(0, dim)) for item in members],
        embedding_model="model-a",
        embedding_dimension=dim,
        rebuild_cursor=cursor,
        complete=complete,
    )


# ---------------------------------------------------------------------------
# _ddl_is_current / init_playbook_aggregation_tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ddl",
    [
        "BEGIN IMMEDIATE;\nBEGIN IMMEDIATE;\nCOMMIT;\n",
        "COMMIT;\n",
        "CREATE TABLE t (a INTEGER);\nCREATE TABLE t (a INTEGER);\n",
        "CREATE TABLE t (a INTEGER);\nCREATE TABLE u (b",
        "BEGIN IMMEDIATE;\nCREATE TABLE t (a INTEGER);\n",
        "",
        "DROP TRIGGER IF EXISTS gone;\nCREATE TABLE t (a INTEGER);\n",
        "CREATE TRIGGER IF NOT EXISTS x AFTER INSERT ON t BEGIN SELECT 1; END;\n"
        "DROP TRIGGER IF EXISTS x;\n",
        "INSERT INTO t VALUES (1);\n",
    ],
    ids=[
        "nested-begin",
        "commit-without-begin",
        "duplicate-object",
        "incomplete-trailing-statement",
        "unterminated-transaction",
        "empty",
        "dropped-trigger-never-recreated",
        "drop-after-create",
        "unsupported-statement",
    ],
)
def test_ddl_recognizer_falls_back_to_execution_for_unrecognized_scripts(ddl):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER)")
    assert _ddl_is_current(conn, ddl) is False


def test_ddl_recognizer_accepts_exact_current_schema(store):
    assert _ddl_is_current(store.conn, AGGREGATION_DDL + AGGREGATION_OPERATIONS_DDL)
    assert _ddl_is_current(store.conn, AGGREGATION_TRIGGER_DDL)
    store.conn.execute("DROP INDEX idx_playbook_aggregation_state_due")
    assert not _ddl_is_current(store.conn, AGGREGATION_DDL + AGGREGATION_OPERATIONS_DDL)


def test_init_adds_columns_missing_from_older_aggregation_tables(store):
    conn = store.conn
    conn.execute("DROP INDEX idx_playbook_aggregation_cluster_rebuild_due")
    conn.execute(
        "ALTER TABLE playbook_aggregation_state DROP COLUMN "
        "intake_floor_user_playbook_id"
    )
    conn.execute(
        "ALTER TABLE playbook_aggregation_cluster DROP COLUMN rebuild_attempt_count"
    )
    conn.execute(
        "ALTER TABLE playbook_aggregation_cluster DROP COLUMN rebuild_next_attempt_at"
    )
    conn.commit()

    init_playbook_aggregation_tables(conn)
    conn.commit()

    state_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(playbook_aggregation_state)")
    }
    cluster_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(playbook_aggregation_cluster)")
    }
    assert "intake_floor_user_playbook_id" in state_columns
    assert {"rebuild_attempt_count", "rebuild_next_attempt_at"} <= cluster_columns
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND "
        "name='idx_playbook_aggregation_cluster_rebuild_due'"
    ).fetchone()
    # The restored columns carry their defaults for new rows.
    store.schedule_playbook_aggregation(VERSION)
    assert (
        conn.execute(
            "SELECT intake_floor_user_playbook_id FROM playbook_aggregation_state "
            "WHERE agent_version=?",
            (VERSION,),
        ).fetchone()[0]
        == 0
    )


# ---------------------------------------------------------------------------
# Capability flags and argument guards
# ---------------------------------------------------------------------------


def test_missing_vector_index_blocks_aggregation_and_nearest_lookup(store):
    store._has_sqlite_vec = False
    assert store.playbook_aggregation_blocked_reason == "blocked_missing_vector_index"
    assert store.supports_durable_playbook_aggregation_operations is False
    assert store.supports_incremental_playbook_aggregation is False
    with pytest.raises(RuntimeError):
        store.find_nearest_playbook_aggregation_clusters(
            VERSION, [(1, [1.0])], embedding_model="m", limit=1
        )
    # Repair still schedules versions with undisposed intake without the index.
    _insert_user_playbook(store, 1)
    store.conn.execute("DELETE FROM playbook_aggregation_state")
    store.conn.commit()
    assert store.repair_playbook_aggregation_pending_state(limit=5) == [VERSION]


def test_incremental_flag_can_be_disabled_independently(vec_store):
    assert vec_store.playbook_aggregation_blocked_reason is None
    assert vec_store.supports_incremental_playbook_aggregation is True
    vec_store.supports_incremental_playbook_aggregation = False
    assert vec_store.supports_incremental_playbook_aggregation is False
    assert vec_store.supports_durable_playbook_aggregation_operations is True


def test_invalid_arguments_are_rejected_before_any_write(store):
    with pytest.raises(ValueError, match="non-empty"):
        store.schedule_playbook_aggregation("   ")
    with pytest.raises(ValueError, match="claim owner"):
        store.claim_due_playbook_aggregation(owner=" ", lease_seconds=10)
    with pytest.raises(ValueError, match="claim owner"):
        store.claim_due_playbook_aggregation(owner="o", lease_seconds=0)
    with pytest.raises(ValueError, match="invalid aggregation bootstrap status"):
        store.set_playbook_aggregation_bootstrap_status(VERSION, "done")
    with pytest.raises(ValueError, match="dimension changed"):
        store.adopt_legacy_playbook_aggregation_cluster_page(
            cluster_id="c",
            agent_version=VERSION,
            agent_playbook_id=1,
            centroid_embedding=[1.0, 0.0],
            member_embeddings=[],
            embedding_model="m",
            embedding_dimension=3,
            rebuild_cursor=0,
            complete=True,
        )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_state"
        ).fetchone()[0]
        == 0
    )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 0
    )


def test_non_positive_limits_and_empty_inputs_are_noops(store):
    _insert_user_playbook(store, 7)
    assert store.repair_playbook_aggregation_pending_state(limit=0) == []
    assert store.stage_playbook_aggregation_intake(VERSION, limit=0) == []
    assert (
        store.stage_playbook_aggregation_intake(VERSION, limit=5, window_limit=0) == []
    )
    assert store.get_playbook_aggregation_residual_ids(VERSION, limit=0) == []
    assert store.get_playbook_aggregation_invalidations(VERSION, limit=0) == []
    snapshot = store.capture_playbook_aggregation_rerun_snapshot(VERSION, limit=0)
    assert snapshot.user_playbooks == () and snapshot.invalidation_ids == ()
    assert snapshot.user_high_watermark is None
    store.stage_playbook_aggregation_snapshot(VERSION, [])
    store.set_playbook_aggregation_disposition(VERSION, [], disposition="terminal_noop")
    store.attach_playbook_aggregation_items(agent_version=VERSION, attachments=[])
    claim = _claim(store)
    assert store.mark_playbook_aggregation_invalidations_processed(claim, []) is True
    assert store.apply_playbook_aggregation_invalidations(claim, []) is True
    # Nothing was staged by any of the no-op calls.
    assert (
        store.conn.execute("SELECT count(*) FROM playbook_aggregation_item").fetchone()[
            0
        ]
        == 0
    )


# ---------------------------------------------------------------------------
# Claim renewal and fencing
# ---------------------------------------------------------------------------


def test_claim_renewal_extends_a_live_claim_and_refuses_a_released_one(store):
    with pytest.raises(ValueError, match="positive"):
        store.renew_playbook_aggregation_claim(
            PlaybookAggregationClaim(VERSION, "o", 1, 0, 0), lease_seconds=0
        )
    claim = _claim(store)
    renewed = store.renew_playbook_aggregation_claim(claim, lease_seconds=600)
    assert renewed is not None
    assert (renewed.owner, renewed.fence, renewed.state_version) == (
        claim.owner,
        claim.fence,
        claim.state_version,
    )
    assert renewed.expires_at >= claim.expires_at
    assert (
        store.conn.execute(
            "SELECT claim_expires_at FROM playbook_aggregation_lease"
        ).fetchone()[0]
        == renewed.expires_at
    )
    # A second claimer is refused while the renewed lease is live.
    assert (
        store.claim_due_playbook_aggregation(owner="owner-b", lease_seconds=60) is None
    )
    assert store.finish_playbook_aggregation_claim(
        renewed,
        success=False,
        retry_after_seconds=5,
        backlog_retry_after_seconds=5,
        min_interval_seconds=5,
    )
    assert store.renew_playbook_aggregation_claim(renewed, lease_seconds=60) is None


def test_claim_returns_none_for_unknown_version_and_dead_claims_cannot_write(store):
    assert (
        store.claim_due_playbook_aggregation(
            owner="o", lease_seconds=30, agent_version="never-scheduled"
        )
        is None
    )
    claim = _claim(store)
    store.append_playbook_aggregation_invalidation(
        agent_version=VERSION, operation="hard_delete", entity_id=3
    )
    pending = store.get_playbook_aggregation_invalidations(VERSION, limit=10)
    ids = [item.invalidation_id for item in pending]
    dead = _dead_claim(claim)
    assert store.validate_playbook_aggregation_claim(dead) is False
    assert store.mark_playbook_aggregation_invalidations_processed(dead, ids) is False
    assert store.apply_playbook_aggregation_invalidations(dead, ids) is False
    assert (
        store.finish_playbook_aggregation_claim(
            dead,
            success=True,
            retry_after_seconds=1,
            backlog_retry_after_seconds=1,
            min_interval_seconds=1,
        )
        is False
    )
    assert len(store.get_playbook_aggregation_invalidations(VERSION, limit=10)) == 1
    assert store.mark_playbook_aggregation_invalidations_processed(claim, ids) is True
    assert store.get_playbook_aggregation_invalidations(VERSION, limit=10) == []


# ---------------------------------------------------------------------------
# Writes joining an outer commit_scope
# ---------------------------------------------------------------------------


def test_aggregation_writes_join_an_outer_scope_and_roll_back_with_it(store):
    _insert_user_playbook(store, 11)
    _insert_user_playbook(store, 12)
    with pytest.raises(RuntimeError, match="abort outer scope"), store.commit_scope():
        store.schedule_playbook_aggregation(VERSION)
        claim = store.claim_due_playbook_aggregation(
            owner="scoped", lease_seconds=60, agent_version=VERSION
        )
        assert claim is not None
        assert store.stage_playbook_aggregation_intake(VERSION, limit=10) == [
            12,
            11,
        ]
        store.set_playbook_aggregation_bootstrap_status(VERSION, "complete")
        store.stage_playbook_aggregation_snapshot(VERSION, [11])
        store.capture_playbook_aggregation_rerun_snapshot(VERSION, limit=5)
        raise RuntimeError("abort outer scope")
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_state"
        ).fetchone()[0]
        == 0
    )
    assert (
        store.conn.execute("SELECT count(*) FROM playbook_aggregation_item").fetchone()[
            0
        ]
        == 0
    )
    assert (
        store.conn.execute(
            "SELECT claim_owner FROM playbook_aggregation_lease"
        ).fetchone()
        is None
    )
    assert store.get_playbook_aggregation_bootstrap_status(VERSION) == "pending"

    with store.commit_scope():
        store.schedule_playbook_aggregation(VERSION)
        claim = store.claim_due_playbook_aggregation(
            owner="scoped", lease_seconds=60, agent_version=VERSION
        )
        assert claim is not None
        store.stage_playbook_aggregation_intake(VERSION, limit=10)
        assert store.get_playbook_aggregation_residual_ids(VERSION, limit=5) == [
            11,
            12,
        ]
        assert store.renew_playbook_aggregation_claim(claim, lease_seconds=30)
        assert store.finish_playbook_aggregation_claim(
            claim,
            success=False,
            retry_after_seconds=0,
            backlog_retry_after_seconds=0,
            min_interval_seconds=0,
        )
    rows = store.conn.execute(
        "SELECT user_playbook_id, attempt_count FROM playbook_aggregation_item "
        "ORDER BY user_playbook_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(11, 1), (12, 1)]


def test_reset_inside_scope_clears_items_clusters_and_vectors(vec_store):
    store = vec_store
    _insert_user_playbook(store, 21)
    store.stage_playbook_aggregation_intake(VERSION, limit=10)
    store.create_playbook_aggregation_cluster(
        cluster_id="reset-c",
        agent_version=VERSION,
        agent_playbook_id=None,
        centroid_embedding=_vec(1),
        member_count=1,
        embedding_model="m",
    )
    rowid = int(_cluster_row(store, "reset-c")["index_rowid"])
    assert rowid in _vec_rowids(store)
    with store.commit_scope():
        store.reset_playbook_aggregation_version(VERSION)
    assert _cluster_row(store, "reset-c") is None
    assert rowid not in _vec_rowids(store)
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_item WHERE agent_version=?",
            (VERSION,),
        ).fetchone()[0]
        == 0
    )
    assert store.get_playbook_aggregation_bootstrap_status(VERSION) == "complete"


# ---------------------------------------------------------------------------
# Legacy cluster adoption
# ---------------------------------------------------------------------------


def test_legacy_adoption_pages_accumulate_then_activate_once(vec_store):
    store = vec_store
    assert store.get_playbook_aggregation_cluster_rebuild_cursor("legacy-c1") is None
    _adopt(store, members=[1, 2], complete=False, cursor=2)
    row = _cluster_row(store, "legacy-c1")
    assert (row["state"], row["member_count"], row["centroid"]) == (
        "rebuilding",
        2,
        None,
    )
    assert store.get_playbook_aggregation_cluster_rebuild_cursor("legacy-c1") == (
        2,
        "rebuilding",
    )
    # Re-delivering a member is idempotent; a new one increments the count.
    _adopt(store, members=[2, 3], complete=True, cursor=3)
    row = _cluster_row(store, "legacy-c1")
    assert (row["state"], row["member_count"]) == ("active", 3)
    assert json.loads(row["centroid"]) == _vec(0)
    assert int(row["index_rowid"]) in _vec_rowids(store)
    # An active cluster ignores later pages entirely.
    _adopt(store, members=[4], complete=True, cursor=9)
    assert _cluster_row(store, "legacy-c1")["member_count"] == 3
    assert store.get_playbook_aggregation_cluster_rebuild_cursor("legacy-c1") == (
        3,
        "active",
    )


def test_legacy_adoption_refuses_changed_embedding_provenance(vec_store):
    _adopt(vec_store, members=[1], complete=False)
    with pytest.raises(RuntimeError, match="provenance changed"):
        _adopt(vec_store, members=[2], complete=False, dim=DIM - 1)
    assert _cluster_row(vec_store, "legacy-c1")["member_count"] == 1


def test_legacy_adoption_inside_scope_commits_with_the_scope(vec_store):
    with vec_store.commit_scope():
        _adopt(vec_store, members=[1], complete=True)
    assert _cluster_row(vec_store, "legacy-c1")["state"] == "active"


# ---------------------------------------------------------------------------
# Rebuilding clusters: complete / discard / defer fences
# ---------------------------------------------------------------------------


def _rebuilding_cluster_with_residuals(store, members=(1, 2)):
    _adopt(store, "rb-c", members=list(members), complete=False)
    store.set_playbook_aggregation_disposition(
        VERSION,
        list(members),
        disposition="residual",
        cluster_id="rb-c",
        reason="rebuild",
    )


def test_complete_rebuild_rejects_stale_or_inconsistent_requests(vec_store):
    store = vec_store
    kwargs = {
        "cluster_id": "rb-c",
        "agent_version": VERSION,
        "expected_agent_playbook_id": 501,
        "replacement_agent_playbook_id": 777,
        "embedding_model": "model-a",
    }
    with pytest.raises(ValueError, match="non-empty"):
        store.complete_playbook_aggregation_cluster_rebuild(
            centroid_embedding=[], **kwargs
        )
    with pytest.raises(RuntimeError, match="cluster changed"):
        store.complete_playbook_aggregation_cluster_rebuild(
            centroid_embedding=_vec(0), **kwargs
        )
    _adopt(store, "rb-c", members=[1], complete=False)
    with pytest.raises(ValueError, match="dimension changed"):
        store.complete_playbook_aggregation_cluster_rebuild(
            centroid_embedding=_vec(0, DIM - 1), **kwargs
        )
    # Adopted rows are cluster members, not residuals: nothing to rebuild from.
    with pytest.raises(RuntimeError, match="no members"):
        store.complete_playbook_aggregation_cluster_rebuild(
            centroid_embedding=_vec(0), **kwargs
        )
    row = _cluster_row(store, "rb-c")
    assert (row["state"], row["agent_playbook_id"]) == ("rebuilding", 501)


def test_discard_rebuild_retires_residuals_and_removes_cluster(vec_store):
    store = vec_store
    with pytest.raises(RuntimeError, match="cluster changed"):
        store.discard_playbook_aggregation_cluster_rebuild(
            cluster_id="rb-c",
            agent_version=VERSION,
            expected_agent_playbook_id=501,
            reason="gone",
        )
    _rebuilding_cluster_with_residuals(store)
    rowid = int(_cluster_row(store, "rb-c")["index_rowid"])
    with pytest.raises(RuntimeError, match="cluster changed"):
        store.discard_playbook_aggregation_cluster_rebuild(
            cluster_id="rb-c",
            agent_version=VERSION,
            expected_agent_playbook_id=999,
            reason="wrong-agent",
        )
    retired = store.discard_playbook_aggregation_cluster_rebuild(
        cluster_id="rb-c",
        agent_version=VERSION,
        expected_agent_playbook_id=501,
        reason="agent_retired",
    )
    assert retired == 2
    assert _cluster_row(store, "rb-c") is None
    assert rowid not in _vec_rowids(store)
    rows = store.conn.execute(
        "SELECT disposition, cluster_id, reason FROM playbook_aggregation_item "
        "WHERE agent_version=? ORDER BY user_playbook_id",
        (VERSION,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [("terminal_noop", None, "agent_retired")] * 2


def test_defer_rebuild_requires_the_expected_agent(vec_store):
    store = vec_store
    _rebuilding_cluster_with_residuals(store, members=(5,))
    with pytest.raises(RuntimeError, match="cluster changed"):
        store.defer_playbook_aggregation_cluster_rebuild(
            cluster_id="rb-c",
            agent_version=VERSION,
            expected_agent_playbook_id=999,
            reason="busy",
        )
    store.defer_playbook_aggregation_cluster_rebuild(
        cluster_id="rb-c",
        agent_version=VERSION,
        expected_agent_playbook_id=501,
        reason="busy",
    )
    row = store.conn.execute(
        "SELECT rebuild_attempt_count FROM playbook_aggregation_cluster "
        "WHERE cluster_id='rb-c'"
    ).fetchone()
    assert row[0] == 1
    # A deferred cluster's residuals are withheld from the next residual page.
    assert store.get_playbook_aggregation_residual_ids(VERSION, limit=10) == []


# ---------------------------------------------------------------------------
# Active clusters: create / attach / replace / nearest
# ---------------------------------------------------------------------------


def test_create_and_attach_reject_invalid_inputs(vec_store):
    store = vec_store
    with pytest.raises(ValueError, match="non-empty"):
        store.create_playbook_aggregation_cluster(
            cluster_id="c",
            agent_version=VERSION,
            agent_playbook_id=None,
            centroid_embedding=[],
            member_count=1,
            embedding_model="m",
        )
    with pytest.raises(ValueError, match="member_count"):
        store.create_playbook_aggregation_cluster(
            cluster_id="c",
            agent_version=VERSION,
            agent_playbook_id=None,
            centroid_embedding=[1.0],
            member_count=0,
            embedding_model="m",
        )
    with pytest.raises(ValueError, match="unique"):
        store.attach_playbook_aggregation_items(
            agent_version=VERSION, attachments=[(1, "c"), (1, "d")]
        )
    with pytest.raises(ValueError, match="unique"):
        store.find_nearest_playbook_aggregation_clusters(
            VERSION, [(1, [1.0]), (1, [0.5])], embedding_model="m", limit=3
        )
    assert (
        store.find_nearest_playbook_aggregation_clusters(
            VERSION, [], embedding_model="m", limit=3
        )
        == {}
    )
    assert _cluster_row(store, "c") is None


def test_attach_rolls_back_when_an_item_is_no_longer_residual(vec_store):
    store = vec_store
    for item in (31, 32):
        _insert_user_playbook(store, item)
    store.stage_playbook_aggregation_intake(VERSION, limit=10)
    store.create_playbook_aggregation_cluster(
        cluster_id="active-c",
        agent_version=VERSION,
        agent_playbook_id=None,
        centroid_embedding=_vec(0),
        member_count=1,
        embedding_model="m",
    )
    store.set_playbook_aggregation_disposition(
        VERSION, [32], disposition="terminal_noop", reason="raced"
    )
    with pytest.raises(RuntimeError, match="lost its state"):
        store.attach_playbook_aggregation_items(
            agent_version=VERSION, attachments=[(31, "active-c"), (32, "active-c")]
        )
    assert _cluster_row(store, "active-c")["member_count"] == 1
    assert store.conn.execute(
        "SELECT disposition, cluster_id FROM playbook_aggregation_item "
        "WHERE user_playbook_id=31"
    ).fetchone()[:] == ("residual", None)


def test_replace_cluster_agent_fences_on_agent_and_dimension(vec_store):
    store = vec_store
    kwargs = {
        "cluster_id": "active-c",
        "agent_version": VERSION,
        "expected_agent_playbook_id": 41,
        "replacement_agent_playbook_id": 42,
        "embedding_model": "m",
    }
    with pytest.raises(ValueError, match="non-empty"):
        store.replace_playbook_aggregation_cluster_agent(
            centroid_embedding=[], **kwargs
        )
    with pytest.raises(RuntimeError, match="agent changed"):
        store.replace_playbook_aggregation_cluster_agent(
            centroid_embedding=_vec(0), **kwargs
        )
    store.create_playbook_aggregation_cluster(
        cluster_id="active-c",
        agent_version=VERSION,
        agent_playbook_id=41,
        centroid_embedding=_vec(0),
        member_count=1,
        embedding_model="m",
    )
    with pytest.raises(ValueError, match="dimension changed"):
        store.replace_playbook_aggregation_cluster_agent(
            centroid_embedding=_vec(0, DIM - 1), **kwargs
        )
    store.replace_playbook_aggregation_cluster_agent(
        centroid_embedding=_vec(1), **kwargs
    )
    row = _cluster_row(store, "active-c")
    assert row["agent_playbook_id"] == 42
    assert json.loads(row["centroid"]) == _vec(1)
    match = store.find_nearest_playbook_aggregation_clusters(
        VERSION, [(9, _vec(1))], embedding_model="m", limit=2
    )
    assert match[9].cluster_id == "active-c"
    assert match[9].similarity == pytest.approx(1.0)
