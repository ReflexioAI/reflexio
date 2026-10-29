"""Opening an older SQLite database upgrades it to the current schema.

Each test rewinds one piece of a real database to an earlier shape (a column
that did not exist yet, a column that has since been retired or renamed, an
old table name, an unbackfilled vector index) and then runs ``migrate()`` --
the same path ``SQLiteStorage.__init__`` takes on startup -- asserting that
the current shape is restored and that existing rows survive.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def _open(db_path: Path, org_id: str) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.1] * 512):
        storage = SQLiteStorage(org_id=org_id, db_path=str(db_path))
    storage._get_embedding = Mock(return_value=[0.1] * 512)  # noqa: SLF001
    return storage


@pytest.fixture
def store(tmp_path, worker_id) -> SQLiteStorage:
    return _open(tmp_path / "legacy.db", f"legacy-schema-{worker_id}")


def _columns(store: SQLiteStorage, table: str) -> set[str]:
    return {
        row["name"]
        for row in store.conn.execute(f"PRAGMA table_info({table})").fetchall()  # noqa: S608
    }


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _drop_columns(store: SQLiteStorage, table: str, columns: list[str]) -> None:
    for column in columns:
        store.conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")  # noqa: S608
    store.conn.commit()
    assert not set(columns) & _columns(store, table)


def test_interactions_regain_columns_added_after_creation(store):
    added = [
        "expert_content",
        "citations",
        "image_encoding",
        "retrieved_learnings",
        "token_count",
    ]
    _drop_columns(store, "interactions", added)
    store.migrate()
    assert set(added) <= _columns(store, "interactions")


def test_runtime_tables_regain_resumable_and_supersession_columns(store):
    _drop_columns(store, "_agent_runs", ["max_steps_remaining"])
    _drop_columns(store, "_pending_tool_calls", ["superseded_by"])
    store.conn.execute("DROP INDEX IF EXISTS idx_apsup_user")
    _drop_columns(
        store, "agent_playbook_source_user_playbooks", ["source_interaction_ids"]
    )
    _drop_columns(
        store,
        "user_playbook_publication_staging",
        ["incumbent_content_digest", "incumbent_trigger", "incumbent_semantic_digest"],
    )
    store.migrate()
    assert "max_steps_remaining" in _columns(store, "_agent_runs")
    assert "superseded_by" in _columns(store, "_pending_tool_calls")
    assert "source_interaction_ids" in _columns(
        store, "agent_playbook_source_user_playbooks"
    )
    assert {
        "incumbent_content_digest",
        "incumbent_trigger",
        "incumbent_semantic_digest",
    } <= _columns(store, "user_playbook_publication_staging")


def _triggers(store: SQLiteStorage) -> set[str]:
    return {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    }


def test_profiles_regain_expanded_terms(store):
    # Playbook tables also carry aggregation triggers over this column; the
    # profile table exercises the same upgrade with only the lexical ones.
    tables = ("profiles",)
    # The Unicode lexical triggers read expanded_terms, so a database from
    # before the column also predates them.
    lexical = {f"{t}_unicode_fts_{op}" for t in tables for op in ("ai", "au", "ad")}
    assert lexical <= _triggers(store)
    for name in lexical:
        store.conn.execute(f"DROP TRIGGER {name}")  # noqa: S608
    for table in tables:
        _drop_columns(store, table, ["expanded_terms"])
    store.migrate()
    for table in tables:
        assert "expanded_terms" in _columns(store, table)
    assert lexical <= _triggers(store)


def test_retired_polarity_column_is_dropped_and_rows_kept(store):
    store.conn.execute(
        "INSERT INTO user_playbooks (user_id, playbook_name, created_at, request_id, "
        "agent_version, content, trigger) VALUES "
        "('u1', 'p', '2026-07-31T00:00:00+00:00', 'r1', 'v1', 'keep me', 't')"
    )
    store.conn.execute(
        "ALTER TABLE user_playbooks ADD COLUMN polarity TEXT DEFAULT 'positive'"
    )
    store.conn.commit()
    store.migrate()
    assert "polarity" not in _columns(store, "user_playbooks")
    assert [row.content for row in store.get_user_playbooks()] == ["keep me"]


def test_renamed_profile_column_is_restored_to_its_current_name(store):
    store.conn.execute("ALTER TABLE profiles RENAME COLUMN content TO profile_content")
    store.conn.commit()
    store.migrate()
    columns = _columns(store, "profiles")
    assert "content" in columns
    assert "profile_content" not in columns


def test_legacy_fts_column_name_recreates_the_fts_tables(store):
    store.conn.executescript(
        """
        DROP TABLE IF EXISTS user_playbooks_fts;
        CREATE VIRTUAL TABLE user_playbooks_fts USING fts5(feedback_content);
        """
    )
    store.conn.commit()
    store.migrate()
    assert "search_text" in _columns(store, "user_playbooks_fts")
    assert "feedback_content" not in _columns(store, "user_playbooks_fts")


def test_pre_flattening_playbook_tables_are_rebuilt(tmp_path, worker_id):
    db_path = tmp_path / "pre-flattening.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE user_playbooks (
            user_playbook_id INTEGER PRIMARY KEY, content TEXT
        );
        CREATE TABLE agent_playbooks (
            agent_playbook_id INTEGER PRIMARY KEY, content TEXT
        );
        INSERT INTO user_playbooks (content) VALUES ('legacy');
        """
    )
    conn.commit()
    conn.close()
    store = _open(db_path, f"pre-flattening-{worker_id}")
    assert "trigger" in _columns(store, "user_playbooks")
    assert "trigger" in _columns(store, "agent_playbooks")
    # SQLite is local-only; the legacy layout is dropped rather than converted.
    assert store.get_user_playbooks() == []


def test_old_feedback_table_names_are_dropped_on_open(tmp_path, worker_id):
    db_path = tmp_path / "old-names.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_feedbacks (raw_feedback_id INTEGER PRIMARY KEY);
        CREATE TABLE feedbacks (feedback_id INTEGER PRIMARY KEY);
        CREATE TABLE feedback_aggregation_change_logs (
            id INTEGER PRIMARY KEY, feedback_name TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    store = _open(db_path, f"old-names-{worker_id}")
    tables = _tables(store.conn)
    assert not {"raw_feedbacks", "feedbacks"} & tables
    assert {"user_playbooks", "agent_playbooks"} <= tables
    # The renamed change-log table is retired by a later startup migration.
    assert "feedback_aggregation_change_logs" not in tables


def test_vector_index_is_backfilled_from_stored_embeddings(store):
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    store.conn.execute(
        "INSERT INTO user_playbooks (user_id, playbook_name, created_at, request_id, "
        "agent_version, content, trigger, embedding) VALUES "
        "('u1', 'p', '2026-07-31T00:00:00+00:00', 'r1', 'v1', 'c', 't', ?)",
        (json.dumps([0.5] * 512),),
    )
    store.conn.execute("DELETE FROM user_playbooks_vec")
    store.conn.commit()
    store.migrate()
    rowids = [
        row[0] for row in store.conn.execute("SELECT rowid FROM user_playbooks_vec")
    ]
    user_playbook_id = store.conn.execute(
        "SELECT user_playbook_id FROM user_playbooks"
    ).fetchone()[0]
    assert rowids == [user_playbook_id]


def test_legacy_vector_sum_clusters_adopt_their_agent_embedding(store):
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    embedding = [0.0] * 511 + [1.0]
    rows = (
        (1, json.dumps(embedding)),
        (2, "not-json"),
        (3, json.dumps([1.0, 2.0])),  # wrong dimension
    )
    for agent_id, stored in rows:
        store.conn.execute(
            "INSERT INTO agent_playbooks (agent_playbook_id, playbook_name, "
            "created_at, agent_version, content, embedding) VALUES "
            "(?, 'p', '2026-07-31T00:00:00+00:00', 'v1', ?, ?)",
            (agent_id, f"agent rule {agent_id}", stored),
        )
        store.conn.execute(
            "INSERT INTO playbook_aggregation_cluster (cluster_id, index_rowid, "
            "agent_version, agent_playbook_id, vector_sum, member_count) "
            "VALUES (?, ?, 'v1', ?, '[1.0]', 1)",
            (f"legacy-{agent_id}", 100 + agent_id, agent_id),
        )
    # An already-backfilled agent vector index (the normal upgrade case) is
    # left alone, so only the centroid adoption reads these embeddings.
    store.conn.execute(
        "INSERT INTO agent_playbooks_vec(rowid, embedding) VALUES (1, ?)",
        (json.dumps(embedding),),
    )
    store.conn.commit()
    store.migrate()
    clusters = {
        row["cluster_id"]: row
        for row in store.conn.execute(
            "SELECT cluster_id, centroid, vector_sum, embedding_dimension "
            "FROM playbook_aggregation_cluster"
        ).fetchall()
    }
    adopted = clusters["legacy-1"]
    assert json.loads(adopted["centroid"]) == embedding
    assert adopted["vector_sum"] is None
    assert adopted["embedding_dimension"] == 512
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_clusters_vec WHERE rowid=101"
        ).fetchone()[0]
        == 1
    )
    for skipped in ("legacy-2", "legacy-3"):
        assert clusters[skipped]["centroid"] is None
        assert clusters[skipped]["vector_sum"] == "[1.0]"
