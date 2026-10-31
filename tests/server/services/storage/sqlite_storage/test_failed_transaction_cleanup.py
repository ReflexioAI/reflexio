"""Failed standalone writes and interrupted scopes release real SQLite locks."""

import sqlite3
import threading
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.error import PostCommitStorageError, StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        store = SQLiteStorage(org_id="cleanup", db_path=str(tmp_path / "data.db"))
    store._execute("CREATE TABLE cleanup_probe (id INTEGER PRIMARY KEY)")
    store._execute("INSERT INTO cleanup_probe VALUES (1)")
    yield store
    store.conn.close()


def _assert_second_writer_succeeds(storage):
    with sqlite3.connect(storage.db_path, timeout=0.05) as conn:
        conn.execute("INSERT INTO cleanup_probe VALUES (2)")
    assert not storage.conn.in_transaction
    with storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (3)")


def test_failed_standalone_execute_releases_writer_and_connection(storage):
    with pytest.raises(sqlite3.IntegrityError):
        storage._execute("INSERT INTO cleanup_probe VALUES (1)")
    _assert_second_writer_succeeds(storage)


def test_failed_decorated_write_releases_writer(storage):
    storage.add_request(
        Request(request_id="duplicate", user_id="alice", session_id="s")
    )
    # add_request_if_absent rejects before writing; create_agent_run is INSERT-only.
    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    run = AgentRunRecord(
        id="duplicate",
        binding=AgentBinding(
            org_id="cleanup",
            extractor_kind="profile",
            user_id="alice",
            request_id="duplicate",
            agent_version="v1",
            source="api",
            source_interaction_ids=[],
            window_start_interaction_id=None,
            window_end_interaction_id=None,
            extractor_config_hash="test",
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={},
    )
    storage.create_agent_run(run)
    with pytest.raises(StorageError):
        storage.create_agent_run(run)
    _assert_second_writer_succeeds(storage)


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_scope_failure_rolls_back_and_releases_writer(storage, exception):
    with pytest.raises(exception), storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (4)")
        raise exception("interrupted")
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=4") is None
    assert storage._scope_depth == 0
    _assert_second_writer_succeeds(storage)


def test_caught_failure_inside_scope_does_not_rollback_prior_writes(storage):
    with storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (4)")
        with pytest.raises(sqlite3.IntegrityError):
            storage._execute("INSERT INTO cleanup_probe VALUES (1)")
        assert storage.conn.in_transaction
        assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=4") is not None
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=4") is not None


@pytest.mark.parametrize(
    "expand, expanded_terms",
    [(False, None), (True, "migration deployment"), (True, None)],
)
def test_prepared_playbook_save_skips_model_calls_inside_writer(
    storage, monkeypatch, expand, expanded_terms
):
    from reflexio.models.api_schema.domain.entities import AgentPlaybook, LineageContext

    playbook = AgentPlaybook(
        agent_version="v1",
        playbook_name="probe",
        content="Run migrations first.",
        trigger="Deploying",
    )
    vector = [0.0] * 512
    probe_lock = threading.Lock()

    def embed(_text):
        assert not storage.conn.in_transaction
        # Another writer can commit while model preparation is running.
        # Serialize the test's independent writers, not the storage connection.
        with probe_lock, sqlite3.connect(storage.db_path, timeout=0.05) as writer:
            writer.execute("INSERT INTO cleanup_probe VALUES (5)")
        return vector

    def expand_document(_text):
        assert not storage.conn.in_transaction
        # Serialize the test's independent writers, not the storage connection.
        with probe_lock, sqlite3.connect(storage.db_path, timeout=0.05) as writer:
            writer.execute("INSERT INTO cleanup_probe VALUES (6)")
        return expanded_terms

    monkeypatch.setattr(storage, "_should_expand_documents", lambda: expand)
    monkeypatch.setattr(storage, "_get_embedding", embed)
    monkeypatch.setattr(storage, "_expand_document", expand_document)
    storage.prepare_agent_playbooks_for_save([playbook])
    assert playbook.expanded_terms == expanded_terms
    monkeypatch.setattr(
        storage,
        "_get_embedding",
        lambda *_: pytest.fail("Model call inside persistence"),
    )
    monkeypatch.setattr(
        storage,
        "_expand_document",
        lambda *_: pytest.fail("Expansion inside persistence"),
    )
    with pytest.raises(RuntimeError), storage.commit_scope():
        saved = storage.save_prepared_agent_playbooks(
            [playbook],
            lineage_contexts=[
                LineageContext(op_kind="aggregate", request_id="run", source_ids=["1"])
            ],
        )
        assert saved[0].embedding == vector
        raise RuntimeError("abort fenced transaction")
    assert storage.get_agent_playbook_by_id(playbook.agent_playbook_id) is None
    assert (
        storage.get_lineage_events(
            entity_type="agent_playbook", entity_id=str(playbook.agent_playbook_id)
        )
        == []
    )
    with storage.commit_scope():
        saved = storage.save_prepared_agent_playbooks(
            [playbook],
            lineage_contexts=[
                LineageContext(op_kind="aggregate", request_id="run", source_ids=["1"])
            ],
        )
    assert (
        storage.get_agent_playbooks_by_ids(
            [saved[0].agent_playbook_id], include_embedding=True
        )[0].embedding
        == vector
    )
    assert (
        len(
            storage.get_lineage_events(
                entity_type="agent_playbook", entity_id=str(saved[0].agent_playbook_id)
            )
        )
        == 1
    )


def test_failed_index_replace_restores_old_index_and_releases_writer(storage):
    storage._execute("CREATE VIRTUAL TABLE cleanup_fts USING fts5(content)")
    storage._fts_upsert_now("cleanup_fts", 1, content="existing")
    with pytest.raises(sqlite3.OperationalError):
        storage._fts_upsert_now("cleanup_fts", 1, missing_column="invalid")
    assert (
        storage._fetchone("SELECT content FROM cleanup_fts WHERE rowid=1")[0]
        == "existing"
    )
    _assert_second_writer_succeeds(storage)


def test_failed_deferred_index_releases_writer_after_primary_commit(storage):
    storage._execute("CREATE VIRTUAL TABLE cleanup_fts USING fts5(content)")
    storage._fts_upsert_now("cleanup_fts", 1, content="existing")
    with pytest.raises(PostCommitStorageError), storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (4)")
        storage._fts_upsert("cleanup_fts", 1, missing_column="invalid")
    # Primary data was committed before index flushing, so rollback applies
    # only to the failed index replacement, not the admitted primary write.
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=4") is not None
    assert (
        storage._fetchone("SELECT content FROM cleanup_fts WHERE rowid=1")[0]
        == "existing"
    )
    assert storage._scope_depth == 0
    assert storage._deferred_index_ops == []
    _assert_second_writer_succeeds(storage)


def test_prepared_save_rejects_missing_embedding_before_writes(storage):
    from reflexio.models.api_schema.domain.entities import AgentPlaybook, LineageContext

    playbook = AgentPlaybook(
        agent_version="v1",
        content="Run migrations first.",
        trigger="Deploying",
        embedding=[],
    )
    with pytest.raises(ValueError, match="prepared agent playbook has no embedding"):
        storage.save_prepared_agent_playbooks(
            [playbook],
            lineage_contexts=[LineageContext(op_kind="aggregate", request_id="run")],
        )
    assert storage.get_agent_playbooks() == []
    assert not storage.conn.in_transaction


def _degrade_embeddings(storage, monkeypatch):
    """Make the real ``_get_embedding`` hit its provider-unavailable fallback."""
    from reflexio.server.llm.providers.embedding_service_provider import (
        EmbeddingUnavailableError,
    )

    def unavailable(*_args, **_kwargs):
        raise EmbeddingUnavailableError("embedding service down")

    monkeypatch.setattr(storage.llm_client, "get_embedding", unavailable)
    monkeypatch.setattr(storage, "_should_expand_documents", lambda: False)


def test_prepared_save_accepts_playbook_prepared_while_embeddings_unavailable(
    storage, monkeypatch
):
    """Regression: aggregation failed on SQLite whenever embeddings were down.

    ``_get_embedding`` deliberately degrades to ``[]`` when the provider is
    unavailable, which ordinary saves accept. The prepared-save guard must
    reject only input that was never prepared, not a prepared playbook whose
    provider was unreachable.
    """
    from reflexio.models.api_schema.domain.entities import AgentPlaybook, LineageContext

    _degrade_embeddings(storage, monkeypatch)
    playbook = AgentPlaybook(
        agent_version="v1",
        content="Run migrations first.",
        trigger="Deploying",
    )
    storage.prepare_agent_playbooks_for_save([playbook])
    assert playbook.embedding == []
    monkeypatch.setattr(
        storage.llm_client,
        "get_embedding",
        lambda *_a, **_k: pytest.fail("Model call inside persistence"),
    )
    with storage.commit_scope():
        saved = storage.save_prepared_agent_playbooks(
            [playbook],
            lineage_contexts=[
                LineageContext(op_kind="aggregate", request_id="run", source_ids=["1"])
            ],
        )
    stored = storage.get_agent_playbooks_by_ids(
        [saved[0].agent_playbook_id], include_embedding=True
    )
    assert [p.content for p in stored] == ["Run migrations first."]
    assert not stored[0].embedding
    assert (
        len(
            storage.get_lineage_events(
                entity_type="agent_playbook", entity_id=str(saved[0].agent_playbook_id)
            )
        )
        == 1
    )


def test_prepared_save_still_rejects_unprepared_playbook_beside_prepared_one(
    storage, monkeypatch
):
    from reflexio.models.api_schema.domain.entities import AgentPlaybook, LineageContext

    _degrade_embeddings(storage, monkeypatch)
    prepared = AgentPlaybook(agent_version="v1", content="Prepared.", trigger="t")
    storage.prepare_agent_playbooks_for_save([prepared])
    unprepared = AgentPlaybook(agent_version="v1", content="Unprepared.", trigger="t")
    with pytest.raises(ValueError, match="prepared agent playbook has no embedding"):
        storage.save_prepared_agent_playbooks(
            [prepared, unprepared],
            lineage_contexts=[
                LineageContext(op_kind="aggregate", request_id="run"),
                LineageContext(op_kind="aggregate", request_id="run"),
            ],
        )
    assert storage.get_agent_playbooks() == []
    assert not storage.conn.in_transaction


def test_rollback_failure_preserves_original_write_error(storage, monkeypatch):
    class BrokenRollback:
        def execute(self, *_args):
            raise sqlite3.IntegrityError("original write failure")

        def rollback(self):
            raise sqlite3.OperationalError("rollback unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(storage, "conn", BrokenRollback())
        with pytest.raises(sqlite3.IntegrityError, match="original write failure"):
            storage._execute("INSERT INTO cleanup_probe VALUES (1)")


def test_failed_scope_rollback_preserves_committed_index_error(storage, monkeypatch):
    connection = storage.conn

    class BrokenRollback:
        def __getattr__(self, name):
            return getattr(connection, name)

        def rollback(self):
            raise sqlite3.OperationalError("rollback unavailable")

    def fail_flush(*_args):
        raise RuntimeError("index unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(storage, "conn", BrokenRollback())
        scoped.setattr(storage, "_flush_index_op", fail_flush)
        with pytest.raises(PostCommitStorageError) as error, storage.commit_scope():
            storage._execute("INSERT INTO cleanup_probe VALUES (4)")
            storage._fts_upsert("cleanup_fts", 1, content="new")
        assert isinstance(error.value.__cause__, RuntimeError)
        assert storage._scope_depth == 0
        assert storage._deferred_index_ops == []
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=4") is not None
    _assert_second_writer_succeeds(storage)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_caught_nested_scope_failure_requires_rollback(storage, failure):
    with pytest.raises(StorageError, match="rollback"), storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (4)")
        try:
            with storage.commit_scope():
                storage._execute("INSERT INTO cleanup_probe VALUES (5)")
                raise failure("nested operation rejected")
        except failure:
            pass
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id IN (4,5)") is None
    with storage.commit_scope():
        storage._execute("INSERT INTO cleanup_probe VALUES (6)")
    assert storage._fetchone("SELECT id FROM cleanup_probe WHERE id=6") is not None
