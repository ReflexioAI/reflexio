from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import reflexio
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    Status,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.components.aggregator import (
    AggregationGenerationOutcome,
    AggregationGenerationStatus,
    PlaybookAggregator,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage.playbook._aggregation import (
    init_playbook_aggregation_tables,
)
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationBacklog,
)


def _store(tmp_path) -> SQLiteStorage:
    return SQLiteStorage(org_id="aggregation-org", db_path=str(tmp_path / "state.db"))


@pytest.fixture
def vec_store(tmp_path) -> SQLiteStorage:
    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    return store


def test_lineage_silent_bulk_delete_callers_only_remove_ineligible_rows() -> None:
    package_root = Path(reflexio.__file__).parent
    call_sites: list[tuple[str, str]] = []
    cleanup_statuses: list[str] = []

    class CallVisitor(ast.NodeVisitor):
        def __init__(self, relative_path: str) -> None:
            self.relative_path = relative_path
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "delete_all_user_playbooks_by_status":
                    function_name = (
                        self.function_stack[-1] if self.function_stack else "<module>"
                    )
                    call_sites.append((self.relative_path, function_name))
                elif node.func.attr == "_delete_items_by_status" and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Attribute):
                        cleanup_statuses.append(argument.attr)
            self.generic_visit(node)

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        CallVisitor(str(path.relative_to(package_root))).visit(tree)

    assert sorted(call_sites) == [
        ("server/services/playbook/service.py", "_delete_items_by_status"),
        ("server/services/playbook/service.py", "_pre_process_rerun"),
    ]
    assert sorted(cleanup_statuses) == ["ARCHIVED"]


def _insert_current(
    store: SQLiteStorage,
    item_id: int,
    *,
    version: str = "v1",
    embedding: str | None = None,
    trigger: str | None = None,
) -> None:
    store.conn.execute(
        "INSERT INTO user_playbooks "
        "(user_playbook_id, user_id, playbook_name, created_at, request_id, "
        "agent_version, content, embedding, trigger) VALUES (?, ?, 'playbook', "
        "'2026-07-31T00:00:00+00:00', ?, ?, ?, ?, ?)",
        (
            item_id,
            f"user-{item_id}",
            f"req-{item_id}",
            version,
            f"rule-{item_id}",
            embedding,
            trigger,
        ),
    )
    store.conn.commit()


def test_create_schedules_intake_without_queueing_invalidation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    playbook = UserPlaybook(
        user_id="new-user",
        agent_version="v1",
        request_id="new-request",
        playbook_name="playbook",
        content="Newly discovered behavior.",
        trigger="when new work arrives",
        source="api",
    )

    store.save_user_playbooks([playbook])

    assert store.get_playbook_aggregation_invalidations("v1", limit=10) == []
    assert tuple(
        store.conn.execute(
            "SELECT pending, next_attempt_at IS NOT NULL "
            "FROM playbook_aggregation_state WHERE agent_version='v1'"
        ).fetchone()
    ) == (1, 1)


@pytest.mark.parametrize(
    ("operation", "expected_reason"),
    [
        ("reject", "agent_playbook_changed"),
        ("edit", "agent_playbook_changed"),
        ("archive", "agent_playbook_changed"),
        ("delete", "agent_playbook_deleted"),
    ],
)
def test_agent_mutation_retires_active_aggregation_cluster(
    vec_store,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_reason: str,
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    _insert_current(
        store,
        1,
        embedding=json.dumps(embedding),
        trigger="same trigger",
    )
    store.stage_playbook_aggregation_intake("v1", limit=1)
    agent = store.save_agent_playbooks(
        [
            AgentPlaybook(
                agent_version="v1",
                playbook_name="playbook",
                content="Current aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.create_playbook_aggregation_cluster(
        cluster_id="mutable-agent-cluster",
        agent_version="v1",
        agent_playbook_id=agent.agent_playbook_id,
        centroid_embedding=agent.embedding,
        member_count=1,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1",
        [1],
        disposition="cluster_member",
        cluster_id="mutable-agent-cluster",
    )
    indexes = {
        str(row[1])
        for row in store.conn.execute(
            "PRAGMA index_list(playbook_aggregation_cluster)"
        ).fetchall()
    }
    assert "idx_playbook_aggregation_cluster_agent" in indexes

    if operation == "reject":
        store.update_agent_playbook_status(
            agent.agent_playbook_id, PlaybookStatus.REJECTED
        )
    elif operation == "edit":
        store.update_agent_playbook(agent.agent_playbook_id, trigger="edited trigger")
    elif operation == "archive":
        store.archive_agent_playbooks_by_ids([agent.agent_playbook_id])
    else:
        store.delete_agent_playbook(agent.agent_playbook_id)

    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 0
    )
    assert tuple(
        store.conn.execute(
            "SELECT disposition, cluster_id, reason, attempt_count, last_attempt_at "
            "FROM playbook_aggregation_item WHERE user_playbook_id=1"
        ).fetchone()
    ) == ("residual", None, expected_reason, 0, None)
    assert (
        store.conn.execute(
            "SELECT pending FROM playbook_aggregation_state WHERE agent_version='v1'"
        ).fetchone()[0]
        == 1
    )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_clusters_vec"
        ).fetchone()[0]
        == 1
    )
    store.repair_playbook_aggregation_pending_state(limit=1)
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_clusters_vec"
        ).fetchone()[0]
        == 0
    )


def test_intake_durably_keeps_only_the_newest_unclustered_window(tmp_path) -> None:
    store = _store(tmp_path)
    for item_id in range(10, 17):
        _insert_current(store, item_id)

    assert store.stage_playbook_aggregation_intake("v1", limit=3, window_limit=5) == [
        16,
        15,
        14,
    ]
    assert store.stage_playbook_aggregation_intake("v1", limit=3, window_limit=5) == [
        13,
        12,
    ]

    # A later low-ID commit is permanently below the cutoff. A genuinely newer
    # row advances the window and terminalizes the residual that fell out.
    _insert_current(store, 2)
    _insert_current(store, 17)
    assert store.stage_playbook_aggregation_intake("v1", limit=3, window_limit=5) == [
        17
    ]
    assert store.stage_playbook_aggregation_intake("v1", limit=3, window_limit=5) == []
    assert (
        store.conn.execute(
            "SELECT intake_floor_user_playbook_id FROM playbook_aggregation_state "
            "WHERE agent_version='v1'"
        ).fetchone()[0]
        == 13
    )
    assert tuple(
        store.conn.execute(
            "SELECT disposition, reason FROM playbook_aggregation_item "
            "WHERE agent_version='v1' AND user_playbook_id=12"
        ).fetchone()
    ) == ("terminal_noop", "outside_recent_clustering_window")
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_item WHERE user_playbook_id=2"
        ).fetchone()[0]
        == 0
    )
    assert store.get_playbook_aggregation_backlog("v1").undisposed == 0


def test_org_claim_is_fenced_and_fence_is_monotonic(tmp_path) -> None:
    store = _store(tmp_path)
    store.schedule_playbook_aggregation("v2")
    store.schedule_playbook_aggregation("v1")
    store.conn.execute(
        "UPDATE playbook_aggregation_state SET next_attempt_at="
        "(SELECT min(next_attempt_at) FROM playbook_aggregation_state)"
    )
    store.conn.commit()

    first = store.claim_due_playbook_aggregation(owner="worker-a", lease_seconds=30)
    assert first is not None
    assert first.agent_version == "v1"
    assert (
        store.claim_due_playbook_aggregation(owner="worker-b", lease_seconds=30) is None
    )
    assert store.finish_playbook_aggregation_claim(
        first,
        success=True,
        retry_after_seconds=5,
        backlog_retry_after_seconds=1,
        min_interval_seconds=60,
    )

    second = store.claim_due_playbook_aggregation(owner="worker-b", lease_seconds=30)
    assert second is not None
    assert second.agent_version == "v2"
    assert second.fence > first.fence
    assert not store.finish_playbook_aggregation_claim(
        first,
        success=True,
        retry_after_seconds=5,
        backlog_retry_after_seconds=1,
        min_interval_seconds=60,
    )


def test_failed_finish_cas_rolls_back_owned_sqlite_transaction(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.schedule_playbook_aggregation("v1")
    claim = store.claim_due_playbook_aggregation(owner="worker", lease_seconds=30)
    assert claim is not None
    external = sqlite3.connect(str(tmp_path / "state.db"), timeout=0.1)
    original_is_live = store._aggregation_claim_is_live

    def race_state_version(candidate) -> bool:
        is_live = original_is_live(candidate)
        external.execute(
            "UPDATE playbook_aggregation_state SET state_version=state_version+1 "
            "WHERE agent_version='v1'"
        )
        external.commit()
        return is_live

    monkeypatch.setattr(store, "_aggregation_claim_is_live", race_state_version)

    assert not store.finish_playbook_aggregation_claim(
        claim,
        success=False,
        retry_after_seconds=60,
        backlog_retry_after_seconds=1,
        min_interval_seconds=3600,
    )
    assert store.conn.in_transaction is False
    external.execute(
        "UPDATE playbook_aggregation_state SET pending=1 WHERE agent_version='v1'"
    )
    external.commit()
    external.close()


def test_successful_bounded_run_with_backlog_is_due_again_immediately(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1)
    store.schedule_playbook_aggregation("v1")
    claim = store.claim_due_playbook_aggregation(owner="worker-a", lease_seconds=30)
    assert claim is not None
    assert store.stage_playbook_aggregation_intake("v1", limit=1) == [1]

    assert store.finish_playbook_aggregation_claim(
        claim,
        success=True,
        retry_after_seconds=60,
        backlog_retry_after_seconds=0,
        min_interval_seconds=3600,
    )

    continuation = store.claim_due_playbook_aggregation(
        owner="worker-b", lease_seconds=30
    )
    assert continuation is not None
    assert continuation.agent_version == "v1"


def test_new_signal_preserves_idle_version_hourly_window(tmp_path) -> None:
    store = _store(tmp_path)
    store.schedule_playbook_aggregation("v1")
    claim = store.claim_due_playbook_aggregation(owner="worker-a", lease_seconds=30)
    assert claim is not None
    assert store.finish_playbook_aggregation_claim(
        claim,
        success=True,
        retry_after_seconds=60,
        backlog_retry_after_seconds=1,
        min_interval_seconds=3600,
        backlog=PlaybookAggregationBacklog(undisposed=0, residual=0, invalidations=0),
    )

    store.schedule_playbook_aggregation("v1")

    assert (
        store.claim_due_playbook_aggregation(owner="worker-b", lease_seconds=30) is None
    )


def test_stale_claim_cannot_commit_incremental_effect(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    _insert_current(store, 1, embedding=encoded, trigger="same trigger")
    _insert_current(store, 2, embedding=encoded, trigger="same trigger")
    store.schedule_playbook_aggregation("v1")
    stale = store.claim_due_playbook_aggregation(owner="stale", lease_seconds=30)
    assert stale is not None
    store.conn.execute(
        "UPDATE playbook_aggregation_lease SET claim_expires_at=0 WHERE singleton=1"
    )
    store.conn.commit()
    replacement = store.claim_due_playbook_aggregation(
        owner="replacement", lease_seconds=30
    )
    assert replacement is not None
    assert replacement.fence > stale.fence

    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
        aggregation_claim=stale,
    )

    with pytest.raises(RuntimeError, match="lost its database fence"):
        aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 0


def test_archive_status_and_invalidation_commit_together(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 7)

    assert store.archive_user_playbook_by_id("user-7", 7)
    status = store.conn.execute(
        "SELECT status FROM user_playbooks WHERE user_playbook_id=7"
    ).fetchone()[0]
    event = store.conn.execute(
        "SELECT operation, entity_id FROM playbook_aggregation_invalidation"
    ).fetchone()
    assert status == "archived"
    assert tuple(event) == ("archive", 7)


def test_archive_rolls_back_when_invalidation_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    _insert_current(store, 8)

    def fail_invalidation(**_kwargs: object) -> None:
        raise RuntimeError("injected invalidation failure")

    monkeypatch.setattr(
        store, "append_playbook_aggregation_invalidation", fail_invalidation
    )
    with pytest.raises(Exception, match="injected invalidation failure"):
        store.archive_user_playbook_by_id("user-8", 8)

    status = store.conn.execute(
        "SELECT status FROM user_playbooks WHERE user_playbook_id=8"
    ).fetchone()[0]
    assert status is None


def test_merge_invalidates_every_source_agent_version(tmp_path) -> None:
    store = _store(tmp_path)
    for item_id, version in ((1, "v2"), (2, "v1"), (3, "v2")):
        _insert_current(store, item_id, version=version)

    store.merge_records(
        entity_type="user_playbook",
        survivor_id="1",
        source_ids=["2", "3"],
        context=LineageContext(
            op_kind="merge", actor="test", request_id="mixed-version-merge"
        ),
    )

    invalidations = store.conn.execute(
        "SELECT agent_version, entity_id, source_ids "
        "FROM playbook_aggregation_invalidation ORDER BY agent_version"
    ).fetchall()
    assert [tuple(row) for row in invalidations] == [
        ("v1", 1, "[2]"),
        ("v2", 1, "[3]"),
    ]
    armed_versions = store.conn.execute(
        "SELECT agent_version FROM playbook_aggregation_state "
        "WHERE pending=1 ORDER BY agent_version"
    ).fetchall()
    assert [str(row[0]) for row in armed_versions] == ["v1", "v2"]


def test_aggregation_schema_init_does_not_rebuild_pending_index(tmp_path) -> None:
    store = _store(tmp_path)
    statements: list[str] = []
    store.conn.set_trace_callback(statements.append)
    try:
        init_playbook_aggregation_tables(store.conn)
    finally:
        store.conn.set_trace_callback(None)

    assert not any(
        "DROP INDEX" in statement
        and "idx_playbook_aggregation_invalidation_pending" in statement
        for statement in statements
    )
    index_sql = store.conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='index' AND name='idx_playbook_aggregation_invalidation_pending'"
    ).fetchone()[0]
    assert "WHERE processed_at IS NULL" in index_sql


def test_semantic_dispositions_are_unique_per_version_and_item(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    store.set_playbook_aggregation_disposition(
        "v1", [1], disposition="terminal_noop", reason="semantic_null"
    )
    store.set_playbook_aggregation_disposition(
        "v1", [1], disposition="residual", reason="retry"
    )

    rows = store.conn.execute(
        "SELECT disposition FROM playbook_aggregation_item "
        "WHERE agent_version='v1' AND user_playbook_id=1"
    ).fetchall()
    assert [row[0] for row in rows] == ["residual"]


def test_residual_selection_reserves_fresh_and_retry_capacity(tmp_path) -> None:
    store = _store(tmp_path)
    for item_id in range(1, 5):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    assert set(store.get_playbook_aggregation_residual_ids("v1", limit=4)) == {
        1,
        2,
        3,
        4,
    }

    for item_id in range(5, 9):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    store.conn.execute(
        "UPDATE playbook_aggregation_item SET last_attempt_at=unixepoch()-61 "
        "WHERE agent_version='v1' AND user_playbook_id <= 4"
    )
    selected = set(store.get_playbook_aggregation_residual_ids("v1", limit=4))
    assert len(selected & {1, 2, 3, 4}) == 2
    assert len(selected & {5, 6, 7, 8}) == 2


def test_residual_retry_is_deferred_until_identical_work_cools_down(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1)
    store.stage_playbook_aggregation_intake("v1", limit=1)

    assert store.get_playbook_aggregation_residual_ids("v1", limit=1) == [1]
    assert store.get_playbook_aggregation_residual_ids("v1", limit=1) == []
    backlog = store.get_playbook_aggregation_backlog("v1")
    assert 1 <= backlog.residual_retry_after_seconds <= 60
    assert backlog.continuation_delay_seconds == backlog.residual_retry_after_seconds


def test_residual_retry_cooldown_is_aggregated_in_sql(tmp_path) -> None:
    store = _store(tmp_path)
    for item_id in range(1, 101):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=100)
    store.conn.execute(
        "UPDATE playbook_aggregation_item SET attempt_count=2, "
        "last_attempt_at=unixepoch() WHERE agent_version='v1'"
    )
    store.conn.commit()
    statements: list[str] = []
    store.conn.set_trace_callback(statements.append)

    backlog = store.get_playbook_aggregation_backlog("v1")

    store.conn.set_trace_callback(None)
    assert 61 <= backlog.residual_retry_after_seconds <= 120
    assert any("COALESCE(MAX(0, MIN(CASE" in statement for statement in statements)


def test_discovery_repairs_lost_post_commit_signal(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 42, version="lost-signal")

    assert store.repair_playbook_aggregation_pending_state() == ["lost-signal"]
    claim = store.claim_due_playbook_aggregation(owner="repair", lease_seconds=30)
    assert claim is not None
    assert claim.agent_version == "lost-signal"


def test_repair_prunes_only_expired_processed_invalidations(tmp_path) -> None:
    store = _store(tmp_path)
    store.conn.executemany(
        "INSERT INTO playbook_aggregation_invalidation "
        "(agent_version, operation, entity_id, processed_at) "
        "VALUES ('v1', 'revise', ?, unixepoch()-?)",
        [(1, 8 * 24 * 60 * 60), (2, 24 * 60 * 60)],
    )

    store.repair_playbook_aggregation_pending_state()

    rows = store.conn.execute(
        "SELECT entity_id FROM playbook_aggregation_invalidation ORDER BY entity_id"
    ).fetchall()
    assert [int(row[0]) for row in rows] == [2]


def test_dirty_cluster_keeps_aggregation_pending() -> None:
    backlog = PlaybookAggregationBacklog(
        undisposed=0,
        residual=0,
        invalidations=0,
        dirty_repairs=1,
    )

    assert backlog.pending is True


def test_sqlite_vec_centroid_lookup_and_attachment(vec_store) -> None:
    store = vec_store
    for item_id in (1, 2, 3):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    cluster_id = "cluster-a"
    store.create_playbook_aggregation_cluster(
        cluster_id=cluster_id,
        agent_version="v1",
        agent_playbook_id=101,
        centroid_embedding=embedding,
        member_count=2,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1", [1, 2], disposition="cluster_member", cluster_id=cluster_id
    )
    vec_schema = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='playbook_aggregation_clusters_vec'"
    ).fetchone()[0]
    assert "distance_metric=cosine" in vec_schema
    scaled_embedding = [value * 10 for value in embedding]

    matches = store.find_nearest_playbook_aggregation_clusters(
        "v1",
        [(3, scaled_embedding)],
        embedding_model=store.embedding_model_name,
        limit=10,
    )
    match = matches[3]
    assert match is not None
    assert match.cluster_id == cluster_id
    assert match.similarity > 0.99
    store.attach_playbook_aggregation_items(
        agent_version="v1",
        attachments=[(3, cluster_id)],
    )
    assert (
        store.conn.execute(
            "SELECT member_count FROM playbook_aggregation_cluster WHERE cluster_id=?",
            (cluster_id,),
        ).fetchone()[0]
        == 3
    )
    replacement_embedding = [0.0] * store.embedding_dimensions
    replacement_embedding[1] = 1.0
    store.replace_playbook_aggregation_cluster_agent(
        cluster_id=cluster_id,
        agent_version="v1",
        expected_agent_playbook_id=101,
        replacement_agent_playbook_id=102,
        centroid_embedding=replacement_embedding,
        embedding_model=store.embedding_model_name,
    )
    refreshed = store.conn.execute(
        "SELECT agent_playbook_id, centroid, vector_sum FROM "
        "playbook_aggregation_cluster WHERE cluster_id=?",
        (cluster_id,),
    ).fetchone()
    assert refreshed[0] == 102
    assert json.loads(refreshed[1]) == replacement_embedding
    assert refreshed[2] is None
    replacement_match = store.find_nearest_playbook_aggregation_clusters(
        "v1",
        [(3, replacement_embedding)],
        embedding_model=store.embedding_model_name,
        limit=10,
    )[3]
    assert replacement_match.agent_playbook_id == 102
    assert replacement_match.similarity > 0.99


def test_centroid_migration_skips_unmigratable_legacy_rows(
    vec_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    agents = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content=f"legacy agent {index}",
                playbook_status=PlaybookStatus.PENDING,
            )
            for index in range(2)
        ]
    )
    store.conn.execute(
        "UPDATE agent_playbooks SET embedding=? WHERE agent_playbook_id=?",
        (json.dumps(embedding), agents[0].agent_playbook_id),
    )
    store.conn.execute(
        "UPDATE agent_playbooks SET embedding='[1.0]' WHERE agent_playbook_id=?",
        (agents[1].agent_playbook_id,),
    )
    store.conn.executemany(
        "INSERT INTO playbook_aggregation_cluster "
        "(cluster_id, index_rowid, agent_version, agent_playbook_id, vector_sum, "
        "member_count, embedding_model, embedding_dimension, state) "
        "VALUES (?, ?, 'v1', ?, '[1.0]', 1, ?, ?, 'active')",
        [
            (
                "missing-rowid",
                None,
                agents[0].agent_playbook_id,
                store.embedding_model_name,
                store.embedding_dimensions,
            ),
            (
                "wrong-dimension",
                4242,
                agents[1].agent_playbook_id,
                store.embedding_model_name,
                store.embedding_dimensions,
            ),
        ],
    )
    store.conn.commit()

    store._migrate_playbook_aggregation_agent_centroids()

    rows = store.conn.execute(
        "SELECT cluster_id, centroid, vector_sum FROM "
        "playbook_aggregation_cluster ORDER BY cluster_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("missing-rowid", None, "[1.0]"),
        ("wrong-dimension", None, "[1.0]"),
    ]
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_clusters_vec WHERE rowid=4242"
        ).fetchone()[0]
        == 0
    )


def test_centroid_batches_are_strictly_isolated_by_agent_version(vec_store) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    other_embedding = [0.0] * store.embedding_dimensions
    other_embedding[1] = 1.0
    for item_id, version in ((1, "v1"), (2, "v2"), (3, "v2")):
        _insert_current(store, item_id, version=version)
        store.stage_playbook_aggregation_intake(version, limit=10)
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-v1",
        agent_version="v1",
        agent_playbook_id=101,
        centroid_embedding=embedding,
        member_count=1,
        embedding_model=store.embedding_model_name,
    )
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-v2",
        agent_version="v2",
        agent_playbook_id=202,
        centroid_embedding=other_embedding,
        member_count=1,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1", [1], disposition="cluster_member", cluster_id="cluster-v1"
    )
    store.set_playbook_aggregation_disposition(
        "v2", [2], disposition="cluster_member", cluster_id="cluster-v2"
    )

    matches = store.find_nearest_playbook_aggregation_clusters(
        "v2",
        [(3, embedding)],
        embedding_model=store.embedding_model_name,
        limit=1,
    )
    assert matches[3].cluster_id == "cluster-v2"
    with pytest.raises(RuntimeError, match="not active for agent version"):
        store.attach_playbook_aggregation_items(
            agent_version="v2",
            attachments=[(3, "cluster-v1")],
        )
    disposition = store.conn.execute(
        "SELECT disposition, cluster_id FROM playbook_aggregation_item "
        "WHERE agent_version='v2' AND user_playbook_id=3"
    ).fetchone()
    assert tuple(disposition) == ("residual", None)


def test_invalidation_dissolves_cluster_and_requeues_unaffected_members(
    vec_store,
) -> None:
    store = vec_store
    for item_id in (1, 2):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-to-dissolve",
        agent_version="v1",
        agent_playbook_id=101,
        centroid_embedding=embedding,
        member_count=2,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1",
        [1, 2],
        disposition="cluster_member",
        cluster_id="cluster-to-dissolve",
    )
    store.append_playbook_aggregation_invalidation(
        agent_version="v1", operation="status_change", entity_id=1
    )
    claim = store.claim_due_playbook_aggregation(owner="worker", lease_seconds=30)
    assert claim is not None
    invalidations = store.get_playbook_aggregation_invalidations("v1", limit=10)

    assert store.apply_playbook_aggregation_invalidations(
        claim, [item.invalidation_id for item in invalidations]
    )
    cluster = store.conn.execute(
        "SELECT state, dirty, member_count FROM playbook_aggregation_cluster"
    ).fetchone()
    assert tuple(cluster) == ("rebuilding", 1, 0)
    rows = store.conn.execute(
        "SELECT user_playbook_id, disposition, cluster_id "
        "FROM playbook_aggregation_item ORDER BY user_playbook_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(2, "residual", "cluster-to-dissolve")]
    assert store.get_playbook_aggregation_backlog("v1").invalidations == 0

    # Rebuilding the same deterministic membership must reactivate the retained
    # cluster row rather than failing its cluster_id uniqueness constraint.
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-to-dissolve",
        agent_version="v1",
        agent_playbook_id=102,
        centroid_embedding=embedding,
        member_count=1,
        embedding_model=store.embedding_model_name,
    )
    rebuilt = store.conn.execute(
        "SELECT state, dirty, member_count FROM playbook_aggregation_cluster"
    ).fetchone()
    assert tuple(rebuilt) == ("active", 0, 1)


def test_hard_delete_trigger_preserves_membership_until_reconciliation(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1)
    store.stage_playbook_aggregation_intake("v1", limit=1)

    store.delete_user_playbook(1)

    assert (
        store.conn.execute(
            "SELECT disposition FROM playbook_aggregation_item WHERE user_playbook_id=1"
        ).fetchone()[0]
        == "residual"
    )
    invalidations = store.get_playbook_aggregation_invalidations("v1", limit=10)
    assert [(item.operation, item.entity_id) for item in invalidations] == [
        ("hard_delete", 1)
    ]
    claim = store.claim_due_playbook_aggregation(owner="worker", lease_seconds=30)
    assert claim is not None
    assert store.apply_playbook_aggregation_invalidations(
        claim, [item.invalidation_id for item in invalidations]
    )
    assert (
        store.conn.execute("SELECT count(*) FROM playbook_aggregation_item").fetchone()[
            0
        ]
        == 0
    )


def test_archive_with_blank_agent_version_does_not_arm_aggregation(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1, version="   ")

    assert store.archive_user_playbook_by_id("user-1", 1)
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_invalidation"
        ).fetchone()[0]
        == 0
    )


def test_nonnumeric_lineage_entity_does_not_abort_on_numeric_source(tmp_path) -> None:
    from reflexio.server.services.storage.sqlite_storage._lineage import (
        _append_event_stmt,
    )

    store = _store(tmp_path)
    _insert_current(store, 1)

    _append_event_stmt(
        store.conn,
        org_id=store.org_id,
        entity_type="user_playbook",
        entity_id="external-id",
        op="revise",
        prov="wasRevisionOf",
        source_ids=["1"],
        actor="test",
        request_id="nonnumeric-entity",
        reason="regression",
    )
    store.conn.commit()

    assert (
        store.conn.execute(
            "SELECT count(*) FROM lineage_event WHERE entity_id='external-id'"
        ).fetchone()[0]
        == 1
    )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_invalidation"
        ).fetchone()[0]
        == 0
    )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_state"
        ).fetchone()[0]
        == 0
    )


def test_incremental_run_refreshes_agent_and_centroid_after_match(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    _insert_current(store, 1, embedding=encoded, trigger="same trigger")
    _insert_current(store, 2, embedding=encoded, trigger="same trigger")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    context = MagicMock(
        org_id="aggregation-org", storage=store, configurator=configurator
    )
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,
        agent_version="v1",
    )
    learning_meter = MagicMock()
    monkeypatch.setattr(aggregator, "_record_learnings_generated", learning_meter)

    first = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))
    assert first["playbooks_generated"] == 1
    first_saved_id = store.conn.execute(
        "SELECT agent_playbook_id FROM agent_playbooks"
    ).fetchone()[0]
    learning_meter.assert_called_once_with(
        learning_ids=[str(first_saved_id)],
        playbook_name="playbook",
        request_id=learning_meter.call_args.kwargs["request_id"],
        metadata=first,
        total_count=1,
    )
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1

    _insert_current(store, 3, embedding=encoded, trigger="same trigger")
    _insert_current(store, 4, embedding=encoded, trigger="same trigger")
    second = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))
    assert second["playbooks_generated"] == 1
    assert second["attachments"] == 2
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 2
    assert (
        store.conn.execute(
            "SELECT member_count FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 4
    )
    agent_rows = store.conn.execute(
        "SELECT agent_playbook_id, status FROM agent_playbooks "
        "ORDER BY agent_playbook_id"
    ).fetchall()
    assert agent_rows[0][1] == Status.SUPERSEDED.value
    assert agent_rows[1][1] is None
    cluster_agent_id = store.conn.execute(
        "SELECT agent_playbook_id FROM playbook_aggregation_cluster"
    ).fetchone()[0]
    assert cluster_agent_id == agent_rows[1][0]

    old_agent_id = int(
        store.conn.execute(
            "SELECT agent_playbook_id FROM agent_playbooks WHERE status IS NULL"
        ).fetchone()[0]
    )
    store.conn.execute(
        "UPDATE user_playbooks SET status=? WHERE user_playbook_id=1",
        (Status.SUPERSEDED.value,),
    )
    _insert_current(store, 5, embedding=encoded, trigger="same trigger")
    store.append_playbook_aggregation_invalidation(
        agent_version="v1", operation="revise", entity_id=5, source_ids=[1]
    )
    claim = store.claim_due_playbook_aggregation(owner="replacement", lease_seconds=30)
    assert claim is not None
    invalidations = store.get_playbook_aggregation_invalidations("v1", limit=10)
    assert store.apply_playbook_aggregation_invalidations(
        claim, [item.invalidation_id for item in invalidations]
    )

    replacement = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,
        agent_version="v1",
        aggregation_claim=claim,
        residual_batch_limit=1,
    ).run(PlaybookAggregatorRequest(agent_version="v1"))
    assert replacement["playbooks_generated"] == 1
    assert replacement["supersessions"] == 1
    assert replacement["rebuilt_members"] == 4
    rows = store.conn.execute(
        "SELECT agent_playbook_id, status FROM agent_playbooks "
        "ORDER BY agent_playbook_id"
    ).fetchall()
    assert rows[1][0] == old_agent_id
    assert rows[1][1] == Status.SUPERSEDED.value
    assert rows[2][1] is None
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 1
    )

    for item_id in (2, 3, 5):
        assert store.archive_user_playbook_by_id(f"user-{item_id}", item_id)
    removal_invalidations = store.get_playbook_aggregation_invalidations("v1", limit=10)
    assert store.apply_playbook_aggregation_invalidations(
        claim, [item.invalidation_id for item in removal_invalidations]
    )
    single_member_rebuild = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,
        agent_version="v1",
        aggregation_claim=claim,
        residual_batch_limit=10,
    ).run(PlaybookAggregatorRequest(agent_version="v1"))
    assert single_member_rebuild["playbooks_generated"] == 1
    assert single_member_rebuild["supersessions"] == 1
    remaining = store.conn.execute(
        "SELECT i.user_playbook_id, i.disposition, c.member_count "
        "FROM playbook_aggregation_item i JOIN playbook_aggregation_cluster c "
        "ON c.cluster_id=i.cluster_id WHERE i.agent_version='v1'"
    ).fetchall()
    assert [tuple(row) for row in remaining] == [(4, "cluster_member", 1)]

    current_agent_id = int(
        store.conn.execute(
            "SELECT agent_playbook_id FROM agent_playbooks WHERE status IS NULL"
        ).fetchone()[0]
    )
    assert store.archive_user_playbook_by_id("user-4", 4)
    final_invalidations = store.get_playbook_aggregation_invalidations("v1", limit=10)
    assert store.apply_playbook_aggregation_invalidations(
        claim, [item.invalidation_id for item in final_invalidations]
    )
    empty_rebuild = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,
        agent_version="v1",
        aggregation_claim=claim,
        residual_batch_limit=10,
    ).run(PlaybookAggregatorRequest(agent_version="v1"))
    assert empty_rebuild["playbooks_generated"] == 0
    assert empty_rebuild["supersessions"] == 1
    assert (
        store.conn.execute(
            "SELECT status FROM agent_playbooks WHERE agent_playbook_id=?",
            (current_agent_id,),
        ).fetchone()[0]
        == Status.SUPERSEDED.value
    )
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 0
    )


def test_rebuild_uses_recent_top_100_but_restores_every_member(
    vec_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    member_ids = list(range(1, 106))
    for item_id in member_ids:
        _insert_current(store, item_id, embedding=encoded, trigger="same trigger")
    store.conn.execute(
        "UPDATE user_playbooks SET created_at='2026-08-01T00:00:00+00:00' "
        "WHERE user_playbook_id=1"
    )
    store.conn.commit()
    store.stage_playbook_aggregation_intake("v1", limit=200)
    current = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Current aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.create_playbook_aggregation_cluster(
        cluster_id="repair-cluster",
        agent_version="v1",
        agent_playbook_id=current.agent_playbook_id,
        centroid_embedding=current.embedding,
        member_count=len(member_ids),
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1",
        member_ids,
        disposition="cluster_member",
        cluster_id="repair-cluster",
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_item SET disposition='residual' "
        "WHERE cluster_id='repair-cluster'"
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_cluster SET state='rebuilding', dirty=1, "
        "centroid=NULL, member_count=0 WHERE cluster_id='repair-cluster'"
    )
    store.conn.commit()

    samples = store.get_playbook_aggregation_rebuild_samples(
        "v1", ["repair-cluster"], limit_per_cluster=100
    )
    assert len(samples) == 1
    assert samples[0].member_ids == (1, *range(105, 6, -1))

    replacement = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Rebuilt aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    rebuilt = store.complete_playbook_aggregation_cluster_rebuild(
        cluster_id="repair-cluster",
        agent_version="v1",
        expected_agent_playbook_id=current.agent_playbook_id,
        replacement_agent_playbook_id=replacement.agent_playbook_id,
        centroid_embedding=replacement.embedding,
        embedding_model=store.embedding_model_name,
    )
    assert rebuilt == 105
    assert tuple(
        store.conn.execute(
            "SELECT state, member_count, agent_playbook_id FROM "
            "playbook_aggregation_cluster WHERE cluster_id='repair-cluster'"
        ).fetchone()
    ) == ("active", 105, replacement.agent_playbook_id)
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_item WHERE "
            "cluster_id='repair-cluster' AND disposition='cluster_member'"
        ).fetchone()[0]
        == 105
    )


def test_rebuild_failure_defers_the_entire_cluster(
    vec_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    for item_id in (1, 2):
        _insert_current(store, item_id, embedding=encoded, trigger="same trigger")
    store.stage_playbook_aggregation_intake("v1", limit=10)
    current = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Current aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.create_playbook_aggregation_cluster(
        cluster_id="deferred-repair",
        agent_version="v1",
        agent_playbook_id=current.agent_playbook_id,
        centroid_embedding=current.embedding,
        member_count=2,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1",
        [1, 2],
        disposition="cluster_member",
        cluster_id="deferred-repair",
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_item SET disposition='residual' "
        "WHERE cluster_id='deferred-repair'"
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_cluster SET state='rebuilding', dirty=1 "
        "WHERE cluster_id='deferred-repair'"
    )
    store.conn.commit()

    store.defer_playbook_aggregation_cluster_rebuild(
        cluster_id="deferred-repair",
        agent_version="v1",
        expected_agent_playbook_id=current.agent_playbook_id,
        reason="llm_retryable_failure",
    )

    assert store.get_playbook_aggregation_residual_ids("v1", limit=10) == []
    backlog = store.get_playbook_aggregation_backlog("v1")
    assert backlog.dirty_repairs == 1
    assert 1 <= backlog.repair_retry_after_seconds <= 60
    assert backlog.continuation_delay_seconds == backlog.repair_retry_after_seconds
    item_rows = store.conn.execute(
        "SELECT reason, attempt_count, last_attempt_at FROM "
        "playbook_aggregation_item WHERE cluster_id='deferred-repair'"
    ).fetchall()
    assert [tuple(row) for row in item_rows] == [
        ("llm_retryable_failure", 0, None),
        ("llm_retryable_failure", 0, None),
    ]

    store.conn.execute(
        "UPDATE playbook_aggregation_cluster SET rebuild_next_attempt_at=0 "
        "WHERE cluster_id='deferred-repair'"
    )
    store.conn.commit()
    assert store.get_playbook_aggregation_residual_ids("v1", limit=10) == [1, 2]


def test_rebuild_completion_rolls_back_items_when_cluster_fence_is_lost(
    vec_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    _insert_current(store, 1, embedding=json.dumps(embedding))
    store.stage_playbook_aggregation_intake("v1", limit=1)
    current = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Current aggregate",
                trigger="current trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.create_playbook_aggregation_cluster(
        cluster_id="rollback-repair",
        agent_version="v1",
        agent_playbook_id=current.agent_playbook_id,
        centroid_embedding=current.embedding,
        member_count=1,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1",
        [1],
        disposition="cluster_member",
        cluster_id="rollback-repair",
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_item SET disposition='residual' "
        "WHERE cluster_id='rollback-repair'"
    )
    store.conn.execute(
        "UPDATE playbook_aggregation_cluster SET state='rebuilding', dirty=1 "
        "WHERE cluster_id='rollback-repair'"
    )
    store.conn.executescript(
        "CREATE TRIGGER ignore_rebuild_completion "
        "BEFORE UPDATE OF agent_playbook_id ON playbook_aggregation_cluster "
        "WHEN OLD.cluster_id='rollback-repair' AND NEW.state='active' BEGIN "
        "SELECT RAISE(IGNORE); END;"
    )
    store.conn.commit()

    with pytest.raises(RuntimeError, match="aggregation rebuilding cluster changed"):
        store.complete_playbook_aggregation_cluster_rebuild(
            cluster_id="rollback-repair",
            agent_version="v1",
            expected_agent_playbook_id=current.agent_playbook_id,
            replacement_agent_playbook_id=current.agent_playbook_id + 1,
            centroid_embedding=embedding,
            embedding_model=store.embedding_model_name,
        )

    assert store.conn.in_transaction is False
    assert tuple(
        store.conn.execute(
            "SELECT disposition, cluster_id FROM playbook_aggregation_item "
            "WHERE user_playbook_id=1"
        ).fetchone()
    ) == ("residual", "rollback-repair")
    assert tuple(
        store.conn.execute(
            "SELECT state, agent_playbook_id FROM playbook_aggregation_cluster "
            "WHERE cluster_id='rollback-repair'"
        ).fetchone()
    ) == ("rebuilding", current.agent_playbook_id)


def test_rebuild_dedup_context_excludes_the_invalidated_agent(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    _insert_current(store, 1, embedding=encoded, trigger="same trigger")
    source = store.get_user_playbooks_by_ids_any_user(
        [1], status_filter=[None], include_embedding=True
    )[0]
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    current, other = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Guidance that was invalidated",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            ),
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Independent existing guidance",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            ),
        ]
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
    )
    captured: dict[str, str] = {}

    def generate(_cluster, approved_playbooks: str, **_kwargs):
        captured["approved"] = approved_playbooks
        return AggregationGenerationOutcome("semantic_null", [source])

    monkeypatch.setattr(aggregator, "_generate_playbook_from_cluster_outcome", generate)
    aggregator._generate_playbook_outcomes_with_source_clusters(
        {0: [source]},
        [current, other],
        excluded_existing_playbook_ids={0: {current.agent_playbook_id}},
    )

    assert "Guidance that was invalidated" not in captured["approved"]
    assert "Independent existing guidance" in captured["approved"]


@pytest.mark.parametrize(
    ("outcome_status", "expected_disposition", "expected_members"),
    [
        ("semantic_null", "cluster_member", 3),
        ("retryable_failure", "residual", 2),
    ],
)
def test_matched_cluster_non_generated_outcomes_preserve_current_agent(
    vec_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    outcome_status: AggregationGenerationStatus,
    expected_disposition: str,
    expected_members: int,
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    for item_id in (1, 2, 3):
        _insert_current(store, item_id, embedding=encoded, trigger="same trigger")
    store.stage_playbook_aggregation_intake("v1", limit=10)
    current = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Current aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    cluster_id = "stable-cluster"
    store.create_playbook_aggregation_cluster(
        cluster_id=cluster_id,
        agent_version="v1",
        agent_playbook_id=current.agent_playbook_id,
        centroid_embedding=current.embedding,
        member_count=2,
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1", [1, 2], disposition="cluster_member", cluster_id=cluster_id
    )
    store.set_playbook_aggregation_bootstrap_status("v1", "complete")

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
    )

    def generate_outcomes(
        clusters,
        _existing,
        *,
        direction_overlap_threshold,
        current_agent_playbooks=None,
    ):
        del direction_overlap_threshold
        if not clusters:
            return []
        assert current_agent_playbooks
        return [
            AggregationGenerationOutcome(outcome_status, members)
            for members in clusters.values()
        ]

    monkeypatch.setattr(
        aggregator,
        "_generate_playbook_outcomes_with_source_clusters",
        generate_outcomes,
    )
    result = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))

    row = store.conn.execute(
        "SELECT disposition, cluster_id FROM playbook_aggregation_item "
        "WHERE user_playbook_id=3"
    ).fetchone()
    assert tuple(row) == (
        expected_disposition,
        cluster_id if expected_disposition == "cluster_member" else None,
    )
    cluster = store.conn.execute(
        "SELECT agent_playbook_id, centroid, member_count FROM "
        "playbook_aggregation_cluster WHERE cluster_id=?",
        (cluster_id,),
    ).fetchone()
    assert cluster[0] == current.agent_playbook_id
    assert json.loads(cluster[1]) == current.embedding
    assert cluster[2] == expected_members
    assert result["attachments"] == (1 if outcome_status == "semantic_null" else 0)
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1


def test_refresh_fence_loss_does_not_discard_healthy_cluster(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    alpha_embedding = [0.0] * store.embedding_dimensions
    alpha_embedding[0] = 1.0
    beta_embedding = [0.0] * store.embedding_dimensions
    beta_embedding[1] = 1.0

    def embed(text: str) -> list[float]:
        return beta_embedding if "beta" in text.lower() else alpha_embedding

    monkeypatch.setattr(store, "_get_embedding", embed)
    _insert_current(store, 1, embedding=json.dumps(alpha_embedding))
    _insert_current(store, 2, embedding=json.dumps(beta_embedding))
    store.stage_playbook_aggregation_intake("v1", limit=10)
    alpha_agent, beta_agent = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Alpha guidance",
                trigger="alpha trigger",
                playbook_status=PlaybookStatus.PENDING,
            ),
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Beta guidance",
                trigger="beta trigger",
                playbook_status=PlaybookStatus.PENDING,
            ),
        ]
    )
    for cluster_id, agent, item_id in (
        ("cluster-alpha", alpha_agent, 1),
        ("cluster-beta", beta_agent, 2),
    ):
        store.create_playbook_aggregation_cluster(
            cluster_id=cluster_id,
            agent_version="v1",
            agent_playbook_id=agent.agent_playbook_id,
            centroid_embedding=agent.embedding,
            member_count=1,
            embedding_model=store.embedding_model_name,
        )
        store.set_playbook_aggregation_disposition(
            "v1",
            [item_id],
            disposition="cluster_member",
            cluster_id=cluster_id,
        )
    store.set_playbook_aggregation_bootstrap_status("v1", "complete")
    _insert_current(store, 3, embedding=json.dumps(alpha_embedding))
    _insert_current(store, 4, embedding=json.dumps(beta_embedding))

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
    )

    def generate_outcomes(
        clusters,
        _existing,
        *,
        direction_overlap_threshold,
        current_agent_playbooks=None,
    ) -> list[AggregationGenerationOutcome]:
        del direction_overlap_threshold
        assert current_agent_playbooks
        return [
            AggregationGenerationOutcome(
                "generated",
                members,
                AgentPlaybook(
                    playbook_name="playbook",
                    agent_version="v1",
                    content=(
                        "Refreshed beta guidance"
                        if current_agent_playbooks[index].agent_playbook_id
                        == beta_agent.agent_playbook_id
                        else "Refreshed alpha guidance"
                    ),
                    trigger=(
                        "refreshed beta trigger"
                        if current_agent_playbooks[index].agent_playbook_id
                        == beta_agent.agent_playbook_id
                        else "refreshed alpha trigger"
                    ),
                    playbook_status=PlaybookStatus.PENDING,
                ),
            )
            for index, members in clusters.items()
        ]

    monkeypatch.setattr(
        aggregator,
        "_generate_playbook_outcomes_with_source_clusters",
        generate_outcomes,
    )
    replace_cluster_agent = store.replace_playbook_aggregation_cluster_agent

    def replace_with_one_fence_loss(**kwargs) -> None:
        if kwargs["cluster_id"] == "cluster-alpha":
            raise RuntimeError("aggregation cluster agent changed")
        replace_cluster_agent(**kwargs)

    monkeypatch.setattr(
        store,
        "replace_playbook_aggregation_cluster_agent",
        replace_with_one_fence_loss,
    )

    result = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))

    assert result["playbooks_generated"] == 1
    assert result["attachments"] == 1
    assert result["supersessions"] == 1
    assert result["cluster_fence_losses"] == 1
    assert result["retryable_failures"] == 1
    items = store.conn.execute(
        "SELECT user_playbook_id, disposition, cluster_id, reason FROM "
        "playbook_aggregation_item WHERE user_playbook_id IN (3, 4) "
        "ORDER BY user_playbook_id"
    ).fetchall()
    assert [tuple(row) for row in items] == [
        (3, "residual", None, "cluster_agent_unavailable"),
        (4, "cluster_member", "cluster-beta", "centroid_match"),
    ]
    clusters = store.conn.execute(
        "SELECT cluster_id, agent_playbook_id, member_count FROM "
        "playbook_aggregation_cluster ORDER BY cluster_id"
    ).fetchall()
    assert tuple(clusters[0]) == (
        "cluster-alpha",
        alpha_agent.agent_playbook_id,
        1,
    )
    assert tuple(clusters[1])[:1] == ("cluster-beta",)
    assert clusters[1][1] != beta_agent.agent_playbook_id
    assert clusters[1][2] == 2
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 3


def test_incremental_run_commits_healthy_clusters_when_one_llm_outcome_fails(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    for item_id in range(1, 5):
        _insert_current(store, item_id, embedding=encoded, trigger=f"trigger-{item_id}")
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    context = MagicMock(
        org_id="aggregation-org", storage=store, configurator=configurator
    )
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,
        agent_version="v1",
    )
    monkeypatch.setattr(
        aggregator,
        "get_clusters",
        lambda playbooks, _config: {0: playbooks[:2], 1: playbooks[2:]},
    )

    def generate_outcomes(
        clusters, _existing, *, direction_overlap_threshold
    ) -> list[AggregationGenerationOutcome]:
        del direction_overlap_threshold
        healthy, failed = clusters.values()
        return [
            AggregationGenerationOutcome(
                "generated",
                healthy,
                AgentPlaybook(
                    playbook_name="playbook",
                    agent_version="v1",
                    content="Healthy aggregate",
                    trigger="healthy trigger",
                    playbook_status=PlaybookStatus.PENDING,
                ),
            ),
            AggregationGenerationOutcome("retryable_failure", failed),
        ]

    monkeypatch.setattr(
        aggregator,
        "_generate_playbook_outcomes_with_source_clusters",
        generate_outcomes,
    )

    result = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))

    assert result["playbooks_generated"] == 1
    assert result["retryable_failures"] == 1
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1
    rows = store.conn.execute(
        "SELECT user_playbook_id, disposition, reason "
        "FROM playbook_aggregation_item ORDER BY user_playbook_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (1, "cluster_member", "generated"),
        (2, "cluster_member", "generated"),
        (3, "residual", "llm_retryable_failure"),
        (4, "residual", "llm_retryable_failure"),
    ]


def test_legacy_fingerprint_is_reembedded_and_adopted_without_regeneration(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    _insert_current(store, 1, trigger="same trigger")
    _insert_current(store, 2, trigger="same trigger")
    existing = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Existing aggregate",
                trigger="same trigger",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.conn.execute(
        "UPDATE agent_playbooks SET embedding='[]' WHERE agent_playbook_id=?",
        (existing.agent_playbook_id,),
    )
    store.conn.commit()
    OperationStateManager(
        store, "aggregation-org", "playbook_aggregator"
    ).update_cluster_fingerprints(
        name="playbook",
        version="v1",
        fingerprints={
            "legacy-fingerprint": {
                "agent_playbook_id": existing.agent_playbook_id,
                "user_playbook_ids": [1, 2],
            }
        },
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
    )

    result = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))

    assert result["playbooks_generated"] == 0
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1
    cluster = store.conn.execute(
        "SELECT agent_playbook_id, member_count, state, embedding_model, "
        "centroid, vector_sum "
        "FROM playbook_aggregation_cluster"
    ).fetchone()
    assert tuple(cluster[:4]) == (
        existing.agent_playbook_id,
        2,
        "active",
        store.embedding_model_name,
    )
    assert json.loads(cluster[4]) == embedding
    assert cluster[5] is None
    assert store.get_playbook_aggregation_bootstrap_status("v1") == "complete"


def test_unembeddable_legacy_agent_does_not_block_bootstrap(
    vec_store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    _insert_current(store, 1, trigger="same trigger")
    existing = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Unembeddable legacy aggregate",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    store.conn.execute(
        "UPDATE agent_playbooks SET embedding='[]' WHERE agent_playbook_id=?",
        (existing.agent_playbook_id,),
    )
    store.conn.commit()
    monkeypatch.setattr(
        store,
        "_get_embedding",
        MagicMock(side_effect=RuntimeError("embedding provider unavailable")),
    )
    OperationStateManager(
        store, "aggregation-org", "playbook_aggregator"
    ).update_cluster_fingerprints(
        name="playbook",
        version="v1",
        fingerprints={
            "broken-legacy-fingerprint": {
                "agent_playbook_id": existing.agent_playbook_id,
                "user_playbook_ids": [1],
            }
        },
    )
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(org_id="aggregation-org", storage=store),
        agent_version="v1",
    )

    assert aggregator._adopt_legacy_aggregation_state(budget=10) == (0, True)

    assert store.get_playbook_aggregation_bootstrap_status("v1") == "complete"
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 0
    )
    assert str(existing.agent_playbook_id) in caplog.text
    assert "could not be re-embedded" in caplog.text


def test_empty_legacy_fingerprint_is_not_activated(vec_store, tmp_path) -> None:
    store = vec_store
    existing = store.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="playbook",
                agent_version="v1",
                content="Legacy aggregate without sources",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )[0]
    OperationStateManager(
        store, "aggregation-org", "playbook_aggregator"
    ).update_cluster_fingerprints(
        name="playbook",
        version="v1",
        fingerprints={
            "empty-legacy-fingerprint": {
                "agent_playbook_id": existing.agent_playbook_id,
                "user_playbook_ids": [],
            }
        },
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
    )

    assert aggregator._adopt_legacy_aggregation_state(budget=10) == (0, True)
    assert store.get_playbook_aggregation_bootstrap_status("v1") == "complete"
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 0
    )


def test_fenced_full_rerun_rebuilds_typed_cluster_state(
    vec_store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = vec_store
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    encoded = json.dumps(embedding)
    _insert_current(store, 1, embedding=encoded, trigger="same trigger")
    _insert_current(store, 2, embedding=encoded, trigger="same trigger")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    monkeypatch.setattr(store, "_get_embedding", lambda _text: embedding)
    store.schedule_playbook_aggregation("v1")
    claim = store.claim_due_playbook_aggregation(owner="admin", lease_seconds=30)
    assert claim is not None
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "state.db")),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )
    configurator = MagicMock()
    configurator.get_config.return_value = config
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=MagicMock(
            org_id="aggregation-org", storage=store, configurator=configurator
        ),
        agent_version="v1",
        aggregation_claim=claim,
    )

    result = aggregator.run(PlaybookAggregatorRequest(agent_version="v1", rerun=True))

    assert result["playbooks_generated"] == 1
    cluster = store.conn.execute(
        "SELECT member_count, state FROM playbook_aggregation_cluster"
    ).fetchone()
    assert tuple(cluster) == (2, "active")
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_item "
            "WHERE disposition='cluster_member'"
        ).fetchone()[0]
        == 2
    )
    assert store.get_playbook_aggregation_bootstrap_status("v1") == "complete"


def test_rerun_snapshot_finishes_only_materialized_work(tmp_path) -> None:
    store = _store(tmp_path)
    _insert_current(store, 1)
    store.append_playbook_aggregation_invalidation(
        agent_version="v1", operation="create", entity_id=1
    )
    store.schedule_playbook_aggregation("v1")
    claim = store.claim_due_playbook_aggregation(owner="admin", lease_seconds=30)
    assert claim is not None

    snapshot = store.capture_playbook_aggregation_rerun_snapshot("v1", limit=10)
    _insert_current(store, 2)
    store.append_playbook_aggregation_invalidation(
        agent_version="v1", operation="create", entity_id=2
    )
    # Intentionally exercise the BEFORE DELETE aggregation-invalidation trigger.
    store.conn.execute("DELETE FROM user_playbooks WHERE user_playbook_id=1")
    store.conn.commit()

    with store.commit_scope():
        store.reset_playbook_aggregation_version("v1")
        store.stage_playbook_aggregation_snapshot(
            "v1", [item.user_playbook_id for item in snapshot.user_playbooks]
        )
        assert store.mark_playbook_aggregation_invalidations_processed(
            claim, list(snapshot.invalidation_ids)
        )

    assert [
        int(row[0])
        for row in store.conn.execute(
            "SELECT user_playbook_id FROM playbook_aggregation_item"
        ).fetchall()
    ] == [1]
    assert [
        (str(row[0]), int(row[1]))
        for row in store.conn.execute(
            "SELECT operation, entity_id FROM playbook_aggregation_invalidation "
            "WHERE processed_at IS NULL ORDER BY invalidation_id"
        ).fetchall()
    ] == [("create", 2), ("hard_delete", 1)]
    assert store.get_playbook_aggregation_backlog("v1").undisposed == 1
