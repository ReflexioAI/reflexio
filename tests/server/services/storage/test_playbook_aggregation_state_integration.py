from __future__ import annotations

import ast
from pathlib import Path

import pytest

import reflexio
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    Status,
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
    PlaybookAggregator,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationBacklog,
)


def _store(tmp_path) -> SQLiteStorage:
    return SQLiteStorage(org_id="aggregation-org", db_path=str(tmp_path / "state.db"))


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
                    call_sites.append((self.relative_path, self.function_stack[-1]))
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
    assert cleanup_statuses == ["ARCHIVED"]


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


def test_bounded_antijoin_discovers_late_lower_id(tmp_path) -> None:
    store = _store(tmp_path)
    for item_id in range(10, 17):
        _insert_current(store, item_id)

    assert store.stage_playbook_aggregation_intake("v1", limit=3) == [10, 11, 12]
    assert store.stage_playbook_aggregation_intake("v1", limit=3) == [13, 14, 15]

    # Simulates a lower sequence value whose allocating transaction committed late.
    _insert_current(store, 2)
    assert store.stage_playbook_aggregation_intake("v1", limit=3) == [2, 16]
    assert store.stage_playbook_aggregation_intake("v1", limit=3) == []
    assert store.get_playbook_aggregation_backlog("v1").undisposed == 0


def test_org_claim_is_fenced_and_fence_is_monotonic(tmp_path) -> None:
    store = _store(tmp_path)
    store.schedule_playbook_aggregation("v2")
    store.schedule_playbook_aggregation("v1")

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


def test_new_signal_advances_an_idle_version_to_due_now(tmp_path) -> None:
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
        backlog=PlaybookAggregationBacklog(0, 0, 0),
    )

    store.schedule_playbook_aggregation("v1")

    assert (
        store.claim_due_playbook_aggregation(owner="worker-b", lease_seconds=30)
        is not None
    )


def test_stale_claim_cannot_commit_incremental_effect(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    import json

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


def test_sqlite_vec_centroid_lookup_and_attachment(tmp_path) -> None:
    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    for item_id in (1, 2, 3):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    cluster_id = "cluster-a"
    store.create_playbook_aggregation_cluster(
        cluster_id=cluster_id,
        agent_version="v1",
        agent_playbook_id=None,
        embeddings=[embedding, embedding],
        embedding_model=store.embedding_model_name,
    )
    store.set_playbook_aggregation_disposition(
        "v1", [1, 2], disposition="cluster_member", cluster_id=cluster_id
    )

    matches = store.find_nearest_playbook_aggregation_clusters(
        "v1",
        [(3, embedding)],
        embedding_model=store.embedding_model_name,
        limit=10,
    )
    match = matches[3]
    assert match is not None
    assert match.cluster_id == cluster_id
    assert match.similarity > 0.99
    store.attach_playbook_aggregation_items(
        agent_version="v1",
        attachments=[(3, cluster_id, embedding)],
    )
    assert (
        store.conn.execute(
            "SELECT member_count FROM playbook_aggregation_cluster WHERE cluster_id=?",
            (cluster_id,),
        ).fetchone()[0]
        == 3
    )


def test_centroid_batches_are_strictly_isolated_by_agent_version(tmp_path) -> None:
    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
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
        agent_playbook_id=None,
        embeddings=[embedding],
        embedding_model=store.embedding_model_name,
    )
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-v2",
        agent_version="v2",
        agent_playbook_id=None,
        embeddings=[other_embedding],
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
        limit=10,
    )
    assert matches[3].cluster_id == "cluster-v2"
    with pytest.raises(RuntimeError, match="not active for agent version"):
        store.attach_playbook_aggregation_items(
            agent_version="v2",
            attachments=[(3, "cluster-v1", embedding)],
        )
    disposition = store.conn.execute(
        "SELECT disposition, cluster_id FROM playbook_aggregation_item "
        "WHERE agent_version='v2' AND user_playbook_id=3"
    ).fetchone()
    assert tuple(disposition) == ("residual", None)


def test_invalidation_dissolves_cluster_and_requeues_unaffected_members(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    for item_id in (1, 2):
        _insert_current(store, item_id)
    store.stage_playbook_aggregation_intake("v1", limit=10)
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    store.create_playbook_aggregation_cluster(
        cluster_id="cluster-to-dissolve",
        agent_version="v1",
        agent_playbook_id=None,
        embeddings=[embedding, embedding],
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
        agent_playbook_id=None,
        embeddings=[embedding],
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
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_state"
        ).fetchone()[0]
        == 0
    )


def test_incremental_run_creates_once_then_attaches_without_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    import json

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

    first = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))
    assert first["playbooks_generated"] == 1
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1

    _insert_current(store, 3, embedding=encoded, trigger="same trigger")
    second = aggregator.run(PlaybookAggregatorRequest(agent_version="v1"))
    assert second["playbooks_generated"] == 0
    assert second["attachments"] == 1
    assert store.conn.execute("SELECT count(*) FROM agent_playbooks").fetchone()[0] == 1
    assert (
        store.conn.execute(
            "SELECT member_count FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 3
    )

    old_agent_id = int(
        store.conn.execute(
            "SELECT agent_playbook_id FROM agent_playbooks WHERE status IS NULL"
        ).fetchone()[0]
    )
    store.append_playbook_aggregation_invalidation(
        agent_version="v1", operation="revise", entity_id=1
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
        work_budget=10,
    ).run(PlaybookAggregatorRequest(agent_version="v1"))
    assert replacement["playbooks_generated"] == 1
    assert replacement["supersessions"] == 1
    rows = store.conn.execute(
        "SELECT agent_playbook_id, status FROM agent_playbooks "
        "ORDER BY agent_playbook_id"
    ).fetchall()
    assert rows[0][0] == old_agent_id
    assert rows[0][1] == Status.SUPERSEDED.value
    assert rows[1][1] is None
    assert (
        store.conn.execute(
            "SELECT count(*) FROM playbook_aggregation_cluster"
        ).fetchone()[0]
        == 1
    )


def test_incremental_run_commits_healthy_clusters_when_one_llm_outcome_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    import json

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
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
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
        "SELECT agent_playbook_id, member_count, state, embedding_model "
        "FROM playbook_aggregation_cluster"
    ).fetchone()
    assert tuple(cluster) == (
        existing.agent_playbook_id,
        2,
        "active",
        store.embedding_model_name,
    )
    assert store.get_playbook_aggregation_bootstrap_status("v1") == "complete"


def test_fenced_full_rerun_rebuilds_typed_cluster_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    store = _store(tmp_path)
    if not store._has_sqlite_vec:
        pytest.skip("sqlite-vec is unavailable")
    embedding = [0.0] * store.embedding_dimensions
    embedding[0] = 1.0
    import json

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
