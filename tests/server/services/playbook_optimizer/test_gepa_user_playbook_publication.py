from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    PlaybookOptimizationEvaluation,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.config_schema import (
    Config,
    PlaybookOptimizerConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.playbook_optimizer.gepa_adapter import (
    PLAYBOOK_CONTENT_COMPONENT,
)
from reflexio.server.services.playbook_optimizer.gepa_publication import (
    GEPA_PROJECTOR_CODE_DIGEST,
    GEPA_PROJECTOR_ID,
    GEPA_PROJECTOR_VERSION,
)
from reflexio.server.services.playbook_optimizer.models import ScenarioWindow
from reflexio.server.services.playbook_optimizer.optimizer import (
    PlaybookOptimizationTarget,
    PlaybookOptimizer,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def _storage(tmp_path) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="gepa-publication-test",
            db_path=str(tmp_path / "reflexio.db"),
        )
    embedding = [0.25, -0.5, *([0.0] * 510)]
    storage._get_embedding = Mock(return_value=embedding)  # noqa: SLF001
    storage.llm_client.get_embeddings = Mock(return_value=[embedding])
    return storage


def _optimizer(storage: SQLiteStorage, tmp_path) -> PlaybookOptimizer:
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        playbook_optimizer_config=PlaybookOptimizerConfig(
            enabled=True,
            optimize_user_playbooks=True,
            auto_update_user_playbooks=True,
            webhook_url="https://assistant.example.test/rollout",
            min_commit_windows=1,
            min_commit_score=0.5,
            min_commit_likert=4,
        ),
    )
    context = SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
    )
    llm_client = SimpleNamespace(config=SimpleNamespace(model="fake-model"))
    return PlaybookOptimizer(cast(Any, context), cast(Any, llm_client))


def _window(incumbent_id: int) -> ScenarioWindow:
    return ScenarioWindow(
        user_playbook_id=incumbent_id,
        source_interaction_ids=[101],
        interactions=[
            Interaction(
                interaction_id=101,
                user_id="u1",
                request_id="request-1",
                role="User",
                content="Help with a refund",
            )
        ],
    )


def _install_winning_gepa(
    optimizer: PlaybookOptimizer,
    storage: SQLiteStorage,
    window: ScenarioWindow,
) -> None:
    optimizer._resolve_windows = Mock(return_value=[window])  # type: ignore[method-assign]

    def fake_run_gepa(config, seed, train, validation, adapter):  # noqa: ARG001
        assert window.user_playbook_id is not None
        candidate = adapter._ensure_candidate("new guidance")  # noqa: SLF001
        storage.insert_playbook_optimization_evaluation(
            PlaybookOptimizationEvaluation(
                job_id=adapter.job_id,
                candidate_id=candidate.candidate_id,
                target_kind="user_playbook",
                target_id=window.user_playbook_id,
                scenario_user_playbook_id=window.user_playbook_id,
                source_interaction_ids=window.source_interaction_ids,
                score=0.9,
                verdict="candidate",
                likert=5,
            )
        )
        return SimpleNamespace(
            best_candidate={PLAYBOOK_CONTENT_COMPONENT: "new guidance"},
            val_aggregate_scores=[0.9],
            best_idx=0,
            to_dict=lambda: {"best_idx": 0},
        )

    optimizer._run_gepa = fake_run_gepa  # type: ignore[method-assign]


def _incumbent(storage: SQLiteStorage) -> UserPlaybook:
    incumbent = UserPlaybook(
        user_id="u1",
        request_id="request-1",
        agent_version="v1",
        playbook_name="refunds",
        content="old guidance",
        trigger="refund request",
    )
    storage.save_user_playbooks([incumbent])
    return incumbent


def test_gepa_user_winner_publishes_exact_projection_then_triggers_aggregation(
    tmp_path,
):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    aggregation_saw_visible_successor: list[bool] = []

    def record_aggregation(**kwargs):  # noqa: ARG001
        current = storage.get_user_playbooks(status_filter=[None])
        aggregation_saw_visible_successor.append(
            any(playbook.content == "new guidance" for playbook in current)
        )

    with patch(
        "reflexio.server.services.playbook_optimizer.optimizer."
        "maybe_trigger_user_playbook_aggregation",
        side_effect=record_aggregation,
    ) as trigger:
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    trigger.assert_called_once()
    assert aggregation_saw_visible_successor == [True]
    job = storage.conn.execute("SELECT * FROM playbook_optimization_jobs").fetchone()
    assert job["stage"] == "applied"
    assert job["terminal_outcome"] == "applied"
    metadata = json.loads(job["metadata_json"])
    assert metadata["publication_subject_epochs"]["subjects"]
    assert metadata["publication_proof_digest"]

    staged = storage.conn.execute(
        "SELECT * FROM user_playbook_publication_staging WHERE job_id = ?",
        (job["job_id"],),
    ).fetchone()
    projection = json.loads(staged["projection_json"])
    assert projection == {
        "candidate_content_digest": staged["content_digest"],
        "embedding": ["0.25", "-0.5", *(["0"] * 510)],
        "embedding_model_id": storage.embedding_model_name,
        "expanded_terms": [],
        "lexical_document": "refund request new guidance",
        "preserved_trigger": "refund request",
        "projector_code_digest": GEPA_PROJECTOR_CODE_DIGEST,
        "projector_id": GEPA_PROJECTOR_ID,
        "projector_version": GEPA_PROJECTOR_VERSION,
        "schema_version": "offline-tuner-candidate-search-projection-v1",
    }
    successor_id = storage.conn.execute(
        "SELECT successor_user_playbook_id FROM user_playbook_publication_results"
    ).fetchone()[0]
    successor = storage.get_user_playbook_by_id(successor_id)
    assert successor is not None
    assert successor.content == "new guidance"
    assert successor.trigger == "refund request"
    stored_embedding = storage.conn.execute(
        "SELECT embedding FROM user_playbooks WHERE user_playbook_id = ?",
        (successor_id,),
    ).fetchone()[0]
    assert json.loads(stored_embedding) == [0.25, -0.5, *([0.0] * 510)]
    fts = storage.conn.execute(
        "SELECT search_text FROM user_playbooks_fts WHERE rowid = ?",
        (successor_id,),
    ).fetchone()
    assert fts["search_text"] == "refund request new guidance"


def test_gepa_user_incumbent_cas_loss_publishes_no_successor_or_aggregation(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_commit = storage.commit_user_playbook_publication

    def lose_incumbent(request):
        storage.conn.execute(
            "UPDATE user_playbooks SET status = ? WHERE user_playbook_id = ?",
            (Status.ARCHIVED.value, incumbent.user_playbook_id),
        )
        storage.conn.commit()
        return original_commit(request)

    with (
        patch.object(
            storage,
            "commit_user_playbook_publication",
            side_effect=lose_incumbent,
        ),
        patch(
            "reflexio.server.services.playbook_optimizer.optimizer."
            "maybe_trigger_user_playbook_aggregation"
        ) as trigger,
    ):
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    trigger.assert_not_called()
    terminal = storage.conn.execute(
        "SELECT outcome, successor_user_playbook_id FROM user_playbook_publication_results"
    ).fetchone()
    assert tuple(terminal) == ("incumbent_changed", None)
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM user_playbooks").fetchone()[0] == 1
    )


def test_gepa_verifier_rechecks_adoption_and_writes_no_staging_on_rejection(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_reject(**kwargs):
        job = original_prepare(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_evaluations SET verdict = 'incumbent'"
        )
        storage.conn.commit()
        return job

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_reject,
        ),
        pytest.raises(ValueError, match="fails adoption rules"),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_staging"
        ).fetchone()[0]
        == 0
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_results"
        ).fetchone()[0]
        == 0
    )
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM user_playbooks").fetchone()[0] == 1
    )


def test_gepa_verifier_rejects_durable_winner_tampering_before_staging(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_tamper(**kwargs):
        job = original_prepare(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_candidates SET content = 'tampered' WHERE is_winner = 1"
        )
        storage.conn.commit()
        return job

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_tamper,
        ),
        pytest.raises(ValueError, match="publication binding changed"),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_staging"
        ).fetchone()[0]
        == 0
    )
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM user_playbooks").fetchone()[0] == 1
    )


def test_gepa_stale_worker_fence_rejects_before_staging(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_claim = storage.claim_user_playbook_publication

    def claim_then_stale(**kwargs):
        claim = original_claim(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_fence = lease_fence + 1"
        )
        storage.conn.commit()
        return claim

    with (
        patch.object(
            storage,
            "claim_user_playbook_publication",
            side_effect=claim_then_stale,
        ),
        pytest.raises(Exception, match="worker fence"),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_staging"
        ).fetchone()[0]
        == 0
    )
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM user_playbooks").fetchone()[0] == 1
    )


def test_gepa_committed_response_loss_recovers_once_and_aggregates_once(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_commit = storage.commit_user_playbook_publication

    def commit_then_lose_response(request):
        original_commit(request)
        raise RuntimeError("response lost after commit")

    with (
        patch.object(
            storage,
            "commit_user_playbook_publication",
            side_effect=commit_then_lose_response,
        ) as commit,
        patch(
            "reflexio.server.services.playbook_optimizer.optimizer."
            "maybe_trigger_user_playbook_aggregation"
        ) as trigger,
    ):
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    commit.assert_called_once()
    trigger.assert_called_once()
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_results"
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbooks WHERE status IS NULL"
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM playbook_optimization_events WHERE event_type = 'publication_applied'"
        ).fetchone()[0]
        == 1
    )


def test_gepa_post_commit_successor_reload_failure_preserves_publication(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_reload = storage.get_user_playbook_by_id

    def fail_only_after_commit(user_playbook_id):
        committed = storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_results"
        ).fetchone()[0]
        if committed:
            raise RuntimeError("successor reload failed")
        return original_reload(user_playbook_id)

    with (
        patch.object(
            storage,
            "get_user_playbook_by_id",
            side_effect=fail_only_after_commit,
        ),
        patch(
            "reflexio.server.services.playbook_optimizer.optimizer."
            "maybe_trigger_user_playbook_aggregation"
        ) as trigger,
    ):
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    trigger.assert_not_called()
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbooks WHERE status IS NULL"
        ).fetchone()[0]
        == 1
    )


def test_gepa_post_commit_aggregation_failure_preserves_publication(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))

    with patch(
        "reflexio.server.services.playbook_optimizer.optimizer."
        "maybe_trigger_user_playbook_aggregation",
        side_effect=RuntimeError("aggregation unavailable"),
    ) as trigger:
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    trigger.assert_called_once()
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbooks WHERE status IS NULL"
        ).fetchone()[0]
        == 1
    )


def test_optimizer_has_no_direct_user_playbook_supersede_route():
    source = Path(__file__).parents[4] / (
        "reflexio/server/services/playbook_optimizer/optimizer.py"
    )
    text = source.read_text(encoding="utf-8")

    assert "_supersede_user_playbook" not in text
    assert "save_user_playbooks" not in text
