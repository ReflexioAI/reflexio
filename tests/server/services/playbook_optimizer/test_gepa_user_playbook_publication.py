from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    PlaybookOptimizationCandidate,
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
    GEPA_PUBLICATION_AUTHORITY_METADATA_KEY,
    _gepa_adoption_result_from_snapshot,
)
from reflexio.server.services.playbook_optimizer.models import ScenarioWindow
from reflexio.server.services.playbook_optimizer.optimizer import (
    PlaybookOptimizationTarget,
    PlaybookOptimizer,
)
from reflexio.server.services.storage.error import StorageError
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
    config = _optimizer_config(tmp_path)
    context = SimpleNamespace(
        org_id=storage.org_id,
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
    )
    llm_client = SimpleNamespace(config=SimpleNamespace(model="fake-model"))
    return PlaybookOptimizer(cast(Any, context), cast(Any, llm_client))


def _optimizer_config(tmp_path) -> Config:
    return Config(
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


def _optimizer_with_config(storage: SQLiteStorage, config: Config) -> PlaybookOptimizer:
    context = SimpleNamespace(
        org_id=storage.org_id,
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
                rationale="candidate handled the refund policy more clearly",
                asi_json='{"score":0.9,"rubric":"refund"}',
                incumbent_rollout_json='[{"role":"Assistant","content":"old"}]',
                candidate_rollout_json='[{"role":"Assistant","content":"new"}]',
            )
        )
        return SimpleNamespace(
            best_candidate={PLAYBOOK_CONTENT_COMPONENT: "new guidance"},
            val_aggregate_scores=[0.9],
            best_idx=0,
            to_dict=lambda: {"best_idx": 0},
        )

    optimizer._run_gepa = fake_run_gepa  # type: ignore[method-assign]


def _candidate(
    *,
    aggregate_score: float = 0.9,
) -> PlaybookOptimizationCandidate:
    return PlaybookOptimizationCandidate(
        candidate_id=11,
        job_id=7,
        content="new guidance",
        aggregate_score=aggregate_score,
        is_winner=True,
    )


def _evaluation(
    *,
    evaluation_id: int,
    candidate_id: int = 11,
    scenario_user_playbook_id: int = 101,
    source_interaction_ids: list[int] | None = None,
    score: float = 0.9,
    likert: int = 5,
) -> PlaybookOptimizationEvaluation:
    return PlaybookOptimizationEvaluation(
        evaluation_id=evaluation_id,
        job_id=7,
        candidate_id=candidate_id,
        target_kind="user_playbook",
        target_id=101,
        scenario_user_playbook_id=scenario_user_playbook_id,
        source_interaction_ids=source_interaction_ids or [1001],
        score=score,
        verdict="candidate",
        likert=likert,
    )


def _authority(
    *,
    min_commit_windows: int,
    min_commit_score: str = "0.5",
    min_commit_likert: int = 4,
    windows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "adoption_policy": {
            "auto_update_user_playbooks": True,
            "min_commit_likert": min_commit_likert,
            "min_commit_score": min_commit_score,
            "min_commit_windows": min_commit_windows,
        },
        "validation_manifest": {
            "digest": "a" * 64,
            "windows": windows
            or [
                {
                    "scenario_user_playbook_id": 101,
                    "source_interaction_ids": [1001],
                    "min_commit_likert": min_commit_likert,
                    "min_commit_score": min_commit_score,
                }
            ],
        },
    }


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


def test_gepa_adoption_counts_duplicate_validation_window_once():
    adoption = _gepa_adoption_result_from_snapshot(
        winner=_candidate(),
        evaluations=[
            _evaluation(evaluation_id=1),
            _evaluation(evaluation_id=2),
        ],
        authority=_authority(min_commit_windows=2),
    )

    assert adoption["passes"] is False


def test_gepa_adoption_counts_distinct_validation_windows():
    adoption = _gepa_adoption_result_from_snapshot(
        winner=_candidate(),
        evaluations=[
            _evaluation(evaluation_id=1, scenario_user_playbook_id=101),
            _evaluation(
                evaluation_id=2,
                scenario_user_playbook_id=102,
                source_interaction_ids=[1002],
            ),
        ],
        authority=_authority(
            min_commit_windows=2,
            windows=[
                {
                    "scenario_user_playbook_id": 101,
                    "source_interaction_ids": [1001],
                    "min_commit_likert": 4,
                    "min_commit_score": "0.5",
                },
                {
                    "scenario_user_playbook_id": 102,
                    "source_interaction_ids": [1002],
                    "min_commit_likert": 4,
                    "min_commit_score": "0.5",
                },
            ],
        ),
    )

    assert adoption["passes"] is True


def test_gepa_adoption_uses_per_window_frozen_thresholds():
    adoption = _gepa_adoption_result_from_snapshot(
        winner=_candidate(),
        evaluations=[
            _evaluation(evaluation_id=1, scenario_user_playbook_id=101, score=0.6),
            _evaluation(
                evaluation_id=2,
                scenario_user_playbook_id=102,
                source_interaction_ids=[1002],
                score=0.6,
            ),
        ],
        authority=_authority(
            min_commit_windows=2,
            windows=[
                {
                    "scenario_user_playbook_id": 101,
                    "source_interaction_ids": [1001],
                    "min_commit_likert": 4,
                    "min_commit_score": "0.5",
                },
                {
                    "scenario_user_playbook_id": 102,
                    "source_interaction_ids": [1002],
                    "min_commit_likert": 4,
                    "min_commit_score": "0.9",
                },
            ],
        ),
    )

    assert adoption["passes"] is False


def test_gepa_authority_uses_round_trippable_float_thresholds(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    config = _optimizer_config(tmp_path)
    threshold = 0.1 + 0.2
    config.playbook_optimizer_config.min_commit_score = threshold
    optimizer = _optimizer_with_config(storage, config)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))

    status = optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="user_playbook", target_id=incumbent.user_playbook_id
        )
    )

    assert status == "completed"
    job = storage.conn.execute(
        "SELECT metadata_json FROM playbook_optimization_jobs"
    ).fetchone()
    authority = json.loads(job["metadata_json"])[
        GEPA_PUBLICATION_AUTHORITY_METADATA_KEY
    ]
    assert authority["adoption_policy"]["min_commit_score"] == repr(threshold)
    assert authority["validation_manifest"]["windows"][0]["min_commit_score"] == repr(
        threshold
    )


def test_gepa_user_job_creation_freezes_complete_sanitized_authority(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    config = _optimizer_config(tmp_path)
    config.playbook_optimizer_config.max_metric_calls = 37
    config.playbook_optimizer_config.max_turns = 6
    config.playbook_optimizer_config.early_stop_score = 0.1 + 0.2
    config.playbook_optimizer_config.reflection_minibatch_size = 3
    config.playbook_optimizer_config.max_validation_windows = 1
    config.playbook_optimizer_config.use_merge = False
    config.playbook_optimizer_config.max_merge_invocations = 0
    config.playbook_optimizer_config.reflection_model = "judge-model"
    config.playbook_optimizer_config.webhook_auth_header = "Bearer secret"
    optimizer = _optimizer_with_config(storage, config)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))

    assert (
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )
        == "completed"
    )

    row = storage.conn.execute(
        "SELECT metadata_json FROM playbook_optimization_jobs"
    ).fetchone()
    authority = json.loads(row["metadata_json"])[
        GEPA_PUBLICATION_AUTHORITY_METADATA_KEY
    ]
    assert authority["budget_settings"] == {
        "max_metric_calls": 37,
        "max_turns": 6,
        "reflection_minibatch_size": 3,
    }
    assert authority["split_settings"] == {"max_validation_windows": 1}
    assert authority["merge_settings"] == {
        "max_merge_invocations": 0,
        "use_merge": False,
    }
    assert authority["stop_settings"] == {
        "early_stop_score": repr(0.1 + 0.2),
        "stopper_class": "gepa.utils.stop_condition.ScoreThresholdStopper",
    }
    assert authority["gepa_algorithm"] == {
        "batch_sampler": "epoch_shuffled",
        "cache_evaluation": True,
        "candidate_selection_strategy": "pareto",
        "display_progress_bar": False,
        "frontier_type": "instance",
        "raise_on_exception": False,
    }
    assert authority["model_identity"] == {
        "default_lm": "fake-model",
        "reflection_lm": "judge-model",
    }
    assert authority["gepa_engine_identity"]["package_name"] == "gepa"
    assert isinstance(authority["gepa_engine_identity"]["package_version"], str)
    assert len(authority["gepa_engine_identity"]["optimize_code_digest"]) == 64
    assert authority["backend_identity"]["backend_kind"] == "webhook"
    assert authority["backend_identity"]["webhook_auth_configured"] is True
    assert (
        authority["backend_identity"]["webhook_url_digest"]
        == sha256(b"https://assistant.example.test/rollout").hexdigest()
    )
    serialized = json.dumps(authority, sort_keys=True)
    assert "https://assistant.example.test/rollout" not in serialized
    assert "Bearer secret" not in serialized


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


def test_gepa_publication_uses_frozen_adoption_policy_when_live_config_drifts(
    tmp_path,
):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    config = _optimizer_config(tmp_path)
    optimizer = _optimizer_with_config(storage, config)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_drift_live_config(**kwargs):
        prepared = original_prepare(**kwargs)
        config.playbook_optimizer_config.min_commit_score = 1.0
        config.playbook_optimizer_config.min_commit_likert = 5
        config.playbook_optimizer_config.min_commit_windows = 2
        config.playbook_optimizer_config.auto_update_user_playbooks = False
        return prepared

    with patch.object(
        storage,
        "prepare_gepa_user_playbook_publication",
        side_effect=prepare_then_drift_live_config,
    ):
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )


@pytest.mark.parametrize(
    ("category", "tamper_sql", "params"),
    [
        (
            "train manifest",
            """UPDATE playbook_optimization_jobs
               SET metadata_json = json_set(
                   metadata_json,
                   '$.gepa_publication_authority.train_manifest.windows[0].source_interaction_ids[0]',
                   999
               )""",
            (),
        ),
        (
            "evaluation rationale",
            "UPDATE playbook_optimization_evaluations SET rationale = 'tampered'",
            (),
        ),
        (
            "evaluation ASI",
            "UPDATE playbook_optimization_evaluations SET asi_json = '{\"tampered\":true}'",
            (),
        ),
        (
            "incumbent rollout",
            "UPDATE playbook_optimization_evaluations SET incumbent_rollout_json = '[]'",
            (),
        ),
        (
            "candidate rollout",
            "UPDATE playbook_optimization_evaluations SET candidate_rollout_json = '[]'",
            (),
        ),
        (
            "candidate identity",
            'UPDATE playbook_optimization_candidates SET metadata_json = \'{"candidate_identity":"tampered"}\'',
            (),
        ),
        (
            "evaluator identity",
            """UPDATE playbook_optimization_jobs
               SET metadata_json = json_set(
                   metadata_json,
                   '$.gepa_publication_authority.evaluator_identity.judge_model_id',
                   'tampered-model'
               )""",
            (),
        ),
        (
            "backend identity",
            """UPDATE playbook_optimization_jobs
               SET metadata_json = json_set(
                   metadata_json,
                   '$.gepa_publication_authority.backend_identity.backend_kind',
                   'tampered-backend'
               )""",
            (),
        ),
        (
            "per-window threshold",
            """UPDATE playbook_optimization_jobs
               SET metadata_json = json_set(
                   metadata_json,
                   '$.gepa_publication_authority.validation_manifest.windows[0].min_commit_score',
                   0.1
               )""",
            (),
        ),
    ],
)
def test_gepa_verifier_rejects_tampered_complete_authority_categories(
    tmp_path,
    category,
    tamper_sql,
    params,
):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_tamper(**kwargs):
        prepared = original_prepare(**kwargs)
        storage.conn.execute(tamper_sql, params)
        storage.conn.commit()
        return prepared

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_tamper,
        ),
        pytest.raises(ValueError, match="GEPA durable decision proof changed"),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_results"
        ).fetchone()[0]
        == 0
    ), category


def test_gepa_recovery_resumes_crash_after_prepare_before_staging(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication
    calls = {"run_gepa": 0, "prepare": 0}
    original_run_gepa = optimizer._run_gepa

    def counted_run_gepa(*args, **kwargs):
        calls["run_gepa"] += 1
        if calls["run_gepa"] > 1:
            raise AssertionError("GEPA search reran instead of resuming")
        return original_run_gepa(*args, **kwargs)

    def prepare_then_crash_once(**kwargs):
        calls["prepare"] += 1
        prepared = original_prepare(**kwargs)
        if calls["prepare"] == 1:
            storage.conn.execute(
                "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
                (prepared.job_id,),
            )
            storage.conn.commit()
            raise RuntimeError("crash after durable prepare before staging")
        return prepared

    optimizer._run_gepa = counted_run_gepa  # type: ignore[method-assign]
    with patch.object(
        storage,
        "prepare_gepa_user_playbook_publication",
        side_effect=prepare_then_crash_once,
    ):
        with pytest.raises(RuntimeError, match="crash after durable prepare"):
            optimizer.optimize(
                PlaybookOptimizationTarget(
                    kind="user_playbook", target_id=incumbent.user_playbook_id
                )
            )
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    assert calls["run_gepa"] == 1
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM playbook_optimization_jobs"
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )


def test_gepa_recovery_resumes_crash_after_staging(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_stage = storage.stage_user_playbook_publication
    calls = {"run_gepa": 0, "stage": 0}
    original_run_gepa = optimizer._run_gepa

    def counted_run_gepa(*args, **kwargs):
        calls["run_gepa"] += 1
        if calls["run_gepa"] > 1:
            raise AssertionError("GEPA search reran instead of resuming")
        return original_run_gepa(*args, **kwargs)

    def stage_then_crash_once(request):
        calls["stage"] += 1
        original_stage(request)
        if calls["stage"] == 1:
            storage.conn.execute(
                "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
                (request.job_id,),
            )
            storage.conn.commit()
            raise RuntimeError("crash after staging")

    optimizer._run_gepa = counted_run_gepa  # type: ignore[method-assign]
    with patch.object(
        storage,
        "stage_user_playbook_publication",
        side_effect=stage_then_crash_once,
    ):
        with pytest.raises(RuntimeError, match="crash after staging"):
            optimizer.optimize(
                PlaybookOptimizationTarget(
                    kind="user_playbook", target_id=incumbent.user_playbook_id
                )
            )
        status = optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    assert status == "completed"
    assert calls["run_gepa"] == 1
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_staging"
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )


def test_gepa_expired_publishing_job_is_reclaimed_before_new_search(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication
    prepared_job_ids: list[int] = []

    def prepare_then_crash(**kwargs):
        prepared = original_prepare(**kwargs)
        prepared_job_ids.append(prepared.job_id)
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
            (prepared.job_id,),
        )
        storage.conn.commit()
        raise RuntimeError("crash after durable prepare")

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_crash,
        ),
        pytest.raises(RuntimeError),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    optimizer._run_gepa = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("expired publication must resume without GEPA")
    )
    status = optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="user_playbook", target_id=incumbent.user_playbook_id
        )
    )

    assert status == "completed"
    assert prepared_job_ids == [
        storage.conn.execute(
            "SELECT job_id FROM playbook_optimization_jobs"
        ).fetchone()[0]
    ]


def test_gepa_live_publishing_lease_excludes_duplicate_worker(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_crash_with_live_lease(**kwargs):
        original_prepare(**kwargs)
        raise RuntimeError("crash with live lease")

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_crash_with_live_lease,
        ),
        pytest.raises(RuntimeError),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    optimizer._run_gepa = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("live publication lease must block duplicate GEPA")
    )
    assert (
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )
        == "skipped"
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM playbook_optimization_jobs"
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("drift", ["disabled_config", "missing_backend"])
def test_gepa_recovery_runs_before_live_config_and_backend_gates(tmp_path, drift):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    config = _optimizer_config(tmp_path)
    optimizer = _optimizer_with_config(storage, config)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_crash(**kwargs):
        prepared = original_prepare(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
            (prepared.job_id,),
        )
        storage.conn.commit()
        raise RuntimeError("crash after durable prepare")

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_crash,
        ),
        pytest.raises(RuntimeError),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    if drift == "disabled_config":
        config.playbook_optimizer_config.enabled = False
        config.playbook_optimizer_config.optimize_user_playbooks = False
        config.playbook_optimizer_config.auto_update_user_playbooks = False
    else:
        config.playbook_optimizer_config.webhook_url = None
    optimizer._run_gepa = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("recovery must not rerun GEPA")
    )

    assert (
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )
        == "completed"
    )
    assert (
        storage.conn.execute(
            "SELECT outcome FROM user_playbook_publication_results"
        ).fetchone()[0]
        == "applied"
    )


def test_gepa_recovery_propagates_non_live_lease_reclaim_storage_error(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_crash(**kwargs):
        prepared = original_prepare(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
            (prepared.job_id,),
        )
        storage.conn.commit()
        raise RuntimeError("crash after durable prepare")

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_crash,
        ),
        pytest.raises(RuntimeError),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    with (
        patch.object(
            storage,
            "reclaim_gepa_user_playbook_publishing_job",
            side_effect=StorageError(
                "optimizer job lease is not expired; sqlite unavailable"
            ),
        ),
        pytest.raises(StorageError, match="sqlite unavailable"),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )


def test_gepa_recovery_reuses_canonical_projection_without_regeneration(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_prepare = storage.prepare_gepa_user_playbook_publication

    def prepare_then_crash(**kwargs):
        prepared = original_prepare(**kwargs)
        storage.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
            (prepared.job_id,),
        )
        storage.conn.commit()
        raise RuntimeError("crash after durable prepare")

    with (
        patch.object(
            storage,
            "prepare_gepa_user_playbook_publication",
            side_effect=prepare_then_crash,
        ),
        pytest.raises(RuntimeError),
    ):
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )

    storage._get_embedding = Mock(  # noqa: SLF001
        side_effect=AssertionError("projection was regenerated")
    )
    optimizer._run_gepa = Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("GEPA search reran")
    )
    status = optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="user_playbook", target_id=incumbent.user_playbook_id
        )
    )

    assert status == "completed"
    staged = storage.conn.execute(
        "SELECT projection_json FROM user_playbook_publication_staging"
    ).fetchone()
    assert '"embedding":["0.25","-0.5"' in staged["projection_json"]


def test_gepa_local_script_identity_binds_script_content_and_code_digests(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    script = tmp_path / "assistant.py"
    script.write_text(
        "#!/usr/bin/env python\n"
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'content': 'ok'}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    config = _optimizer_config(tmp_path)
    config.playbook_optimizer_config.webhook_url = None
    config.playbook_optimizer_config.assistant_script_path = str(script)
    optimizer = _optimizer_with_config(storage, config)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))

    assert (
        optimizer.optimize(
            PlaybookOptimizationTarget(
                kind="user_playbook", target_id=incumbent.user_playbook_id
            )
        )
        == "completed"
    )

    job = storage.conn.execute(
        "SELECT metadata_json FROM playbook_optimization_jobs"
    ).fetchone()
    authority = json.loads(job["metadata_json"])["gepa_publication_authority"]
    backend_identity = authority["backend_identity"]
    assert backend_identity["backend_kind"] == "local_script"
    assert "backend_class_code_digest" in backend_identity
    assert (
        backend_identity["script_content_digest"]
        == sha256(script.read_bytes()).hexdigest()
    )
    assert "script_path_digest" not in backend_identity
    assert "adapter_code_digest" in authority["optimizer_identity"]
    assert "rollout_code_digest" in authority["optimizer_identity"]
    assert "judge_code_digest" in authority["evaluator_identity"]


def test_sqlite_storage_uses_shared_publication_metadata_key_constants():
    source = Path(__file__).parents[4] / (
        "reflexio/server/services/storage/sqlite_storage/playbook/_optimization.py"
    )
    text = source.read_text(encoding="utf-8")

    assert "PUBLICATION_PROOF_JSON_METADATA_KEY" in text
    assert "PUBLICATION_PROJECTION_JSON_METADATA_KEY" in text
    assert (
        '_GEPA_PUBLICATION_PROOF_JSON_METADATA_KEY = "publication_proof_json"'
        not in text
    )
    assert (
        '_GEPA_PUBLICATION_PROJECTION_JSON_METADATA_KEY = "publication_projection_json"'
        not in text
    )


def test_gepa_recovery_and_retry_create_one_successor_event_and_aggregation(tmp_path):
    storage = _storage(tmp_path)
    incumbent = _incumbent(storage)
    optimizer = _optimizer(storage, tmp_path)
    _install_winning_gepa(optimizer, storage, _window(incumbent.user_playbook_id))
    original_stage = storage.stage_user_playbook_publication
    original_run_gepa = optimizer._run_gepa
    calls = {"stage": 0, "aggregation": 0, "run_gepa": 0}

    def counted_run_gepa(*args, **kwargs):
        calls["run_gepa"] += 1
        if calls["run_gepa"] > 1:
            raise AssertionError("GEPA search reran instead of resuming")
        return original_run_gepa(*args, **kwargs)

    def stage_then_crash_once(request):
        calls["stage"] += 1
        original_stage(request)
        if calls["stage"] == 1:
            storage.conn.execute(
                "UPDATE playbook_optimization_jobs SET lease_expires_at = 0 WHERE job_id = ?",
                (request.job_id,),
            )
            storage.conn.commit()
            raise RuntimeError("crash after staging")

    def aggregate_once(**kwargs):  # noqa: ARG001
        calls["aggregation"] += 1

    optimizer._run_gepa = counted_run_gepa  # type: ignore[method-assign]
    with (
        patch.object(
            storage,
            "stage_user_playbook_publication",
            side_effect=stage_then_crash_once,
        ),
        patch(
            "reflexio.server.services.playbook_optimizer.optimizer."
            "maybe_trigger_user_playbook_aggregation",
            side_effect=aggregate_once,
        ),
    ):
        with pytest.raises(RuntimeError):
            optimizer.optimize(
                PlaybookOptimizationTarget(
                    kind="user_playbook", target_id=incumbent.user_playbook_id
                )
            )
        assert (
            optimizer.optimize(
                PlaybookOptimizationTarget(
                    kind="user_playbook", target_id=incumbent.user_playbook_id
                )
            )
            == "completed"
        )
        assert (
            optimizer.optimize(
                PlaybookOptimizationTarget(
                    kind="user_playbook", target_id=incumbent.user_playbook_id
                )
            )
            == "skipped"
        )

    assert calls["aggregation"] == 1
    assert calls["run_gepa"] == 1
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM playbook_optimization_jobs"
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
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbook_publication_results"
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
