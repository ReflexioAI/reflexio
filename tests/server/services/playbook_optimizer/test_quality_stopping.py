"""Search satisfaction and publication use the same persisted quality evidence."""

from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import AgentPlaybook, PlaybookOptimizationJob
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.playbook_optimizer.gepa_adapter import (
    PLAYBOOK_CONTENT_COMPONENT,
    ReflexioPlaybookGEPAAdapter,
)
from reflexio.server.services.playbook_optimizer.gepa_publication import (
    gepa_adoption_authority_from_config,
)
from reflexio.server.services.playbook_optimizer.models import (
    CandidateEvaluationOutput,
    ScenarioWindow,
)
from reflexio.server.services.playbook_optimizer.rollout import MultiTurnRollout
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def search(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="quality-stop", db_path=str(tmp_path / "db.sqlite")
        )
    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(
            optimizer_kind="gepa", target_kind="agent_playbook", target_id=1
        )
    )
    adapter = ReflexioPlaybookGEPAAdapter(
        storage=storage,
        job_id=job.job_id,
        target_kind="agent_playbook",
        target_id=1,
        incumbent=AgentPlaybook(
            agent_playbook_id=1,
            playbook_name="test",
            agent_version="v1",
            content="seed",
        ),
        rollout=MultiTurnRollout(Mock()),
        judge=Mock(),
        max_turns=1,
    )
    windows = [ScenarioWindow(source_interaction_ids=[i]) for i in (1, 2)]
    authority = gepa_adoption_authority_from_config(
        config=PlaybookOptimizerConfig(),
        validation_windows=windows,
        adoption_enabled=True,
    )
    try:
        yield adapter, windows, authority
    finally:
        storage.conn.close()


def _state(*scores):
    return SimpleNamespace(
        program_full_scores_val_set=list(scores),
        program_candidates=[
            {PLAYBOOK_CONTENT_COMPONENT: f"candidate-{i}"} for i in range(len(scores))
        ],
    )


def _evaluate(
    adapter,
    windows,
    index=0,
    *,
    score=0.95,
    likert=5,
    verdict: Literal["candidate", "incumbent", "tie", "aborted"] = "candidate",
):
    adapter._evaluate_one = Mock(
        return_value=CandidateEvaluationOutput(
            score=score,
            likert=likert,
            verdict=verdict,
            rationale="local evidence",
        )
    )
    adapter.evaluate(windows, {PLAYBOOK_CONTENT_COMPONENT: f"candidate-{index}"})


@pytest.mark.parametrize(
    "score,likert,verdict,expected",
    [
        (0.95, 1, "candidate", False),
        (0.8, 4, "candidate", True),
        (0.7, 5, "candidate", False),
        (0.95, 5, "tie", False),
        (0.95, 5, "incumbent", False),
    ],
)
def test_actual_winner_must_meet_both_quality_dimensions(
    search, score, likert, verdict, expected
):
    adapter, windows, authority = search
    _evaluate(adapter, windows, score=score, likert=likert, verdict=verdict)
    # Once evidence is persisted, the callback must perform no database read.
    with patch.object(
        adapter.storage,
        "list_playbook_optimization_evaluations",
        side_effect=AssertionError("callback DB read"),
    ):
        assert (
            adapter.should_stop_on_quality(_state(score), authority=authority)
            is expected
        )


@pytest.mark.parametrize("scores", [(0.95, 0.8), (0.95, 0.95)])
def test_eligible_lower_ranked_or_tied_candidate_cannot_stop_actual_winner(
    search, scores
):
    adapter, windows, authority = search
    _evaluate(adapter, windows, index=0, likert=1)
    _evaluate(adapter, windows, index=1, score=scores[1])
    assert adapter.should_stop_on_quality(_state(*scores), authority=authority) is False


def test_validation_windows_must_be_distinct_and_complete(search):
    adapter, windows, authority = search
    _evaluate(adapter, [windows[0], windows[0]])
    assert adapter.should_stop_on_quality(_state(0.95), authority=authority) is False
    _evaluate(adapter, [windows[1]])
    assert adapter.should_stop_on_quality(_state(0.95), authority=authority) is True


def test_any_persisted_abort_vetoes_early_success_even_after_retry(search):
    adapter, windows, authority = search
    _evaluate(adapter, [windows[0]], verdict="aborted", score=0, likert=0)
    _evaluate(adapter, windows)
    assert adapter.should_stop_on_quality(_state(0.95), authority=authority) is False


def test_failed_insert_cannot_make_phantom_quality_evidence(search):
    adapter, windows, authority = search
    with (
        patch.object(
            adapter.storage,
            "insert_playbook_optimization_evaluation",
            side_effect=RuntimeError("write refused"),
        ),
        pytest.raises(RuntimeError, match="write refused"),
    ):
        _evaluate(adapter, windows)
    assert adapter.should_stop_on_quality(_state(0.95), authority=authority) is False


def test_empty_or_unevaluated_winner_does_not_stop(search):
    adapter, _, authority = search
    assert adapter.should_stop_on_quality(_state(), authority=authority) is False
    assert adapter.should_stop_on_quality(_state(0.95), authority=authority) is False


@pytest.mark.parametrize(
    "score,likert,winner,proposals", [(0.95, 1, "next", 2), (0.8, 5, "first", 1)]
)
def test_real_gepa_search_uses_publication_quality(
    search, score, likert, winner, proposals
):
    """Real GEPA, adapter, rollouts and SQLite; only external judgement/proposals are local."""
    from typing import Any, cast

    from reflexio.models.api_schema.domain import Interaction
    from reflexio.server.services.playbook_optimizer.models import JudgeOutput
    from reflexio.server.services.playbook_optimizer.optimizer import PlaybookOptimizer

    adapter, validation_windows, _ = search
    train_windows = [ScenarioWindow(source_interaction_ids=[3])]
    for window in [*validation_windows, *train_windows]:
        window.interactions = [
            Interaction(
                role="User", content="Summarize", user_id="test", request_id="test"
            )
        ]
    adapter.rollout.assistant = Mock(return_value="local assistant reply")

    def judge(**kwargs):
        judged_score, judged_likert = {
            "seed": (0.5, 3),
            "first": (score, likert),
            "next": (1.0, 5),
        }[kwargs["candidate"].content]
        return JudgeOutput(
            verdict="candidate",
            score=judged_score,
            likert=judged_likert,
            rationale="deterministic local judge",
        )

    adapter.judge.judge = Mock(side_effect=judge)
    adapter.propose_new_texts = Mock(
        side_effect=[
            {PLAYBOOK_CONTENT_COMPONENT: "first"},
            {PLAYBOOK_CONTENT_COMPONENT: "next"},
        ]
    )
    optimizer = PlaybookOptimizer(
        cast(Any, SimpleNamespace(storage=adapter.storage)),
        cast(
            Any, SimpleNamespace(config=SimpleNamespace(model="unused-local-proposer"))
        ),
    )
    result = optimizer._run_gepa(
        PlaybookOptimizerConfig(
            max_metric_calls=20, max_merge_invocations=0, reflection_minibatch_size=1
        ),
        "seed",
        train_windows,
        validation_windows,
        adapter,
    )
    assert result.best_candidate == {PLAYBOOK_CONTENT_COMPONENT: winner}
    assert adapter.propose_new_texts.call_count == proposals
    rows = adapter.storage.list_playbook_optimization_evaluations(adapter.job_id)
    assert {row.likert for row in rows if row.score == score} == {likert}
    # A training cache entry must never stand in for either held-out window.
    winning_id = next(
        candidate.candidate_id
        for candidate in adapter.storage.list_playbook_optimization_candidates(
            adapter.job_id
        )
        if candidate.content == winner
    )
    assert {
        (row.source_interaction_ids[0])
        for row in rows
        if row.candidate_id == winning_id
    } >= {1, 2}
    assert result.total_metric_calls <= 20
    assert adapter.rollout.assistant.call_count > 0
