from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    Interaction,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationJob,
    PlaybookStatus,
    Request,
    Status,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookOptimizerConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.playbook_optimizer import (
    PlaybookOptimizationScheduler,
    PlaybookOptimizationTarget,
)
from reflexio.server.services.playbook_optimizer.assistant_webhook import (
    LocalScriptAssistant,
    LocalScriptFailedError,
)
from reflexio.server.services.playbook_optimizer.gepa_adapter import (
    PLAYBOOK_CONTENT_COMPONENT,
    ReflexioPlaybookGEPAAdapter,
)
from reflexio.server.services.playbook_optimizer.judge import JudgeOutput
from reflexio.server.services.playbook_optimizer.models import (
    ChatMessage,
    JudgeASI,
    ScenarioWindow,
)
from reflexio.server.services.playbook_optimizer.optimizer import PlaybookOptimizer
from reflexio.server.services.playbook_optimizer.rollout import MultiTurnRollout
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def _sqlite_storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="opt-test", db_path=str(tmp_path / "reflexio.db")
        )
    storage._get_embedding = Mock(return_value=[0.0] * 512)  # noqa: SLF001
    storage.llm_client.get_embeddings = Mock(return_value=[[0.0] * 512])
    return storage


def _optimizer_for_test(storage, config) -> PlaybookOptimizer:
    context = SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
    )
    llm_client = SimpleNamespace(config=SimpleNamespace(model="fake-model"))
    return PlaybookOptimizer(cast(Any, context), cast(Any, llm_client))


def test_local_script_assistant_sends_payload_and_parses_content(tmp_path):
    script = tmp_path / "assistant.py"
    record = tmp_path / "payload.json"
    script.write_text(
        """
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
Path(sys.argv[1]).write_text(json.dumps(payload))
json.dump({"content": "script:" + payload["playbooks"][0]["content"] + ":" + sys.argv[2]}, sys.stdout)
""".strip(),
        encoding="utf-8",
    )
    assistant = LocalScriptAssistant(
        script_path=sys.executable,
        script_args=[str(script), str(record), "extra-arg"],
        timeout_s=5,
        max_retries=0,
        backoff_base_s=0.0,
    )

    content = assistant(
        [ChatMessage(role="user", content="hello")],
        [
            AgentPlaybook(
                agent_playbook_id=7,
                playbook_name="support",
                agent_version="v1",
                content="candidate content",
                trigger="when asked",
            )
        ],
    )

    assert content == "script:candidate content:extra-arg"
    payload = record.read_text(encoding="utf-8")
    assert "hello" in payload
    assert '"playbooks"' in payload
    assert "candidate content" in payload


@pytest.mark.parametrize(
    ("script_body", "match"),
    [
        ("import sys; print('bad', file=sys.stderr); raise SystemExit(2)", "code 2"),
        ("print('not json')", "invalid JSON"),
        (
            "import json; json.dump({'text': 'missing'}, __import__('sys').stdout)",
            "no content",
        ),
    ],
)
def test_local_script_assistant_reports_script_failures(tmp_path, script_body, match):
    script = tmp_path / "assistant.py"
    script.write_text(script_body, encoding="utf-8")
    assistant = LocalScriptAssistant(
        script_path=sys.executable,
        script_args=[str(script)],
        timeout_s=5,
        max_retries=0,
        backoff_base_s=0.0,
    )

    with pytest.raises(LocalScriptFailedError, match=match):
        assistant([], [])


def test_local_script_assistant_times_out(tmp_path):
    script = tmp_path / "assistant.py"
    script.write_text("import time; time.sleep(5)", encoding="utf-8")
    assistant = LocalScriptAssistant(
        script_path=sys.executable,
        script_args=[str(script)],
        timeout_s=1,
        max_retries=0,
        backoff_base_s=0.0,
    )

    with pytest.raises(LocalScriptFailedError, match="timed out"):
        assistant([], [])


def test_adapter_calls_assistant_with_same_seed_and_different_content(tmp_path):
    storage = _sqlite_storage(tmp_path)
    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(target_kind="agent_playbook", target_id=1)
    )
    incumbent = AgentPlaybook(
        agent_playbook_id=1,
        playbook_name="support",
        agent_version="v1",
        content="incumbent content",
    )
    assistant = Mock(return_value="assistant response")
    judge = Mock()
    judge.judge.return_value = JudgeOutput(
        verdict="candidate",
        score=0.9,
        likert=5,
        rationale="candidate wins",
        asi=JudgeASI(winning_behaviors=["clearer response"]),
    )
    adapter = ReflexioPlaybookGEPAAdapter(
        storage=storage,
        job_id=job.job_id,
        target_kind="agent_playbook",
        target_id=1,
        incumbent=incumbent,
        rollout=MultiTurnRollout(assistant),
        judge=judge,
        max_turns=1,
    )

    batch = adapter.evaluate(
        [
            ScenarioWindow(
                user_playbook_id=11,
                source_interaction_ids=[101],
                interactions=[
                    Interaction(
                        interaction_id=101,
                        user_id="u1",
                        request_id="r1",
                        role="User",
                        content="Please summarize this.",
                    )
                ],
            )
        ],
        {PLAYBOOK_CONTENT_COMPONENT: "candidate content"},
        capture_traces=True,
    )

    assert batch.scores == [0.9]
    assert assistant.call_count == 2
    incumbent_call, candidate_call = assistant.call_args_list
    assert incumbent_call.args[0] == candidate_call.args[0]
    assert incumbent_call.args[1][0].content == "incumbent content"
    assert candidate_call.args[1][0].content == "candidate content"
    evaluations = storage.list_playbook_optimization_evaluations(job.job_id)
    assert len(evaluations) == 1
    assert evaluations[0].verdict == "candidate"


def test_optimizer_skips_approved_agent_playbooks(tmp_path):
    storage = _sqlite_storage(tmp_path)
    saved = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="approved content",
                playbook_status=PlaybookStatus.APPROVED,
            )
        ]
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        playbook_optimizer_config=PlaybookOptimizerConfig(
            enabled=True,
            webhook_url="https://assistant.example.test/rollout",
        ),
    )
    optimizer = _optimizer_for_test(storage, config)
    optimizer._run_gepa = Mock(side_effect=AssertionError("GEPA should not run"))  # type: ignore[method-assign]

    optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="agent_playbook", target_id=saved[0].agent_playbook_id
        )
    )

    current = storage.get_agent_playbook_by_id(saved[0].agent_playbook_id)
    assert current is not None
    assert current.playbook_status == PlaybookStatus.APPROVED
    assert current.status is None
    optimizer._run_gepa.assert_not_called()  # type: ignore[attr-defined]


def test_optimizer_skips_when_no_assistant_backend_configured(tmp_path):
    storage = _sqlite_storage(tmp_path)
    saved = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="pending content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        playbook_optimizer_config=PlaybookOptimizerConfig(enabled=True),
    )
    optimizer = _optimizer_for_test(storage, config)
    optimizer._run_gepa = Mock(side_effect=AssertionError("GEPA should not run"))  # type: ignore[method-assign]

    optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="agent_playbook", target_id=saved[0].agent_playbook_id
        )
    )

    optimizer._run_gepa.assert_not_called()  # type: ignore[attr-defined]


def test_optimizer_selects_local_script_assistant(tmp_path):
    storage = _sqlite_storage(tmp_path)
    config = PlaybookOptimizerConfig(
        enabled=True,
        assistant_script_path=sys.executable,
        assistant_script_args=["assistant.py"],
    )
    optimizer = _optimizer_for_test(storage, config)

    assistant = optimizer._create_assistant(config)

    assert isinstance(assistant, LocalScriptAssistant)
    assert assistant.command == [sys.executable, "assistant.py"]


def test_optimizer_runs_local_script_assistant_and_persists_evaluation(tmp_path):
    storage = _sqlite_storage(tmp_path)
    script = tmp_path / "assistant.py"
    record = tmp_path / "assistant_calls.jsonl"
    script.write_text(
        """
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
with Path(sys.argv[1]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\\n")
json.dump({"content": "response for " + payload["playbooks"][0]["content"]}, sys.stdout)
""".strip(),
        encoding="utf-8",
    )
    with (
        patch.object(storage, "_get_embedding", return_value=[0.0] * 512),
        patch.object(storage.llm_client, "get_embeddings", return_value=[[0.0] * 512]),
    ):
        storage.add_request(
            Request(
                request_id="request-1",
                user_id="u1",
                source="test",
                agent_version="v1",
            )
        )
        storage.add_user_interactions_bulk(
            "u1",
            [
                Interaction(
                    interaction_id=1,
                    user_id="u1",
                    request_id="request-1",
                    role="User",
                    content="Please summarize this.",
                )
            ],
        )
        user_playbook = UserPlaybook(
            user_id="u1",
            agent_version="v1",
            request_id="request-1",
            playbook_name="support",
            content="source content",
            source_interaction_ids=[1],
        )
        storage.save_user_playbooks([user_playbook])
        [agent_playbook] = storage.save_agent_playbooks(
            [
                AgentPlaybook(
                    playbook_name="support",
                    agent_version="v1",
                    content="incumbent content",
                    playbook_status=PlaybookStatus.PENDING,
                )
            ]
        )
    storage.set_source_user_playbook_ids_for_agent_playbook(
        agent_playbook.agent_playbook_id, [user_playbook.user_playbook_id]
    )
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        playbook_optimizer_config=PlaybookOptimizerConfig(
            enabled=True,
            assistant_script_path=sys.executable,
            assistant_script_args=[str(script), str(record)],
            allow_single_window_commit=True,
            auto_update_pending_agent_playbooks=False,
            min_commit_windows=1,
            min_commit_score=0.1,
            min_commit_likert=1,
        ),
    )
    optimizer = _optimizer_for_test(storage, config)

    def fake_run_gepa(config, seed_content, windows, adapter):  # noqa: ARG001
        adapter.judge = Mock()
        adapter.judge.judge.return_value = JudgeOutput(
            verdict="candidate",
            score=0.9,
            likert=5,
            rationale="candidate wins",
            asi=JudgeASI(winning_behaviors=["clearer response"]),
        )
        adapter.evaluate(
            windows,
            {PLAYBOOK_CONTENT_COMPONENT: "candidate content"},
            capture_traces=True,
        )
        return SimpleNamespace(
            best_candidate={PLAYBOOK_CONTENT_COMPONENT: "candidate content"},
            val_aggregate_scores=[0.9],
            best_idx=0,
            to_dict=lambda: {"best_idx": 0},
        )

    optimizer._run_gepa = fake_run_gepa  # type: ignore[method-assign]

    optimizer.optimize(
        PlaybookOptimizationTarget(
            kind="agent_playbook", target_id=agent_playbook.agent_playbook_id
        )
    )

    calls = record.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "incumbent content" in calls[0]
    assert "candidate content" in calls[1]
    jobs = storage.conn.execute("SELECT * FROM playbook_optimization_jobs").fetchall()
    assert jobs[0]["status"] == "completed"
    evaluations = storage.list_playbook_optimization_evaluations(jobs[0]["job_id"])
    assert len(evaluations) == 1
    assert evaluations[0].verdict == "candidate"


def test_sqlite_persists_source_mapping_and_winner_candidate(tmp_path):
    storage = _sqlite_storage(tmp_path)
    storage.set_source_user_playbook_ids_for_agent_playbook(10, [2, 3, 2])
    assert storage.get_source_user_playbook_ids_for_agent_playbook(10) == [2, 3]

    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(target_kind="agent_playbook", target_id=10)
    )
    candidate = storage.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(
            job_id=job.job_id,
            candidate_index=0,
            content="candidate",
        )
    )
    storage.update_playbook_optimization_candidate(
        candidate.candidate_id,
        aggregate_score=0.86,
        is_winner=True,
    )

    [persisted] = storage.list_playbook_optimization_candidates(job.job_id)
    assert persisted.aggregate_score == 0.86
    assert persisted.is_winner is True


def test_commit_thresholds_only_count_winner_candidate(tmp_path):
    storage = _sqlite_storage(tmp_path)
    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(target_kind="agent_playbook", target_id=10)
    )
    losing_candidate = storage.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(job_id=job.job_id, content="losing")
    )
    winner_candidate = storage.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(job_id=job.job_id, content="winner")
    )
    storage.insert_playbook_optimization_evaluation(
        PlaybookOptimizationEvaluation(
            job_id=job.job_id,
            candidate_id=losing_candidate.candidate_id,
            target_kind="agent_playbook",
            target_id=10,
            scenario_user_playbook_id=1,
            score=0.95,
            verdict="candidate",
            likert=5,
        )
    )
    config = PlaybookOptimizerConfig(
        min_commit_windows=1,
        min_commit_score=0.75,
        min_commit_likert=4,
    )
    optimizer = _optimizer_for_test(
        storage,
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
            playbook_optimizer_config=config,
        ),
    )

    assert not optimizer._passes_commit_thresholds(  # noqa: SLF001
        job.job_id, winner_candidate.candidate_id, 0.95, config
    )

    storage.insert_playbook_optimization_evaluation(
        PlaybookOptimizationEvaluation(
            job_id=job.job_id,
            candidate_id=winner_candidate.candidate_id,
            target_kind="agent_playbook",
            target_id=10,
            scenario_user_playbook_id=2,
            score=0.9,
            verdict="candidate",
            likert=5,
        )
    )
    assert optimizer._passes_commit_thresholds(  # noqa: SLF001
        job.job_id, winner_candidate.candidate_id, 0.95, config
    )


def test_disk_any_user_playbook_lookup_none_status_filter_means_all(tmp_path):
    from reflexio.server.services.storage.disk_storage import DiskStorage

    with patch(
        "reflexio.server.services.storage.disk_storage._base.QMDClient",
        return_value=MagicMock(),
    ):
        storage = DiskStorage(org_id="opt-disk-test", base_dir=str(tmp_path))

    archived = UserPlaybook(
        user_playbook_id=7,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        playbook_name="support",
        content="archived source",
        status=Status.ARCHIVED,
    )
    storage.save_user_playbooks([archived])

    assert storage.get_user_playbooks_by_ids_any_user([7], status_filter=None) == [
        archived
    ]
    assert storage.get_user_playbooks_by_ids_any_user([7], status_filter=[None]) == []


def test_scheduler_applies_abort_cooldown():
    scheduler = PlaybookOptimizationScheduler.__new__(PlaybookOptimizationScheduler)
    scheduler._scheduled = {}  # noqa: SLF001
    scheduler._heap = []  # noqa: SLF001
    scheduler._mutex = threading.Lock()  # noqa: SLF001
    scheduler._wake_event = threading.Event()  # noqa: SLF001
    scheduler._abort_counts = {}  # noqa: SLF001
    key = ("org", "agent_playbook", 1)

    scheduler._record_abort(key, abort_threshold=2, cooldown_seconds=60)  # noqa: SLF001
    with scheduler._mutex:  # noqa: SLF001
        assert scheduler._cooldown_remaining_locked(key, time.monotonic()) == 0  # noqa: SLF001

    scheduler._record_abort(key, abort_threshold=2, cooldown_seconds=60)  # noqa: SLF001
    with scheduler._mutex:  # noqa: SLF001
        assert scheduler._cooldown_remaining_locked(key, time.monotonic()) > 0  # noqa: SLF001

    target = PlaybookOptimizationTarget(kind="agent_playbook", target_id=1)
    scheduler.enqueue(
        org_id="org",
        target=target,
        callback=lambda: "completed",
        abort_cooldown_threshold=2,
        cooldown_after_aborts_seconds=60,
    )
    assert scheduler._scheduled == {}  # noqa: SLF001
