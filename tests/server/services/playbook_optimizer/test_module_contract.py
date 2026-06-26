from __future__ import annotations

from pathlib import Path


def test_playbook_optimizer_public_imports_work() -> None:
    from reflexio.server.services.playbook_optimizer import (
        PlaybookOptimizationScheduler,
        PlaybookOptimizationTarget,
        PlaybookOptimizer,
    )
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptAssistant,
        WebhookAssistant,
    )
    from reflexio.server.services.playbook_optimizer.gepa_adapter import (
        ReflexioPlaybookGEPAAdapter,
    )
    from reflexio.server.services.playbook_optimizer.judge import PairwiseJudge
    from reflexio.server.services.playbook_optimizer.models import (
        CandidateEvaluationOutput,
        ScenarioWindow,
    )
    from reflexio.server.services.playbook_optimizer.optimizer import (
        optimizer_run_request_id,
    )
    from reflexio.server.services.playbook_optimizer.rollout import MultiTurnRollout
    from reflexio.server.services.playbook_optimizer.scenario_resolver import (
        ScenarioResolver,
    )

    assert PlaybookOptimizer.__name__ == "PlaybookOptimizer"
    assert PlaybookOptimizationScheduler.__name__ == "PlaybookOptimizationScheduler"
    assert PlaybookOptimizationTarget.__name__ == "PlaybookOptimizationTarget"
    assert LocalScriptAssistant.__name__ == "LocalScriptAssistant"
    assert WebhookAssistant.__name__ == "WebhookAssistant"
    assert ReflexioPlaybookGEPAAdapter.__name__ == "ReflexioPlaybookGEPAAdapter"
    assert PairwiseJudge.__name__ == "PairwiseJudge"
    assert CandidateEvaluationOutput.__name__ == "CandidateEvaluationOutput"
    assert ScenarioWindow.__name__ == "ScenarioWindow"
    assert optimizer_run_request_id(7) == "optjob_7"
    assert MultiTurnRollout.__name__ == "MultiTurnRollout"
    assert ScenarioResolver.__name__ == "ScenarioResolver"


def test_playbook_optimizer_keeps_mature_flat_layout() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    module_dir = repo_root / "reflexio/server/services/playbook_optimizer"

    assert not (module_dir / "components").exists()
    assert (module_dir / "optimizer.py").exists()
    assert (module_dir / "scheduler.py").exists()
    assert (module_dir / "models.py").exists()
    assert (module_dir / "judge.py").exists()
    assert (module_dir / "rollout.py").exists()
    assert (module_dir / "gepa_adapter.py").exists()
    assert (module_dir / "assistant_webhook.py").exists()
    assert (module_dir / "scenario_resolver.py").exists()
