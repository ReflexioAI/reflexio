from pathlib import Path


def test_agent_success_service_canonical_imports_work() -> None:
    from reflexio.server.services.agent_success_evaluation.components.evaluator import (
        AgentSuccessEvaluator,
    )
    from reflexio.server.services.agent_success_evaluation.service import (
        AgentSuccessEvaluationService,
        AgentSuccessGenerationServiceConfig,
    )

    assert AgentSuccessEvaluationService.__name__ == "AgentSuccessEvaluationService"
    assert AgentSuccessGenerationServiceConfig.__name__ == (
        "AgentSuccessGenerationServiceConfig"
    )
    assert AgentSuccessEvaluator.__name__ == "AgentSuccessEvaluator"


def test_agent_success_legacy_service_and_evaluator_files_removed() -> None:
    module_dir = (
        Path(__file__).resolve().parents[4]
        / "reflexio"
        / "server"
        / "services"
        / "agent_success_evaluation"
    )
    assert not (module_dir / "agent_success_evaluation_service.py").exists()
    assert not (module_dir / "agent_success_evaluator.py").exists()


def test_agent_success_runner_and_scheduler_canonical_imports_work() -> None:
    from reflexio.server.services.agent_success_evaluation.runner import (
        run_group_evaluation,
    )
    from reflexio.server.services.agent_success_evaluation.scheduler import (
        GroupEvaluationScheduler,
    )

    assert run_group_evaluation.__name__ == "run_group_evaluation"
    assert GroupEvaluationScheduler.__name__ == "GroupEvaluationScheduler"


def test_agent_success_legacy_runner_and_scheduler_files_removed() -> None:
    module_dir = (
        Path(__file__).resolve().parents[4]
        / "reflexio"
        / "server"
        / "services"
        / "agent_success_evaluation"
    )
    assert not (module_dir / "group_evaluation_runner.py").exists()
    assert not (module_dir / "delayed_group_evaluator.py").exists()
