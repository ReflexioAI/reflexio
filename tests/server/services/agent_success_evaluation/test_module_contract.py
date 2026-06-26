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
