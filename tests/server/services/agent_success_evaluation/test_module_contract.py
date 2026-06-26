from pathlib import Path

_PY_SUFFIX = "." + "py"
# Build legacy filenames from fragments so stale-path greps can detect real
# reintroductions without matching this contract test itself.
_LEGACY_SERVICE_FILES = (
    "agent_success_evaluation_" + "service" + _PY_SUFFIX,
    "agent_success_" + "evaluator" + _PY_SUFFIX,
)
_LEGACY_RUNNER_FILES = (
    "group_evaluation_" + "runner" + _PY_SUFFIX,
    "delayed_group_" + "evaluator" + _PY_SUFFIX,
)
_MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "reflexio"
    / "server"
    / "services"
    / "agent_success_evaluation"
)


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


def test_agent_success_package_root_init_exists() -> None:
    assert (_MODULE_DIR / "__init__.py").exists()


def test_agent_success_legacy_service_and_evaluator_files_removed() -> None:
    for filename in _LEGACY_SERVICE_FILES:
        assert not (_MODULE_DIR / filename).exists()


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
    for filename in _LEGACY_RUNNER_FILES:
        assert not (_MODULE_DIR / filename).exists()
