"""Consolidated deploy guard for durable service imports and module layout."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy_guard

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "reflexio"
_SERVICES_ROOT = _SOURCE_ROOT / "server" / "services"

_IMPORT_CONTRACTS = {
    "reflexio.server.services.agent_success_evaluation.components.evaluator": (
        "AgentSuccessEvaluator",
    ),
    "reflexio.server.services.agent_success_evaluation.runner": (
        "run_group_evaluation",
    ),
    "reflexio.server.services.agent_success_evaluation.scheduler": (
        "GroupEvaluationScheduler",
    ),
    "reflexio.server.services.agent_success_evaluation.service": (
        "AgentSuccessEvaluationService",
        "AgentSuccessGenerationServiceConfig",
    ),
    "reflexio.server.services.evaluation_overview.components.distribution": (
        "bucket_corrections",
    ),
    "reflexio.server.services.evaluation_overview.components.hero_state": (
        "HeroState",
        "compute_hero_state",
    ),
    "reflexio.server.services.evaluation_overview.components.rule_attribution": (
        "RuleAttribution",
        "compute_net_sessions",
    ),
    "reflexio.server.services.evaluation_overview.components.shadow_aggregation": (
        "compute_shadow_win_rate_trend",
    ),
    "reflexio.server.services.evaluation_overview.eval_sampler": (
        "SampleCandidate",
        "sample_candidates",
    ),
    "reflexio.server.services.evaluation_overview.service": (
        "EvaluationOverviewService",
    ),
    "reflexio.server.services.extraction.agent_run_records": (
        "build_extractor_agent_run_record",
    ),
    "reflexio.server.services.extraction.outcome": ("ExtractionOutcome",),
    "reflexio.server.services.extraction.pending_tool_call_dispatch": (
        "PendingToolCallToolContext",
        "create_ask_human_tool",
        "create_attach_pending_info_request_tool",
    ),
    "reflexio.server.services.extraction.prior_answer_search": (
        "append_prior_knowledge_context",
    ),
    "reflexio.server.services.extraction.resumable_agent": (
        "AgentRunResult",
        "ResumableExtractionAgent",
        "run_resumable_extraction_agent",
    ),
    "reflexio.server.services.extraction.resume_scheduler": (
        "ExtractionResumeScheduler",
        "maybe_start_resume_scheduler",
    ),
    "reflexio.server.services.extraction.resume_worker": (
        "ExtractionResumeWorker",
        "ResumeWorkerError",
    ),
    "reflexio.server.services.playbook.components.aggregator": ("PlaybookAggregator",),
    "reflexio.server.services.playbook.components.consolidator": (
        "PlaybookConsolidator",
    ),
    "reflexio.server.services.playbook.components.extractor": ("PlaybookExtractor",),
    "reflexio.server.services.playbook.service": (
        "PlaybookGenerationService",
        "PlaybookGenerationServiceConfig",
        "read_user_playbook_as_of_for_learning",
    ),
    "reflexio.server.services.playbook_optimizer": (
        "PlaybookOptimizationScheduler",
        "PlaybookOptimizationTarget",
        "PlaybookOptimizer",
    ),
    "reflexio.server.services.playbook_optimizer.assistant_webhook": (
        "LocalScriptAssistant",
        "WebhookAssistant",
    ),
    "reflexio.server.services.playbook_optimizer.gepa_adapter": (
        "ReflexioPlaybookGEPAAdapter",
    ),
    "reflexio.server.services.playbook_optimizer.judge": ("PairwiseJudge",),
    "reflexio.server.services.playbook_optimizer.models": (
        "CandidateEvaluationOutput",
        "ScenarioWindow",
    ),
    "reflexio.server.services.playbook_optimizer.rollout": ("MultiTurnRollout",),
    "reflexio.server.services.playbook_optimizer.scenario_resolver": (
        "ScenarioResolver",
    ),
    "reflexio.server.services.pre_retrieval": (
        "DocumentExpander",
        "ExpansionResult",
        "QueryReformulator",
        "ReformulationResult",
        "ReformulationSearchResult",
    ),
    "reflexio.server.services.profile.components.consolidator": (
        "ProfileConsolidator",
        "ProfileDeduplicationOutput",
        "ProfileDeletionDirective",
        "ProfileDuplicateGroup",
    ),
    "reflexio.server.services.profile.components.extractor": ("ProfileExtractor",),
    "reflexio.server.services.profile.service": (
        "ProfileGenerationService",
        "ProfileGenerationServiceConfig",
    ),
    "reflexio.server.services.shadow_comparison.judge": ("ShadowComparisonJudge",),
    "reflexio.server.services.shadow_comparison.outcome": (
        "Outcome",
        "assign_positions",
        "derive_reflexio_outcome",
    ),
    "reflexio.server.services.tagging.service": ("TaggingService", "TagsOutput"),
    "reflexio.server.services.tagging.tagging_scheduler": (
        "TaggingScheduler",
        "schedule_tagging",
    ),
}


@pytest.mark.parametrize(
    ("module_name", "public_names"),
    _IMPORT_CONTRACTS.items(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_canonical_service_symbols_are_importable(module_name, public_names):
    module = importlib.import_module(module_name)

    missing = [name for name in public_names if not hasattr(module, name)]
    assert missing == []


_LAYOUT_CONTRACTS = {
    "agent_success_evaluation": (
        {"__init__.py", "service.py", "runner.py", "scheduler.py"},
        {
            "agent_success_evaluation_service.py",
            "agent_success_evaluator.py",
            "group_evaluation_runner.py",
            "delayed_group_evaluator.py",
        },
    ),
    "evaluation_overview": (
        {
            "service.py",
            "eval_sampler.py",
            "components/distribution.py",
            "components/hero_state.py",
            "components/rule_attribution.py",
            "components/shadow_aggregation.py",
        },
        {
            "distribution.py",
            "hero_state.py",
            "rule_attribution.py",
            "shadow_aggregation.py",
        },
    ),
    "extraction": (
        {
            "__init__.py",
            "agent_run_records.py",
            "outcome.py",
            "pending_tool_call_dispatch.py",
            "prior_answer_search.py",
            "resumable_agent.py",
            "resume_scheduler.py",
            "resume_worker.py",
            "README.md",
        },
        {"tools.py", "plan.py", "invariants.py"},
    ),
    "playbook": (
        {
            "service.py",
            "playbook_service_utils.py",
            "playbook_service_constants.py",
            "components/__init__.py",
            "components/extractor.py",
            "components/consolidator.py",
            "components/aggregator.py",
        },
        {
            "playbook_generation_service.py",
            "playbook_extractor.py",
            "playbook_consolidator.py",
            "playbook_aggregator.py",
        },
    ),
    "playbook_optimizer": (
        {
            "optimizer.py",
            "scheduler.py",
            "models.py",
            "judge.py",
            "rollout.py",
            "gepa_adapter.py",
            "assistant_webhook.py",
            "scenario_resolver.py",
        },
        {"components"},
    ),
    "pre_retrieval": (
        {"_query_reformulator.py", "_document_expander.py"},
        {"components"},
    ),
    "profile": (
        {
            "service.py",
            "profile_generation_service_utils.py",
            "components/__init__.py",
            "components/extractor.py",
            "components/consolidator.py",
        },
        {
            "profile_generation_service.py",
            "profile_extractor.py",
            "profile_deduplicator.py",
        },
    ),
    "shadow_comparison": (
        {"judge.py", "outcome.py"},
        {"components"},
    ),
    "tagging": ({"service.py", "tagging_scheduler.py"}, {"tagging_service.py"}),
}


@pytest.mark.parametrize(
    ("service_name", "required_paths", "retired_paths"),
    [
        (service_name, required_paths, retired_paths)
        for service_name, (required_paths, retired_paths) in _LAYOUT_CONTRACTS.items()
    ],
)
def test_service_layout_keeps_canonical_and_retired_paths(
    service_name, required_paths, retired_paths
):
    service_dir = _SERVICES_ROOT / service_name

    assert {
        path for path in required_paths if not (service_dir / path).exists()
    } == set()
    assert {path for path in retired_paths if (service_dir / path).exists()} == set()


def test_durable_service_identifiers_remain_stable():
    from reflexio.models.config_schema import Config, UserPlaybookExtractorConfig
    from reflexio.server.services.evaluation_overview.components.hero_state import (
        HeroState,
    )
    from reflexio.server.services.playbook.playbook_service_constants import (
        PlaybookServiceConstants,
    )
    from reflexio.server.services.playbook_optimizer.optimizer import (
        optimizer_run_request_id,
    )
    from reflexio.server.services.profile.components.consolidator import (
        ProfileConsolidator,
        ProfileDeduplicationOutput,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        ProfileGenerationServiceConstants,
    )
    from reflexio.server.services.shadow_comparison.outcome import Outcome
    from reflexio.test_support.llm_model_registry import _build_registry

    assert HeroState.FULL == "full"
    assert Outcome.WIN == "win"
    assert optimizer_run_request_id(7) == "optjob_7"
    assert (
        PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_PROMPT_ID
        == "playbook_should_generate"
    )
    assert (
        PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID
        == "playbook_extraction_context"
    )
    assert (
        PlaybookServiceConstants.PLAYBOOK_EXTRACTION_PROMPT_ID
        == "playbook_extraction_main"
    )
    assert (
        PlaybookServiceConstants.PLAYBOOK_AGGREGATION_PROMPT_ID
        == "playbook_aggregation"
    )
    assert "profile_extractor_config" in Config.model_fields
    assert "deduplication_config" in UserPlaybookExtractorConfig.model_fields
    assert (
        ProfileGenerationServiceConstants.PROFILE_SHOULD_GENERATE_PROMPT_ID
        == "profile_should_generate"
    )
    assert (
        ProfileGenerationServiceConstants.PROFILE_UPDATE_MAIN_PROMPT_ID
        == "profile_update_main"
    )
    assert ProfileConsolidator.DEDUPLICATION_PROMPT_ID == "profile_deduplication"
    assert (
        _build_registry()["profile_deduplication"].model_class
        is ProfileDeduplicationOutput
    )


def test_playbook_service_keeps_consolidator_import_lazy():
    script = """
import importlib
import sys

module = importlib.import_module("reflexio.server.services.playbook.service")
assert hasattr(module, "PlaybookGenerationService")
assert "reflexio.server.services.playbook.components.consolidator" not in sys.modules
"""

    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
