"""Registry mapping LLM operations to their expected Pydantic output models.

Each entry pairs a descriptive key with the Pydantic model class the service
expects and a minimal valid JSON instance that ``model_validate()`` must accept.
This serves as the single source of truth for mock response shapes, used by
both the heuristic mock (``llm_mock.py``) and schema compliance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelRegistryEntry:
    """A registry entry pairing a Pydantic model with a minimal valid instance.

    Args:
        model_class: The Pydantic model class, or None for raw string responses.
        minimal_valid: A dict (or raw value) that ``model_class.model_validate()`` accepts.
    """

    model_class: type[BaseModel] | None
    minimal_valid: dict[str, Any] | str


def _build_registry() -> dict[str, ModelRegistryEntry]:
    """Build the model registry with lazy imports to avoid circular dependencies."""
    from reflexio_ext.server.services.offline_tuner.models import (
        ModeBAdoptionJudgeOutput,
        OfflinePairwiseJudgeOutput,
        OfflinePlaybookEditResponse,
        SuccessLabelSelfConsistencyOutput,
    )

    from reflexio.models.api_schema.eval_overview_schema import ShadowComparisonOutput
    from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_constants import (
        AgentSuccessEvaluationOutput,
    )
    from reflexio.server.services.playbook.playbook_consolidator import (
        PlaybookConsolidationOutput,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        PlaybookAggregationOutput,
        StructuredPlaybookList,
    )
    from reflexio.server.services.playbook_optimizer.models import JudgeOutput
    from reflexio.server.services.profile.profile_deduplicator import (
        ProfileDeduplicationOutput,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        ProfileUpdateOutput,
        StructuredProfilesOutput,
    )
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionOutput,
    )
    from reflexio.server.services.tagging.tagging_service import TagsOutput

    return {
        "playbook_extraction": ModelRegistryEntry(
            model_class=StructuredPlaybookList,
            minimal_valid={
                "playbooks": [
                    {
                        "content": "When user asks a question, provide a detailed answer rather than a brief response.",
                        "trigger": "when user asks a question",
                    },
                ],
            },
        ),
        "playbook_aggregation": ModelRegistryEntry(
            model_class=PlaybookAggregationOutput,
            minimal_valid={
                "playbook": {
                    "content": "When user asks about implementation, provide step-by-step explanations rather than high-level overviews.",
                    "trigger": "when user asks about implementation",
                },
            },
        ),
        "playbook_consolidation": ModelRegistryEntry(
            model_class=PlaybookConsolidationOutput,
            minimal_valid={
                "decisions": [
                    {"kind": "independent", "new_id": "NEW-0"},
                ],
            },
        ),
        "profile_extraction": ModelRegistryEntry(
            model_class=StructuredProfilesOutput,
            minimal_valid={
                "profiles": [
                    {"content": "likes sushi", "time_to_live": "one_month"},
                ],
            },
        ),
        "profile_update": ModelRegistryEntry(
            model_class=ProfileUpdateOutput,
            minimal_valid={
                "add": [
                    {"content": "prefers dark mode", "time_to_live": "one_month"},
                ],
                "delete": [],
                "mention": [],
            },
        ),
        "profile_deduplication": ModelRegistryEntry(
            model_class=ProfileDeduplicationOutput,
            minimal_valid={
                "duplicate_groups": [],
                "unique_ids": ["NEW-0"],
            },
        ),
        "agent_success_evaluation": ModelRegistryEntry(
            model_class=AgentSuccessEvaluationOutput,
            minimal_valid={
                "is_success": True,
                "is_escalated": False,
            },
        ),
        "tagging": ModelRegistryEntry(
            model_class=TagsOutput,
            minimal_valid={"tags": ["example_tag"]},
        ),
        "reflection": ModelRegistryEntry(
            model_class=ReflectionOutput,
            minimal_valid={
                "decisions": [
                    {
                        "target_kind": "profile",
                        "target_id": "PROFILE-0",
                        "reason": "no change",
                    },
                ],
            },
        ),
        "playbook_optimizer_judge": ModelRegistryEntry(
            model_class=JudgeOutput,
            minimal_valid={
                "verdict": "candidate",
                "score": 0.5,
                "likert": 3,
            },
        ),
        "offline_tuner_corrective_proposer": ModelRegistryEntry(
            model_class=OfflinePlaybookEditResponse,
            minimal_valid={
                "proposed_edit": {
                    "op": "revise",
                    "target_user_playbook_id": 101,
                    "new_content": "Ask for the charge date and correct billing errors before discussing refund policy.",
                    "rationale": "Corrective evidence shows billing-error failures tied to no-refund wording.",
                }
            },
        ),
        "offline_tuner_generative_proposer": ModelRegistryEntry(
            model_class=OfflinePlaybookEditResponse,
            minimal_valid={
                "proposed_edit": {
                    "op": "split",
                    "target_user_playbook_id": 101,
                    "replacements": [
                        {
                            "trigger": "billing error or double charge",
                            "content": "Issue a billing correction or escalate to billing.",
                            "rationale": "Observed in similar successful billing-error sessions.",
                            "expanded_terms": ["billing error", "double charge"],
                        },
                        {
                            "trigger": "buyer's-remorse refund request",
                            "content": "Explain the no-refund policy.",
                            "rationale": "Preserves the known-good no-refund path.",
                            "expanded_terms": ["refund request"],
                        },
                    ],
                    "rationale": "Generative support suggests splitting billing errors from true no-refund cases.",
                }
            },
        ),
        "offline_tuner_pairwise_judge": ModelRegistryEntry(
            model_class=OfflinePairwiseJudgeOutput,
            minimal_valid={
                "verdict": "candidate_better",
                "confidence": 0.81,
                "preserves_successes": True,
                "fixes_failures": True,
                "rationale": "The candidate fixes billing-error failures without hurting preserved no-refund successes.",
            },
        ),
        "offline_tuner_mode_b_adoption_judge": ModelRegistryEntry(
            model_class=ModeBAdoptionJudgeOutput,
            minimal_valid={
                "adoption_labels": [
                    {
                        "user_playbook_id": 101,
                        "label": "violated",
                        "rationale": "The agent repeated no-refund language when the transcript showed a billing error.",
                    }
                ]
            },
        ),
        "offline_tuner_success_label_self_consistency": ModelRegistryEntry(
            model_class=SuccessLabelSelfConsistencyOutput,
            minimal_valid={
                "session_label": {
                    "session_id": "sess-1",
                    "label": "success",
                    "rationale": "The user explicitly confirmed the billing issue was resolved.",
                }
            },
        ),
        "shadow_comparison": ModelRegistryEntry(
            model_class=ShadowComparisonOutput,
            minimal_valid={
                "better_request": "1",
                "is_significantly_better": True,
            },
        ),
        # F1 cleanup: ``agent_success_evaluation_comparison`` was removed along
        # with the session-level shadow comparison branch. The per-turn shadow
        # comparison entry is registered above (``shadow_comparison``).
        "boolean_evaluation": ModelRegistryEntry(
            model_class=None,
            minimal_valid="true",
        ),
    }


_REGISTRY: dict[str, ModelRegistryEntry] | None = None


def get_model_registry() -> dict[str, ModelRegistryEntry]:
    """Return the model registry, building it on first access."""
    global _REGISTRY  # noqa: PLW0603
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY
