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
    from reflexio.server.services.profile.components.consolidator import (
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
