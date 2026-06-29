from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AggregationPromptProcessingContext:
    """Per-cluster state shared across aggregation prompt-processing hooks.

    ``changed`` controls whether contextual prompt guidance is requested.
    ``data`` is an opaque deployment-owned mapping. The aggregator initializes
    ``agent_version`` and ``org_id`` keys when those values are available.
    """

    changed: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptPreprocessResult:
    text: str
    changed: bool = False


@dataclass(frozen=True)
class PromptPostprocessResult:
    value: object
    artifacts_removed: int = 0


class AggregationPromptProcessor(Protocol):
    """Optional deployment hook for the playbook aggregation boundary.

    Implementations may transform prompt text before the aggregation LLM call,
    provide contextual prompt instructions when preprocessing changed prompt
    input, and post-process aggregation output before storage or response
    logging. Deployments supply one by overriding
    ``BaseConfigurator.create_aggregation_prompt_processor()``.
    """

    def preprocess_prompt_text(
        self,
        text: str,
        *,
        shared_state: dict[str, Any] | None = None,
        context: AggregationPromptProcessingContext | None = None,
    ) -> PromptPreprocessResult:
        """Return text to send to the aggregation prompt."""
        ...

    def prompt_instructions(
        self,
        context: AggregationPromptProcessingContext,
    ) -> str | None:
        """Return extra prompt guidance for this aggregation context, if any."""
        ...

    def postprocess_aggregation_output(
        self,
        value: object,
        *,
        context: AggregationPromptProcessingContext | None = None,
    ) -> PromptPostprocessResult:
        """Return output to store or log after the aggregation LLM call."""
        ...


class PassthroughPromptProcessor:
    def preprocess_prompt_text(
        self,
        text: str,
        *,
        shared_state: dict[str, Any] | None = None,  # noqa: ARG002
        context: AggregationPromptProcessingContext | None = None,  # noqa: ARG002
    ) -> PromptPreprocessResult:
        return PromptPreprocessResult(text=text)

    def prompt_instructions(
        self,
        context: AggregationPromptProcessingContext,  # noqa: ARG002
    ) -> str | None:
        return None

    def postprocess_aggregation_output(
        self,
        value: object,
        *,
        context: AggregationPromptProcessingContext | None = None,  # noqa: ARG002
    ) -> PromptPostprocessResult:
        return PromptPostprocessResult(value=value)
