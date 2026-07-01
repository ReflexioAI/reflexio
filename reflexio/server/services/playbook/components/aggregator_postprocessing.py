from __future__ import annotations

import logging

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AggregationPromptProcessingContext,
    AggregationPromptProcessor,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
)

logger = logging.getLogger(__name__)


class AggregationPostProcessing:
    """Pre/post-processing for playbook aggregation prompts and LLM output.

    Groups the aggregation prompt-processor seam (the enterprise redaction
    Protocol integration point). Holds the single injected collaborator,
    ``aggregation_prompt_processor``, by shared reference from the owning
    ``PlaybookAggregator`` — the SAME object resolved from the
    ``AGGREGATION_PROMPT_PROCESSOR`` ServiceKey — so the enterprise
    ``EnterpriseAggregationPromptProcessor`` injection is preserved unchanged.
    """

    def __init__(
        self,
        aggregation_prompt_processor: AggregationPromptProcessor | None,
    ) -> None:
        self.aggregation_prompt_processor = aggregation_prompt_processor

    @staticmethod
    def _format_prompt_extra_instructions(instructions: str | None) -> str:
        if not instructions or not instructions.strip():
            return ""
        return f"{instructions.strip()}\n"

    def _preprocess_prompt_field(
        self,
        text: str | None,
        shared_state: dict[str, object],
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> str | None:
        if text is None or self.aggregation_prompt_processor is None:
            return text
        result = self.aggregation_prompt_processor.preprocess_prompt_text(
            text,
            shared_state=shared_state,
            context=processing_context,
        )
        if processing_context is not None and (result.changed or result.text != text):
            processing_context.changed = True
        return result.text

    def _preprocess_user_playbook_for_prompt(
        self,
        playbook: UserPlaybook,
        shared_state: dict[str, object],
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> UserPlaybook:
        if self.aggregation_prompt_processor is None:
            return playbook
        return playbook.model_copy(
            update={
                "content": self._preprocess_prompt_field(
                    playbook.content, shared_state, processing_context
                )
                or "",
                "trigger": self._preprocess_prompt_field(
                    playbook.trigger, shared_state, processing_context
                ),
                "rationale": self._preprocess_prompt_field(
                    playbook.rationale, shared_state, processing_context
                ),
            }
        )

    def _aggregation_prompt_extra_instructions_for_context(
        self,
        processing_context: AggregationPromptProcessingContext | None,
    ) -> str:
        if (
            processing_context is None
            or not processing_context.changed
            or self.aggregation_prompt_processor is None
        ):
            return ""

        context_instructions = self.aggregation_prompt_processor.prompt_instructions(
            processing_context
        )
        if not isinstance(context_instructions, str):
            context_instructions = None
        return self._format_prompt_extra_instructions(context_instructions)

    def _postprocess_aggregation_output(
        self,
        value: object,
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> tuple[object, int]:
        if self.aggregation_prompt_processor is None:
            return value, 0
        result = self.aggregation_prompt_processor.postprocess_aggregation_output(
            value,
            context=processing_context,
        )
        return result.value, result.artifacts_removed

    def _postprocess_aggregation_response(
        self,
        response: PlaybookAggregationOutput,
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> tuple[PlaybookAggregationOutput, int]:
        processed, artifact_count = self._postprocess_aggregation_output(
            response,
            processing_context,
        )
        if not isinstance(processed, PlaybookAggregationOutput):
            return response, 0
        return processed, artifact_count

    def _record_postprocessing_artifacts(self, artifact_count: int) -> None:
        if artifact_count <= 0:
            return
        logger.warning(
            "Post-processed %d residual artifacts in aggregated playbook output",
            artifact_count,
        )
