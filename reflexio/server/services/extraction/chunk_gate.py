"""Bounded gates: a negative decision requires every original source chunk."""

from collections.abc import Callable

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.input_preparation import (
    PreparedExtractionInput,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    present_extraction_chunk,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord


def judge_extraction_chunks(
    *,
    coordinator: OverflowCoordinator,
    prepared: PreparedExtractionInput,
    record: ProtectedRecord,
    client: LiteLLMClient,
    model: str,
    build_prompt: Callable[[], str | None],
    minimum_after_positive: int,
) -> bool:
    """Retain configured domain prompts; errors never become negative votes."""
    if prepared.source_digest != coordinator.source_digest:
        raise ValueError("Extraction gate source binding changed")
    unfinished = coordinator.unfinished_extraction_chunks(prepared)
    # Preserve the caller's downstream reserve when new extraction is needed.
    # Fully cached children let the pipeline derive its remaining cached stages;
    # charging the fresh-run reserve would prevent recovery at exhausted quota.
    follow_up = max(0, minimum_after_positive - len(prepared.chunks))
    minimum_after_positive = unfinished + (follow_up if unfinished else 0)
    with coordinator.provider_scope():
        for index, chunk in enumerate(prepared.chunks):
            with present_extraction_chunk(prepared, chunk, record):
                prompt = build_prompt()
                presentation = current_overflow_presentation.get()
                if prompt and (
                    presentation is None
                    or not presentation.is_present_in(
                        [{"role": "user", "content": prompt}]
                    )
                ):
                    raise InputBudgetConfigurationError(
                        "Extraction gate template omits prepared source"
                    )
                if not prompt:
                    coordinator.require_remaining_work(minimum_after_positive)
                    return True
                messages = [{"role": "user", "content": prompt}]
                key = coordinator.step_key(
                    "gate-v1", {"model": model, "messages": messages}
                )
                saved = coordinator.completed_step(key)
                if saved is None:
                    # Either all remaining votes are negative, or the current vote
                    # admits a full extraction. Both branches must remain feasible.
                    coordinator.require_remaining_work(
                        max(len(prepared.chunks) - index, 1 + minimum_after_positive)
                    )
                    result = client.generate_chat_response_with_provenance(
                        messages=messages, model=model
                    )
                    if not isinstance(
                        result.value, str
                    ) or result.value.strip().lower() not in {"true", "false"}:
                        raise ValueError(
                            "Extraction gate returned no valid boolean decision"
                        )
                    decision = result.value.strip().lower() == "true"
                    coordinator.complete_step(key, decision, result.provenance)
                else:
                    decision = saved["output"]
                    if type(decision) is not bool:
                        raise ValueError("Saved extraction gate decision is invalid")
                if decision:
                    coordinator.require_remaining_work(minimum_after_positive)
                    return True
    checkpoint = coordinator.checkpoint
    checkpoint["source_coverage"] = [
        coordinator.candidate_key(chunk.text) for chunk in prepared.chunks
    ]
    checkpoint["stage"] = "gated_false"
    coordinator.save_checkpoint(checkpoint)
    return False
