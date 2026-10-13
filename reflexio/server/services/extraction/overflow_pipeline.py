"""Full-source chunk extraction followed by one source-verified reconciliation."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel

from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    present_extraction_chunk,
)
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    decode_committed_output,
    encode_committed_output,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus


@dataclass(frozen=True)
class OverflowExtractionCall:
    messages: list[dict[str, Any]]
    schema: type[BaseModel]
    validator: Callable[[BaseModel], Sequence[str]] | None = None


def _recover_saved_output(
    coordinator: OverflowCoordinator,
    build_call: Callable[[], OverflowExtractionCall],
    final_validator: Callable[[BaseModel], Sequence[str]] | None,
) -> AgentRunResult:
    """Replay verified output without changing its observed attribution."""
    prepared = coordinator.require_prepared()
    field = "profiles" if coordinator.window.kind == "profile" else "playbooks"
    saved_run = coordinator.saved_run
    if saved_run is None or saved_run.committed_output is None:
        raise InputBudgetConfigurationError("Saved extraction output is unavailable")
    if not prepared.chunks:
        raise InputBudgetConfigurationError("Extraction has no prepared chunks")
    with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
        cached_call = build_call()
    payload, cached_provenance = decode_committed_output(saved_run.committed_output)
    output = cached_call.schema.model_validate(payload)
    if final_validator is not None and final_validator(output):
        raise ProviderRequestGuardError("Saved candidates lack original evidence")
    coverage = [coordinator.candidate_key(chunk.text) for chunk in prepared.chunks]
    checkpoint = coordinator.checkpoint
    evidence = checkpoint.get("candidate_evidence", {})
    protocol_output = bool(checkpoint.get("extraction_policy"))
    if protocol_output:
        with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
            current_policy = _extraction_policy(coordinator, cached_call)
        if current_policy != checkpoint["extraction_policy"]:
            raise InputBudgetConfigurationError(
                "Extraction policy changed after checkpoint"
            )
    if (protocol_output and checkpoint.get("source_coverage") != coverage) or (
        not protocol_output
        and (
            checkpoint.get("protected_coverage") != coverage
            or any(
                not evidence.get(f"C{index}", {}).get("evidence")
                for index, _ in enumerate(payload.get(field) or [], 1)
            )
        )
    ):
        raise ProviderRequestGuardError(
            "Saved candidates lack complete verified evidence scan"
        )
    keys = {
        coordinator.candidate_key(candidate)
        for candidate in output.model_dump(mode="json").get(field) or []
    }
    if "candidate_models" in checkpoint:
        if set(checkpoint["candidate_models"]) != keys:
            raise InputBudgetConfigurationError(
                "Saved candidate attribution contract changed"
            )
    else:
        # Legacy envelopes only have scalar attribution. New protocol
        # outputs retain the observed model of each unchanged child.
        checkpoint["candidate_models"] = {
            key: asdict(cached_provenance or ModelProvenance()) for key in keys
        }
    checkpoint["source_coverage"] = coverage
    checkpoint["stage"] = "recovered"
    coordinator.save_checkpoint(checkpoint)
    return AgentRunResult(
        run_id=saved_run.id,
        output=output,
        pending_tool_call_ids=saved_run.pending_tool_call_ids,
        messages=[],
        trace=ToolLoopTrace.model_validate(
            saved_run.committed_output.get("trace", {"finished": True})
        ),
        finished_reason="structured_output",
        model_provenance=cached_provenance,
    )


def extract_overflow_window(
    *,
    coordinator: OverflowCoordinator,
    client: LiteLLMClient,
    build_call: Callable[[], OverflowExtractionCall],
    execute_call: Callable[[OverflowExtractionCall], AgentRunResult],
    final_validator: Callable[[BaseModel], Sequence[str]] | None,
) -> AgentRunResult:
    """No child can finalize; every original chunk precedes reconciliation."""
    prepared = coordinator.require_prepared()
    field = "profiles" if coordinator.window.kind == "profile" else "playbooks"
    saved_run = coordinator.saved_run
    if saved_run is not None and saved_run.committed_output is not None:
        return _recover_saved_output(coordinator, build_call, final_validator)
    candidates: list[dict[str, Any]] = []
    models: dict[str, ModelProvenance] = {}
    schema: type[BaseModel] | None = None
    policy_key: str | None = None
    for index, chunk in enumerate(prepared.chunks):
        with present_extraction_chunk(prepared, chunk, coordinator.record):
            call = build_call()
            if schema is not None and schema is not call.schema:
                raise InputBudgetConfigurationError(
                    "Extraction schema changed between chunks"
                )
            schema = call.schema
            current_policy = _extraction_policy(coordinator, call)
            previous = coordinator.checkpoint.get("extraction_policy")
            if (previous is not None and previous != current_policy) or (
                policy_key is not None and policy_key != current_policy
            ):
                raise InputBudgetConfigurationError(
                    "Extraction policy changed after checkpoint"
                )
            policy_key = current_policy
            if previous is None:
                checkpoint = coordinator.checkpoint
                checkpoint["extraction_policy"] = policy_key
                coordinator.save_checkpoint(checkpoint)
            coordinator.require_remaining_work(
                coordinator.unfinished_extraction_chunks(prepared, start=index)
            )
            result = execute_call(call)
        if result.output is None:
            raise ProviderRequestGuardError("Extraction chunk lacks completed output")
        for candidate in result.output.model_dump(mode="json").get(field) or []:
            candidates.append(candidate)
            models.setdefault(
                coordinator.candidate_key(candidate),
                result.model_provenance or ModelProvenance(),
            )
    if schema is None:
        raise InputBudgetConfigurationError("Extraction has no prepared chunks")
    if not candidates:
        output = schema.model_validate({field: []})
        provenance = ModelProvenance()
    elif len(prepared.chunks) == 1:
        output = schema.model_validate({field: candidates})
        provenance = next(iter(models.values()))
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "Reconcile source-grounded extraction candidates from a fully scanned window. "
                    "Preserve distinct supported lessons, scoped approvals, contradictions and uncertainty. "
                    "Merge exact duplicates and compatible partial lessons only when original quoted evidence supports every clause. "
                    "Do not invent evidence, widen permission, turn reported success into observed success, or discard a supported lesson merely to shorten output. "
                    "Return the supplied extraction schema. Cite only original evidence references/quotations already carried by candidates or protected context."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "protected_record": coordinator.record.model_dump(mode="json"),
                        "human_answers": coordinator.checkpoint.get(
                            "answer_witnesses", {}
                        ),
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        key = coordinator.step_key(
            "reconcile-v1", {"messages": messages, "schema": schema.model_json_schema()}
        )
        saved = coordinator.completed_step(key)
        if saved is None:
            coordinator.require_remaining_work(1)
            response = client.generate_chat_response_with_provenance(
                messages=messages,
                model_role=ModelRole.EXTRACTION_AGENT,
                response_format=schema,
                structured_output_validator=final_validator,
            )
            if not isinstance(response.value, schema):
                raise ProviderRequestGuardError(
                    "Reconciliation lacks structured output"
                )
            output, provenance = response.value, response.provenance
            coordinator.complete_step(key, output.model_dump(mode="json"), provenance)
        else:
            output = schema.model_validate(saved["output"])
            provenance = ModelProvenance(**saved["model_provenance"])
    if final_validator is not None and final_validator(output):
        raise ProviderRequestGuardError("Reconciled candidates lack original evidence")
    checkpoint = coordinator.checkpoint
    checkpoint["candidate_models"] = {
        coordinator.candidate_key(candidate): asdict(
            models.get(coordinator.candidate_key(candidate), provenance)
        )
        for candidate in output.model_dump(mode="json").get(field) or []
    }
    checkpoint["source_coverage"] = [
        coordinator.candidate_key(chunk.text) for chunk in prepared.chunks
    ]
    checkpoint["stage"] = "extracted"
    coordinator.save_checkpoint(checkpoint)
    run_id = f"window:{coordinator.window.window_id}"
    envelope = encode_committed_output(output, provenance)
    with coordinator.storage.commit_scope():
        coordinator.storage.fence_user_extraction(
            coordinator.window.user_id, coordinator.token
        )
        coordinator.storage.validate_extraction_source_digest(
            coordinator.window, coordinator.source_digest
        )
        stored = coordinator.storage.update_agent_run_status(
            run_id,
            AgentRunStatus.AGENT_COMPLETED,
            committed_output=envelope,
            expected_statuses=(AgentRunStatus.RUNNING, AgentRunStatus.FAILED),
        )
    if stored is None:
        raise ProviderRequestGuardError("Extraction parent completion lost ownership")
    payload, observed = decode_committed_output(envelope)
    return AgentRunResult(
        run_id=run_id,
        output=schema.model_validate(payload),
        pending_tool_call_ids=[],
        messages=[],
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        model_provenance=observed,
    )


def _extraction_policy(
    coordinator: OverflowCoordinator, call: OverflowExtractionCall
) -> str:
    """Use the same complete prompt contract for fresh and cached output."""
    presentation = current_overflow_presentation.get()
    if presentation is None or not presentation.is_present_in(call.messages):
        raise InputBudgetConfigurationError("Extraction template omits prepared source")
    return coordinator.step_key(
        "extraction-policy-v1",
        {
            "messages": _stable_messages(call.messages, presentation.text),
            "schema": call.schema.model_json_schema(),
        },
    )


def _stable_messages(
    messages: list[dict[str, Any]], frame: str
) -> list[dict[str, Any]]:
    """Remove only the complete prepared frame from text or multimodal content."""
    variants = (frame, json.dumps(frame, ensure_ascii=False)[1:-1])

    def normalize(value: str) -> str:
        for variant in variants:
            value = value.replace(variant, "<prepared-source>")
        return value

    stable = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = normalize(content)
        elif isinstance(content, list):
            content = [
                {**part, "text": normalize(part["text"])}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        stable.append({**message, "content": content})
    return stable
