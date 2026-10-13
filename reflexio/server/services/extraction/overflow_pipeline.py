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
from reflexio.server.services.extraction.input_preparation import (
    SourceSpan,
    resolve_evidence_span,
    resolve_visible_quote,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    present_extraction_chunk,
    protected_quote_view,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    decode_committed_output,
    encode_committed_output,
)
from reflexio.server.services.playbook.playbook_evidence import (
    resolve_verbatim_source_span,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus


@dataclass(frozen=True)
class OverflowExtractionCall:
    messages: list[dict[str, Any]]
    schema: type[BaseModel]
    validator: Callable[[BaseModel], Sequence[str]] | None = None


def _carried_evidence_errors(
    output: BaseModel,
    *,
    field: str,
    candidates: list[dict[str, Any]],
    protected: ProtectedRecord,
    protected_spans: tuple[SourceSpan, ...],
) -> list[str]:
    """Reconciliation can reuse shown proof, never discover unshown evidence."""
    refs = {ref.strip() for item in candidates for ref in item.get("evidence_refs", [])}
    quotes: dict[str, list[str]] = {}
    profile_quotes = [
        item["source_span"] for item in candidates if item.get("source_span")
    ]
    for item in candidates:
        for evidence in item.get("evidence", []):
            quotes.setdefault(evidence["turn_ref"].strip(), []).append(
                evidence["source_span"]
            )
    by_ref = {span.evidence_ref: span for span in protected_spans}
    for evidence in protected.evidence:
        refs.update(evidence.evidence_refs)
        first = by_ref[evidence.evidence_refs[0]]
        quotes.setdefault(f"T{first.source_order}", []).append(evidence.quote)
        profile_quotes.append(evidence.quote)
    errors = []
    for index, candidate in enumerate(output.model_dump(mode="json").get(field) or []):
        if field == "profiles":
            quote = candidate.get("source_span")
            if not quote or not any(quote in shown for shown in profile_quotes):
                errors.append(
                    f"profiles[{index}]: cite one contiguous carried original quotation."
                )
        else:
            if any(
                ref.strip() not in refs for ref in candidate.get("evidence_refs", [])
            ):
                errors.append(
                    f"playbooks[{index}]: cite only carried original evidence references."
                )
            if any(
                not item.get("source_span")
                or resolve_verbatim_source_span(
                    item["source_span"].strip(),
                    tuple(quotes.get(item.get("turn_ref", "").strip(), [])),
                )
                is None
                for item in candidate.get("evidence", [])
            ):
                errors.append(
                    f"playbooks[{index}]: cite a carried quotation from the same original turn."
                )
    return errors


def _candidate_source_spans(
    candidates: list[dict[str, Any]], field: str, visible: tuple[SourceSpan, ...]
) -> tuple[SourceSpan, ...]:
    """Retain only physical proof actually cited in each raw child call."""
    selected = {}
    for candidate in candidates:
        if field == "profiles":
            spans = resolve_visible_quote(
                candidate["source_span"], visible, require_unique=False
            )
            selected.update({span.evidence_ref: span for span in spans})
        else:
            for ref in candidate.get("evidence_refs", []):
                span = resolve_evidence_span(ref.strip(), visible)
                selected[span.evidence_ref] = span
            for item in candidate.get("evidence", []):
                turn = item["turn_ref"].strip()
                texts = tuple(
                    span.text for span in visible if f"T{span.source_order}" == turn
                )
                quote = (
                    resolve_verbatim_source_span(item["source_span"].strip(), texts)
                    or item["source_span"]
                )
                spans = resolve_visible_quote(quote, visible, turn_ref=turn)
                selected.update({span.evidence_ref: span for span in spans})
    return tuple(selected.values())


def _saved_raw_candidates(
    coordinator: OverflowCoordinator,
    field: str,
    build_call: Callable[[], OverflowExtractionCall],
) -> tuple[list[dict[str, Any]], tuple[SourceSpan, ...]]:
    """Recover proof only from frozen children of the current prepared plan."""
    candidates = []
    selected = {}
    checkpoint = coordinator.checkpoint
    children = checkpoint.get("extraction_children", {})
    receipts = checkpoint.get("completed_steps", {})
    for chunk in coordinator.require_prepared().chunks:
        contract = children.get(coordinator.candidate_key(chunk.text), {})
        receipt = receipts.get(contract.get("key"))
        if (
            contract.get("policy") != checkpoint.get("extraction_policy")
            or receipt is None
        ):
            raise ProviderRequestGuardError(
                "Saved extraction child proof is unavailable"
            )
        with present_extraction_chunk(
            coordinator.require_prepared(), chunk, coordinator.record
        ):
            call = build_call()
            output = call.schema.model_validate(receipt["output"]["output"])
            if call.validator is not None and call.validator(output):
                raise ProviderRequestGuardError(
                    "Saved child lacks visible original evidence"
                )
            presentation = current_overflow_presentation.get()
            if presentation is None:
                raise InputBudgetConfigurationError(
                    "Saved child presentation is unavailable"
                )
            raw = output.model_dump(mode="json").get(field) or []
            spans = _candidate_source_spans(raw, field, presentation.visible_spans)
            selected.update({span.evidence_ref: span for span in spans})
            candidates.extend(raw)
    return candidates, tuple(selected.values())


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
    stored_schema = saved_run.generation_request_snapshot.get("output_schema_name")
    if stored_schema != cached_call.schema.__name__ and not (
        coordinator.window.kind == "profile"
        and stored_schema is None
        and cached_call.schema.__name__ == "StructuredProfilesOutput"
    ):
        raise InputBudgetConfigurationError("Saved extraction schema is incompatible")
    payload, cached_provenance = decode_committed_output(saved_run.committed_output)
    output = cached_call.schema.model_validate(payload)
    coverage = [coordinator.candidate_key(chunk.text) for chunk in prepared.chunks]
    checkpoint = coordinator.checkpoint
    evidence = checkpoint.get("candidate_evidence", {})
    protocol_output = bool(checkpoint.get("extraction_policy"))
    if protocol_output:
        protected, protected_spans = protected_quote_view(prepared, coordinator.record)
        raw_candidates, raw_spans = _saved_raw_candidates(
            coordinator, field, build_call
        )
        coordinator.carried_source_spans = (*raw_spans, *protected_spans)
        if _carried_evidence_errors(
            output,
            field=field,
            candidates=raw_candidates,
            protected=protected,
            protected_spans=protected_spans,
        ):
            raise ProviderRequestGuardError(
                "Saved candidates exceed their carried evidence"
            )
        with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
            current_policy = _extraction_policy(coordinator, cached_call)
        if current_policy != checkpoint["extraction_policy"]:
            raise InputBudgetConfigurationError(
                "Extraction policy changed after checkpoint"
            )
    if final_validator is not None and final_validator(output):
        raise ProviderRequestGuardError("Saved candidates lack original evidence")
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
    carried = {}
    coordinator.carried_source_spans = None
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
            if (
                result.output is not None
                and call.validator is not None
                and call.validator(result.output)
            ):
                raise ProviderRequestGuardError(
                    "Extraction child lacks visible original evidence"
                )
            if result.output is not None:
                presentation = current_overflow_presentation.get()
                if presentation is None:
                    raise InputBudgetConfigurationError(
                        "Extraction child presentation is unavailable"
                    )
                spans = _candidate_source_spans(
                    result.output.model_dump(mode="json").get(field) or [],
                    field,
                    presentation.visible_spans,
                )
                carried.update({span.evidence_ref: span for span in spans})
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
    protected, protected_spans = protected_quote_view(prepared, coordinator.record)
    coordinator.carried_source_spans = (*carried.values(), *protected_spans)

    def validate_final(output: BaseModel) -> Sequence[str]:
        return [
            *(final_validator(output) if final_validator is not None else []),
            *_carried_evidence_errors(
                output,
                field=field,
                candidates=candidates,
                protected=protected,
                protected_spans=protected_spans,
            ),
        ]

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
                        "protected_record": protected.model_dump(mode="json"),
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
                structured_output_validator=validate_final,
            )
            if not isinstance(response.value, schema):
                raise ProviderRequestGuardError(
                    "Reconciliation lacks structured output"
                )
            output, provenance = response.value, response.provenance
            if validate_final(output):
                raise ProviderRequestGuardError(
                    "Reconciled candidates exceed their carried evidence"
                )
            coordinator.complete_step(key, output.model_dump(mode="json"), provenance)
        else:
            output = schema.model_validate(saved["output"])
            provenance = ModelProvenance(**saved["model_provenance"])
    if validate_final(output):
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
