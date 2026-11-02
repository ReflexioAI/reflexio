"""Existing LiteLLM dispatch and usage accounting for input compression."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from reflexio.server.llm import StructuredOutputRepairError
from reflexio.server.llm.input_budget import check_request_input_budget
from reflexio.server.llm.litellm_client import LiteLLMClient, ProviderRequestGuardError
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionCall,
    CompressionChoice,
    CompressionError,
    CompressionNodeInput,
    CompressionPlan,
    CompressionPoint,
    CompressionPolicy,
    CompressionResult,
    EvidencePacket,
    EvidenceSource,
    OmissionCounts,
    packet_tokens,
    plan_compression_input,
)

logger = logging.getLogger(__name__)


class _ReferenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_refs: list[str] = Field(min_length=1)


class _ReferenceSelection(BaseModel):
    """No generated facts, summaries, omission claims or quoted source text."""

    model_config = ConfigDict(extra="forbid", strict=True)
    points: list[_ReferenceGroup] = Field(max_length=128)

    @model_validator(mode="before")
    @classmethod
    def wrap_one_group(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"evidence_refs"}:
            return {"points": [value]}
        return value

    def choice(self) -> CompressionChoice:
        return CompressionChoice(
            points=[
                CompressionPoint(
                    category="selected_evidence", evidence_refs=point.evidence_refs
                )
                for point in self.points
            ]
        )


class CompressionUsageCapture(RunTokenCapture):
    """Attribute this operation locally and forward billing exactly once."""

    def __init__(self, parent: RunTokenCapture | None = None) -> None:
        super().__init__()
        self.parent = parent
        self.prompt_complete = True
        self.output_complete = True

    def observe(self, **usage: Any) -> None:
        self.prompt_complete &= usage.get("prompt_tokens") is not None
        self.output_complete &= usage.get("completion_tokens") is not None
        super().observe(**usage)
        if self.parent is not None:
            self.parent.observe(**usage)

    def known_usage(self, expected_attempts: int) -> tuple[int | None, int | None]:
        """Unknown fields/missing responses must not masquerade as zero cost."""
        observed_all = self.completions == expected_attempts
        return (
            self.totals.prompt_tokens
            if observed_all and self.prompt_complete
            else None,
            self.totals.completion_tokens
            if observed_all and self.output_complete
            else None,
        )


@dataclass(frozen=True)
class ClientCompressionPlan:
    """Transient native preparation; persist its witness, never private sources.

    The native witness adds actual generation policy and capacity declarations
    to the core witness. Existing v6 selection cache identities remain exact.
    Callers must still own source authorization and durable dispatch accounting.
    """

    compression: CompressionPlan
    witness: str
    request_contract_digest: str
    _execute: Callable[..., CompressionResult] = field(repr=False, compare=False)

    def execute(
        self,
        *,
        before_call: Callable[[], None] = lambda: None,
        load: Callable[[str], Mapping[str, Any] | None] = lambda _key: None,
        save: Callable[[str, dict[str, Any]], None] = lambda _key, _value: None,
        inspect_node: Callable[[CompressionNodeInput], None] = lambda _node: None,
    ) -> CompressionResult:
        return self._execute(
            before_call=before_call, load=load, save=save, inspect_node=inspect_node
        )


def plan_with_client(
    sources: Sequence[EvidenceSource],
    *,
    scope: Mapping[str, str],
    policy: CompressionPolicy,
    budget: CompressionBudget,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    measure_output: Callable[[EvidencePacket], int] = packet_tokens,
    output_contract: str = "packet-json-v1",
    estimate_output: Callable[[EvidencePacket], int] | None = None,
) -> ClientCompressionPlan:
    """Prepare and price actual initial native requests without provider/cache I/O.

    Context declarations, model/route/policy, schema and output reservation bind
    the new witness. Rebuild and compare it before resuming an admitted parent.
    Planning observes no usage and invokes no lease/attempt callbacks.
    """
    preparation_started = time.monotonic()
    context_limits = dict(context_limits)
    # The provider emits IDs, while the host copies original passages. Their
    # rendered packet budget must never raise the configured completion cap.
    output_reservation = client.config.max_tokens or 8192
    # Reuse the actual request builder; it owns custom-route precedence, seed,
    # top_p and provider-specific output keys. Never fingerprint API keys.
    probe = client._build_completion_params(
        messages=[],
        response_format=_ReferenceSelection,
        max_tokens=output_reservation,
        temperature=0,
        fallback_models=[],
    )[0]
    contract_keys = (
        "model",
        "api_base",
        "base_url",
        "api_version",
        "custom_llm_provider",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
    )
    contract = {key: probe.get(key) for key in contract_keys}
    model = str(contract["model"])
    direct_answer = model in {"minimax/MiniMax-M3", "zai/glm-5.1"}

    def request_options() -> dict[str, Any]:
        return {
            "response_format": _ReferenceSelection,
            "max_tokens": output_reservation,
            "temperature": 0,
            "fallback_models": [],
            **(
                {"extra_body": {"thinking": {"type": "disabled"}}}
                if direct_answer
                else {}
            ),
        }

    native_contract = client.get_completion_request_contract(**request_options())

    def check_params(params: dict[str, Any]) -> int:
        if {
            key: params.get(key) for key in contract_keys
        } != contract or LiteLLMClient._completion_request_contract_from_params(
            params
        ) != native_contract:
            raise ProviderRequestGuardError("Compression model contract changed")
        return check_request_input_budget(
            params, context_limits=context_limits, prompt_target=budget.input_tokens
        )

    compression = plan_compression_input(
        sources,
        scope=scope,
        policy=policy,
        budget=budget,
        model_contract=json.dumps(
            {"dispatch": contract, "direct_answer": direct_answer}, sort_keys=True
        ),
        measure_output=measure_output,
        output_contract=output_contract,
        estimate_output=estimate_output,
        response_schema=_ReferenceSelection.model_json_schema(),
    )

    def check_leaf_requests() -> tuple[tuple[str, int], ...]:
        checked: list[tuple[str, int]] = []

        def check_messages(messages: list[dict[str, str]]) -> None:
            params = client._build_completion_params(
                messages=messages, **request_options()
            )[0]
            cost = check_params(params)
            messages_digest = hashlib.sha256(
                json.dumps(
                    params["messages"],
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            checked.append((messages_digest, cost))

        compression.check_requests(check_messages)
        return tuple(checked)

    native_leaves = check_leaf_requests()
    ladder = client.get_completion_model_ladder(**request_options())
    if ladder != (model,):
        raise ProviderRequestGuardError("Compression requires a pinned model ladder")
    witness = hashlib.sha256(
        json.dumps(
            {
                "compression": compression.witness,
                "native": native_contract,
                "context_limits": context_limits,
                "initial_native_requests": native_leaves,
                "ladder": ladder,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()

    preparation_seconds = time.monotonic() - preparation_started
    request_contract_digest = hashlib.sha256(
        json.dumps(
            {"native": native_contract, "ladder": ladder},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()

    def execute(
        *,
        before_call: Callable[[], None],
        load: Callable[[str], Mapping[str, Any] | None],
        save: Callable[[str, dict[str, Any]], None],
        inspect_node: Callable[[CompressionNodeInput], None],
    ) -> CompressionResult:
        execution_started = time.monotonic()
        # Check drift before even reading a cached selection. Physical requests
        # are checked again, including every repair, at the native guard.
        if (
            client.get_completion_request_contract(**request_options())
            != native_contract
            or client.get_completion_model_ladder(**request_options()) != ladder
            or check_leaf_requests() != native_leaves
        ):
            raise ProviderRequestGuardError("Compression model contract changed")
        attempts = 0
        reference_recoveries = 0

        def guard(params: dict[str, Any], *_args: Any) -> None:
            nonlocal attempts
            check_params(params)
            if attempts >= budget.max_calls:
                logger.warning(
                    "event=compression_provider_attempt_budget_exhausted "
                    "attempts_used=%d max_calls=%d sections_planned=%d",
                    attempts,
                    budget.max_calls,
                    compression.sections,
                )
                raise ProviderRequestGuardError(
                    "Compression provider attempt budget exhausted"
                )
            before_call()
            attempts += 1

        def complete(call: CompressionCall) -> CompressionChoice:
            nonlocal reference_recoveries

            def validate(value: BaseModel) -> list[str]:
                if not isinstance(value, _ReferenceSelection):
                    return ["Expected structured compression selections"]
                return call.validate_selection(value.choice())

            try:
                result = client.generate_chat_response(
                    messages=call.messages,
                    **request_options(),
                    provider_request_guard=guard,
                    structured_output_validator=validate,
                )
            except StructuredOutputRepairError as error:
                # Recovery is only for COMPLETE schema-valid selections after the
                # client's bounded repair. Never salvage truncated JSON, guess a
                # reference, sever one group's qualifiers, or mask integrity errors.
                if (
                    error.failure_kind != "semantic"
                    or error.model != model
                    or not isinstance(error.raw_content, str)
                ):
                    raise
                try:
                    candidate = _ReferenceSelection.model_validate_json(
                        error.raw_content
                    )
                except ValidationError:
                    raise error from None
                visible = set(call.visible_refs)
                retained = [
                    point
                    for point in candidate.points
                    if all(ref in visible for ref in point.evidence_refs)
                ]
                discarded = len(candidate.points) - len(retained)
                choice = _ReferenceSelection(points=retained).choice()
                capacity_loss = bool(call.validate_selection(choice))
                if not discarded and not capacity_loss:
                    raise
                choice.omissions = (
                    [
                        f"Host discarded {discarded} entire groups with unseen references after bounded repair; source selection is incomplete."
                    ]
                    if discarded
                    else [
                        "Host: Selection still exceeds capacity after bounded repair; whole points may be omitted."
                    ]
                )
                choice.omission_counts = OmissionCounts(
                    unseen_reference_groups=discarded
                )
                if call.validate(choice):
                    raise
                reference_recoveries += bool(discarded)
                return choice
            if not isinstance(result, _ReferenceSelection):
                raise CompressionError("Provider returned no structured compression")
            return result.choice()

        capture = CompressionUsageCapture(run_token_capture.get())
        token = run_token_capture.set(capture)
        try:
            result = compression.execute(
                complete=complete, load=load, save=save, inspect_node=inspect_node
            )
            prompt, completion = capture.known_usage(attempts)
            return replace(
                result,
                elapsed_seconds=preparation_seconds
                + time.monotonic()
                - execution_started,
                provider_attempts=attempts,
                provider_input_tokens=prompt,
                provider_output_tokens=completion,
                provider_reference_recoveries=reference_recoveries,
            )
        finally:
            run_token_capture.reset(token)

    return ClientCompressionPlan(compression, witness, request_contract_digest, execute)


def compress_with_client(
    sources: Sequence[EvidenceSource],
    *,
    scope: Mapping[str, str],
    policy: CompressionPolicy,
    budget: CompressionBudget,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    before_call: Callable[[], None] = lambda: None,
    load: Callable[[str], Mapping[str, Any] | None] = lambda _key: None,
    save: Callable[[str, dict[str, Any]], None] = lambda _key, _value: None,
    measure_output: Callable[[EvidencePacket], int] = packet_tokens,
    output_contract: str = "packet-json-v1",
    estimate_output: Callable[[EvidencePacket], int] | None = None,
) -> CompressionResult:
    """Prepare and execute through the same native plan and usage guards.

    Durable callers can prepare separately with ``plan_with_client``. Every
    physical dispatch still checks capacity, the frozen contract and attempt
    budget. Usage is forwarded to the parent capture exactly once.
    """
    return plan_with_client(
        sources,
        scope=scope,
        policy=policy,
        budget=budget,
        client=client,
        context_limits=context_limits,
        measure_output=measure_output,
        output_contract=output_contract,
        estimate_output=estimate_output,
    ).execute(before_call=before_call, load=load, save=save)
