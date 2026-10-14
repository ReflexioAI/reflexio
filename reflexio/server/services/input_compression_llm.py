"""Existing LiteLLM dispatch and usage accounting for input compression."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.server.llm.input_budget import check_request_input_budget
from reflexio.server.llm.litellm_client import LiteLLMClient, ProviderRequestGuardError
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionCall,
    CompressionChoice,
    CompressionError,
    CompressionPoint,
    CompressionPolicy,
    CompressionResult,
    EvidencePacket,
    EvidenceSource,
    compress_input,
    packet_tokens,
)


class _ReferenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: str = Field(min_length=1, max_length=32)
    evidence_refs: list[str] = Field(min_length=1)


class _ReferenceSelection(BaseModel):
    """No generated facts, summaries, omission claims or quoted source text."""

    model_config = ConfigDict(extra="forbid", strict=True)
    points: list[_ReferenceGroup] = Field(max_length=128)

    @model_validator(mode="before")
    @classmethod
    def wrap_one_group(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"category", "evidence_refs"}:
            return {"points": [value]}
        return value

    def choice(self) -> CompressionChoice:
        return CompressionChoice(
            points=[
                CompressionPoint(
                    category=point.category, evidence_refs=point.evidence_refs
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
    """Keep parent billing capture and provider guard, including repair attempts.

    Use an isolated local capture that forwards usage to the parent. Missing
    provider observations remain None; estimates are never reported as usage.
    Durable callers supply their lease check in before_call and quota in the
    scoped provider guard, plus authorized checkpoint load/save functions.
    """
    output_reservation = max(client.config.max_tokens or 8192, budget.output_tokens)
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
    attempts = 0

    def guard(params: dict[str, Any], *_args: Any) -> None:
        nonlocal attempts
        if {key: params.get(key) for key in contract_keys} != contract:
            raise ProviderRequestGuardError("Compression model contract changed")
        if direct_answer and params.get("extra_body", {}).get("thinking") != {
            "type": "disabled"
        }:
            raise ProviderRequestGuardError("Compression thinking mode changed")
        check_request_input_budget(
            params, context_limits=context_limits, prompt_target=budget.input_tokens
        )
        if attempts >= budget.max_calls:
            raise ProviderRequestGuardError(
                "Compression provider attempt budget exhausted"
            )
        before_call()
        attempts += 1

    def complete(call: CompressionCall) -> CompressionChoice:
        def validate(value: BaseModel) -> list[str]:
            if not isinstance(value, _ReferenceSelection):
                return ["Expected structured compression selections"]
            return call.validate(value.choice())

        request_options: dict[str, Any] = (
            {"extra_body": {"thinking": {"type": "disabled"}}} if direct_answer else {}
        )
        result = client.generate_chat_response(
            messages=call.messages,
            response_format=_ReferenceSelection,
            max_tokens=output_reservation,
            temperature=0,
            fallback_models=[],
            provider_request_guard=guard,
            structured_output_validator=validate,
            **request_options,
        )
        if not isinstance(result, _ReferenceSelection):
            raise CompressionError("Provider returned no structured compression")
        return result.choice()

    capture = CompressionUsageCapture(run_token_capture.get())
    token = run_token_capture.set(capture)
    try:
        result = compress_input(
            sources,
            scope=scope,
            policy=policy,
            budget=budget,
            model_contract=json.dumps(
                {"dispatch": contract, "direct_answer": direct_answer}, sort_keys=True
            ),
            complete=complete,
            load=load,
            save=save,
            measure_output=measure_output,
            output_contract=output_contract,
            estimate_output=estimate_output,
            response_schema=_ReferenceSelection.model_json_schema(),
        )
        prompt, completion = capture.known_usage(attempts)
        return replace(
            result,
            provider_attempts=attempts,
            provider_input_tokens=prompt,
            provider_output_tokens=completion,
        )
    finally:
        run_token_capture.reset(token)
