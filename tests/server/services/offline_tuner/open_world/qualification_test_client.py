"""Deterministic OpenWorldModelClient used by qualification tests and gates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.offline_tuner.open_world.analyst import ChatMessage
from reflexio.server.services.offline_tuner.open_world.models import (
    ComponentIdentity,
    DiscoveryMemo,
    HeldOutSessionMemo,
    ObservedStructuredCompletion,
)

_RESPONSES_PATH = (
    Path(__file__).resolve().parents[5]
    / "reflexio/server/services/offline_tuner/open_world/qualification_responses/v1.json"
)


@dataclass(frozen=True)
class RecordedModelCall:
    messages: tuple[ChatMessage, ...]
    response_format: type[StrictStructuredOutput]
    model_revision: str
    idempotency_key: str


def qualification_outputs(
    role: str,
) -> tuple[DiscoveryMemo | HeldOutSessionMemo, ...]:
    raw = json.loads(_RESPONSES_PATH.read_text(encoding="utf-8"))
    outputs = []
    for response in raw["responses"]:
        if response["role"] != role:
            continue
        output_type = DiscoveryMemo if role == "discovery" else HeldOutSessionMemo
        payload = dict(response["output"])
        tuple_fields = (
            "observations",
            "citations",
            "alternative_explanations",
            "counterevidence",
            "uncertainties",
        )
        for field in tuple_fields:
            if field in payload:
                payload[field] = tuple(payload[field])
        outputs.append(output_type.model_validate(payload))
    return tuple(outputs)


class DeterministicQualificationClient:
    """A normal model-client double with deterministic structured completions."""

    def __init__(
        self,
        *,
        identity: ComponentIdentity,
        outputs: Sequence[DiscoveryMemo | HeldOutSessionMemo] | None = None,
        observed_provider: str | None = None,
        observed_model_revision: str | None = None,
    ) -> None:
        self.identity = identity
        self.outputs = tuple(
            qualification_outputs(identity.component_role)
            if outputs is None and identity.component_role != "candidate"
            else outputs or ()
        )
        self.observed_provider = observed_provider or identity.provider
        self.observed_model_revision = (
            observed_model_revision or identity.immutable_model_revision
        )
        self.calls: list[RecordedModelCall] = []

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        response_format: type[StrictStructuredOutput],
        model_revision: str,
        idempotency_key: str,
    ) -> ObservedStructuredCompletion:
        call = RecordedModelCall(
            messages=tuple(messages),
            response_format=response_format,
            model_revision=model_revision,
            idempotency_key=idempotency_key,
        )
        self.calls.append(call)
        index = len(self.calls) - 1
        if index >= len(self.outputs):
            raise AssertionError("deterministic client received an unexpected call")
        output = self.outputs[index]
        if type(output) is not response_format:
            raise AssertionError("deterministic output does not match requested schema")
        return ObservedStructuredCompletion(
            value=output,
            observed_provider=self.observed_provider,
            observed_model_revision=self.observed_model_revision,
            called_at=100 + len(self.calls),
            latency_ms=1,
            usage_json=None,
        )
