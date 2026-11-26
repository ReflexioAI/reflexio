"""Bounded return-only declarations through the product's owned LLM client."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, ToolCallingChatResponse
from reflexio.server.llm.tools import (
    ToolLoopTrace,
    ToolLoopTurn,
)

from .models import ContractError, Finish

MAX_CALLS = 6


class IncompleteBatchError(ContractError):
    """A valid adverse declaration: never a structural-repair opportunity."""


def _return_call(raw: Any) -> dict[str, Any]:
    """Read SDK wire fields without repairing malformed provider envelopes."""

    def field(value: Any, name: str) -> Any:
        return (
            value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        )

    function = field(raw, "function")
    return {
        "id": field(raw, "id"),
        "type": field(raw, "type"),
        "function": {
            "name": field(function, "name"),
            "arguments": field(function, "arguments"),
        },
    }


def _declarations(
    calls: list[dict[str, Any]],
    models: dict[str, type[BaseModel]],
    evidence_choices: list[str] | None,
) -> list[tuple[str, BaseModel]]:
    functions = [call.get("function", {}) for call in calls]
    endings = [
        i
        for i, function in enumerate(functions)
        if function.get("name") == "finish_batch"
    ]
    if endings and endings != [len(calls) - 1]:
        raise ContractError("Completion must occur exactly once at the end")
    if endings:
        Finish.model_validate_json(functions[-1].get("arguments", ""))
    batch = []
    for call in calls[:-1] if endings else calls:
        function = call.get("function", {})
        name, arguments = function.get("name"), function.get("arguments")
        if (
            call.get("type") != "function"
            or name not in models
            or not isinstance(arguments, str)
        ):
            raise ContractError("Only declared data-return functions are allowed")
        parsed = models[name].model_validate_json(arguments)
        if (
            evidence_choices is not None
            and "evidence" in models[name].model_fields
            and (getattr(parsed, "evidence", None) not in evidence_choices)
        ):
            raise ContractError("Evidence must be exactly one offered witness")
        batch.append((name, parsed))
    return batch


def _used(usage: Any, name: str) -> int | None:
    raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return raw if type(raw) is int and raw >= 0 else None


class StageClient:
    """One V8 operation; no returned function is a tool with execution authority.

    Native provider routing, credentials, guards, deadlines and token accounting
    remain with LiteLLMClient. Normal configured provider fallbacks remain owned
    there; they cannot exceed this operation's physical dispatch ceiling.
    """

    def __init__(self, client: LiteLLMClient, model: str) -> None:
        self.client = client
        self.calls = 0
        self.physical_calls = 0
        self.latest_provenance: ModelProvenance | None = None
        self.trace = ToolLoopTrace()
        compiler = (
            "minimax/MiniMax-M3.1-Flash" if model == "minimax/MiniMax-M3" else model
        )
        max_tokens = client.config.max_tokens
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ContractError("Invalid configured procedural output limit")
        output_limit = min(16_384, max_tokens) if max_tokens is not None else 16_384
        timeout = min(240, client.config.timeout)
        self.profiles: dict[str, dict[str, Any]] = {
            stage: {
                "model": compiler if stage == "maintain" else model,
                "temperature": 0.7,
                "max_tokens": output_limit,
                "timeout": timeout,
                "seed": 42,
                **(
                    {"reasoning_effort": "low"}
                    if stage == "maintain" and compiler != model
                    else {}
                ),
            }
            for stage in ("source", "maintain", "review")
        }
        # Freeze resolved native ladders, not only the preferred model name.
        self.model_ladders = {
            stage: list(client.get_completion_model_ladder(**options))
            for stage, options in self.profiles.items()
        }

    def _dispatch_guard(self, *_args: Any) -> None:
        if self.physical_calls >= MAX_CALLS:
            raise ProviderRequestGuardError(
                "Procedural operation dispatch allowance exhausted"
            )
        self.physical_calls += 1

    def ask[T](
        self,
        stage: str,
        prompt: str,
        body: dict[str, Any],
        models: dict[str, type[BaseModel]],
        assemble: Callable[[list[tuple[str, Any]]], T],
        validate: Callable[[T], Any],
        *,
        evidence_choices: list[str] | None = None,
    ) -> T:
        if stage not in self.profiles:
            raise ContractError("Unknown procedural stage")
        remaining = MAX_CALLS - self.calls
        if remaining < 1:
            raise ContractError("Procedural allowance exhausted before review")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Submit data without executing a tool or changing stored state.",
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in {**models, "finish_batch": Finish}.items()
        ]
        if evidence_choices is not None:
            for tool in tools:
                properties = tool["function"]["parameters"]["properties"]
                if "evidence" in properties:
                    properties["evidence"] = {
                        "type": "string",
                        "enum": evidence_choices,
                        "description": "Exactly one offered original-source witness identifier.",
                    }
        contract = (
            "Return all proposal declarations in one terminal provider function-call batch. "
            "For a nonempty proposal batch, no extra finish_batch is needed: the provider's "
            "complete tool_calls response ends delivery, not a claim of perfect lesson coverage. "
            "For NOOP return only finish_batch(status=complete); declare incomplete explicitly "
            "when you cannot finish. Truncated responses are failures. Nothing executes. "
            "Each evidence field is exactly ONE offered enum value, with no SEGMENT, "
            "request/session labels, numbers, commentary or joined values. In compilation, "
            "cite_evidence adds another offered witness to an existing change group."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt + "\n" + contract},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
        ]
        for attempt in range(min(1, remaining - 1) + 1):
            self.calls += 1
            started = time.monotonic()
            response = self.client.generate_chat_response_with_provenance(
                messages=deepcopy(messages),
                tools=tools,
                tool_choice="required",
                provider_request_guard=self._dispatch_guard,
                **self.profiles[stage],
            )
            self.latest_provenance = response.provenance
            value = response.value
            usage = response.usage

            self.trace.turns.append(
                ToolLoopTurn(
                    tool_name="procedural_" + stage,
                    args={},
                    result={"structural_repair": bool(attempt)},
                    latency_ms=round((time.monotonic() - started) * 1000),
                    model=response.provenance.model_name,
                    prompt_tokens=_used(usage, "prompt_tokens"),
                    completion_tokens=_used(usage, "completion_tokens"),
                    total_tokens=_used(usage, "total_tokens"),
                )
            )
            if (
                not isinstance(value, ToolCallingChatResponse)
                or value.finish_reason != "tool_calls"
            ):
                raise ContractError(
                    "Procedural delivery requires a complete terminal tool-call response"
                )
            calls = [_return_call(call) for call in (value.tool_calls or [])]
            # A valid terminal incomplete wins over malformed prior declarations.
            if calls and calls[-1].get("type") == "function":
                terminal = calls[-1].get("function", {})
                if terminal.get("name") == "finish_batch":
                    try:
                        finish = Finish.model_validate_json(
                            terminal.get("arguments", "")
                        )
                    except ValidationError:
                        pass
                    else:
                        if finish.status == "incomplete":
                            raise IncompleteBatchError(
                                "Incomplete extraction is a method failure, not NOOP"
                            )
            identities = [call.get("id") for call in calls]
            if (
                not calls
                or any(not isinstance(key, str) or not key for key in identities)
                or len(set(identities)) != len(identities)
            ):
                raise ContractError("Procedural calls need unique nonempty return IDs")
            try:
                batch = _declarations(calls, models, evidence_choices)
                result = assemble(batch)
                validate(result)
            except (ValidationError, ContractError, json.JSONDecodeError) as error:
                if attempt or remaining == 1:
                    raise
                feedback = (
                    error.errors(include_input=False, include_url=False)
                    if isinstance(error, ValidationError)
                    else str(error)
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": value.content,
                        "tool_calls": calls,
                        **(
                            {"reasoning_content": value.reasoning_content}
                            if value.reasoning_content
                            else {}
                        ),
                    }
                )
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": identity,
                        "content": "No changes applied. Correct only structural/binding defects "
                        "against original source and return the complete replacement batch. "
                        + json.dumps(feedback, default=str),
                    }
                    for identity in identities
                )
                continue
            return result
        raise ContractError("No complete batch within bounded structural repair")
