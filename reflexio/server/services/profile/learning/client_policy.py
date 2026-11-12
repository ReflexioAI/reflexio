"""Explicit policy on the two profile-producing call boundaries."""

from __future__ import annotations

import copy
from contextvars import ContextVar
from typing import Any

InputError = ValueError
CURRENT_COMPONENT: ContextVar[dict | None] = ContextVar(
    "profile_request_component", default=None
)


def producing_request_options(client: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Preserve measured M3 policy for an M3-only native model ladder.

    Mixed-provider fallback ladders retain native provider policy throughout;
    a MiniMax request field must not leak into another provider's retry.
    """
    ladder = client.get_completion_model_ladder(**options)
    if ladder and set(ladder) == {"minimax/MiniMax-M3"}:
        body = copy.deepcopy(options.get("extra_body") or {})
        if "thinking" in body and body["thinking"] != {"type": "disabled"}:
            raise InputError("Conflicting component thinking setting")
        body["thinking"] = {"type": "disabled"}
        return {**options, "extra_body": body}
    return options


class ComponentClient:
    """Delegate native behavior except the explicitly wired structured output."""

    def __init__(self, client: Any, component: str, output_schema: Any):
        if component not in {
            "native_profile_extraction",
            "unified_change_planner",
        }:
            raise InputError("Unknown profile-producing component")
        self._client = client
        self._component = component
        self._output_schema = output_schema

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def generate_chat_response_with_provenance(self, *args, **kwargs) -> Any:
        component = None
        if kwargs.get("response_format") is self._output_schema:
            component = {
                "component": self._component,
                "schema": self._output_schema.model_json_schema(),
            }
            kwargs = producing_request_options(self._client, kwargs)
        token = CURRENT_COMPONENT.set(component)
        try:
            return self._client.generate_chat_response_with_provenance(*args, **kwargs)
        finally:
            CURRENT_COMPONENT.reset(token)
