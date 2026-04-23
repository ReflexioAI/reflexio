"""Tool-calling primitives shared by agentic extraction and search pipelines."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class Tool(BaseModel):
    """A single LLM-callable tool.

    Arguments are defined by a Pydantic model (its schema goes to the LLM,
    its docstring becomes the tool description). The handler takes a
    validated args instance plus a caller-supplied context object and
    returns a JSON-serialisable dict that is fed back as the tool result.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, Any], dict]

    def openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.args_model.__doc__ or "").strip(),
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def openai_specs(self) -> list[dict]:
        return [t.openai_spec() for t in self._tools.values()]

    def handle(self, name: str, args_json: str, ctx: Any) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        try:
            raw = json.loads(args_json or "{}")
            args = tool.args_model.model_validate(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            return {"error": f"invalid args for {name}: {e}"}
        try:
            return tool.handler(args, ctx)
        except Exception as e:  # handler errors are recoverable tool-turn errors
            return {"error": f"handler error: {type(e).__name__}: {e}"}
