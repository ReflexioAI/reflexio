import json

from pydantic import BaseModel

from reflexio.server.llm.tools import Tool, ToolRegistry


class EmitProfileArgs(BaseModel):
    """Emit a candidate user profile item."""

    content: str
    time_to_live: str


class Ctx:
    def __init__(self):
        self.calls = []
        self.finished = False

    def emit(self, args, ctx):
        self.calls.append(args)
        return {"ok": True}


def test_tool_openai_spec_uses_docstring_and_schema():
    t = Tool(name="emit_profile", args_model=EmitProfileArgs, handler=lambda _a, _c: {})
    spec = t.openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "emit_profile"
    assert "Emit a candidate user profile item." in spec["function"]["description"]
    assert spec["function"]["parameters"]["properties"]["content"]["type"] == "string"


def test_registry_handle_parses_and_dispatches():
    ctx = Ctx()
    t = Tool(name="emit_profile", args_model=EmitProfileArgs, handler=ctx.emit)
    reg = ToolRegistry()
    reg.register(t)
    result = reg.handle(
        "emit_profile", json.dumps({"content": "hi", "time_to_live": "persistent"}), ctx
    )
    assert result == {"ok": True}
    assert ctx.calls[0].content == "hi"


def test_registry_handle_converts_validation_error_to_tool_error():
    ctx = Ctx()
    reg = ToolRegistry()
    reg.register(
        Tool(name="emit_profile", args_model=EmitProfileArgs, handler=ctx.emit)
    )
    # Missing required field.
    result = reg.handle("emit_profile", json.dumps({"content": "hi"}), ctx)
    assert "error" in result
    assert "time_to_live" in result["error"]
    assert ctx.calls == []


def test_registry_rejects_unknown_tool():
    reg = ToolRegistry()
    result = reg.handle("not_a_tool", "{}", None)
    assert "error" in result
    assert "unknown tool" in result["error"].lower()


def test_openai_specs_lists_all_registered_tools():
    reg = ToolRegistry()
    reg.register(Tool(name="a", args_model=EmitProfileArgs, handler=lambda *_: {}))
    reg.register(Tool(name="b", args_model=EmitProfileArgs, handler=lambda *_: {}))
    specs = reg.openai_specs()
    assert {s["function"]["name"] for s in specs} == {"a", "b"}
