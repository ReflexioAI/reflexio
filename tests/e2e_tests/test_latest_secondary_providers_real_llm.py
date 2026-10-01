"""Opt-in, bounded provider smoke tests; every call uses Reflexio's worker.

Set RUN_LOW_PRIORITY=1 and load MINIMAX_API_KEY/ZAI_API_KEY into the process
before pytest starts. Never run this file as part of an ordinary offline gate.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop
from reflexio.test_support.llm_credentials import real_provider_key
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from reflexio.test_support.skip_decorators import skip_low_priority

pytestmark = [pytest.mark.e2e, pytest.mark.requires_credentials]

MODELS = [
    "minimax/MiniMax-M3",
    "minimax/MiniMax-M3.1-Flash-Preview",
    "zai/glm-5.2",
    "zai/glm-5.3",
    "zai/glm-5.3-flash",
]
# FlashX requires the general API rather than the coding-plan endpoint. Its
# transport is qualified offline; this smoke matrix targets subscription keys.


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class LookupArgs(BaseModel):
    """Look up the answer for a key; use the returned answer verbatim."""

    key: str


@pytest.fixture(params=MODELS)
def client(request, monkeypatch):
    model = request.param
    key_name = "MINIMAX_API_KEY" if model.startswith("minimax/") else "ZAI_API_KEY"
    if not real_provider_key(key_name):
        pytest.skip(f"{key_name} is not populated with a real key")
    assert_litellm_unpatched()
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setattr(
        "reflexio.server.llm.tools.resolve_model_name", lambda **_: model
    )
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda *_a, **_k: model,
    )
    result = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=4096, timeout=60, fallback_models=[])
    )
    assert result._should_process_isolate_completion(60, 5)
    # Limit thinking on models that document effort. The legacy baselines keep
    # their existing behavior, and the runtime provider defaults stay intact.
    if model not in {"minimax/MiniMax-M3", "zai/glm-5.2"}:
        generate = result.generate_chat_response_with_provenance

        def with_low_effort(*args, **kwargs):
            kwargs.setdefault("reasoning_effort", "low")
            return generate(*args, **kwargs)

        monkeypatch.setattr(
            result, "generate_chat_response_with_provenance", with_low_effort
        )
    return result


@skip_low_priority
def test_live_text(client):
    result = client.generate_chat_response(
        [{"role": "user", "content": "Reply with exactly the word READY."}]
    )
    assert isinstance(result, str)
    assert result.strip().strip('"').rstrip(".") == "READY"


@skip_low_priority
def test_live_structured_json(client):
    result = client.generate_chat_response(
        [{"role": "user", "content": "Set answer to exactly done."}],
        response_format=Answer,
    )
    assert result == Answer(answer="done")


@skip_low_priority
def test_live_tool_exchange(client):
    calls = []
    registry = ToolRegistry()

    def lookup(args, _ctx):
        calls.append(args.key)
        return {"answer": "done"}

    registry.register(
        Tool(
            name="lookup",
            args_model=LookupArgs,
            handler=lookup,
        )
    )
    result = run_tool_loop(
        client,
        [
            {
                "role": "user",
                "content": "Call lookup once with key test. After observing its result, return a JSON object with answer set to the returned answer. Do not guess the answer before calling the tool.",
            }
        ],
        registry,
        ModelRole.EXTRACTION_AGENT,
        response_format=Answer,
        max_steps=2,
    )
    assert calls == ["test"]
    assert result.structured_output == Answer(answer="done")
