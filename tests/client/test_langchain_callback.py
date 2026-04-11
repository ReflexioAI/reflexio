"""Tests for the optional LangChain callback integration."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from uuid import uuid4


class _FakeClient:
    """Capture publish_interaction calls for assertions."""

    def __init__(self) -> None:
        self.publish_calls: list[dict] = []

    def publish_interaction(self, **kwargs: object) -> None:
        self.publish_calls.append(kwargs)


@dataclass
class _FakeMessage:
    type: str
    content: str


def _load_callback_module(monkeypatch) -> types.ModuleType:
    """Import the callback module with a stubbed langchain_core package."""
    callbacks_mod = types.ModuleType("langchain_core.callbacks")
    outputs_mod = types.ModuleType("langchain_core.outputs")
    root_mod = types.ModuleType("langchain_core")

    class BaseCallbackHandler:  # noqa: D401
        """Minimal stub for LangChain's callback base class."""

    class ChatGeneration:
        def __init__(self, text: str = "", message: _FakeMessage | None = None) -> None:
            self.text = text
            self.message = message

    class LLMResult:
        def __init__(self, generations: list[list[ChatGeneration]]) -> None:
            self.generations = generations

    callbacks_mod.BaseCallbackHandler = BaseCallbackHandler
    outputs_mod.ChatGeneration = ChatGeneration
    outputs_mod.LLMResult = LLMResult
    root_mod.callbacks = callbacks_mod
    root_mod.outputs = outputs_mod

    monkeypatch.setitem(sys.modules, "langchain_core", root_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.outputs", outputs_mod)
    monkeypatch.delitem(
        sys.modules, "reflexio.integrations.langchain.callback", raising=False
    )

    module = importlib.import_module("reflexio.integrations.langchain.callback")
    return importlib.reload(module)


def test_callback_buffers_user_once_and_skips_prompt_side_messages(monkeypatch) -> None:
    """User messages should be buffered once and published on chain end only."""
    callback_mod = _load_callback_module(monkeypatch)
    client = _FakeClient()
    handler = callback_mod.ReflexioCallbackHandler(client=client, user_id="user-1")

    root_run_id = uuid4()
    first_llm_run_id = uuid4()
    second_llm_run_id = uuid4()

    history = [
        _FakeMessage("system", "synthetic context"),
        _FakeMessage("human", "How do I reset a password?"),
        _FakeMessage("ai", "previous assistant response"),
    ]

    handler.on_chat_model_start(
        serialized={},
        messages=[history],
        run_id=first_llm_run_id,
        parent_run_id=root_run_id,
    )
    handler.on_llm_end(
        callback_mod.LLMResult(
            generations=[[callback_mod.ChatGeneration(text="I'll look that up.")]]
        ),
        run_id=first_llm_run_id,
        parent_run_id=root_run_id,
    )

    handler.on_chat_model_start(
        serialized={},
        messages=[history],
        run_id=second_llm_run_id,
        parent_run_id=root_run_id,
    )
    handler.on_llm_end(
        callback_mod.LLMResult(
            generations=[[callback_mod.ChatGeneration(text="Reset it from Settings.")]]
        ),
        run_id=second_llm_run_id,
        parent_run_id=root_run_id,
    )

    assert client.publish_calls == []

    handler.on_chain_end(outputs={}, run_id=root_run_id)

    assert len(client.publish_calls) == 1
    published = client.publish_calls[0]["interactions"]
    assert [interaction.role for interaction in published] == [
        "user",
        "assistant",
        "assistant",
    ]
    assert [interaction.content for interaction in published] == [
        "How do I reset a password?",
        "I'll look that up.",
        "Reset it from Settings.",
    ]


def test_callback_keeps_concurrent_root_runs_isolated(monkeypatch) -> None:
    """Ending one root run should not publish or clear another run's buffer."""
    callback_mod = _load_callback_module(monkeypatch)
    client = _FakeClient()
    handler = callback_mod.ReflexioCallbackHandler(client=client, user_id="user-1")

    root_a = uuid4()
    root_b = uuid4()
    llm_a = uuid4()
    llm_b = uuid4()

    handler.on_chat_model_start(
        serialized={},
        messages=[[_FakeMessage("human", "request-a")]],
        run_id=llm_a,
        parent_run_id=root_a,
    )
    handler.on_llm_end(
        callback_mod.LLMResult(
            generations=[[callback_mod.ChatGeneration(text="response-a")]]
        ),
        run_id=llm_a,
        parent_run_id=root_a,
    )

    handler.on_chat_model_start(
        serialized={},
        messages=[[_FakeMessage("human", "request-b")]],
        run_id=llm_b,
        parent_run_id=root_b,
    )
    handler.on_llm_end(
        callback_mod.LLMResult(
            generations=[[callback_mod.ChatGeneration(text="response-b")]]
        ),
        run_id=llm_b,
        parent_run_id=root_b,
    )

    handler.on_chain_end(outputs={}, run_id=root_a)

    assert len(client.publish_calls) == 1
    published_a = client.publish_calls[0]["interactions"]
    assert [interaction.content for interaction in published_a] == [
        "request-a",
        "response-a",
    ]

    handler.on_chain_end(outputs={}, run_id=root_b)

    assert len(client.publish_calls) == 2
    published_b = client.publish_calls[1]["interactions"]
    assert [interaction.content for interaction in published_b] == [
        "request-b",
        "response-b",
    ]
