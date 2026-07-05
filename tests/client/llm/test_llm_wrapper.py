"""Unit tests for the auto-publishing LLM-client wrapper (``wrap_llm_client``).

Uses fake provider clients (OpenAI Chat / Responses, Anthropic) and a fake
``ReflexioClient`` whose thread pool runs inline so publishing is deterministic.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from pydantic import ValidationError

from reflexio import ReflexioParams
from reflexio import wrap_llm_client as _real_wrap
from reflexio.client import ReflexioClient
from reflexio.client.llm.adapters import BaseAdapter
from reflexio.client.llm.proxy import _AttrPathProxy
from reflexio.server.services.service_utils import (
    format_interactions_to_history_string,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _InlinePool:
    def submit(self, fn: Any) -> None:
        fn()


class FakeReflexioClient:
    def __init__(self) -> None:
        self._thread_pool = _InlinePool()
        self.calls: list[tuple[str, list[Any], dict[str, Any]]] = []

    def publish_interaction(
        self, user_id: str, interactions: Any, **kwargs: Any
    ) -> None:
        self.calls.append((user_id, list(interactions), kwargs))


class RaisingReflexioClient(FakeReflexioClient):
    def publish_interaction(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")


# --- OpenAI Chat Completions shape ---


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str, index: int = 0) -> None:
        self.function = _Fn(name, arguments)
        self.index = index
        self.id = f"call_{index}"


class _Msg:
    def __init__(
        self, content: str | None = None, tool_calls: list[_ToolCall] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, message: _Msg) -> None:
        self.message = message


class _ChatResp:
    def __init__(self, message: _Msg) -> None:
        self.choices = [_Choice(message)]


class _Delta:
    def __init__(
        self, content: str | None = None, tool_calls: list[Any] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Chunk:
    def __init__(self, delta: _Delta) -> None:
        self.choices = [_Choice.__new__(_Choice)]
        self.choices[0].delta = delta  # type: ignore[attr-defined]


class _Completions:
    def __init__(self, resp: Any = None, stream: list[Any] | None = None) -> None:
        self._resp = resp
        self._stream = stream
        self.received: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.received = kwargs
        if kwargs.get("stream"):
            return iter(self._stream or [])
        return self._resp


class _Chat:
    def __init__(self, resp: Any = None, stream: list[Any] | None = None) -> None:
        self.completions = _Completions(resp, stream)


class _Responses:
    def __init__(self, resp: Any = None, stream: list[Any] | None = None) -> None:
        self._resp = resp
        self._stream = stream
        self.received: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.received = kwargs
        if kwargs.get("stream"):
            return iter(self._stream or [])
        return self._resp


class FakeOpenAI:
    def __init__(
        self,
        chat_resp: Any = None,
        chat_stream: list[Any] | None = None,
        responses_resp: Any = None,
        responses_stream: list[Any] | None = None,
    ) -> None:
        self.chat = _Chat(chat_resp, chat_stream)
        self.responses = _Responses(responses_resp, responses_stream)
        self.api_key = "sk-test"


class _RespAPI:
    def __init__(
        self, output_text: str | None = "", output: list[Any] | None = None
    ) -> None:
        self.output_text = output_text
        self.output = output or []


# --- Async OpenAI shape ---


class _AsyncCompletions:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    async def create(self, **kwargs: Any) -> Any:
        return self._resp


class _AsyncChat:
    def __init__(self, resp: Any) -> None:
        self.completions = _AsyncCompletions(resp)


class AsyncFakeOpenAI:
    def __init__(self, resp: Any) -> None:
        self.chat = _AsyncChat(resp)


# --- Anthropic shape ---


class _AntResp:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _Messages:
    def __init__(self, resp: Any = None, stream: list[Any] | None = None) -> None:
        self._resp = resp
        self._stream = stream
        self.received: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.received = kwargs
        if kwargs.get("stream"):
            return iter(self._stream or [])
        return self._resp


class FakeAnthropic:
    def __init__(self, resp: Any = None, stream: list[Any] | None = None) -> None:
        self.messages = _Messages(resp, stream)


def _wrap(client: Any, **kwargs: Any) -> Any:
    """Call wrap_llm_client, casting the fake Reflexio client to satisfy typing."""
    rc = kwargs.get("reflexio_client")
    if rc is not None:
        kwargs["reflexio_client"] = cast(ReflexioClient, rc)
    return _real_wrap(client, **kwargs)


def _ident(**extra: Any) -> dict[str, Any]:
    base = {"user_id": "u", "session_id": "s"}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forwarded_and_identical_object_returned() -> None:
    resp = _ChatResp(_Msg(content="hi"))
    client = _wrap(
        FakeOpenAI(chat_resp=resp), reflexio_client=FakeReflexioClient()
    )
    out = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}], reflexio=_ident()
    )
    assert out is resp


def test_non_completion_passthrough_and_namespace_walk() -> None:
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("hi"))),
        reflexio_client=FakeReflexioClient(),
    )
    # plain attribute passes through raw
    assert client.api_key == "sk-test"
    # namespace nodes keep walking as proxies
    assert isinstance(client.chat, _AttrPathProxy)
    assert isinstance(client.chat.completions, _AttrPathProxy)


def test_payload_user_content_not_from_prompt_and_identity() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg(content="The weather is sunny"))),
        reflexio_client=rc,
    )
    client.chat.completions.create(
        model="m",
        messages=[
            {"role": "system", "content": "system framing"},
            {"role": "user", "content": "FULL TEMPLATED PROMPT WITH CONTEXT ..."},
        ],
        reflexio=_ident(user_content="what's the weather?", source="app"),
    )
    assert len(rc.calls) == 1
    user_id, interactions, kwargs = rc.calls[0]
    assert user_id == "u"
    assert kwargs["session_id"] == "s"
    assert kwargs["source"] == "app"
    # Exact roles + clean user content (NOT the templated prompt).
    assert [(i.role, i.content) for i in interactions] == [
        ("User", "what's the weather?"),
        ("Assistant", "The weather is sunny"),
    ]


def test_merge_semantics_per_call_wins_and_wrap_default_applies() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("ok"))),
        reflexio={"source": "app", "agent_version": "v9"},
        reflexio_client=rc,
    )
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        reflexio=_ident(source="override", user_content="hi"),
    )
    _, _, kwargs = rc.calls[0]
    assert kwargs["source"] == "override"  # per-call wins
    assert kwargs["agent_version"] == "v9"  # wrap-time default still applies


def test_reflexio_kwarg_stripped_from_forwarded_call() -> None:
    fake = FakeOpenAI(chat_resp=_ChatResp(_Msg("ok")))
    client = _wrap(fake, reflexio_client=FakeReflexioClient())
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}], reflexio=_ident()
    )
    assert "reflexio" not in (fake.chat.completions.received or {})


def test_missing_identity_when_publishing_raises_before_call() -> None:
    fake = FakeOpenAI(chat_resp=_ChatResp(_Msg("ok")))
    client = _wrap(fake, reflexio_client=FakeReflexioClient())
    with pytest.raises(ValueError, match="user_id"):
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            reflexio={"user_content": "hi"},  # no user_id/session_id, publish defaults True
        )
    assert fake.chat.completions.received is None  # provider never called


def test_publish_false_without_identity_does_not_require_it() -> None:
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_resp=_ChatResp(_Msg("ok"))), reflexio_client=rc)
    out = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        reflexio={"publish": False},  # no identity needed when not publishing
    )
    assert out is not None
    assert rc.calls == []


def test_reflexio_params_object_accepted() -> None:
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_resp=_ChatResp(_Msg("ans"))), reflexio_client=rc)
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        reflexio=ReflexioParams(user_id="u", session_id="s", user_content="hi"),
    )
    user_id, interactions, _ = rc.calls[0]
    assert user_id == "u"
    assert [(i.role, i.content) for i in interactions] == [
        ("User", "hi"),
        ("Assistant", "ans"),
    ]


def test_reflexio_params_missing_identity_raises_at_construction() -> None:
    with pytest.raises(ValidationError):
        ReflexioParams(user_content="hi")  # type: ignore[call-arg]


def test_reflexio_params_empty_identity_rejected() -> None:
    with pytest.raises(ValidationError):
        ReflexioParams(user_id="", session_id="s")


def test_unknown_key_raises_typeerror_before_provider_call() -> None:
    fake = FakeOpenAI(chat_resp=_ChatResp(_Msg("ok")))
    client = _wrap(fake, reflexio_client=FakeReflexioClient())
    with pytest.raises(TypeError):
        client.chat.completions.create(model="m", messages=[], reflexio=_ident(bogus=1))
    assert fake.chat.completions.received is None  # provider never called


def test_invalid_flag_combo_raises_valueerror_before_provider_call() -> None:
    fake = FakeOpenAI(chat_resp=_ChatResp(_Msg("ok")))
    client = _wrap(fake, reflexio_client=FakeReflexioClient())
    with pytest.raises(ValueError):
        client.chat.completions.create(
            model="m",
            messages=[],
            reflexio=_ident(evaluation_only=True, force_extraction=True),
        )
    assert fake.chat.completions.received is None


def test_evaluation_only_requires_session_id() -> None:
    fake = FakeOpenAI(chat_resp=_ChatResp(_Msg("ok")))
    client = _wrap(fake, reflexio_client=FakeReflexioClient())
    with pytest.raises(ValueError):
        client.chat.completions.create(
            model="m",
            messages=[],
            reflexio={"user_id": "u", "evaluation_only": True},
        )


def test_publish_failure_does_not_propagate() -> None:
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("ok"))),
        reflexio_client=RaisingReflexioClient(),
    )
    out = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        reflexio=_ident(user_content="hi"),
    )
    assert out is not None  # no raise into the caller


def test_wrap_time_extractor_supplies_user_turn() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("ans"))),
        reflexio_client=rc,
        user_content_extractor=lambda ctx: ctx.adapter.default_user_content(ctx),
    )
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi there"}],
        reflexio=_ident(),
    )
    _, interactions, _ = rc.calls[0]
    assert interactions[0].role == "User"
    assert interactions[0].content == "hi there"


def test_extractor_returning_none_yields_no_user_turn() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("ans"))),
        reflexio_client=rc,
        user_content_extractor=lambda _ctx: None,
    )
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], reflexio=_ident()
    )
    _, interactions, _ = rc.calls[0]
    assert [i.role for i in interactions] == ["Assistant"]


def test_multi_turn_publishes_each_turn_independently() -> None:
    rc = FakeReflexioClient()
    c1 = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a1"))), reflexio_client=rc
    )
    c1.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q1"),
    )
    c2 = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a2"))), reflexio_client=rc
    )
    c2.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "y"}],
        reflexio=_ident(user_content="q2"),
    )
    assert len(rc.calls) == 2
    assert [(i.role, i.content) for i in rc.calls[0][1]] == [
        ("User", "q1"),
        ("Assistant", "a1"),
    ]
    assert [(i.role, i.content) for i in rc.calls[1][1]] == [
        ("User", "q2"),
        ("Assistant", "a2"),
    ]


def test_user_content_plus_tool_call_response_uniform_path() -> None:
    resp = _ChatResp(
        _Msg(content=None, tool_calls=[_ToolCall("get_weather", '{"city": "SF"}')])
    )
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_resp=resp), reflexio_client=rc)
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="weather?"),
    )
    _, interactions, _ = rc.calls[0]
    assert (interactions[0].role, interactions[0].content) == ("User", "weather?")
    assistant = interactions[1]
    assert assistant.role == "Assistant"
    assert assistant.content == "(tool call)"
    assert assistant.tools_used[0].tool_name == "get_weather"
    assert assistant.tools_used[0].tool_data == {"input": {"city": "SF"}}


def test_function_call_continuation_publishes_assistant_only_and_survives_renderer() -> (
    None
):
    resp = _ChatResp(_Msg(content=None, tool_calls=[_ToolCall("do_x", "{}")]))
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_resp=resp), reflexio_client=rc)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], reflexio=_ident()
    )
    _, interactions, _ = rc.calls[0]
    assert [i.role for i in interactions] == ["Assistant"]
    assert interactions[0].content == "(tool call)"
    rendered = format_interactions_to_history_string(interactions)
    assert "used tool: do_x" in rendered  # tool-only turn not dropped by renderer


def test_all_empty_batch_skips_publish() -> None:
    resp = _ChatResp(_Msg(content=None))  # no content, no tools, no user_content
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_resp=resp), reflexio_client=rc)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], reflexio=_ident()
    )
    assert rc.calls == []


def test_publish_gate_publish_false_and_filter() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a"))), reflexio_client=rc
    )
    out = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q", publish=False),
    )
    assert out is not None
    assert rc.calls == []

    rc2 = FakeReflexioClient()
    client2 = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a"))),
        reflexio_client=rc2,
        publish_filter=lambda _ctx: False,
    )
    client2.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q"),
    )
    assert rc2.calls == []


def test_extraction_passthroughs_forwarded() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a"))), reflexio_client=rc
    )
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q", skip_aggregation=True, force_extraction=True),
    )
    _, _, kwargs = rc.calls[0]
    assert kwargs["skip_aggregation"] is True
    assert kwargs["force_extraction"] is True


def test_streaming_publishes_on_normal_exhaustion() -> None:
    chunks = [_Chunk(_Delta(content="Hel")), _Chunk(_Delta(content="lo"))]
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_stream=chunks), reflexio_client=rc)
    gen = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        reflexio=_ident(user_content="q"),
    )
    received = list(gen)
    assert len(received) == 2
    _, interactions, _ = rc.calls[0]
    assert interactions[-1].content == "Hello"


def test_streaming_early_break_skips_publish() -> None:
    chunks = [_Chunk(_Delta(content="Hel")), _Chunk(_Delta(content="lo"))]
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_stream=chunks), reflexio_client=rc)
    gen = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        reflexio=_ident(user_content="q"),
    )
    for _ in gen:
        break
    gen.close()
    assert rc.calls == []


def test_streaming_partial_opt_in_publishes_accumulated() -> None:
    chunks = [_Chunk(_Delta(content="Hel")), _Chunk(_Delta(content="lo"))]
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_stream=chunks), reflexio_client=rc)
    gen = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        reflexio=_ident(user_content="q", publish_partial_stream=True),
    )
    next(gen)  # consume only the first chunk
    gen.close()
    assert len(rc.calls) == 1
    _, interactions, _ = rc.calls[0]
    assert interactions[-1].content == "Hel"


def test_async_await_returns_identical_and_publishes() -> None:
    resp = _ChatResp(_Msg("async ans"))
    rc = FakeReflexioClient()
    client = _wrap(AsyncFakeOpenAI(resp), reflexio_client=rc)
    out = asyncio.run(
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            reflexio=_ident(user_content="q"),
        )
    )
    assert out is resp
    _, interactions, _ = rc.calls[0]
    assert interactions[-1].content == "async ans"


def test_multi_adapter_one_client_chat_and_responses() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(
            chat_resp=_ChatResp(_Msg("chat ans")),
            responses_resp=_RespAPI(output_text="resp ans"),
        ),
        reflexio_client=rc,
    )
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q1"),
    )
    client.responses.create(
        model="m", input="hello", reflexio=_ident(user_content="q2")
    )
    assert rc.calls[0][1][-1].content == "chat ans"
    assert rc.calls[1][1][-1].content == "resp ans"


def test_responses_adapter_parses_function_call() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(
            responses_resp=_RespAPI(
                output_text=None,
                output=[
                    {"type": "function_call", "name": "f", "arguments": '{"a": 1}'}
                ],
            )
        ),
        reflexio_client=rc,
    )
    client.responses.create(model="m", input="hi", reflexio=_ident())
    assistant = rc.calls[0][1][-1]
    assert assistant.content == "(tool call)"
    assert assistant.tools_used[0].tool_name == "f"
    assert assistant.tools_used[0].tool_data == {"input": {"a": 1}}


def test_responses_adapter_stream_text_deltas() -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
    ]
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(responses_stream=events), reflexio_client=rc)
    gen = client.responses.create(
        model="m", input="hi", stream=True, reflexio=_ident(user_content="q")
    )
    list(gen)
    assert rc.calls[0][1][-1].content == "Hello"


def test_anthropic_adapter_parses_text_content() -> None:
    rc = FakeReflexioClient()
    client = _wrap(
        FakeAnthropic(resp=_AntResp([{"type": "text", "text": "hi there"}])),
        reflexio_client=rc,
    )
    client.messages.create(
        model="c",
        max_tokens=10,
        messages=[{"role": "user", "content": "q"}],
        reflexio=_ident(user_content="hello"),
    )
    assistant = rc.calls[0][1][-1]
    assert assistant.role == "Assistant"
    assert assistant.content == "hi there"


def test_anthropic_adapter_stream_text_deltas() -> None:
    events = [
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}},
    ]
    rc = FakeReflexioClient()
    client = _wrap(FakeAnthropic(stream=events), reflexio_client=rc)
    gen = client.messages.create(
        model="c",
        max_tokens=10,
        messages=[{"role": "user", "content": "q"}],
        stream=True,
        reflexio=_ident(user_content="q"),
    )
    list(gen)
    assert rc.calls[0][1][-1].content == "Hello"


def test_custom_adapter_prepended_wins_over_builtin() -> None:
    class CustomAdapter(BaseAdapter):
        namespace_prefixes = frozenset({("chat",), ("chat", "completions")})

        def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool:
            return path[-3:] == ("chat", "completions", "create")

        def _assistant_from_response(self, response: Any):
            return "CUSTOM", []

        def _assistant_from_chunks(self, chunks: list[Any]):
            return "CUSTOM", []

    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("orig"))),
        reflexio_client=rc,
        adapters=[CustomAdapter()],
    )
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q"),
    )
    assert rc.calls[0][1][-1].content == "CUSTOM"


def test_bare_module_callable_publishes() -> None:
    # wrap_llm_client(litellm.completion)(...) — bare callable, empty path.
    def completion(**kwargs: Any) -> Any:
        return _ChatResp(_Msg("bare ans"))

    rc = FakeReflexioClient()
    wrapped = _wrap(completion, reflexio_client=rc)
    out = wrapped(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q"),
    )
    assert out.choices[0].message.content == "bare ans"
    assert len(rc.calls) == 1
    assert rc.calls[0][1][-1].content == "bare ans"


def test_user_content_extractor_exception_does_not_raise() -> None:
    def boom(ctx: Any) -> str:
        raise RuntimeError("extractor boom")

    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a"))),
        reflexio_client=rc,
        user_content_extractor=boom,
    )
    out = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], reflexio=_ident()
    )
    assert out is not None  # hook failure must not raise into the call
    # Fail-safe: extractor error => no User turn, but the assistant turn still publishes.
    assert len(rc.calls) == 1
    assert [i.role for i in rc.calls[0][1]] == ["Assistant"]


def test_publish_filter_exception_does_not_raise() -> None:
    def boom(ctx: Any) -> bool:
        raise RuntimeError("filter boom")

    rc = FakeReflexioClient()
    client = _wrap(
        FakeOpenAI(chat_resp=_ChatResp(_Msg("a"))),
        reflexio_client=rc,
        publish_filter=boom,
    )
    out = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q"),
    )
    assert out is not None
    assert rc.calls == []


# --- Async streaming + tool-accumulation fakes ---


class _AsyncStreamCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def create(self, **kwargs: Any) -> Any:
        async def _agen() -> Any:
            for chunk in self._chunks:
                yield chunk

        return _agen()


class _AsyncStreamChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = _AsyncStreamCompletions(chunks)


class AsyncStreamFakeOpenAI:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = _AsyncStreamChat(chunks)


class _ToolCallDelta:
    def __init__(
        self, index: int = 0, name: str | None = None, arguments: str | None = None
    ) -> None:
        self.index = index
        self.function = _Fn(name or "", arguments or "")
        if name is None:
            self.function.name = None  # type: ignore[assignment]
        if arguments is None:
            self.function.arguments = None  # type: ignore[assignment]


def test_async_streaming_tee_publishes_on_exhaustion() -> None:
    chunks = [_Chunk(_Delta(content="Hel")), _Chunk(_Delta(content="lo"))]
    rc = FakeReflexioClient()
    client = _wrap(AsyncStreamFakeOpenAI(chunks), reflexio_client=rc)

    async def run() -> list[Any]:
        stream = await client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
            reflexio=_ident(user_content="q"),
        )
        return [chunk async for chunk in stream]

    received = asyncio.run(run())
    assert len(received) == 2
    assert rc.calls[0][1][-1].content == "Hello"


def test_litellm_module_path_publishes() -> None:
    # wrap_llm_client(litellm).completion(...) — matched via __getattr__ path.
    class FakeLitellm:
        def completion(self, **kwargs: Any) -> Any:
            return _ChatResp(_Msg("litellm ans"))

    rc = FakeReflexioClient()
    client = _wrap(FakeLitellm(), reflexio_client=rc)
    out = client.completion(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reflexio=_ident(user_content="q"),
    )
    assert out.choices[0].message.content == "litellm ans"
    assert rc.calls[0][1][-1].content == "litellm ans"


def test_openai_streaming_tool_accumulation() -> None:
    # Tool name arrives in the first chunk; arguments are split across chunks.
    chunks = [
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(0, name="get_weather", arguments='{"ci')])),
        _Chunk(_Delta(tool_calls=[_ToolCallDelta(0, arguments='ty": "SF"}')])),
    ]
    rc = FakeReflexioClient()
    client = _wrap(FakeOpenAI(chat_stream=chunks), reflexio_client=rc)
    list(
        client.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            stream=True,
            reflexio=_ident(user_content="q"),
        )
    )
    assistant = rc.calls[0][1][-1]
    assert assistant.content == "(tool call)"
    assert assistant.tools_used[0].tool_name == "get_weather"
    assert assistant.tools_used[0].tool_data == {"input": {"city": "SF"}}


def test_anthropic_streaming_tool_accumulation() -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "name": "get_weather"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ' "SF"}'},
        },
    ]
    rc = FakeReflexioClient()
    client = _wrap(FakeAnthropic(stream=events), reflexio_client=rc)
    list(
        client.messages.create(
            model="c",
            max_tokens=10,
            messages=[{"role": "user", "content": "x"}],
            stream=True,
            reflexio=_ident(user_content="q"),
        )
    )
    assistant = rc.calls[0][1][-1]
    assert assistant.content == "(tool call)"
    assert assistant.tools_used[0].tool_name == "get_weather"
    assert assistant.tools_used[0].tool_data == {"input": {"city": "SF"}}
