"""Completion identity, response metadata, and private request behavior."""

import json
import logging
import pickle
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from threading import Thread
from unittest.mock import Mock

import pytest
from pydantic import HttpUrl

from reflexio.models.config_schema import (
    APIKeyConfig,
    CustomEndpointConfig,
    OpenAIConfig,
)
from reflexio.server.llm.failure import Terminal, Transient, classify_llm_failure
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)
from reflexio.server.llm.model_defaults import ModelRole

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "status,upstream_type,verdict",
    [
        (400, None, Terminal),
        (529, None, Transient),
        (None, "RateLimitError", Transient),
    ],
)
def test_private_provider_failure_keeps_classification_without_body(
    monkeypatch, caplog, status, upstream_type, verdict
):
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    private_marker = "synthetic provider echoed customer evidence"

    def reject(params, _timeout):
        assert "redact_provider_errors" not in params
        assert params["no-log"] is True
        raise LiteLLMClientError(private_marker) from LiteLLMClientError(
            private_marker,
            upstream_status_code=status,
            upstream_error_type=upstream_type,
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", reject)
    with pytest.raises(LiteLLMClientError, match="Provider request failed") as caught:
        client._make_request(
            [],
            redact_provider_errors=True,
            parse_structured_output=False,
            fallback_models=[],
        )
    assert private_marker not in caplog.text
    assert private_marker not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    classified = classify_llm_failure(caught.value, consult_breaker=False)
    assert isinstance(classified, verdict)
    if upstream_type is not None:
        assert caught.value.upstream_error_type == upstream_type


def test_plain_completion_preserves_finish_reason_and_unsolicited_tool_call(
    monkeypatch,
):
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    from litellm import ModelResponse

    response = ModelResponse(
        model="gpt-4o",
        choices=[
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "synthetic body",
                    "tool_calls": [
                        {
                            "id": "synthetic",
                            "type": "function",
                            "function": {"name": "synthetic", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", lambda *_: response)
    result = client._make_request([])
    assert result.value == "synthetic body"
    assert result.finish_reason == "length"
    assert result.has_tool_calls


def test_private_request_skips_litellm_failure_callback(monkeypatch):
    import httpx
    import litellm
    from litellm.caching.caching import Cache
    from litellm.integrations.custom_logger import CustomLogger

    from reflexio.server.llm._litellm_subprocess import (
        _litellm_completion_worker,
        _raise_completion_error,
    )
    from reflexio.test_support.llm_mock import unpatched_litellm

    seen = []

    class TransportSpy(httpx.HTTPTransport):
        def handle_request(self, request):
            seen.append("sync-transport")
            return super().handle_request(request)

    class FailureSpy(CustomLogger):
        def log_failure_event(self, *_args, **_kwargs):
            seen.append("failure")

        def log_post_api_call(self, *_args, **_kwargs):
            seen.append("body")

    for key in (
        "callbacks",
        "success_callback",
        "service_callback",
        "_async_input_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, key, [])
    monkeypatch.setattr(litellm, "failure_callback", [FailureSpy()])
    monkeypatch.setattr(litellm, "input_callback", [FailureSpy()])
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    monkeypatch.setattr(litellm, "set_verbose", False)
    monkeypatch.setattr(litellm, "global_disable_no_log_param", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)

    response_reasons = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if response_reasons:
                body = json.dumps(
                    {
                        "model": "MiniMax-M3",
                        "id": "synthetic",
                        "object": "chat.completion",
                        "created": 1,
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "content": "synthetic answer",
                                    "role": "assistant",
                                },
                                "finish_reason": response_reasons.pop(0),
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
            else:
                body = b'{"error":{"message":"synthetic provider error"}}'
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))

    def run_test_worker(params, timeout):
        queue = SimpleQueue()
        previous_logging = logging.root.manager.disable
        try:
            _litellm_completion_worker(params, queue, time.monotonic() + timeout)
        finally:
            logging.disable(previous_logging)
        status, value = queue.get_nowait()
        if status == "error":
            _raise_completion_error(value)
        return pickle.loads(pickle.dumps(value))  # noqa: S301 -- local worker fixture

    monkeypatch.setattr(client, "_completion_with_hard_timeout", run_test_worker)
    with (
        unpatched_litellm(),
        ThreadingHTTPServer(("127.0.0.1", 0), Handler) as provider,
        httpx.Client(
            event_hooks={"request": [lambda _: seen.append("http-hook")]}
        ) as http,
        TransportSpy() as inherited_transport,
    ):
        monkeypatch.setattr(litellm, "client_session", http)
        monkeypatch.setattr(litellm, "aclient_session", None)
        thread = Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        try:
            messages = [{"role": "user", "content": "synthetic evidence"}]
            params = {
                "api_base": f"http://127.0.0.1:{provider.server_port}/v1",
                "api_key": "synthetic-key",
                "fallback_models": [],
                "logger_fn": lambda _: seen.append("per-request"),
            }
            with pytest.raises(LiteLLMClientError):
                client._make_request(messages, **params)
            assert {"failure", "body", "per-request", "http-hook"} <= set(seen)
            monkeypatch.setattr(
                litellm, "sync_transport", inherited_transport, raising=False
            )
            response_reasons.append("stop")
            client._make_request(messages, model="minimax/MiniMax-M3", **params)
            assert "sync-transport" in seen
            seen.clear()
            with pytest.raises(LiteLLMClientError, match="Provider request failed"):
                client._make_request(
                    messages,
                    redact_provider_errors=True,
                    parse_structured_output=False,
                    **params,
                )
            assert seen == []
            for raw_reason, expected in [
                ("stop", "stop"),
                ("length", "length"),
                (None, None),
                ("error", None),
            ]:
                monkeypatch.setattr(litellm, "sync_transport", inherited_transport)
                cache = Mock(spec=Cache)
                monkeypatch.setattr(litellm, "cache", cache)
                response_reasons.append(raw_reason)
                result = client._make_request(
                    messages,
                    redact_provider_errors=True,
                    parse_structured_output=False,
                    model="minimax/MiniMax-M3",
                    caching=True,
                    **params,
                )
                assert result.value == "synthetic answer"
                assert result.finish_reason == expected
                assert not cache.mock_calls
            assert seen == []
            with pytest.raises(ValueError, match="native IO"):
                client._make_request(
                    messages,
                    redact_provider_errors=True,
                    parse_structured_output=False,
                    client=object(),
                    **params,
                )
        finally:
            provider.shutdown()
            thread.join()


KEYS = (
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


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "minimax/MiniMax-M3"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"max_tokens": 1024, "temperature": 0},
        {"model_role": ModelRole.EXTRACTION_AGENT},
        {"max_tokens": None},
        {"fallback_models": None},
        {"fallback_models": []},
        {"seed": 19, "temperature": 0.3},
    ],
)
def test_contract_matches_actual_builder_and_preserves_none_keys(
    monkeypatch, model, overrides
):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("Identity resolution cannot dispatch"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(
            model=model,
            max_tokens=2048,
            top_p=0.8,
            api_key_config=APIKeyConfig(
                openai=OpenAIConfig(api_key="contract-test-key")
            ),
        )
    )
    forwarded = Mock(return_value=object())
    monkeypatch.setattr(client, "_make_request", forwarded)
    client.generate_chat_response_with_provenance([], **overrides)
    params = client._build_completion_params(messages=[], **forwarded.call_args.kwargs)[
        0
    ]
    identity = client.get_completion_request_contract(**overrides)
    assert {key: identity[key] for key in KEYS} == {
        key: params.get(key) for key in KEYS
    }
    assert tuple(identity) == (*KEYS, "generation_policy_digest")
    assert len(identity["generation_policy_digest"]) == 64
    assert identity["seed"] == overrides.get("seed")
    assert identity["base_url"] is None
    assert identity["custom_llm_provider"] is None
    assert "api_key" not in identity and "messages" not in identity
    assert "contract-test-key" not in identity.values()
    if overrides.get("max_tokens", 2048) is not None:
        cap = identity["max_completion_tokens"] or identity["max_tokens"]
        assert cap == overrides.get("max_tokens", 2048)
    dispatch.assert_not_called()


def test_custom_route_wins_role_and_descriptor_is_for_hashing_not_logging(monkeypatch):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("Identity resolution cannot dispatch"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/custom-contract",
            max_tokens=2048,
            api_key_config=APIKeyConfig(
                custom_endpoint=CustomEndpointConfig(
                    api_key="custom-key",
                    api_base=HttpUrl(
                        "https://example.invalid/v1?access_token=route-credential"
                    ),
                )
            ),
        )
    )
    identity = client.get_completion_request_contract(
        model_role=ModelRole.EXTRACTION_AGENT, max_tokens=777, temperature=0
    )
    params = client._build_completion_params(
        messages=[],
        model_role=ModelRole.EXTRACTION_AGENT,
        max_tokens=777,
        temperature=0,
    )[0]
    assert {key: identity[key] for key in KEYS} == {
        key: params.get(key) for key in KEYS
    }
    assert identity["model"] == "openai/custom-contract"
    assert "route-credential" in identity["api_base"]
    assert identity["max_tokens"] == 777
    assert "api_key" not in identity and "custom-key" not in identity.values()
    dispatch.assert_not_called()


def test_unset_seed_and_explicit_temperature_none_keep_builder_semantics(monkeypatch):
    monkeypatch.delenv("REFLEXIO_LLM_SEED", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini", temperature=0.7))
    assert client.get_completion_request_contract()["temperature"] == 0.7
    changed = client.get_completion_request_contract(temperature=0.2)
    assert changed["temperature"] == 0.2 and changed["seed"] is None
    assert (
        client.get_completion_request_contract(temperature=None)["temperature"] is None
    )
    assert client.config.temperature == 0.7


@pytest.mark.parametrize(
    "override", [None, [], ["gpt-4o-mini", "local/nomic", "gpt-4o", "gpt-4o"]]
)
def test_public_ladder_uses_actual_deduped_resolver_without_dispatch(
    monkeypatch, override
):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("No provider for ladder resolution"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-4o-mini",
            fallback_models=["gpt-4o-mini", "gpt-4o", "gpt-4o", "local/nomic"],
        )
    )
    assert client.get_completion_model_ladder() == ("gpt-4o-mini", "gpt-4o")
    options = {"fallback_models": override}
    forwarded = Mock(return_value=object())
    monkeypatch.setattr(client, "_make_request", forwarded)
    client.generate_chat_response_with_provenance([], **options)
    effective = forwarded.call_args.kwargs
    assert ("fallback_models" in effective) is (override is not None)
    assert client.get_completion_model_ladder(**options) == tuple(
        client._resolve_ladder(**effective)
    )
    assert client.get_completion_model_ladder(**options) == (
        ("gpt-4o-mini",) if override == [] else ("gpt-4o-mini", "gpt-4o")
    )
    dispatch.assert_not_called()


def test_public_ladder_uses_role_precedence_and_custom_endpoint_pin(monkeypatch):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("No provider for ladder resolution"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", fallback_models=["gpt-4o", "gpt-4o"])
    )
    overrides = {"model": "ignored-model", "model_role": ModelRole.EXTRACTION_AGENT}
    primary = client.get_completion_request_contract(**overrides)["model"]
    assert client.get_completion_model_ladder(**overrides) == tuple(
        dict.fromkeys([primary, "gpt-4o"])
    )
    assert primary != "ignored-model"
    client.config.api_key_config = APIKeyConfig(
        custom_endpoint=CustomEndpointConfig(
            api_key="test-private",
            api_base=HttpUrl("https://example.invalid/v1"),
        )
    )
    assert client.get_completion_model_ladder(**overrides) == ("ignored-model",)
    assert (
        client.get_completion_request_contract(**overrides)["model"] == "ignored-model"
    )
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "reasoning_effort",
        "reasoning",
        "thinking",
        "reasoning_split",
        "output_config",
        "extra_body",
    ],
)
def test_contract_binds_actual_reasoning_policy_without_exporting_native_custom_data(
    monkeypatch, field
):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("No dispatch for contract resolution"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    options = {
        "reasoning_effort": ("low", "high"),
        "reasoning": ({"effort": "low"}, {"effort": "high"}),
        "thinking": (
            {"type": "enabled", "budget_tokens": 100},
            {"type": "enabled", "budget_tokens": 200},
        ),
        "reasoning_split": (False, True),
        "output_config": ({"effort": "low"}, {"effort": "high"}),
        "extra_body": (
            {"reasoning_effort": "low", "private_key": "test-custom-secret"},
            {"reasoning_effort": "high", "private_key": "test-custom-secret"},
        ),
    }
    low, high = (
        client.get_completion_request_contract(**{field: value})
        for value in options[field]
    )
    assert {key: low[key] for key in KEYS} == {key: high[key] for key in KEYS}
    assert low["generation_policy_digest"] != high["generation_policy_digest"]
    assert "test-custom-secret" not in str(low)
    assert "extra_body" not in low
    assert low == client.get_completion_request_contract(**{field: options[field][0]})
    dispatch.assert_not_called()


def test_reasoning_digest_uses_canonical_native_mapping_order():
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    left = client.get_completion_request_contract(
        extra_body={"reasoning_effort": "low", "thinking": {"type": "enabled"}}
    )
    right = client.get_completion_request_contract(
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    )
    assert left == right


@pytest.mark.parametrize(
    ("key", "left", "right"),
    [
        ("frequency_penalty", 0, 1),
        ("presence_penalty", 0, 1),
        ("stop", ["stop-a"], ["stop-b"]),
        ("n", 1, 2),
        ("logprobs", False, True),
        ("drop_params", False, True),
        ("allowed_openai_params", ["a"], ["b"]),
        ("tool_choice", "auto", "none"),
        (
            "tools",
            [{"type": "function", "function": {"name": "a"}}],
            [{"type": "function", "function": {"name": "b"}}],
        ),
        ("response_format", {"type": "json_object"}, {"type": "text"}),
        ("future_provider_generation_knob", 1, 2),
    ],
)
def test_contract_binds_every_built_generation_parameter(key, left, right):
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    one = client.get_completion_request_contract(**{key: left})
    two = client.get_completion_request_contract(**{key: right})
    assert one["generation_policy_digest"] != two["generation_policy_digest"]


def test_contract_excludes_credentials_observations_and_dispatch_controls():
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    base = client.get_completion_request_contract()
    changed = client.get_completion_request_contract(
        api_key="private-override",
        extra_headers={"Authorization": "private-header"},
        metadata={"private": "private-trace"},
        callbacks=[lambda: None],
        provider_request_guard=lambda *_args: None,
        structured_output_validator=lambda _value: [],
        timeout=999,
    )
    assert changed == base
    assert "private-" not in str(changed)


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf")])
def test_contract_rejects_unsupported_policy_values_without_stringification(value):
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    with pytest.raises((TypeError, ValueError)):
        client.get_completion_request_contract(future_provider_generation_knob=value)


def test_contract_canonicalizes_actual_pydantic_passthrough_schema(monkeypatch):
    from pydantic import BaseModel, ConfigDict

    class Count(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: int

    class Text(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answer: str

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    monkeypatch.setattr(
        client, "_structured_output_strategy", lambda **_kwargs: "pydantic_passthrough"
    )
    assert (
        client._build_completion_params(messages=[], response_format=Count)[0][
            "response_format"
        ]
        is Count
    )
    one = client.get_completion_request_contract(response_format=Count)
    two = client.get_completion_request_contract(response_format=Text)
    assert one["generation_policy_digest"] != two["generation_policy_digest"]
    assert one == client.get_completion_request_contract(response_format=Count)


@pytest.mark.parametrize("model", ["minimax/MiniMax-M3", "zai/glm-4.7"])
@pytest.mark.parametrize("route", ["api_base", "base_url"])
def test_custom_extraction_route_cannot_inherit_native_output_default(model, route):
    from unittest.mock import Mock

    from reflexio.server.llm.input_budget import (
        InputBudgetConfigurationError,
        extraction_provider_guard,
    )

    client = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=None, fallback_models=[])
    )
    token = extraction_provider_guard.set(Mock())
    try:
        with pytest.raises(
            InputBudgetConfigurationError, match="explicit output reservation"
        ):
            client._build_completion_params([], **{route: "https://custom.invalid/v1"})
        params = client._build_completion_params(
            [], **{route: "https://custom.invalid/v1", "max_tokens": 1024}
        )[0]
        assert params["max_tokens"] == 1024
    finally:
        extraction_provider_guard.reset(token)
