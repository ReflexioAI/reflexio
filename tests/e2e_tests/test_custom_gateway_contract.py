"""The documented gateway config must reach a real OpenAI-compatible transport.

Local HTTP only, with a fake key. No vendor account or paid calls. Nested gateway
model ids must survive LiteLLM's single OpenAI routing-prefix removal.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import litellm
import pytest
from litellm.main import completion
from pydantic import BaseModel

from reflexio.models.config_schema import APIKeyConfig, CustomEndpointConfig
from reflexio.server.llm._litellm_text_generation import TextGenerationMixin
from reflexio.server.llm.litellm_client import (
    ISOLATED_COMPLETION_EXECUTION,
    LiteLLMClient,
    LiteLLMConfig,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop
from reflexio.test_support.llm_mock import assert_litellm_unpatched, unpatched_litellm

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "gateway_model", ["orcarouter/auto", "anthropic/claude-sonnet-4"]
)
@pytest.mark.parametrize(
    "per_call,tool_mode",
    [(False, None), (True, None), (False, "empty"), (False, "finish")],
)
def test_custom_gateway_keeps_nested_model_id_and_bearer_key(
    gateway_model, per_call, tool_mode, monkeypatch
):
    # This contract intentionally targets loopback, regardless of the caller's
    # production URL policy. monkeypatch restores that policy after the test.
    monkeypatch.delenv("REFLEXIO_BLOCK_PRIVATE_URLS", raising=False)
    # Keep inherited observability settings from sending this local probe to
    # a remote telemetry service. Completion transport itself stays real.
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "telemetry", False)
    for callback_list in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "input_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "_async_input_callback",
    ):
        monkeypatch.setattr(litellm, callback_list, [])
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers.get("Authorization"), body))
            message = {"role": "assistant", "content": "gateway-ok"}
            if tool_mode == "finish":
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "finish-local",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": '{"answer":"gateway-ok"}',
                            },
                        }
                    ],
                }
            payload = json.dumps(
                {
                    "id": "local-contract",
                    "object": "chat.completion",
                    "created": 1,
                    "model": gateway_model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls"
                            if tool_mode == "finish"
                            else "stop",
                            "message": message,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        config = APIKeyConfig(
            custom_endpoint=CustomEndpointConfig(
                api_key="local-contract-key",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",  # type: ignore[arg-type]
            )
        )
        client = LiteLLMClient(
            LiteLLMConfig(
                model="openai/not-requested" if per_call else f"openai/{gateway_model}",
                api_key_config=config,
                timeout=10,
                max_retries=0,
                fallback_models=[],
            )
        )
        with unpatched_litellm(), monkeypatch.context() as transport:
            # Lift the per-test fixture patch too; the only endpoint is local.
            transport.setattr(litellm, "completion", completion)
            transport.setattr(
                TextGenerationMixin,
                "_execute_completion",
                ISOLATED_COMPLETION_EXECUTION,
            )
            assert_litellm_unpatched()
            if tool_mode:

                class Finish(BaseModel):
                    answer: str

                def finish(args, ctx):
                    ctx.append(args.answer)
                    return {}

                handled = []
                registry = ToolRegistry(
                    [Tool(name="finish", args_model=Finish, handler=finish)]
                    if tool_mode == "finish"
                    else []
                )
                result = run_tool_loop(
                    client=client,
                    messages=[{"role": "user", "content": "Local contract check."}],
                    registry=registry,
                    model_role=ModelRole.EXTRACTION_AGENT,
                    max_steps=1,
                    ctx=handled,
                    finish_tool_name="finish",
                )
                assert result.finished_reason == (
                    "finish_tool" if tool_mode == "finish" else "no_tool_call"
                )
                assert handled == (["gateway-ok"] if tool_mode == "finish" else [])
            else:
                result = client.generate_chat_response(
                    [{"role": "user", "content": "Local contract check."}],
                    model_role=ModelRole.GENERATION,
                    model=f"openai/{gateway_model}" if per_call else None,
                )
                assert result == "gateway-ok"
        assert len(received) == 1
        path, authorization, body = received[0]
        assert path == "/v1/chat/completions"
        assert authorization == "Bearer local-contract-key"
        assert body["model"] == gateway_model
        if tool_mode == "finish":
            assert body["tools"][0]["function"]["name"] == "finish"
        elif tool_mode == "empty":
            assert "tools" not in body
        assert body["messages"] == [
            {"role": "user", "content": "Local contract check."}
        ]
        # Custom completion routing must not repoint the embedding provider.
        assert client._resolve_api_key(for_embedding=True) == (None, None, None)
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
