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

from reflexio.models.config_schema import APIKeyConfig, CustomEndpointConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.test_support.llm_mock import assert_litellm_unpatched, unpatched_litellm

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "gateway_model", ["orcarouter/auto", "anthropic/claude-sonnet-4"]
)
def test_custom_gateway_keeps_nested_model_id_and_bearer_key(gateway_model, monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers.get("Authorization"), body))
            payload = json.dumps(
                {
                    "id": "local-contract",
                    "object": "chat.completion",
                    "created": 1,
                    "model": gateway_model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "gateway-ok"},
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
    config = APIKeyConfig(
        custom_endpoint=CustomEndpointConfig(
            model=f"openai/{gateway_model}",
            api_key="local-contract-key",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",  # type: ignore[arg-type]
        )
    )
    client = LiteLLMClient(
        LiteLLMConfig(
            model="unused-model",
            api_key_config=config,
            timeout=10,
            max_retries=0,
            fallback_models=[],
        )
    )
    try:
        with unpatched_litellm(), monkeypatch.context() as transport:
            # Lift the per-test fixture patch too; the only endpoint is local.
            transport.setattr(litellm, "completion", completion)
            assert_litellm_unpatched()
            result = client.generate_chat_response(
                [{"role": "user", "content": "Local contract check."}],
                model_role=ModelRole.GENERATION,
            )
        assert result == "gateway-ok"
        assert len(received) == 1
        path, authorization, body = received[0]
        assert path == "/v1/chat/completions"
        assert authorization == "Bearer local-contract-key"
        assert body["model"] == gateway_model
        assert body["messages"] == [
            {"role": "user", "content": "Local contract check."}
        ]
        # Custom completion routing must not repoint the embedding provider.
        assert client._resolve_api_key(for_embedding=True)[1] is None
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
