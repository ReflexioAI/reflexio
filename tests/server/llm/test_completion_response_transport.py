"""Provider models must be usable after crossing the hard-timeout process seam."""

import multiprocessing
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

from reflexio.server.llm._litellm_subprocess import _picklable_completion_result


@pytest.mark.parametrize("response", ["plain text", {"provider": "plain mapping"}])
def test_plain_provider_returns_keep_existing_transport(response: object) -> None:
    assert _picklable_completion_result(response) is response
    assert pickle.loads(pickle.dumps(response)) == response  # noqa: S301 - trusted fixture


def test_terminal_structured_content_keeps_sdk_response_shape() -> None:
    from litellm.types.utils import ModelResponse

    response = ModelResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": '{"answer":"synthetic"}'},
                "finish_reason": "stop",
            }
        ],
        model="synthetic-model",
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )
    copied = pickle.loads(pickle.dumps(_picklable_completion_result(response)))  # noqa: S301
    assert isinstance(copied, ModelResponse)
    assert copied.model_dump() == response.model_dump()


def test_streaming_chunks_keep_existing_transport() -> None:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    chunk = ModelResponseStream(
        choices=[StreamingChoices(delta=Delta(content="synthetic"), finish_reason=None)]
    )
    assert _picklable_completion_result(chunk) is chunk
    copied = pickle.loads(pickle.dumps(chunk))  # noqa: S301 - trusted fixture
    assert copied.model_dump() == chunk.model_dump()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="The production completion worker uses fork on supported platforms",
)
def test_deferred_provider_models_serialize_in_parent_after_real_worker() -> None:
    # A fresh interpreter is essential: another test constructing Function in
    # the parent would build its deferred serializer and hide this regression.
    script = textwrap.dedent(
        """
        import multiprocessing
        import litellm
        from litellm.types.utils import Function, ModelResponse
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        multiprocessing.set_start_method("fork", force=True)
        assert not Function.__pydantic_complete__, "fixture was already warmed"

        def completion(**params):
            # Only the forked worker constructs these provider models. The
            # parent's deferred serializers remain unbuilt during transport.
            response = ModelResponse(
                model="gpt-4o",
                choices=[{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant", "content": "synthetic text",
                        "tool_calls": [{
                            "id": "synthetic-call", "type": "function",
                            "function": {"name": "search_playbooks",
                                         "arguments": '{"query":"synthetic"}'},
                            "provider_specific_fields": {"signature": "opaque-tool"},
                        }],
                        "reasoning_content": "synthetic reasoning",
                        "thinking_blocks": [{"type": "thinking", "signature": "opaque"}],
                        "reasoning_items": [{"type": "reasoning", "encrypted_content": "opaque"}],
                        "provider_specific_fields": {"continuation": "opaque"},
                    },
                }],
                usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                       "prompt_tokens_details": {"cached_tokens": 3},
                       "completion_tokens_details": {"reasoning_tokens": 2}},
            )
            response._hidden_params = {
                "response_cost": 0.125, "reflexio_provider": "synthetic",
                "reflexio_served_model": "synthetic-served",
                "additional_headers": {"synthetic-header": "preserved"},
            }
            return response

        litellm.completion = completion
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
        client._should_process_isolate_completion = lambda *_: True
        response = client._completion_with_hard_timeout(
            {"model": "gpt-4o", "timeout": 5}, 20
        )
        assert isinstance(response, ModelResponse)
        message = response.choices[0].message
        call = message.tool_calls[0]
        # This is the exact operation that failed in audited reconciliation.
        dumped = call.model_dump()
        assert dumped["id"] == "synthetic-call"
        assert dumped["function"] == {
            "name": "search_playbooks", "arguments": '{"query":"synthetic"}'
        }
        assert dumped["provider_specific_fields"] == {"signature": "opaque-tool"}
        assert response.model_dump()["choices"][0]["message"]["tool_calls"][0] == dumped
        assert message.content == "synthetic text"
        assert message.reasoning_content == "synthetic reasoning"
        assert message.thinking_blocks[0]["signature"] == "opaque"
        assert message.reasoning_items[0]["encrypted_content"] == "opaque"
        assert message.provider_specific_fields == {"continuation": "opaque"}
        assert response.usage.total_tokens == 18
        assert response.usage.prompt_tokens_details.cached_tokens == 3
        assert response.usage.completion_tokens_details.reasoning_tokens == 2
        assert response._hidden_params == {
            "response_cost": 0.125, "reflexio_provider": "synthetic",
            "reflexio_served_model": "synthetic-served",
            "additional_headers": {"synthetic-header": "preserved"},
        }
        print("deferred provider response round-trip passed")
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and synthetic fixture
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "PYTHON_DOTENV_DISABLED": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
        },
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deferred provider response round-trip passed" in result.stdout
