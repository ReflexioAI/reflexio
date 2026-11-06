"""Opt-in live check of the reasoning-exhausted resend's request shape.

ONE real MiniMax-M3 call with thinking disabled, as the client's resend sends
it: a structured answer must come back non-empty with no reasoning tokens.
Set RUN_LOW_PRIORITY=1 and provide MINIMAX_API_KEY; never part of an offline gate.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from reflexio.server.llm._model_compat import reasoning_exhausted_retry_extra_body
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.test_support.llm_credentials import real_provider_key
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from reflexio.test_support.skip_decorators import skip_low_priority

pytestmark = [pytest.mark.e2e, pytest.mark.requires_credentials]

MODEL = "minimax/MiniMax-M3"


class Capital(BaseModel):
    model_config = ConfigDict(extra="forbid")
    country: str
    capital: str


@skip_low_priority
def test_m3_answers_directly_with_thinking_disabled(monkeypatch):
    if not real_provider_key("MINIMAX_API_KEY"):
        pytest.skip("MINIMAX_API_KEY is not populated with a real key")
    assert_litellm_unpatched()
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    extra_body = reasoning_exhausted_retry_extra_body({"model": MODEL})
    assert extra_body == {"thinking": {"type": "disabled"}}
    client = LiteLLMClient(
        LiteLLMConfig(model=MODEL, max_tokens=1024, timeout=60, fallback_models=[])
    )

    result = client.generate_chat_response_with_provenance(
        [{"role": "user", "content": "What is the capital of France?"}],
        response_format=Capital,
        extra_body=extra_body,
        max_retries=1,
        fallback_models=[],
    )

    assert isinstance(result.value, Capital)
    assert result.value.capital.strip().lower() == "paris"
    details = getattr(result.usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    print(
        f"live_check model={MODEL} thinking=disabled "
        f"completion_tokens={getattr(result.usage, 'completion_tokens', None)} "
        f"reasoning_tokens={reasoning_tokens}"
    )
    assert not reasoning_tokens
