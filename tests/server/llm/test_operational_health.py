"""Logical LLM measurements do not turn cancellations into provider outages."""

from unittest.mock import Mock

import pytest

from reflexio.server.llm._litellm_text_generation import (
    ProviderRequestGuardError,
    TextGenerationMixin,
)
from reflexio.server.llm._litellm_types import LiteLLMClientError


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (LiteLLMClientError("exhausted"), "failure"),
        (ProviderRequestGuardError("lease lost"), "cancelled"),
        (ValueError("invalid options"), "aborted"),
    ],
)
def test_logical_outcomes_preserve_errors(monkeypatch, error, outcome):
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    client = TextGenerationMixin()
    monkeypatch.setattr(client, "_make_request_inner", Mock(side_effect=error))
    with pytest.raises(type(error)) as raised:
        client._make_request([])
    assert raised.value is error
    logical = [
        call for call in sink.record.call_args_list if call.args[0] == "llm.requests"
    ]
    assert len(logical) == 1
    assert logical[0].kwargs["attributes"] == {"outcome": outcome}
    failures = [
        call for call in sink.record.call_args_list if call.args[0] == "llm.failures"
    ]
    assert len(failures) == int(outcome == "failure")
