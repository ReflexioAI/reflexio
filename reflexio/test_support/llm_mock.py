"""Global LiteLLM mock for reflexio test suites.

Patches ``litellm.completion`` with a deterministic mock that returns
JSON responses based on prompt content heuristics.  E2E tests are
excluded so they can make real API calls.

Response payloads are sourced from the model registry
(``llm_model_registry.py``) so they always validate against the Pydantic
models that services expect.

Usage in conftest.py::

    from reflexio.test_support.llm_mock import configure_llm_mock, cleanup_llm_mock

    def pytest_configure(config):
        configure_llm_mock(config)

    def pytest_unconfigure(config):
        cleanup_llm_mock(config)
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

from reflexio.test_support.llm_model_registry import get_model_registry


def _create_mock_completion(
    prompt_content: str, parse_structured_output: bool = False
) -> MagicMock:
    """Create a mock LiteLLM completion response.

    Routes on prompt content heuristics; payloads come from the model
    registry to guarantee schema validity.
    """
    registry = get_model_registry()

    if "Output just a boolean value" in prompt_content:
        content = str(registry["boolean_evaluation"].minimal_valid)
    elif "policy consolidation" in prompt_content:
        content = json.dumps(registry["playbook_aggregation"].minimal_valid)
    elif '"playbooks"' in prompt_content:
        # Anchor on the schema marker every playbook-extraction prompt MUST
        # describe (the JSON output key the LLM is told to return). This is
        # intrinsic to the schema, not stylistic, so it survives prompt
        # rewrites better than matching on a tagline like "policy mining".
        content = json.dumps(registry["playbook_extraction"].minimal_valid)
    elif parse_structured_output:
        content = json.dumps(registry["profile_extraction"].minimal_valid)
    else:
        default_payload = json.dumps(registry["profile_update"].minimal_valid)
        content = f"```json\n{default_payload}\n```"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    return mock_response


def _mock_completion(*args: Any, **kwargs: Any) -> MagicMock:
    """Mock implementation for litellm.completion."""
    messages = kwargs.get("messages", args[0] if args else [])
    prompt_content = ""
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            prompt_content += str(message["content"])

    parse_structured = kwargs.get("response_format") is not None
    return _create_mock_completion(prompt_content, parse_structured)


def _is_e2e_test_run(config: Any) -> bool:
    """Check if this pytest run includes e2e tests.

    Returns True if any of the test paths contain 'e2e_tests'.
    """
    args = config.args if hasattr(config, "args") else []
    for arg in args:
        if "e2e_tests" in str(arg):
            return True

    if hasattr(config, "workerinput"):
        worker_args = config.workerinput.get("args", [])
        for arg in worker_args:
            if "e2e_tests" in str(arg):
                return True

    return False


# Global patcher reference kept alive across the test session.
_litellm_patcher = None


def configure_llm_mock(config: Any) -> None:
    """Call from ``pytest_configure`` to patch litellm for non-e2e tests."""
    global _litellm_patcher  # noqa: PLW0603

    if _is_e2e_test_run(config):
        return

    os.environ["MOCK_LLM_RESPONSE"] = "true"
    _litellm_patcher = patch("litellm.completion", side_effect=_mock_completion)
    _litellm_patcher.start()


def cleanup_llm_mock(config: Any) -> None:  # noqa: ARG001
    """Call from ``pytest_unconfigure`` to stop the patcher."""
    global _litellm_patcher  # noqa: PLW0603

    if _litellm_patcher:
        _litellm_patcher.stop()
        _litellm_patcher = None


def make_tool_call_response(tool_name: str, args: dict[str, Any]) -> MagicMock:
    """Build a litellm ModelResponse-shaped mock with a single tool_call.

    Used by unit tests that drive tool loops against the patched
    ``litellm.completion``. Not routed automatically by prompt
    heuristics — callers install it explicitly with ``side_effect``.

    Args:
        tool_name (str): The name the assistant is calling.
        args (dict[str, Any]): JSON-serialisable arguments passed to the tool.

    Returns:
        MagicMock: A response object shaped like a litellm ModelResponse
            whose first choice has ``finish_reason="tool_calls"`` and a
            single tool call matching the given name and args.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = "tool_calls"
    resp.choices[0].message.content = None
    tc = MagicMock()
    tc.id = f"tc_{tool_name}"
    tc.type = "function"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)
    resp.choices[0].message.tool_calls = [tc]
    return resp


def make_finish_response(text: str = "done") -> MagicMock:
    """Build a normal (non-tool-call) assistant message.

    Used to terminate a tool loop that was driven by repeated
    ``make_tool_call_response`` mocks.

    Args:
        text (str): Content of the terminal message.

    Returns:
        MagicMock: A response object with ``finish_reason="stop"``,
            the given text, and ``tool_calls=None``.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].finish_reason = "stop"
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp
