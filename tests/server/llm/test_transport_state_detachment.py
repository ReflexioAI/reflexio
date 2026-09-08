"""The transport must never be handed the caller's own request objects.

WHY THIS FILE EXISTS

litellm's provider transports mutate request state IN PLACE.
``OpenAIGPTConfig.validate_environment`` -- inherited by every route dispatched
through litellm's own HTTP handler, ``minimax/*`` among them -- does::

    headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"

on whatever ``extra_headers`` dict it is given. Reproduced against the
installed litellm: the ``minimax/MiniMax-M3`` route grows a caller's one-key
``{"Idempotency-Key": ...}`` into three keys; the plain ``openai`` SDK routes
leave it alone, which is exactly why the exposure reads as provider-specific
from any one call site and is not.

``_make_request`` treats the caller's kwargs as the STANDING description of the
request -- it rebuilds params for every rung of the ladder and re-reads
``extra_headers`` in ``_corrective_kwargs`` to derive the corrective retry's
idempotency key. So an in-place write on attempt 1 is carried into attempt 2
and into every later rung as though the CALLER had asked for it. In production
that made a strict ``provider_request_guard`` refuse the corrective retry, and
the offline tuner recorded every parse failure as an infrastructure failure.

These tests use a transport that mutates exactly as litellm's does. Tests whose
fake transport is polite cannot observe any of this -- which is how the defect
survived a test that asserts the same contract.
"""

from typing import Any
from unittest.mock import MagicMock

import litellm
import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

_MODEL = "gpt-4o-2024-08-06"


def _response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.refusal = None
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.model = _MODEL
    response._hidden_params = {"custom_llm_provider": "openai"}
    response.usage = None
    return response


def _mutating_transport(seen: list[dict[str, Any]], *, content: str = "ok") -> Any:
    """A transport that scribbles on the request state it is handed.

    Verbatim shape of litellm's ``validate_environment``: it writes into the
    dict it received rather than into a copy, and hands the same object on.
    """

    def serve(**params: Any) -> MagicMock:
        seen.append(params)
        headers = params.get("extra_headers")
        if isinstance(headers, dict):
            headers["Authorization"] = "Bearer sk-not-a-real-key"
            headers.setdefault("Content-Type", "application/json")
        messages = params.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                first["_transport_scribble"] = True
        metadata = params.get("metadata")
        if isinstance(metadata, dict):
            metadata["_transport_scribble"] = True
        return _response(content)

    return serve


def test_a_mutating_transport_cannot_reach_the_callers_request_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing the caller passed in comes back changed.

    This is the whole class, not the one header that was observed to break:
    ``_build_completion_params`` forwards every unrecognised kwarg into
    ``params`` by reference (``params.update(kwargs)``), and ``messages`` is
    forwarded by reference too whenever prompt caching does not rewrite it
    (every non-Anthropic model). Any of them is exposed the moment a provider
    mutates in place, so the copy is taken where params are BUILT.
    """
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(litellm, "completion", _mutating_transport(seen))

    headers = {"Idempotency-Key": "k" * 64}
    messages = [{"role": "user", "content": "hi"}]
    metadata = {"project_name": "reflexio"}

    client.generate_chat_response(
        messages,
        extra_headers=headers,
        metadata=metadata,
    )

    # The transport really did scribble -- otherwise this asserts nothing. All
    # THREE mutations are confirmed to have landed, not just the header one.
    # The ``messages`` and ``metadata`` writes sit behind ``isinstance`` guards
    # in the fake, so if either value ever stops being the shape the fake
    # expects (``messages`` is rewritten wholesale by the Anthropic prompt-cache
    # path, for one), that write silently no-ops and the matching "untouched"
    # assertion below would pass while measuring nothing at all.
    assert len(seen) == 1
    assert sorted(seen[0]["extra_headers"]) == [
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
    ]
    assert seen[0]["messages"][0]["_transport_scribble"] is True
    assert seen[0]["metadata"]["_transport_scribble"] is True

    # ...and it scribbled on copies. The caller's objects are untouched.
    assert headers == {"Idempotency-Key": "k" * 64}
    assert messages == [{"role": "user", "content": "hi"}]
    assert metadata == {"project_name": "reflexio"}


def test_a_later_ladder_rung_does_not_inherit_the_previous_rungs_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rung 2 must send the caller's headers, not rung 1's provider's.

    ``rung_kwargs = {**original_kwargs, ...}`` rebuilds only the TOP level per
    rung, so before the fix every rung of the ladder shared one
    ``extra_headers`` dict and rung 2 sent the credential rung 1's provider had
    injected -- a cross-provider credential leak, not merely a guard failure.
    """
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))
    seen: list[dict[str, Any]] = []
    sent_headers: list[dict[str, Any]] = []
    mutating = _mutating_transport(seen)

    def serve(**params: Any) -> MagicMock:
        sent_headers.append(dict(params.get("extra_headers") or {}))
        response = mutating(**params)
        if len(seen) == 1:  # fail the first rung so the walk advances
            raise litellm.exceptions.APIConnectionError(
                message="rung 1 down", llm_provider="openai", model=_MODEL
            )
        return response

    monkeypatch.setattr(litellm, "completion", serve)

    headers = {"Idempotency-Key": "k" * 64}
    client.generate_chat_response(
        [{"role": "user", "content": "hi"}],
        extra_headers=headers,
        fallback_models=["gpt-4o-mini-2024-07-18"],
    )

    assert len(sent_headers) == 2
    assert sent_headers[0] == {"Idempotency-Key": "k" * 64}
    assert sent_headers[1] == {"Idempotency-Key": "k" * 64}
    assert headers == {"Idempotency-Key": "k" * 64}
