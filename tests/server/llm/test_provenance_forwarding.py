"""Guard the forwarding contract between the two text-generation entry points.

``generate_chat_response`` and ``generate_chat_response_with_provenance`` share a
copy-pasted block that forwards optional keyword arguments into ``_make_request``.
When a new option is added to one and the forwarding line is pasted into the other
without also adding the parameter, the result is a ``NameError`` that only fires at
runtime, on the exact call path that uses it.

That happened in production: ``provider_request_guard`` was added to
``generate_chat_response`` and its forwarding block pasted into
``generate_chat_response_with_provenance``, whose signature never gained the
parameter. Every tool-loop call raised
``NameError: name 'provider_request_guard' is not defined``, which surfaced as
"Playbook extraction failed without structured output".

The signature-parity test below fails on any future instance of that class; the
call test pins the specific regression.
"""

import inspect
from unittest.mock import patch

from reflexio.server.llm._litellm_text_generation import TextGenerationMixin

# Forwarded options both entry points must accept. `generate_chat_response_with_provenance`
# forwards each of these by name, so a missing parameter is a latent NameError.
FORWARDED_OPTIONS = (
    "tools",
    "tool_choice",
    "model_role",
    "max_retries",
    "fallback_models",
    "structured_output_validator",
    "provider_request_guard",
)


def _params(func):
    return set(inspect.signature(func).parameters)


def test_both_entry_points_declare_every_forwarded_option():
    """Neither entry point may forward a name it does not declare."""
    plain = _params(TextGenerationMixin.generate_chat_response)
    provenance = _params(TextGenerationMixin.generate_chat_response_with_provenance)
    for option in FORWARDED_OPTIONS:
        assert option in plain, f"generate_chat_response is missing {option!r}"
        assert option in provenance, (
            f"generate_chat_response_with_provenance forwards {option!r} but does not "
            f"declare it — this is a NameError at runtime, not a type error"
        )


def test_forwarded_names_are_locals_not_globals():
    """Every forwarded name must be a real parameter, not an accidental global lookup."""
    code = TextGenerationMixin.generate_chat_response_with_provenance.__code__
    for option in FORWARDED_OPTIONS:
        assert option in code.co_varnames, (
            f"{option!r} is not a local in generate_chat_response_with_provenance; "
            f"referencing it would raise NameError"
        )


def test_provenance_call_without_a_guard_does_not_raise_nameerror():
    """The exact production call path: no guard supplied, tool loop invokes it."""
    inst = TextGenerationMixin.__new__(TextGenerationMixin)
    with patch.object(TextGenerationMixin, "_make_request", return_value="ok"):
        result = TextGenerationMixin.generate_chat_response_with_provenance(
            inst, [{"role": "user", "content": "hi"}]
        )
    assert result == "ok"


def test_provenance_call_forwards_the_guard_when_supplied():
    inst = TextGenerationMixin.__new__(TextGenerationMixin)

    def guard(_params):
        return None

    with patch.object(
        TextGenerationMixin, "_make_request", return_value="ok"
    ) as make_request:
        TextGenerationMixin.generate_chat_response_with_provenance(
            inst, [{"role": "user", "content": "hi"}], provider_request_guard=guard
        )
    assert make_request.call_args.kwargs["provider_request_guard"] is guard


def test_plain_entry_point_does_not_silently_drop_the_guard():
    """``generate_chat_response`` accepted the guard and never passed it on.

    The parameter was declared but omitted from the delegation, so a caller's guard
    was accepted and silently ignored — no error, no guard, just a request that was
    never inspected.
    """
    inst = TextGenerationMixin.__new__(TextGenerationMixin)

    def guard(_params):
        return None

    with patch.object(
        TextGenerationMixin, "generate_chat_response_with_provenance"
    ) as provenance:
        provenance.return_value.value = "ok"
        TextGenerationMixin.generate_chat_response(
            inst, [{"role": "user", "content": "hi"}], provider_request_guard=guard
        )
    assert provenance.call_args.kwargs["provider_request_guard"] is guard
