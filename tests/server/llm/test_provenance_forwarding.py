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

import pytest

from reflexio.server.llm._litellm_text_generation import TextGenerationMixin

PLAIN = TextGenerationMixin.generate_chat_response
PROVENANCE = TextGenerationMixin.generate_chat_response_with_provenance

# Structural, not hand-maintained: the options are DERIVED from the signatures, so a
# future option added to one entry point and forgotten in the other fails here without
# anyone remembering to update this file. A hardcoded list would need the same manual
# sync step that produced the original bug.
_PLUMBING = {"self", "messages", "system_message", "kwargs"}


def _options(func):
    return set(inspect.signature(func).parameters) - _PLUMBING


def test_both_entry_points_declare_the_same_options():
    """The two entry points must stay in lockstep.

    `generate_chat_response` delegates to `generate_chat_response_with_provenance`
    and forwards each option by name, so any asymmetry is either a NameError (declared
    on neither side but forwarded) or a silently dropped argument.
    """
    plain, provenance = _options(PLAIN), _options(PROVENANCE)
    assert plain == provenance, (
        "signatures drifted — "
        f"only in generate_chat_response: {sorted(plain - provenance)}; "
        f"only in generate_chat_response_with_provenance: {sorted(provenance - plain)}"
    )


@pytest.mark.parametrize("option", sorted(_options(PLAIN)))
def test_plain_entry_point_forwards_every_declared_option(option):
    """Declaring an option is not enough — the delegation must actually pass it on.

    Signature parity alone would still pass if both entry points declared an option
    and one dropped it in the call, which is exactly what `generate_chat_response`
    did with `provider_request_guard`.
    """
    inst = TextGenerationMixin.__new__(TextGenerationMixin)
    sentinel = object()
    with patch.object(TextGenerationMixin, "generate_chat_response_with_provenance") as p:
        p.return_value.value = "ok"
        PLAIN(inst, [{"role": "user", "content": "hi"}], **{option: sentinel})
    assert p.call_args.kwargs.get(option) is sentinel, (
        f"generate_chat_response declares {option!r} but does not forward it — "
        f"a caller's value is accepted and silently discarded"
    )


@pytest.mark.parametrize("option", sorted(_options(PROVENANCE)))
def test_provenance_entry_point_forwards_every_declared_option(option):
    """Each declared option must reach `_make_request`, exercising the real line.

    This runs the forwarding block itself, so a name referenced there but missing
    from the signature raises NameError here rather than in production.
    """
    inst = TextGenerationMixin.__new__(TextGenerationMixin)
    sentinel = object()
    with patch.object(TextGenerationMixin, "_make_request", return_value="ok") as m:
        PROVENANCE(inst, [{"role": "user", "content": "hi"}], **{option: sentinel})
    assert m.call_args.kwargs.get(option) is sentinel, (
        f"generate_chat_response_with_provenance declares {option!r} but never "
        f"forwards it into _make_request"
    )


def test_provider_request_guard_is_still_wired():
    """Pin the specific regression the derived checks above generalise."""
    assert "provider_request_guard" in _options(PLAIN)
    assert "provider_request_guard" in _options(PROVENANCE)


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
