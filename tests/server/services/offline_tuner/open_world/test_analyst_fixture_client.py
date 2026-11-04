"""The sealed qualification fixture client: build, bind, require and complete.

The packaged qualification gate answers the suite from frozen responses through
a client that only its factory can mint. These tests drive that client through
its whole lifecycle and pin each refusal: an unbound call, a second binding, a
call that drifts from the canonical case, an extra call, a forged client, and
fixture outputs that do not cover (or do not parse into) the production schema.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.offline_tuner.open_world import (
    analyst as analyst_module,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    ChatMessage,
    _bind_qualification_fixture_model_client,
    _case_digest,
    _complete_qualification_fixture_case,
    _messages_digest,
    _QualificationCallBinding,
    _request_digest,
    _require_qualification_fixture_model_client,
    compose_litellm_open_world_model_client,
    qualification_fixture_model_client,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    ComponentIdentity,
    DiscoveryMemo,
    HeldOutSessionMemo,
)
from reflexio.server.services.offline_tuner.open_world.qualification import (
    QualificationSuite,
    load_qualification_suite,
)

_Role = Literal["discovery", "held_out"]
_MESSAGES = (
    ChatMessage(role="system", content="policy"),
    ChatMessage(role="user", content="data"),
)
_KEY = "0" * 64


def _gate_identity(role: _Role) -> ComponentIdentity:
    return analyst_module._FROZEN_REFERENCE_IDENTITIES[role]


def _bindings(
    suite: QualificationSuite,
    role: _Role,
    messages: tuple[ChatMessage, ...] = _MESSAGES,
) -> tuple[_QualificationCallBinding, ...]:
    return tuple(
        _QualificationCallBinding(
            case_digest=_case_digest(case),
            request_digest=_request_digest(case.to_production_request(suite.seed)),
            messages_digest=_messages_digest(messages),
        )
        for case in suite.cases_for_role(role)
    )


def _bound_client(role: _Role = "discovery", **build_kwargs: object):
    suite = load_qualification_suite()
    identity = _gate_identity(role)
    client = qualification_fixture_model_client(
        suite=suite,
        role=role,
        identity=identity,
        **build_kwargs,  # type: ignore[arg-type]
    )
    _bind_qualification_fixture_model_client(
        client,
        suite=suite,
        role=role,
        identity=identity,
        bindings=_bindings(suite, role),
    )
    return suite, identity, client


def _complete_case(
    client: object,
    *,
    suite: QualificationSuite,
    role: _Role,
    identity: ComponentIdentity,
    index: int,
    **overrides: object,
):
    case = suite.cases_for_role(role)[index]
    kwargs: dict[str, object] = {
        "case": case,
        "request": case.to_production_request(suite.seed),
        "messages": _MESSAGES,
        "response_format": DiscoveryMemo if role == "discovery" else HeldOutSessionMemo,
        "model_revision": identity.immutable_model_revision,
        "idempotency_key": _KEY,
    }
    kwargs.update(overrides)
    return _complete_qualification_fixture_case(
        client,
        suite=suite,
        role=role,
        identity=identity,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("role", ["discovery", "held_out"])
def test_every_role_case_is_answered_by_its_frozen_response_in_order(
    role: _Role,
) -> None:
    suite, identity, client = _bound_client(role)
    frozen = {
        item.case_id: item.output for item in analyst_module._FROZEN_FIXTURE_RESPONSES
    }
    cases = suite.cases_for_role(role)

    observed = [
        _complete_case(client, suite=suite, role=role, identity=identity, index=index)
        for index in range(len(cases))
    ]

    assert [item.value for item in observed] == [frozen[case.case_id] for case in cases]
    # Observation metadata is the packaged gate's own, never a caller's.
    assert {item.observed_provider for item in observed} == {identity.provider}
    assert {item.observed_model_revision for item in observed} == {
        identity.immutable_model_revision
    }
    assert [item.called_at for item in observed] == [
        101 + index for index in range(len(cases))
    ]
    assert len(client.calls) == len(cases)
    assert all(call.idempotency_key == _KEY for call in client.calls)
    assert all(call.messages == _MESSAGES for call in client.calls)

    with pytest.raises(ValueError, match="extra model call"):
        _complete_case(client, suite=suite, role=role, identity=identity, index=0)


def test_complete_before_binding_is_refused_on_both_entry_points() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")
    client = qualification_fixture_model_client(
        suite=suite, role="discovery", identity=identity
    )

    with pytest.raises(ValueError, match="must be bound before execution"):
        _complete_case(
            client, suite=suite, role="discovery", identity=identity, index=0
        )
    with pytest.raises(ValueError, match="must be bound before execution"):
        client.complete(
            messages=_MESSAGES,
            response_format=DiscoveryMemo,
            model_revision=identity.immutable_model_revision,
            idempotency_key=_KEY,
        )
    assert client.calls == []


def test_bindings_must_cover_the_role_cases_and_bind_only_once() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")
    client = qualification_fixture_model_client(
        suite=suite, role="discovery", identity=identity
    )
    bindings = _bindings(suite, "discovery")

    with pytest.raises(ValueError, match="does not cover role cases"):
        _bind_qualification_fixture_model_client(
            client,
            suite=suite,
            role="discovery",
            identity=identity,
            bindings=bindings[:-1],
        )
    _bind_qualification_fixture_model_client(
        client, suite=suite, role="discovery", identity=identity, bindings=bindings
    )
    with pytest.raises(ValueError, match="already bound"):
        _bind_qualification_fixture_model_client(
            client, suite=suite, role="discovery", identity=identity, bindings=bindings
        )


@pytest.mark.parametrize(
    "overrides",
    [
        # The case is not the one this call slot was bound to.
        pytest.param({"case_index": 1}, id="case-out-of-order"),
        # A request that is not the canonical case's production request.
        pytest.param({"request_index": 1}, id="request-substituted"),
        pytest.param(
            {
                "messages": (
                    ChatMessage(role="system", content="policy"),
                    ChatMessage(role="user", content="different data"),
                )
            },
            id="messages-drifted",
        ),
        pytest.param({"response_format": HeldOutSessionMemo}, id="wrong-schema"),
        pytest.param({"model_revision": "other-model"}, id="wrong-model"),
        pytest.param({"idempotency_key": "short"}, id="short-idempotency-key"),
    ],
)
def test_a_call_that_drifts_from_the_bound_case_is_refused(
    overrides: dict[str, object],
) -> None:
    suite, identity, client = _bound_client("discovery")
    cases = suite.cases_for_role("discovery")
    kwargs = dict(overrides)
    if "case_index" in kwargs:
        kwargs["case"] = cases[int(kwargs.pop("case_index"))]  # type: ignore[arg-type]
    if "request_index" in kwargs:
        kwargs["request"] = cases[
            int(kwargs.pop("request_index"))  # type: ignore[arg-type]
        ].to_production_request(suite.seed)

    with pytest.raises(ValueError, match="does not match the canonical case"):
        _complete_case(
            client, suite=suite, role="discovery", identity=identity, index=0, **kwargs
        )
    # A refused call is not consumed: nothing was recorded.
    assert client.calls == []


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(object(), id="not-a-sealed-client"),
        pytest.param(None, id="none"),
    ],
)
def test_require_refuses_anything_but_the_sealed_client(candidate: object) -> None:
    suite = load_qualification_suite()
    with pytest.raises(ValueError, match="sealed deterministic qualification client"):
        _require_qualification_fixture_model_client(
            candidate,
            suite=suite,
            role="discovery",
            identity=_gate_identity("discovery"),
        )


def test_require_refuses_a_client_sealed_for_another_role_or_identity() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")
    client = qualification_fixture_model_client(
        suite=suite, role="discovery", identity=identity
    )

    assert (
        _require_qualification_fixture_model_client(
            client, suite=suite, role="discovery", identity=identity
        )
        is client
    )
    with pytest.raises(ValueError, match="sealed deterministic qualification client"):
        _require_qualification_fixture_model_client(
            client,
            suite=suite,
            role="held_out",
            identity=_gate_identity("held_out"),
        )
    with pytest.raises(ValueError, match="sealed deterministic qualification client"):
        _require_qualification_fixture_model_client(
            client,
            suite=suite,
            role="discovery",
            identity=replace(identity, prompt_template_digest="e" * 64),
        )


def test_the_sealed_client_cannot_be_constructed_outside_its_factory() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")
    # Any: the call below deliberately bypasses the factory's typed surface.
    sealed_type: Any = type(
        qualification_fixture_model_client(
            suite=suite, role="discovery", identity=identity
        )
    )

    with pytest.raises(TypeError, match="must be created by its factory"):
        sealed_type(
            suite=suite,
            role="discovery",
            identity=identity,
            outputs=(),
            allow_malformed_raw_output=False,
            observed_provider=identity.provider,
            observed_model_revision=identity.immutable_model_revision,
            subject_transport=None,
            token=object(),
        )


def test_fixture_outputs_must_exactly_cover_the_role_cases() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")
    frozen = tuple(
        item.output
        for item in analyst_module._FROZEN_FIXTURE_RESPONSES
        if item.role == "discovery"
    )

    with pytest.raises(ValueError, match="exactly cover the role cases"):
        qualification_fixture_model_client(
            suite=suite, role="discovery", identity=identity, outputs=frozen[:-1]
        )
    held_out_output = next(
        item.output
        for item in analyst_module._FROZEN_FIXTURE_RESPONSES
        if item.role == "held_out"
    )
    with pytest.raises(ValueError, match="exactly cover the role cases"):
        qualification_fixture_model_client(
            suite=suite,
            role="discovery",
            identity=identity,
            outputs=(*frozen[:-1], held_out_output),
        )
    # A raw mapping is admitted ONLY on the explicit malformed-output path.
    raw = frozen[0].model_dump(mode="json")
    with pytest.raises(ValueError, match="exactly cover the role cases"):
        qualification_fixture_model_client(
            suite=suite,
            role="discovery",
            identity=identity,
            outputs=(raw, *frozen[1:]),
        )


def test_a_valid_raw_output_parses_through_the_production_schema() -> None:
    frozen = tuple(
        item.output
        for item in analyst_module._FROZEN_FIXTURE_RESPONSES
        if item.role == "discovery"
    )
    raw = frozen[0].model_dump(mode="python")
    suite, identity, client = _bound_client(
        "discovery",
        outputs=(raw, *frozen[1:]),
        _allow_malformed_raw_output=True,
    )

    observed = _complete_case(
        client, suite=suite, role="discovery", identity=identity, index=0
    )

    assert type(observed.value) is DiscoveryMemo
    assert observed.value == frozen[0]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda raw: {**raw, "invented_field": "x"}, id="extra-key"),
        pytest.param(lambda raw: {**raw, "decision": "maybe"}, id="bad-literal"),
        pytest.param(
            lambda raw: {k: v for k, v in raw.items() if k != "scope"},
            id="missing-field",
        ),
    ],
)
def test_a_malformed_raw_output_is_rejected_not_coerced(mutate: object) -> None:
    frozen = tuple(
        item.output
        for item in analyst_module._FROZEN_FIXTURE_RESPONSES
        if item.role == "discovery"
    )
    raw = mutate(frozen[0].model_dump(mode="python"))  # type: ignore[operator]
    suite, identity, client = _bound_client(
        "discovery",
        outputs=(raw, *frozen[1:]),
        _allow_malformed_raw_output=True,
    )

    with pytest.raises(
        analyst_module._MalformedQualificationFixtureOutputError,
        match="did not match production schema",
    ):
        _complete_case(
            client, suite=suite, role="discovery", identity=identity, index=0
        )


def test_observed_provenance_override_is_reported_verbatim() -> None:
    """The override exists so a test can make the gate LIE about provenance;
    whatever it says must reach the completion unchanged, so the downstream
    echo check is what refuses it."""
    suite, identity, client = _bound_client(
        "discovery",
        _observed_provider="provider-substituted",
        _observed_model_revision="model-substituted",
    )

    observed = _complete_case(
        client, suite=suite, role="discovery", identity=identity, index=0
    )

    assert observed.observed_provider == "provider-substituted"
    assert observed.observed_model_revision == "model-substituted"


def test_build_requires_the_exact_frozen_suite() -> None:
    suite = load_qualification_suite()
    identity = _gate_identity("discovery")

    with pytest.raises(TypeError, match="exact qualification contract"):
        qualification_fixture_model_client(
            suite=suite.model_dump(),  # type: ignore[arg-type]
            role="discovery",
            identity=identity,
        )
    reordered = suite.model_copy(update={"cases": tuple(reversed(suite.cases))})
    with pytest.raises(ValueError, match="exact canonical qualification suite"):
        qualification_fixture_model_client(
            suite=reordered, role="discovery", identity=identity
        )


def test_build_requires_a_component_identity_subject() -> None:
    with pytest.raises(TypeError, match="must be a ComponentIdentity"):
        qualification_fixture_model_client(
            suite=load_qualification_suite(),
            role="discovery",
            identity=analyst_module._expected_identity_payload(  # type: ignore[arg-type]
                _gate_identity("discovery")
            ),
        )


def test_a_composed_transport_attests_a_matching_production_identity() -> None:
    """The success arm of subject admission: provider and model come from the
    composed transport, every other field from the packaged gate contract."""
    model = "gpt-4o-2024-08-06"
    transport = compose_litellm_open_world_model_client(
        LiteLLMClient(LiteLLMConfig(model=model))
    )
    identity = replace(
        _gate_identity("discovery"),
        provider=transport.transport_identity.provider,
        immutable_model_revision=model,
    )
    suite = load_qualification_suite()

    client = qualification_fixture_model_client(
        suite=suite,
        role="discovery",
        identity=identity,
        subject_transport=transport,
    )

    assert (
        _require_qualification_fixture_model_client(
            client, suite=suite, role="discovery", identity=identity
        )
        is client
    )
