"""Source-integrity refusals of the native rule-evidence input and edit replay.

Complements `test_rule_evidence_native.py`: each test perturbs exactly one
binding between the frozen bundle, the incumbent, the source documents, the
trajectory preimages and the durable model outputs, and asserts the method
refuses with the specific reason rather than proceeding on unbound evidence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.offline_tuner.open_world.rule_evidence import (
    native_rule_evidence_input,
    observed_output_bytes,
    validate_evidence_edit,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)


def _input_kwargs(source: dict) -> dict:
    return {
        key: source[key]
        for key in ("bundle", "incumbent", "documents", "trajectory_json_by_session")
    }


def test_the_bundle_eligibility_check_runs_on_a_validated_bundle(source, monkeypatch):
    import reflexio.server.services.offline_tuner.open_world.rule_evidence as mod

    # Isolate the eligibility guard from the bundle's own digest validation.
    monkeypatch.setattr(mod, "validate_open_world_evidence_bundle", lambda _b: None)
    ineligible = replace(source["bundle"], eligible=False)
    with pytest.raises(ValueError, match="needs an eligible source bundle"):
        native_rule_evidence_input(**{**_input_kwargs(source), "bundle": ineligible})


@pytest.mark.parametrize(
    "update",
    [
        {"user_id": ""},
        {"user_playbook_id": 8},
        {"status": Status.ARCHIVED},
        {"superseded_by": 99},
        {"retired_at": 1_700_000_000},
    ],
)
def test_an_incumbent_that_is_not_the_frozen_live_version_is_stale(source, update):
    incumbent = source["incumbent"].model_copy(update=update)
    with pytest.raises(OpenWorldEvidenceTargetStaleError, match="frozen current"):
        native_rule_evidence_input(**{**_input_kwargs(source), "incumbent": incumbent})


def test_documents_must_cover_the_exact_population_without_duplicates(source):
    document = source["documents"][0]
    for documents in ([], [document, document]):
        with pytest.raises(ValueError, match="exact frozen source population"):
            native_rule_evidence_input(
                **{**_input_kwargs(source), "documents": documents}
            )


def test_a_document_for_another_session_is_refused(source):
    document = source["documents"][0]
    stranger = replace(
        document,
        session_id="session-other",
        spans=tuple(
            replace(span, session_id="session-other") for span in document.spans
        ),
    )
    with pytest.raises(ValueError, match="exact frozen source population"):
        native_rule_evidence_input(**{**_input_kwargs(source), "documents": [stranger]})


def test_trajectory_preimages_must_cover_the_exact_population(source):
    preimages = dict(source["trajectory_json_by_session"])
    for changed in ({}, {**preimages, ("user-x", "session-x"): "{}"}):
        with pytest.raises(ValueError, match="trajectory preimages must cover"):
            native_rule_evidence_input(
                **{**_input_kwargs(source), "trajectory_json_by_session": changed}
            )


def test_a_non_string_trajectory_preimage_is_refused(source):
    preimages = dict.fromkeys(source["trajectory_json_by_session"], b"{}")
    with pytest.raises(ValueError, match="preimage digest differs"):
        native_rule_evidence_input(
            **{**_input_kwargs(source), "trajectory_json_by_session": preimages}
        )


def _with_agent_span(source, **changes):
    document = source["documents"][0]
    spans = list(document.spans)
    index = next(i for i, span in enumerate(spans) if span.speaker == "agent")
    spans[index] = replace(spans[index], **changes)
    source["documents"] = [replace(document, spans=tuple(spans))]


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "request-forged"},
        {"interaction_id": 999},
        {"speaker": "user"},
        {"field_path": "requests[9].interactions[9].content"},
    ],
)
def test_a_span_whose_coordinates_differ_from_the_original_is_refused(source, changes):
    _with_agent_span(source, **changes)
    with pytest.raises(ValueError, match="coordinates differ from original"):
        native_rule_evidence_input(**_input_kwargs(source))


@pytest.mark.parametrize(
    "changes", [{"manifest_role": "held_out"}, {"trajectory_digest": "f" * 64}]
)
def test_a_document_bound_to_another_role_or_trajectory_is_refused(source, changes):
    document = source["documents"][0]
    span_changes = {k: v for k, v in changes.items() if k == "manifest_role"}
    source["documents"] = [
        replace(
            document,
            spans=tuple(replace(span, **span_changes) for span in document.spans),
            **changes,
        )
    ]
    with pytest.raises(ValueError, match="must bind the owned frozen trajectory"):
        native_rule_evidence_input(**_input_kwargs(source))


def test_a_population_owned_by_another_user_than_the_incumbent_is_refused(source):
    incumbent = source["incumbent"].model_copy(update={"user_id": "user-2"})
    # Re-fingerprinting would make the incumbent stale first; bypass only the
    # fingerprint so the document-ownership check is what decides.
    import reflexio.server.services.offline_tuner.open_world.rule_evidence as mod

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            mod,
            "user_playbook_full_version_fingerprint",
            lambda _p: source["bundle"].target.incumbent_full_version_fingerprint,
        )
        with pytest.raises(ValueError, match="must bind the owned frozen trajectory"):
            native_rule_evidence_input(
                **{**_input_kwargs(source), "incumbent": incumbent}
            )


def test_a_document_without_any_agent_span_is_refused(source):
    document = source["documents"][0]
    source["documents"] = [
        replace(
            document,
            spans=tuple(span for span in document.spans if span.speaker != "agent"),
        )
    ]
    with pytest.raises(ValueError, match="must bind the owned frozen trajectory"):
        native_rule_evidence_input(**_input_kwargs(source))


def test_a_duplicated_span_identity_is_refused(source):
    document = source["documents"][0]
    agent = next(span for span in document.spans if span.speaker == "agent")
    source["documents"] = [replace(document, spans=(*document.spans, agent))]
    with pytest.raises(ValueError, match="tenant or immutable identity"):
        native_rule_evidence_input(**_input_kwargs(source))


def test_a_span_id_not_derived_from_the_trajectory_is_refused(source):
    _with_agent_span(source, span_id="forged:span")
    with pytest.raises(ValueError, match="tenant or immutable identity"):
        native_rule_evidence_input(**_input_kwargs(source))


def test_observed_output_bytes_accepts_only_the_two_method_contracts(source):
    with pytest.raises(TypeError, match="must be Diagnostic or Proposal"):
        observed_output_bytes({"payload": "not a contract"})  # type: ignore[arg-type]
    diagnostic = observed_output_bytes(source["diagnostics"][0])
    proposal = observed_output_bytes(source["proposal"])
    assert b'"contract_type":"rule_evidence_diagnostic"' in diagnostic
    assert b'"contract_type":"rule_evidence_proposal"' in proposal


def test_durable_outputs_in_the_wrong_schema_order_are_refused(source):
    # The method asks for a Diagnostic first; a Proposal in that slot means the
    # durable observations were not produced by this method's call sequence.
    with pytest.raises(ValueError, match="schema/order does not match"):
        validate_evidence_edit(
            **{
                **source,
                "diagnostics": [source["proposal"]],
                "proposal": source["diagnostics"][0],
            }
        )


def test_an_extra_durable_output_the_method_never_requested_is_refused(
    source, monkeypatch
):
    import reflexio.server.services.offline_tuner.open_world.rule_evidence as mod

    def diagnostics_only(data, call):
        # A method run that requests only the per-session diagnostics: the
        # durable proposal is then an output with no corresponding call.
        for session in data.sessions:
            call("system", {"session_id": session.session_id}, Diagnostic)
        return

    monkeypatch.setattr(mod, "run_rule_evidence", diagnostics_only)
    with pytest.raises(ValueError, match="include a call the method did not request"):
        validate_evidence_edit(**source)
