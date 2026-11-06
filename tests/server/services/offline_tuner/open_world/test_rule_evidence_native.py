"""Frozen native source and exact-edit policy checks, without model/storage I/O."""

from dataclasses import replace

import pytest

from reflexio.server.services.offline_tuner.open_world.rule_evidence import (
    diagnostic_population_digest,
    native_rule_evidence_input,
    validate_evidence_edit,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)
from reflexio.test_support.open_world_rule_evidence import source as source


def test_exact_edit_uses_owned_unlabeled_source_and_inherits_native_citations(source):
    validation = validate_evidence_edit(**source)
    assert validation.policy_id == "evidence-grounded-exact-edit-v1"
    assert validation.result.revised_content == (
        "Send once. Report queued as queued. Keep the recipient."
    )
    assert validation.verified_citations
    assert all(
        citation.user_id == "user-1" for citation in validation.verified_citations
    )
    data = native_rule_evidence_input(
        bundle=source["bundle"],
        incumbent=source["incumbent"],
        documents=source["documents"],
        trajectory_json_by_session=source["trajectory_json_by_session"],
    )
    assert data.tool_context_status == "missing"
    assert not data.available_tools
    assert data.agent_context == ""


def test_exact_edit_refuses_another_incumbent_version(source):
    source["incumbent"] = source["incumbent"].model_copy(update={"content": "Changed"})
    with pytest.raises(
        OpenWorldEvidenceTargetStaleError, match="frozen current version"
    ):
        validate_evidence_edit(**source)


def test_exact_edit_refuses_another_tenant_source(source):
    document = source["documents"][0]
    source["documents"] = [
        replace(
            document,
            spans=tuple(replace(span, tenant_id="other") for span in document.spans),
        )
    ]
    with pytest.raises(ValueError, match="tenant"):
        validate_evidence_edit(**source)


def test_exact_edit_requires_all_session_diagnostics(source):
    source["diagnostics"] = []
    with pytest.raises(ValueError, match="every frozen input session"):
        validate_evidence_edit(**source)


def test_exact_edit_refuses_invented_rule_quote(source):
    diagnostic = source["diagnostics"][0]
    source["diagnostics"] = [
        diagnostic.model_copy(
            update={
                "rule_assessment": diagnostic.rule_assessment.model_copy(
                    update={"clause_start": "Invented clause"}
                )
            }
        )
    ]
    with pytest.raises(ValueError, match="literal_error|available ordered clause"):
        validate_evidence_edit(**source)


def test_no_gap_still_requires_a_durable_tuner_audit(source):
    diagnostic = source["diagnostics"][0]
    source["diagnostics"] = [
        diagnostic.model_copy(
            update={
                "rule_assessment": diagnostic.rule_assessment.model_copy(
                    update={"attribution": "adequate_guidance_ignored"}
                )
            }
        )
    ]
    with pytest.raises(ValueError, match="required durable"):
        validate_evidence_edit(**{**source, "proposal": None})
    source["proposal"] = Proposal(
        hypothesis="", edits=[], abstention_reason="No grounded defect"
    )
    validation = validate_evidence_edit(**source)
    assert (
        validation.result.decision == "abstain"
        and validation.result.revised_content is None
    )


def test_diagnostic_population_binds_exact_observation_and_unique_membership(source):
    diagnostics = source["diagnostics"]
    assert diagnostic_population_digest(diagnostics) != diagnostic_population_digest(
        [diagnostics[0].model_copy(update={"missing_information": ["missing receipt"]})]
    )
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic_population_digest([*diagnostics, *diagnostics])


def _replace_source_span(source, text, *, field_path=None):
    from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
        _span_id,
    )

    document = source["documents"][0]
    index = next(i for i, span in enumerate(document.spans) if span.speaker == "agent")
    original = document.spans[index]
    changed = replace(
        original, immutable_text=text, field_path=field_path or original.field_path
    )
    changed = replace(
        changed,
        span_id=_span_id(
            trajectory_digest=document.trajectory_digest,
            tenant_id=changed.tenant_id,
            role=changed.manifest_role,
            user_id=changed.user_id,
            session_id=changed.session_id,
            request_id=changed.request_id,
            interaction_id=changed.interaction_id,
            field_path=changed.field_path,
            immutable_text=changed.immutable_text,
            speaker=changed.speaker,
        ),
    )
    spans = list(document.spans)
    spans[index] = changed
    source["documents"] = [replace(document, spans=tuple(spans))]
    return original


def test_self_consistent_forged_span_cannot_authorize_invented_evidence(source):
    _replace_source_span(source, "An invented statement never present in the source.")
    with pytest.raises(ValueError, match="text differs from original trajectory"):
        validate_evidence_edit(**source)


def test_changed_trajectory_preimage_cannot_match_frozen_digest(source):
    source["trajectory_json_by_session"] = {
        key: value + " " for key, value in source["trajectory_json_by_session"].items()
    }
    with pytest.raises(ValueError, match="preimage digest"):
        validate_evidence_edit(**source)


def test_original_substring_compaction_retains_authenticated_citations(source):
    document = source["documents"][0]
    original = next(span for span in document.spans if span.speaker == "agent")
    text = original.immutable_text
    start, end = 1, len(text) - 1
    _replace_source_span(
        source, text[start:end], field_path=f"{original.field_path}[{start}:{end}]"
    )
    validation = validate_evidence_edit(**source)
    assert validation.verified_citations
    assert any(
        citation.exact_quote == text[start:end]
        for citation in validation.verified_citations
    )


def test_compacted_span_offsets_must_match_its_original_substring(source):
    document = source["documents"][0]
    original = next(span for span in document.spans if span.speaker == "agent")
    _replace_source_span(
        source,
        original.immutable_text[1:],
        field_path=f"{original.field_path}[0:{len(original.immutable_text) - 1}]",
    )
    with pytest.raises(ValueError, match="text differs from original trajectory"):
        validate_evidence_edit(**source)


def test_native_qualification_rejects_the_provider_enum_parsing_regression(monkeypatch):
    from pydantic import ValidationError

    from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
        _structural_qualification,
    )
    from reflexio.server.services.playbook.rule_evidence import wire

    assert _structural_qualification()["passed"] is True
    # Remove numeric adaptation to reproduce the deployed strict-enum defect.
    # Qualification must exercise decoded JSON against the input-bound schema.
    monkeypatch.setattr(wire, "BeforeValidator", lambda _validator: None)
    with pytest.raises(ValidationError):
        _structural_qualification()
