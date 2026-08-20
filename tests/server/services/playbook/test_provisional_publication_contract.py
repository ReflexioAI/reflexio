from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from inspect import signature

import pytest

from reflexio.models.api_schema.domain import UserPlaybook
from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    ProvisionalPublicationRequest,
    ProvisionalPublicationResult,
    PublicationClaim,
    PublicationSearchProjection,
    QualificationAuthorityRef,
    UserPlaybookPublicationStore,
)


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _incumbent_snapshot(*, trigger: str | None = "refund") -> str:
    incumbent = UserPlaybook(
        user_playbook_id=101,
        user_id="user-1",
        agent_version="agent-v1",
        request_id="request-1",
        playbook_name="Refund handling",
        created_at=123,
        content="old content",
        trigger=trigger,
        rationale="Original rationale",
        source="manual",
        source_interaction_ids=[1, 2],
        source_span="support",
        notes="Keep concise",
        reader_angle="operator",
        tags=["billing"],
        governance_subject_ref="subject-1",
    )
    payload = {
        "schema_version": "user-playbook-full-version-v1",
        "user_playbook": incumbent.model_dump(mode="json", exclude={"embedding"})
        | {
            "governance_subject_ref": incumbent.governance_subject_ref,
            "retired_at": incumbent.retired_at,
        },
    }
    return _canonical(payload)


def _authority() -> QualificationAuthorityRef:
    return QualificationAuthorityRef(
        epoch=7,
        authority_digest="a" * 64,
        discovery_component_identity_digest="b" * 64,
        discovery_qualification_suite_digest="c" * 64,
        discovery_qualification_result_digest="d" * 64,
        held_out_component_identity_digest="e" * 64,
        held_out_qualification_suite_digest="f" * 64,
        held_out_qualification_result_digest="0" * 64,
        candidate_generator_identity_digest="1" * 64,
        candidate_generator_authorization_digest="2" * 64,
    )


def _projection() -> PublicationSearchProjection:
    canonical = _canonical(
        {
            "candidate_content_digest": _digest("new content"),
            "embedding": ["0.125", "0.5"],
            "embedding_model_id": "test-embedding-v1",
            "expanded_terms": ["refund", "escalation"],
            "lexical_document": "refund escalation exact projection",
            "preserved_trigger": "refund",
            "projector_code_digest": "3" * 64,
            "projector_id": "reflexio.search.user-playbook",
            "projector_version": "1",
            "schema_version": "offline-tuner-candidate-search-projection-v1",
        }
    )
    return PublicationSearchProjection(
        schema_version="offline-tuner-candidate-search-projection-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        projector_id="reflexio.search.user-playbook",
        projector_version="1",
        projector_code_digest="3" * 64,
        candidate_content_digest=_digest("new content"),
        preserved_trigger="refund",
        embedding_model_id="test-embedding-v1",
        embedding=("0.125", "0.5"),
        expanded_terms=("refund", "escalation"),
        lexical_document="refund escalation exact projection",
    )


def _proof() -> DecisionProofEnvelope:
    canonical = _canonical(
        {
            "decision": "apply",
            "optimizer_kind": "offline_tuner_open_world",
            "schema_version": "offline-tuner-open-world-publication-proof-v1",
        }
    )
    return DecisionProofEnvelope(
        optimizer_kind="offline_tuner_open_world",
        schema_version="offline-tuner-open-world-publication-proof-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        decision="apply",
    )


def _request(**changes: object) -> ProvisionalPublicationRequest:
    snapshot = _incumbent_snapshot()
    values: dict[str, object] = {
        "optimizer_kind": "offline_tuner_open_world",
        "job_id": 7,
        "attempt_key": "attempt-7",
        "publication_claim": PublicationClaim(job_id=7, owner="worker-a", fence=3),
        "worker_fence": 11,
        "incumbent_user_playbook_id": 101,
        "incumbent_full_version_fingerprint": _digest(snapshot),
        "incumbent_snapshot_json": snapshot,
        "revised_content": "new content",
        "projection": _projection(),
        "decision_proof": _proof(),
        "subject_epochs_json": _canonical(
            {"subjects": [{"epoch": 0, "ref": "subject:1"}]}
        ),
        "qualification_authority": _authority(),
        "evidence_bundle_digest": "4" * 64,
        "candidate_digest": "5" * 64,
    }
    values.update(changes)
    return ProvisionalPublicationRequest(**values)  # type: ignore[arg-type]


def test_provisional_publication_contract_accepts_exact_content_only_bindings() -> None:
    request = _request()

    assert request.optimizer_kind == "offline_tuner_open_world"
    assert request.incumbent_full_version_fingerprint == _digest(
        request.incumbent_snapshot_json
    )
    assert request.qualification_authority.epoch == 7


def test_provisional_publication_result_accepts_complete_terminal_shapes() -> None:
    assert (
        ProvisionalPublicationResult(
            job_id=7,
            outcome="provisional",
            successor_user_playbook_id=10,
            deployment_lifecycle_id=9,
            observation_deadline=1_209_600,
        ).successor_user_playbook_id
        == 10
    )
    assert (
        ProvisionalPublicationResult(
            job_id=7,
            outcome="incumbent_changed",
            successor_user_playbook_id=None,
            deployment_lifecycle_id=None,
            observation_deadline=None,
        ).outcome
        == "incumbent_changed"
    )


@pytest.mark.parametrize(
    "field",
    [
        "authority_digest",
        "discovery_component_identity_digest",
        "discovery_qualification_suite_digest",
        "discovery_qualification_result_digest",
        "held_out_component_identity_digest",
        "held_out_qualification_suite_digest",
        "held_out_qualification_result_digest",
        "candidate_generator_identity_digest",
        "candidate_generator_authorization_digest",
    ],
)
def test_qualification_authority_requires_exact_digest_references(field: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_authority(), **{field: "invalid"})


@pytest.mark.parametrize("optimizer_kind", ["gepa", "offline_tuner_replay"])
def test_provisional_publication_rejects_non_open_world_optimizers(
    optimizer_kind: str,
) -> None:
    with pytest.raises(ValueError, match="offline_tuner_open_world"):
        _request(optimizer_kind=optimizer_kind)


def test_provisional_publication_rejects_claim_projection_and_authority_drift() -> None:
    with pytest.raises(ValueError, match="claim job_id"):
        _request(
            publication_claim=PublicationClaim(job_id=8, owner="worker-a", fence=3)
        )

    with pytest.raises(ValueError, match="revised content digest"):
        _request(revised_content="different content")

    with pytest.raises(ValueError, match="qualification authority"):
        _request(qualification_authority=replace(_authority(), epoch=0))


def test_provisional_publication_rejects_full_snapshot_fingerprint_and_field_drift() -> (
    None
):
    snapshot = _incumbent_snapshot()
    with pytest.raises(ValueError, match="full version fingerprint"):
        _request(incumbent_full_version_fingerprint="6" * 64)

    with pytest.raises(ValueError, match="incumbent_user_playbook_id"):
        _request(incumbent_user_playbook_id=102)

    changed_trigger_snapshot = _incumbent_snapshot(trigger="billing")
    with pytest.raises(ValueError, match="preserve incumbent trigger"):
        _request(
            incumbent_snapshot_json=changed_trigger_snapshot,
            incumbent_full_version_fingerprint=_digest(changed_trigger_snapshot),
        )

    changed_snapshot = snapshot.replace("old content", "drifted content")
    with pytest.raises(ValueError, match="full version fingerprint"):
        _request(incumbent_snapshot_json=changed_snapshot)


@pytest.mark.parametrize(
    ("outcome", "successor_id", "lifecycle_id", "deadline"),
    [
        ("provisional", None, 9, 1_209_600),
        ("provisional", 10, None, 1_209_600),
        ("provisional", 10, 9, None),
        ("incumbent_changed", 10, None, None),
        ("incumbent_changed", None, 9, None),
        ("incumbent_changed", None, None, 1_209_600),
    ],
)
def test_provisional_publication_result_rejects_incomplete_terminal_shapes(
    outcome: str,
    successor_id: int | None,
    lifecycle_id: int | None,
    deadline: int | None,
) -> None:
    with pytest.raises(ValueError):
        ProvisionalPublicationResult(
            job_id=7,
            outcome=outcome,  # type: ignore[arg-type]
            successor_user_playbook_id=successor_id,
            deployment_lifecycle_id=lifecycle_id,
            observation_deadline=deadline,
        )


def test_provisional_publication_store_exposes_distinct_durable_operations() -> None:
    assert tuple(
        signature(
            UserPlaybookPublicationStore.claim_user_playbook_provisional_publication
        ).parameters
    ) == ("self", "job_id", "owner", "worker_fence")
    assert callable(
        UserPlaybookPublicationStore.claim_user_playbook_provisional_publication
    )
    assert callable(
        UserPlaybookPublicationStore.stage_user_playbook_provisional_publication
    )
    assert callable(
        UserPlaybookPublicationStore.commit_user_playbook_provisional_publication
    )
    assert callable(
        UserPlaybookPublicationStore.load_user_playbook_provisional_publication_result
    )
