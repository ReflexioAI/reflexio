from __future__ import annotations

import json
from hashlib import sha256

import pytest

from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    PublicationClaim,
    PublicationRequest,
    PublicationSearchProjection,
    UserPlaybookPublicationService,
    canonical_json_bytes,
    incumbent_user_playbook_semantic_digest,
)


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _projection() -> PublicationSearchProjection:
    canonical = _canonical(
        {
            "candidate_content_digest": _digest("new content"),
            "embedding": ["0.125", "0.5"],
            "embedding_model_id": "test-embedding-v1",
            "expanded_terms": ["refund", "escalation"],
            "lexical_document": "refund escalation exact projection",
            "preserved_trigger": "refund",
            "projector_code_digest": "a" * 64,
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
        projector_code_digest="a" * 64,
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
            "optimizer_kind": "gepa",
            "schema_version": "gepa-publication-proof-v1",
            "source": "playbook_optimizer",
        }
    )
    return DecisionProofEnvelope(
        optimizer_kind="gepa",
        schema_version="gepa-publication-proof-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        decision="apply",
    )


def test_publication_models_accept_strict_canonical_payloads() -> None:
    request = PublicationRequest(
        optimizer_kind="gepa",
        job_id=7,
        attempt_key="attempt-7",
        publication_claim=PublicationClaim(job_id=7, owner="worker-a", fence=3),
        worker_fence=11,
        incumbent_user_playbook_id=101,
        incumbent_content_digest=_digest("old content"),
        incumbent_trigger="refund",
        incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
            content_digest=_digest("old content"), trigger="refund"
        ),
        revised_content="new content",
        projection=_projection(),
        decision_proof=_proof(),
        subject_epochs_json=_canonical({"subjects": [{"ref": "user:u1", "epoch": 0}]}),
        request_id="request-7",
    )

    assert request.projection.digest == _digest(request.projection.canonical_json)
    assert request.decision_proof.digest == _digest(
        request.decision_proof.canonical_json
    )


@pytest.mark.parametrize(
    "bad_digest",
    [
        "A" * 64,
        "0" * 63,
        "g" * 64,
    ],
)
def test_publication_digest_validation_rejects_non_lowercase_sha256(
    bad_digest: str,
) -> None:
    canonical = _canonical({"decision": "apply"})

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DecisionProofEnvelope(
            optimizer_kind="gepa",
            schema_version="gepa-publication-proof-v1",
            canonical_json=canonical,
            digest=bad_digest,
            decision="apply",
        )


def test_publication_canonical_json_must_match_digest_and_bytes() -> None:
    noncanonical = json.dumps({"b": 1, "a": 2})

    with pytest.raises(ValueError, match="canonical JSON"):
        DecisionProofEnvelope(
            optimizer_kind="gepa",
            schema_version="gepa-publication-proof-v1",
            canonical_json=noncanonical,
            digest=_digest(noncanonical),
            decision="apply",
        )


def test_publication_envelopes_bind_their_declared_fields() -> None:
    proof = _proof()
    with pytest.raises(ValueError, match="optimizer_kind"):
        DecisionProofEnvelope(
            optimizer_kind="offline_tuner_replay",
            schema_version=proof.schema_version,
            canonical_json=proof.canonical_json,
            digest=proof.digest,
            decision="apply",
        )

    projection = _projection()
    with pytest.raises(ValueError, match="projection fields"):
        PublicationSearchProjection(
            schema_version=projection.schema_version,
            canonical_json=projection.canonical_json,
            digest=projection.digest,
            projector_id=projection.projector_id,
            projector_version=projection.projector_version,
            projector_code_digest=projection.projector_code_digest,
            candidate_content_digest=projection.candidate_content_digest,
            preserved_trigger=projection.preserved_trigger,
            embedding_model_id=projection.embedding_model_id,
            embedding=projection.embedding,
            expanded_terms=("changed",),
            lexical_document=projection.lexical_document,
        )


def test_publication_request_binds_content_optimizer_and_canonical_epochs() -> None:
    claim = PublicationClaim(job_id=7, owner="worker-a", fence=3)
    common = {
        "optimizer_kind": "gepa",
        "job_id": 7,
        "attempt_key": "attempt-7",
        "publication_claim": claim,
        "worker_fence": 11,
        "incumbent_user_playbook_id": 101,
        "incumbent_content_digest": _digest("old content"),
        "incumbent_trigger": "refund",
        "incumbent_semantic_digest": incumbent_user_playbook_semantic_digest(
            content_digest=_digest("old content"), trigger="refund"
        ),
        "projection": _projection(),
        "decision_proof": _proof(),
        "request_id": "request-7",
    }

    with pytest.raises(ValueError, match="content digest"):
        PublicationRequest(
            **common,
            revised_content="different content",
            subject_epochs_json=_canonical({"subjects": []}),
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        PublicationRequest(
            **common,
            revised_content="new content",
            subject_epochs_json=json.dumps({"subjects": []}),
        )
    with pytest.raises(ValueError, match="optimizer_kind"):
        PublicationRequest(
            **{**common, "optimizer_kind": "offline_tuner_replay"},
            revised_content="new content",
            subject_epochs_json=_canonical({"subjects": []}),
        )

    canonical = _canonical({"a": 2, "b": 1})
    with pytest.raises(ValueError, match="digest"):
        DecisionProofEnvelope(
            optimizer_kind="gepa",
            schema_version="gepa-publication-proof-v1",
            canonical_json=canonical,
            digest="0" * 64,
            decision="apply",
        )


def test_publication_request_rejects_wrong_claim_kind_and_non_apply_decision() -> None:
    canonical = _canonical(
        {
            "decision": "abstain",
            "optimizer_kind": "gepa",
            "schema_version": "gepa-publication-proof-v1",
        }
    )
    with pytest.raises(ValueError, match="decision"):
        DecisionProofEnvelope(
            optimizer_kind="gepa",
            schema_version="gepa-publication-proof-v1",
            canonical_json=canonical,
            digest=_digest(canonical),
            decision="abstain",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="job_id"):
        PublicationRequest(
            optimizer_kind="gepa",
            job_id=7,
            attempt_key="attempt-7",
            publication_claim=PublicationClaim(job_id=8, owner="worker-a", fence=1),
            worker_fence=11,
            incumbent_user_playbook_id=101,
            incumbent_content_digest=_digest("old content"),
            incumbent_trigger="refund",
            incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
                content_digest=_digest("old content"), trigger="refund"
            ),
            revised_content="new content",
            projection=_projection(),
            decision_proof=_proof(),
            subject_epochs_json=_canonical({"subjects": []}),
            request_id="request-7",
        )


@pytest.mark.parametrize(
    "subjects",
    [
        [],
        ["not-an-object"],
        [{"epoch": -1, "ref": "subject:a"}],
        [{"epoch": 0, "ref": ""}],
        [{"epoch": 0, "ref": "subject:a"}, {"epoch": 1, "ref": "subject:a"}],
        [{"epoch": 0, "subject_ref": "subject:a"}],
        [{"epoch": 0, "ref": "subject:a", "unexpected": True}],
    ],
)
def test_publication_request_rejects_invalid_subject_epoch_vectors(
    subjects: list[object],
) -> None:
    with pytest.raises(ValueError, match="subject epochs"):
        PublicationRequest(
            optimizer_kind="gepa",
            job_id=7,
            attempt_key="attempt-7",
            publication_claim=PublicationClaim(job_id=7, owner="worker-a", fence=3),
            worker_fence=11,
            incumbent_user_playbook_id=101,
            incumbent_content_digest=_digest("old content"),
            incumbent_trigger="refund",
            incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
                content_digest=_digest("old content"), trigger="refund"
            ),
            revised_content="new content",
            projection=_projection(),
            decision_proof=_proof(),
            subject_epochs_json=_canonical({"subjects": subjects}),
            request_id="request-7",
        )


def test_publication_projection_accepts_exact_task6_bytes_and_digest() -> None:
    canonical = (
        '{"candidate_content_digest":"fe32608c9ef5b6cf7e3f946480253ff76f24f4ec0678f3d0f07f9844cbff9601",'
        '"embedding":["0.25","-1","0"],"embedding_model_id":"test-embedding-v1",'
        '"expanded_terms":["refund","escalation"],'
        '"lexical_document":"refund escalation exact projection",'
        '"preserved_trigger":"refund",'
        '"projector_code_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"projector_id":"reflexio.search.user-playbook","projector_version":"1",'
        '"schema_version":"offline-tuner-candidate-search-projection-v1"}'
    )

    projection = PublicationSearchProjection(
        schema_version="offline-tuner-candidate-search-projection-v1",
        canonical_json=canonical,
        digest="f092c63fec1376c1e20089086427092613c4f63f16dab317f44eb8f71622b338",
        projector_id="reflexio.search.user-playbook",
        projector_version="1",
        projector_code_digest="a" * 64,
        candidate_content_digest=_digest("new content"),
        preserved_trigger="refund",
        embedding_model_id="test-embedding-v1",
        embedding=("0.25", "-1", "0"),
        expanded_terms=("refund", "escalation"),
        lexical_document="refund escalation exact projection",
    )

    assert projection.canonical_json.encode() == canonical.encode()
    assert (
        projection.digest
        == "f092c63fec1376c1e20089086427092613c4f63f16dab317f44eb8f71622b338"
    )


@pytest.mark.parametrize("value", [-0.0, 0.0, 1.5])
def test_rfc8785_encoder_rejects_all_floats(value: float) -> None:
    with pytest.raises(TypeError, match="Unsupported RFC 8785 value: float"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize("value", [-(2**53), 2**53])
def test_rfc8785_encoder_rejects_inexact_integer_bounds(value: int) -> None:
    with pytest.raises(ValueError, match="exactly representable"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize("value", [-(2**53) + 1, 2**53 - 1])
def test_rfc8785_encoder_accepts_exact_integer_bounds(value: int) -> None:
    assert canonical_json_bytes({"value": value})


def test_rfc8785_encoder_rejects_surrogates_and_uses_utf16_key_order() -> None:
    with pytest.raises(ValueError, match="surrogate"):
        canonical_json_bytes({"value": "\ud800"})

    assert canonical_json_bytes({"\ue000": "bmp", "\U00010000": "astral"}) == (
        '{"\U00010000":"astral","\ue000":"bmp"}'.encode()
    )


def test_publication_service_requires_explicit_verifier() -> None:
    with pytest.raises(TypeError):
        UserPlaybookPublicationService(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PublicationDecisionVerifier"):
        UserPlaybookPublicationService(
            object(),  # type: ignore[arg-type]
            verifier=None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="PublicationDecisionVerifier"):
        UserPlaybookPublicationService(
            object(),  # type: ignore[arg-type]
            verifier=object(),  # type: ignore[arg-type]
        )
