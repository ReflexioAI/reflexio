from __future__ import annotations

import json
from hashlib import sha256

import pytest

from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    PublicationClaim,
    PublicationRequest,
    PublicationSearchProjection,
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
            "content_digest": _digest("new content"),
            "embedding": ["0.125", "0.5"],
            "embedding_model_id": "test-embedding-v1",
            "expanded_terms": ["refund", "escalation"],
            "lexical_document": "refund escalation exact projection",
            "schema_version": "publication-search-projection-v1",
            "trigger": "refund",
        }
    )
    return PublicationSearchProjection(
        schema_version="publication-search-projection-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        content_digest=_digest("new content"),
        trigger="refund",
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
            content_digest=projection.content_digest,
            trigger=projection.trigger,
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
            revised_content="new content",
            projection=_projection(),
            decision_proof=_proof(),
            subject_epochs_json=_canonical({"subjects": []}),
            request_id="request-7",
        )


@pytest.mark.parametrize(
    "subjects",
    [
        ["not-an-object"],
        [{"epoch": -1, "ref": "subject:a"}],
        [{"epoch": 0, "ref": ""}],
        [{"epoch": 0, "ref": "subject:a"}, {"epoch": 1, "ref": "subject:a"}],
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
            revised_content="new content",
            projection=_projection(),
            decision_proof=_proof(),
            subject_epochs_json=_canonical({"subjects": subjects}),
            request_id="request-7",
        )
