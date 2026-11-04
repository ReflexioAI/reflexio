from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from reflexio.server.services.offline_tuner.search_projection import (
    CandidateSearchProjection,
    CandidateSearchProjectionRequest,
    SearchProjectorKeyIdentity,
    publication_search_projection,
)


def _projection() -> CandidateSearchProjection:
    return CandidateSearchProjection(
        projector_id="user-playbook-search",
        projector_version="v1",
        projector_code_digest="a" * 64,
        candidate_content_digest="b" * 64,
        preserved_trigger="refund requests",
        embedding_model_id="synthetic-embedding-v1",
        embedding=("0.25", "-0.5", "1"),
        expanded_terms=("return", "reimbursement"),
        lexical_document="refund requests Revised refund procedure",
    )


def test_projection_is_deeply_immutable() -> None:
    projection = _projection()

    with pytest.raises(FrozenInstanceError):
        projection.lexical_document = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        projection.expanded_terms[0] = "changed"  # type: ignore[index]


def test_projection_digest_is_stable_and_round_trips_exact_bytes() -> None:
    projection = _projection()
    artifact_bytes = projection.canonical_bytes()

    restored = CandidateSearchProjection.from_canonical_bytes(artifact_bytes)

    assert restored == projection
    assert restored.canonical_bytes() == artifact_bytes
    assert restored.digest == projection.digest
    assert restored.key_identity == SearchProjectorKeyIdentity(
        projector_id="user-playbook-search",
        projector_version="v1",
        code_digest="a" * 64,
    )


def test_replay_and_publication_can_reuse_one_projection_artifact() -> None:
    projection = _projection()

    replay_overlay_artifact = projection.canonical_bytes()
    publication_staging_artifact = projection.canonical_bytes()

    assert replay_overlay_artifact is publication_staging_artifact
    assert projection.digest == CandidateSearchProjection.digest_bytes(
        publication_staging_artifact
    )


def test_projection_request_is_replay_neutral_and_derives_candidate_digest() -> None:
    request = CandidateSearchProjectionRequest(
        incumbent_user_playbook_id=41,
        incumbent_content_digest="c" * 64,
        candidate_content="Revised refund procedure",
        preserved_trigger="refund requests",
    )

    assert request.candidate_content_digest == CandidateSearchProjection.digest_bytes(
        b"Revised refund procedure"
    )


def test_shared_publication_projection_preserves_exact_artifact() -> None:
    projection = _projection()

    shared = publication_search_projection(projection)

    assert shared.canonical_json.encode() == projection.canonical_bytes()
    assert shared.digest == projection.digest
    assert shared.candidate_content_digest == projection.candidate_content_digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("incumbent_user_playbook_id", 0),
        ("incumbent_content_digest", "not-a-digest"),
        ("candidate_content", " "),
        ("preserved_trigger", ""),
    ],
)
def test_projection_request_rejects_invalid_inputs(field: str, value: object) -> None:
    values: dict[str, object] = {
        "incumbent_user_playbook_id": 41,
        "incumbent_content_digest": "c" * 64,
        "candidate_content": "Revised refund procedure",
        "preserved_trigger": "refund requests",
    }
    values[field] = value

    with pytest.raises(ValueError):
        CandidateSearchProjectionRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_content_digest", "c" * 64),
        ("preserved_trigger", None),
        ("embedding", ("0.5",)),
        ("expanded_terms", ("different",)),
        ("lexical_document", "different"),
    ],
)
def test_every_search_projection_field_changes_the_digest(
    field: str, value: object
) -> None:
    projection = _projection()

    changed = replace(projection, **{field: value})

    assert changed.digest != projection.digest


def test_projection_rejects_noncanonical_or_mismatched_artifact_bytes() -> None:
    with pytest.raises(ValueError, match="canonical"):
        CandidateSearchProjection.from_canonical_bytes(
            b'{"projector_version":"v1", "projector_id":"x"}'
        )


def test_projection_rejects_unsupported_schema_at_runtime_and_round_trip() -> None:
    with pytest.raises(ValueError, match="schema"):
        replace(_projection(), schema_version="offline-tuner-projection-v2")  # type: ignore[arg-type]

    artifact = (
        _projection()
        .canonical_bytes()
        .replace(
            b"offline-tuner-candidate-search-projection-v1",
            b"offline-tuner-candidate-search-projection-v2",
        )
    )
    with pytest.raises(ValueError, match="candidate search projection"):
        CandidateSearchProjection.from_canonical_bytes(artifact)


@pytest.mark.parametrize(
    "embedding",
    [
        (),
        ("NaN",),
        ("Infinity",),
        ("-Infinity",),
        ("not-a-number",),
        ("01",),
        ("-0",),
        ("1.0",),
        ("0.50",),
        ("1e0",),
        ("+1",),
        (".5",),
    ],
)
def test_projection_rejects_empty_nonfinite_nonnumeric_or_noncanonical_vectors(
    embedding: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="embedding"):
        replace(_projection(), embedding=embedding)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("projector_id", 1),
        ("projector_version", None),
        ("projector_code_digest", 1),
        ("candidate_content_digest", None),
        ("preserved_trigger", 1),
        ("embedding_model_id", False),
        ("embedding", ["1"]),
        ("expanded_terms", ["term"]),
        ("expanded_terms", ("",)),
        ("lexical_document", ""),
        ("lexical_document", 1),
    ],
)
def test_projection_rejects_invalid_runtime_field_types_and_lexical_content(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(_projection(), **{field: value})


def test_projection_accepts_the_documented_canonical_decimal_grammar() -> None:
    projection = replace(
        _projection(),
        embedding=("0", "-0.5", "0.25", "1", "-12", "123.456"),
    )

    assert projection.embedding == ("0", "-0.5", "0.25", "1", "-12", "123.456")
