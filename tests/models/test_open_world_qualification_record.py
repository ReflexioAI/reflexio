"""Strictness coverage for the shared open-world qualification record."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import (
    OPEN_WORLD_QUALIFICATION_CLASSES,
    OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION,
    OpenWorldQualificationClassCount,
    OpenWorldQualificationRecord,
)

_COMPONENT_DIGEST = "a" * 64
_SUITE_DIGEST = "b" * 64
_RESULT_DIGEST = "c" * 64


def _class_counts(
    **overrides: tuple[int, int],
) -> tuple[OpenWorldQualificationClassCount, ...]:
    return tuple(
        OpenWorldQualificationClassCount(
            qualification_class=qualification_class,
            required=overrides.get(qualification_class, (2, 2))[0],
            passed_required=overrides.get(qualification_class, (2, 2))[1],
        )
        for qualification_class in OPEN_WORLD_QUALIFICATION_CLASSES
    )


def _record(**overrides: object) -> OpenWorldQualificationRecord:
    payload: dict[str, object] = {
        "component_identity_digest": _COMPONENT_DIGEST,
        "suite_digest": _SUITE_DIGEST,
        "result_digest": _RESULT_DIGEST,
        "class_counts": _class_counts(),
        "passed": True,
        "observation_digests": ("0" * 64, "1" * 64),
        "created_at": 1_700_000_000,
    }
    payload.update(overrides)
    return OpenWorldQualificationRecord(**payload)  # type: ignore[arg-type]


def test_seven_ordered_classes_round_trip() -> None:
    record = _record()

    assert record.schema_version == OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION
    assert record.schema_version == "offline-tuner-open-world-qualification-result-v1"
    assert len(OPEN_WORLD_QUALIFICATION_CLASSES) == 7
    assert OPEN_WORLD_QUALIFICATION_CLASSES == (
        "citation_fidelity",
        "abstention",
        "support",
        "refutation",
        "insufficiency",
        "unsupported_causal_claim_rejection",
        "prompt_injection_resistance",
    )
    assert tuple(count.qualification_class for count in record.class_counts) == (
        OPEN_WORLD_QUALIFICATION_CLASSES
    )


def test_schema_version_must_be_the_exact_v1_value() -> None:
    with pytest.raises(ValidationError):
        _record(schema_version="offline-tuner-open-world-qualification-result-v2")


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(rationale="free-form model prose")


def test_missing_class_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(class_counts=_class_counts()[:-1])


def test_reordered_classes_are_rejected() -> None:
    reordered = _class_counts()
    with pytest.raises(ValidationError):
        _record(class_counts=(reordered[1], reordered[0], *reordered[2:]))


def test_duplicated_class_is_rejected() -> None:
    counts = _class_counts()
    with pytest.raises(ValidationError):
        _record(class_counts=(counts[0], *counts))


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(class_counts=_class_counts(abstention=(-1, 0)))
    with pytest.raises(ValidationError):
        _record(class_counts=_class_counts(abstention=(2, -1)))


def test_passed_required_may_not_exceed_required() -> None:
    with pytest.raises(ValidationError):
        _record(class_counts=_class_counts(support=(1, 2)))


def test_zero_required_class_is_accepted() -> None:
    record = _record(class_counts=_class_counts(refutation=(0, 0)))

    assert record.class_counts[3].required == 0


@pytest.mark.parametrize("field", ["component_identity_digest", "suite_digest"])
def test_identity_digests_must_be_sha256(field: str) -> None:
    with pytest.raises(ValidationError):
        _record(**{field: "not-a-digest"})


def test_result_digest_must_be_sha256() -> None:
    with pytest.raises(ValidationError):
        _record(result_digest="C" * 64)


def test_observation_digests_must_be_sha256() -> None:
    with pytest.raises(ValidationError):
        _record(observation_digests=("0" * 63,))


def test_observation_digests_must_be_sorted() -> None:
    with pytest.raises(ValidationError):
        _record(observation_digests=("1" * 64, "0" * 64))


def test_observation_digests_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        _record(observation_digests=("0" * 64, "0" * 64))


def test_empty_observation_digests_are_accepted() -> None:
    assert _record(observation_digests=()).observation_digests == ()


def test_negative_created_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(created_at=-1)


def test_record_is_immutable() -> None:
    record = _record()

    with pytest.raises(ValidationError):
        record.passed = False  # type: ignore[misc]
