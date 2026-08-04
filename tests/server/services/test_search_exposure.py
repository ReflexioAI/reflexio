"""Direct behavior tests for user-playbook search exposure identities."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reflexio.models.api_schema.domain import BlockingIssue, UserPlaybook
from reflexio.models.api_schema.domain.enums import BlockingIssueKind, Status
from reflexio.server.services.search_exposure import (
    SearchExposureBatch,
    build_user_playbook_exposure_event,
    user_playbook_full_version_fingerprint,
)


def _playbook() -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=101,
        user_id="user-1",
        agent_version="agent-v1",
        request_id="source-request-1",
        playbook_name="Support policy",
        created_at=1_700_000_000,
        content="Escalate refund requests after verification.",
        trigger="refund escalation",
        rationale="Historical resolution pattern.",
        blocking_issue=BlockingIssue(
            kind=BlockingIssueKind.MISSING_TOOL, details="CRM access is absent."
        ),
        status=Status.ARCHIVED,
        source="support-import",
        source_interaction_ids=[11, 12],
        expanded_terms="refund return escalation",
        tags=["support", "refund"],
        embedding=[0.25] * 512,
        source_span="messages 4-6",
        notes="Reviewed by ops.",
        reader_angle="customer impact",
        merged_into=88,
        superseded_by=99,
        governance_subject_ref="subject:user-1",
        retired_at=1_700_000_050,
    )


def _batch(
    playbook: UserPlaybook,
    *,
    request_id: str | None = "request-1",
    session_id: str | None = "session-1",
    interaction_id: int | None = 41,
    invocation_id: str = "invocation-1",
) -> SearchExposureBatch:
    return SearchExposureBatch(
        org_id="org-1",
        request_id=request_id,
        session_id=session_id,
        interaction_id=interaction_id,
        user_id="user-1",
        user_playbooks=(playbook,),
        invocation_id=invocation_id,
    )


def _event(batch: SearchExposureBatch, playbook: UserPlaybook):
    return build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref="user:user-1",
        playbook_owner_governance_subject_ref="owner:user-1",
    )


def test_correlated_retries_keep_one_exposure_event_id_despite_invocation_id() -> None:
    playbook = _playbook()
    initial = _event(_batch(playbook, invocation_id="invocation-a"), playbook)
    retry = _event(_batch(playbook, invocation_id="invocation-b"), playbook)

    assert initial.exposure_event_id == retry.exposure_event_id


def test_correlation_free_invocations_get_distinct_exposure_event_ids() -> None:
    playbook = _playbook()
    first = _event(
        _batch(
            playbook,
            request_id=None,
            session_id=None,
            interaction_id=None,
            invocation_id="invocation-a",
        ),
        playbook,
    )
    second = _event(
        _batch(
            playbook,
            request_id=None,
            session_id=None,
            interaction_id=None,
            invocation_id="invocation-b",
        ),
        playbook,
    )

    assert first.exposure_event_id != second.exposure_event_id


def test_unscoped_exposure_keeps_unknown_subject_separate_from_playbook_owner() -> None:
    playbook = _playbook()
    batch = replace(_batch(playbook), user_id=None)

    event = build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref=None,
        playbook_owner_governance_subject_ref="owner:user-1",
    )

    assert event.user_id is None
    assert event.governance_subject_ref is None
    assert event.playbook_owner_user_id == "user-1"
    assert event.playbook_owner_governance_subject_ref == "owner:user-1"


@pytest.mark.parametrize("user_id", ["", " \t\n"], ids=["empty", "whitespace"])
def test_blank_retrieval_subject_normalizes_to_unscoped(user_id: str) -> None:
    playbook = _playbook()
    batch = replace(_batch(playbook), user_id=user_id)

    event = build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref=None,
        playbook_owner_governance_subject_ref="owner:user-1",
    )

    assert batch.user_id is None
    assert event.user_id is None
    assert event.governance_subject_ref is None
    assert event.playbook_owner_user_id == "user-1"


def test_scoped_exposure_rejects_a_playbook_owned_by_another_user() -> None:
    playbook = _playbook().model_copy(update={"user_id": "user-2"})
    batch = _batch(playbook)

    with pytest.raises(ValueError, match="does not match retrieval subject"):
        _event(batch, playbook)


def test_request_and_session_correlation_ids_normalize_whitespace_consistently() -> (
    None
):
    playbook = _playbook()
    whitespace = _batch(
        playbook,
        request_id=" \trequest-1\n",
        session_id="\tsession-1 ",
        interaction_id=None,
    )
    normalized = _batch(
        playbook,
        request_id="request-1",
        session_id="session-1",
        interaction_id=None,
    )
    blank = _batch(
        playbook,
        request_id=" \t",
        session_id="\n ",
        interaction_id=None,
        invocation_id="fallback-invocation",
    )
    absent = _batch(
        playbook,
        request_id=None,
        session_id=None,
        interaction_id=None,
        invocation_id="fallback-invocation",
    )

    assert (
        (whitespace.request_id, whitespace.session_id)
        == (
            normalized.request_id,
            normalized.session_id,
        )
        == ("request-1", "session-1")
    )
    assert (
        _event(whitespace, playbook).exposure_event_id
        == _event(normalized, playbook).exposure_event_id
    )
    assert (blank.request_id, blank.session_id) == (None, None)
    assert (
        _event(blank, playbook).exposure_event_id
        == _event(absent, playbook).exposure_event_id
    )


def test_embedding_changes_do_not_change_full_version_fingerprint() -> None:
    playbook = _playbook()
    reembedded = playbook.model_copy(update={"embedding": [0.5] * 512})

    assert user_playbook_full_version_fingerprint(playbook) == (
        user_playbook_full_version_fingerprint(reembedded)
    )


def test_internal_fields_are_excluded_from_user_playbook_serialization() -> None:
    serialized = _playbook().model_dump(mode="json")

    assert "governance_subject_ref" not in serialized
    assert "retired_at" not in serialized


# Every current UserPlaybook field is persisted except its derived embedding vector.
_PERSISTED_FIELD_CHANGES = [
    ("user_playbook_id", 102),
    ("user_id", "user-2"),
    ("agent_version", "agent-v2"),
    ("request_id", "source-request-2"),
    ("playbook_name", "Returns policy"),
    ("created_at", 1_700_000_001),
    ("content", "Verify returns before escalating."),
    ("trigger", "returns escalation"),
    ("rationale", "Updated resolution pattern."),
    (
        "blocking_issue",
        BlockingIssue(
            kind=BlockingIssueKind.PERMISSION_DENIED,
            details="CRM access was denied.",
        ),
    ),
    ("status", Status.PENDING),
    ("source", "returns-import"),
    ("source_interaction_ids", [11, 13]),
    ("expanded_terms", "return exchange escalation"),
    ("tags", ["support", "returns"]),
    ("source_span", "messages 7-9"),
    ("notes", "Needs legal review."),
    ("reader_angle", "policy compliance"),
    ("merged_into", 87),
    ("superseded_by", 100),
    ("governance_subject_ref", "subject:user-2"),
    ("retired_at", 1_700_000_051),
]


@pytest.mark.parametrize(("field", "value"), _PERSISTED_FIELD_CHANGES)
def test_full_version_fingerprint_changes_for_each_persisted_non_embedding_field(
    field: str, value: object
) -> None:
    playbook = _playbook()
    changed = playbook.model_copy(update={field: value})

    assert user_playbook_full_version_fingerprint(playbook) != (
        user_playbook_full_version_fingerprint(changed)
    )


def test_full_version_fingerprint_coverage_includes_each_persisted_model_field() -> (
    None
):
    assert {field for field, _value in _PERSISTED_FIELD_CHANGES} == (
        set(UserPlaybook.model_fields) - {"embedding"}
    )


def test_semantic_digest_and_fallback_identity_are_deterministic_and_domain_separated() -> (
    None
):
    playbook = _playbook()
    batch = _batch(
        playbook,
        request_id=None,
        session_id=None,
        interaction_id=None,
        invocation_id="invocation-a",
    )

    first = _event(batch, playbook)
    repeated = _event(replace(batch), playbook)

    assert (
        first.exposure_event_id
        == repeated.exposure_event_id
        == ("80b011b78df4a90e2238a7150d091c4d2f8c0e38d4343ccc7616e3536d40ca49")
    )
    assert (
        first.served_semantic_digest
        == repeated.served_semantic_digest
        == ("d321988fa077b43deb4df4c89753b98c757c7c3167a12ea26af9247e665cc942")
    )
    assert first.exposure_event_id != first.served_semantic_digest
