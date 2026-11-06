from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from typing import Any

import pytest

from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceEvent,
    OpenWorldEvidenceSnapshot,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    select_open_world_evidence,
)
from reflexio.server.services.offline_tuner.open_world.citations import verify_citations
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world.models import EvidenceCitation
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)
from reflexio.test_support.open_world_rule_evidence import (
    evidence_digest as _digest,
)
from reflexio.test_support.open_world_rule_evidence import (
    evidence_event as _event,
)
from reflexio.test_support.open_world_rule_evidence import (
    frozen_evidence as _frozen,
)


def test_rehydrate_builds_byte_stable_spans_for_exact_frozen_manifest_union() -> None:
    frozen, snapshot = _frozen()

    first = rehydrate_frozen_evidence(frozen, snapshot)
    second = rehydrate_frozen_evidence(frozen, snapshot)

    assert first == second
    assert first.bundle_digest == frozen.bundle.digest
    assert set(first.documents_by_session) == {
        (session.user_id, session.session_id)
        for manifest in (
            frozen.bundle.discovery,
            frozen.bundle.held_out,
            frozen.bundle.safety_only,
        )
        for session in manifest.sessions
    }
    span_ids = [
        span.span_id
        for document in first.documents_by_session.values()
        for span in document.spans
    ]
    assert span_ids
    assert len(span_ids) == len(set(span_ids))


def test_unlabeled_source_round_trips_frozen_bundle_and_original_citations() -> None:
    from reflexio.server.services.offline_tuner.evidence.open_world_store import (
        _parse_canonical_bundle,
    )

    original, snapshot = _frozen()
    event = replace(
        snapshot.events[0],
        outcome=None,
        outcome_id=None,
        outcome_revision=None,
        outcome_contract_digest=None,
    )
    snapshot = OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=(event,))
    selection = select_open_world_evidence(snapshot, original.bundle.target)
    assert selection.eligible
    bundle = _parse_canonical_bundle(selection.bundle.canonical_bytes().decode())
    assert bundle == selection.bundle
    frozen = replace(original, bundle=bundle)
    document = rehydrate_frozen_evidence(frozen, snapshot).documents_by_session[
        ("user-1", "session-1")
    ]
    assert document.session_outcome is None
    assert any(span.speaker == "agent" for span in document.spans)
    assert event.trajectory_digest is not None
    assert all(
        span.span_id.startswith(event.trajectory_digest + ":")
        for span in document.spans
    )


def _trajectory_with_changed_field(
    event: OpenWorldEvidenceEvent, path: tuple[str | int, ...]
) -> object:
    trajectory = deepcopy(event.trajectory)
    assert trajectory is not None
    value: Any = trajectory
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = "changed"
    return trajectory


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, tenant_id="tenant-2"),
        lambda event: replace(event, user_id="user-2"),
        lambda event: replace(event, session_id="session-other"),
        lambda event: replace(event, user_playbook_id=8),
        lambda event: replace(event, served_semantic_digest=_digest("other")),
        lambda event: replace(event, served_full_version_fingerprint="b" * 64),
        lambda event: replace(event, outcome_id="outcome-other"),
        lambda event: replace(event, outcome="success"),
        lambda event: replace(event, outcome_revision=2),
        lambda event: replace(event, outcome_contract_digest=_digest("other-contract")),
        lambda event: replace(event, trajectory_digest="b" * 64),
        lambda event: replace(
            event,
            trajectory=_trajectory_with_changed_field(
                event, ("requests", 0, "interactions", 0, "content")
            ),
        ),
        lambda event: replace(
            event,
            trajectory=_trajectory_with_changed_field(
                event, ("requests", 0, "request", "request_id")
            ),
        ),
        lambda event: replace(event, governance_subject_refs=()),
        lambda event: replace(event, governance_valid=False),
        lambda event: replace(event, trajectory_byte_count=0),
        lambda event: replace(event, trajectory_token_count=0),
    ],
)
def test_rehydrate_rejects_changed_frozen_dependencies(mutate: object) -> None:
    frozen, snapshot = _frozen()
    changed = list(snapshot.events)
    changed[0] = mutate(changed[0])  # type: ignore[operator]

    with pytest.raises(ValueError):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=tuple(changed)),
        )


def test_digest_match_with_stale_counts_still_rehydrates() -> None:
    """A tiktoken version change must not read as corruption.

    The digest pins the exact bytes; both stored counts are functions of those
    same bytes. So the only way a recomputation can disagree while the digest
    agrees is that the thing doing the counting changed -- a tokenizer bump --
    which would then fail every rehydrate in the fleet and report it as
    tampering. Drift between the counts and the FROZEN MANIFEST is still
    refused, above; that is the comparison which detects tampering.
    """
    frozen, snapshot = _frozen(count_drift=7)

    rehydrated = rehydrate_frozen_evidence(frozen, snapshot)

    assert rehydrated.bundle_digest == frozen.bundle.digest
    assert rehydrated.documents_by_session


def test_tampered_trajectory_bytes_still_raise() -> None:
    """Detection is unchanged: the digest alone still catches changed bytes."""
    frozen, snapshot = _frozen()
    changed = list(snapshot.events)
    changed[0] = replace(
        changed[0],
        trajectory=_trajectory_with_changed_field(
            changed[0], ("requests", 0, "interactions", 0, "content")
        ),
    )

    with pytest.raises(ValueError, match="snapshot trajectory bytes or tokens changed"):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=tuple(changed)),
        )


def test_rehydrate_rejects_missing_or_extra_frozen_session_events() -> None:
    frozen, snapshot = _frozen()

    with pytest.raises(ValueError, match="exact frozen manifest union"):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(
                tenant_id="tenant-1", events=snapshot.events[:-1]
            ),
        )
    with pytest.raises(ValueError, match="exact frozen manifest union"):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(
                tenant_id="tenant-1",
                events=(
                    *snapshot.events,
                    _event(session_id="session-6", outcome="failure", content="extra"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="exact frozen manifest union"):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(
                tenant_id="tenant-1", events=(*snapshot.events, snapshot.events[0])
            ),
        )


def test_rehydrate_builds_each_session_trajectory_spans_once_for_two_exposures() -> (
    None
):
    frozen, snapshot = _frozen(duplicate_first_exposure=True)

    document = rehydrate_frozen_evidence(frozen, snapshot).documents_by_session[
        ("user-1", "session-1")
    ]
    citation = EvidenceCitation(
        span_id=document.spans[0].span_id,
        exact_quote=document.spans[0].immutable_text,
    )

    assert len(document.spans) == len({span.span_id for span in document.spans})
    assert (
        verify_citations((citation,), allowed_spans=document.spans)[0].span_id
        == citation.span_id
    )


def test_session_allowlists_exclude_other_frozen_source_documents() -> None:
    frozen, snapshot = _frozen()
    rehydrated = rehydrate_frozen_evidence(frozen, snapshot)
    discovery, *held_out = rehydrated.documents_by_session.values()
    assert len(held_out) >= 2
    held_out_citation = EvidenceCitation(
        span_id=held_out[0].spans[0].span_id,
        exact_quote=held_out[0].spans[0].immutable_text,
    )
    other_session_citation = EvidenceCitation(
        span_id=held_out[1].spans[0].span_id,
        exact_quote=held_out[1].spans[0].immutable_text,
    )

    with pytest.raises(ValueError, match="unavailable"):
        verify_citations((held_out_citation,), allowed_spans=discovery.spans)
    with pytest.raises(ValueError, match="unavailable"):
        verify_citations((other_session_citation,), allowed_spans=held_out[0].spans)


def test_rehydration_preserves_optional_completion_metadata() -> None:
    """Existing labels remain honest metadata without partitioning the sources."""
    frozen, snapshot = _frozen()

    rehydrated = rehydrate_frozen_evidence(frozen, snapshot)
    outcome_by_session = {
        (session.user_id, session.session_id): session.outcome
        for manifest in (
            frozen.bundle.discovery,
            frozen.bundle.held_out,
            frozen.bundle.safety_only,
        )
        for session in manifest.sessions
    }
    discovery = {
        key: document
        for key, document in rehydrated.documents_by_session.items()
        if document.manifest_role == "discovery"
    }

    assert discovery
    for key, document in discovery.items():
        assert document.session_outcome == outcome_by_session[key]
    assert len(discovery) == len(outcome_by_session)
    assert {document.session_outcome for document in discovery.values()} == {
        "failure",
        "success",
    }


# ---------------------------------------------------------------------------
# Cutover judging: an inferred EXTENSION is not a displacement.
# ---------------------------------------------------------------------------


def _extended(
    event: OpenWorldEvidenceEvent, **changes: object
) -> OpenWorldEvidenceEvent:
    """What an event read through a FROZEN cutover reports after the tuner
    extended the session's
    inferred outcome: the trajectory is still read through the FROZEN prefix
    (same bytes, same digest), while the outcome row moved on -- a new value,
    its own digest no longer the frozen one, same id and revision."""
    values: dict[str, object] = {
        "outcome": "success" if event.outcome == "failure" else "failure",
        "trajectory_digest_matches": False,
        "outcome_is_inferred": True,
        "outcome_covers_trajectory_prefix": True,
    }
    values.update(changes)
    return replace(event, **values)  # type: ignore[arg-type]


def _with_first_event(
    snapshot: OpenWorldEvidenceSnapshot, event: OpenWorldEvidenceEvent
) -> OpenWorldEvidenceSnapshot:
    return OpenWorldEvidenceSnapshot(
        tenant_id="tenant-1", events=(event, *snapshot.events[1:])
    )


def test_an_inferred_extension_after_the_freeze_still_rehydrates() -> None:
    """A live session keeps reacting; its refreshed outcome must not kill an
    analysis already in flight. The analyst still sees the FROZEN label."""
    frozen, snapshot = _frozen()
    first = snapshot.events[0]

    rehydrated = rehydrate_frozen_evidence(
        frozen, _with_first_event(snapshot, _extended(first))
    )

    assert (first.user_id, first.session_id) in rehydrated.documents_by_session
    # Every discovery label is the FROZEN outcome, the extended one included.
    for session in frozen.bundle.discovery.sessions:
        document = rehydrated.documents_by_session[
            (session.user_id, session.session_id)
        ]
        assert document.session_outcome == session.outcome


@pytest.mark.parametrize(
    "changes",
    [
        # Not ours any more: a customer's report is never an extension.
        {"outcome_is_inferred": False},
        # The outcome no longer covers the prefix the manifest froze.
        {"outcome_covers_trajectory_prefix": False},
        # A different outcome row, or a bumped revision, is a displacement.
        {"outcome_id": "outcome-other"},
        {"outcome_revision": 2},
        # The reload read a DIFFERENT prefix than the manifest froze.
        {"trajectory_through_request_id": "request-elsewhere"},
        # The frozen trajectory itself changed: the digest check still runs.
        {"trajectory_digest": "b" * 64},
    ],
)
def test_anything_beyond_an_inferred_extension_still_refuses(
    changes: dict[str, object],
) -> None:
    frozen, snapshot = _frozen()
    with pytest.raises(ValueError):
        rehydrate_frozen_evidence(
            frozen,
            _with_first_event(snapshot, _extended(snapshot.events[0], **changes)),
        )


def test_ordinary_projection_keeps_tool_content_but_omits_tools_used_metadata():
    from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
        project_trajectory_spans,
    )

    frozen, _ = _frozen()
    event = _event(
        session_id="session-1", outcome="unknown", content="Observed agent action"
    )
    trajectory = deepcopy(event.trajectory)
    assert trajectory is not None
    interactions = trajectory["requests"][0]["interactions"]
    interactions[0].update(role="Tool", content="Captured tool result")
    interactions[1]["tools_used"] = [
        {"name": "lookup_account", "arguments": {"id": "a"}}
    ]
    original_bytes = canonical_trajectory_bytes(trajectory)
    event = replace(
        event,
        trajectory=trajectory,
        trajectory_digest=sha256(original_bytes).hexdigest(),
        trajectory_byte_count=len(original_bytes),
        trajectory_token_count=count_input_tokens(original_bytes.decode()),
    )
    snapshot = OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=(event,))
    bundle = select_open_world_evidence(snapshot, frozen.bundle.target).bundle
    assert not bundle.expected.events[0].compaction_required
    evidence = rehydrate_frozen_evidence(replace(frozen, bundle=bundle), snapshot)
    document = evidence.documents_by_session[("user-1", "session-1")]
    assert any(
        span.immutable_text == "Captured tool result" and span.speaker == "other"
        for span in document.spans
    )
    assert not any(span.field_path.endswith(".tools_used") for span in document.spans)
    assert event.trajectory_digest is not None
    compaction_sources = project_trajectory_spans(
        trajectory,
        trajectory_digest=event.trajectory_digest,
        tenant_id="tenant-1",
        role="discovery",
        user_id="user-1",
        session_id="session-1",
        include_tools=True,
    )
    assert any(
        span.field_path.endswith(".tools_used")
        and "lookup_account" in span.immutable_text
        and span.speaker == "other"
        for span in compaction_sources
    )


def test_a_whitespace_only_turn_is_skipped_rather_than_failing_rehydration() -> None:
    # EvidenceSpan refuses blank text, so projecting a blank turn used to fail
    # the whole frozen attempt even when the rest of the session was citable.
    frozen, _ = _frozen()
    event = _event(
        session_id="session-1", outcome="unknown", content="Observed agent action"
    )
    trajectory = deepcopy(event.trajectory)
    assert trajectory is not None
    interactions = trajectory["requests"][0]["interactions"]
    interactions[0]["content"] = " \n\t "
    original_bytes = canonical_trajectory_bytes(trajectory)
    event = replace(
        event,
        trajectory=trajectory,
        trajectory_digest=sha256(original_bytes).hexdigest(),
        trajectory_byte_count=len(original_bytes),
        trajectory_token_count=count_input_tokens(original_bytes.decode()),
    )
    snapshot = OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=(event,))
    bundle = select_open_world_evidence(snapshot, frozen.bundle.target).bundle

    evidence = rehydrate_frozen_evidence(replace(frozen, bundle=bundle), snapshot)

    document = evidence.documents_by_session[("user-1", "session-1")]
    paths = [span.field_path for span in document.spans]
    assert "requests[0].interactions[0].content" not in paths
    assert "requests[0].interactions[1].content" in paths
    assert any(span.speaker == "agent" for span in document.spans)
