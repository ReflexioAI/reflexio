"""Rehydrate immutable frozen open-world evidence into citation documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    EvidenceRole,
    OpenWorldEvidenceEvent,
    OpenWorldEvidenceSchemaUnsupportedError,
    OpenWorldEvidenceSession,
    OpenWorldEvidenceSnapshot,
    OpenWorldExpectedEvent,
    TurnSpeaker,
    classify_turn_speaker,
    validate_open_world_evidence_bundle,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    FrozenOpenWorldEvidence,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceDocument,
    EvidenceSpan,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
    OpenWorldRefusalError,
)
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
    canonical_trajectory_bytes,
)


class OpenWorldEvidenceRehydrationError(ValueError, OpenWorldRefusalError):
    """Frozen snapshot bytes do not match the retained evidence contract."""

    refusal = OpenWorldRefusal.REHYDRATION_FAILED


@dataclass(frozen=True)
class RehydratedEvidence:
    bundle_digest: str
    documents_by_session: Mapping[tuple[str, str], EvidenceDocument]
    trajectory_json_by_session: Mapping[tuple[str, str], str]


def rehydrate_frozen_evidence(
    frozen: FrozenOpenWorldEvidence,
    snapshot: OpenWorldEvidenceSnapshot,
) -> RehydratedEvidence:
    """Validate a fresh exact-ID snapshot and derive immutable citation spans."""
    if type(frozen) is not FrozenOpenWorldEvidence:
        raise TypeError("frozen must be FrozenOpenWorldEvidence")
    if type(snapshot) is not OpenWorldEvidenceSnapshot:
        raise TypeError("snapshot must be OpenWorldEvidenceSnapshot")
    try:
        return _rehydrate_frozen_evidence(frozen, snapshot)
    except OpenWorldEvidenceRehydrationError:
        raise
    except OpenWorldEvidenceSchemaUnsupportedError:
        # ABOVE the widening handler below, which it would otherwise match --
        # it is a ``ValueError``. Re-wrapping it as a rehydration error is what
        # made the runner report ``infrastructure_failure`` for a bundle no
        # retry can load; the class has to survive to the runner intact for the
        # refusal to keep its name.
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OpenWorldEvidenceRehydrationError(str(exc)) from exc


def _rehydrate_frozen_evidence(
    frozen: FrozenOpenWorldEvidence,
    snapshot: OpenWorldEvidenceSnapshot,
) -> RehydratedEvidence:
    bundle = frozen.bundle
    validate_open_world_evidence_bundle(bundle)
    if snapshot.tenant_id != bundle.target.tenant_id:
        raise ValueError("snapshot tenant does not match frozen evidence")

    expected_by_id = {
        event.exposure_event_id: event for event in bundle.expected.events
    }
    frozen_by_event_id: dict[str, tuple[EvidenceRole, OpenWorldEvidenceSession]] = {}
    for manifest in (bundle.discovery, bundle.held_out, bundle.safety_only):
        for session in manifest.sessions:
            for event_id in session.exposure_event_ids:
                if event_id in frozen_by_event_id:
                    raise ValueError(
                        "frozen manifests contain duplicate exposure events"
                    )
                frozen_by_event_id[event_id] = (manifest.role, session)
                if event_id not in expected_by_id:
                    raise ValueError(
                        "frozen manifest event is absent from expected manifest"
                    )

    snapshot_by_id = {event.exposure_event_id: event for event in snapshot.events}
    if len(snapshot_by_id) != len(snapshot.events) or set(snapshot_by_id) != set(
        frozen_by_event_id
    ):
        raise ValueError("snapshot is not the exact frozen manifest union")

    events_by_session: dict[tuple[str, str], list[OpenWorldEvidenceEvent]] = {}
    for event_id, event in snapshot_by_id.items():
        role, session = frozen_by_event_id[event_id]
        _validate_event(
            event,
            expected=expected_by_id[event_id],
            session=session,
            tenant_id=bundle.target.tenant_id,
            user_playbook_id=bundle.target.user_playbook_id,
            incumbent_semantic_digest=bundle.target.incumbent_semantic_digest,
            incumbent_full_version_fingerprint=(
                bundle.target.incumbent_full_version_fingerprint
            ),
        )
        events_by_session.setdefault((session.user_id, session.session_id), []).append(
            event
        )

    documents: dict[tuple[str, str], EvidenceDocument] = {}
    trajectories: dict[tuple[str, str], str] = {}
    for manifest in (bundle.discovery, bundle.held_out, bundle.safety_only):
        for session in manifest.sessions:
            key = (session.user_id, session.session_id)
            if key in documents:
                raise ValueError("frozen manifests contain duplicate sessions")
            session_events = events_by_session.get(key)
            if session_events is None:
                raise ValueError("snapshot is not the exact frozen manifest union")
            original_event = _canonical_session_event(session_events)
            trajectories[key] = canonical_trajectory_bytes(
                original_event.trajectory
            ).decode()
            spans = _spans_for_event(
                original_event,
                manifest.role,
                include_tools=any(
                    expected_by_id[e.exposure_event_id].compaction_required
                    for e in session_events
                ),
            )
            documents[key] = EvidenceDocument(
                manifest_role=manifest.role,
                user_id=session.user_id,
                session_id=session.session_id,
                trajectory_digest=session.trajectory_digest,
                spans=spans,
                # Selection already adjudicated this session's outcome and
                # froze it on the manifest; this is where it used to be
                # dropped, leaving the discovery analyst asked for a
                # failed-vs-successful contrast over unlabelled sessions.
                # Held-out and safety-only stay unlabelled on purpose -- see
                # EvidenceDocument.
                session_outcome=(
                    session.outcome if manifest.role == "discovery" else None
                ),
            )
    return RehydratedEvidence(
        bundle_digest=bundle.digest,
        documents_by_session=MappingProxyType(documents),
        trajectory_json_by_session=MappingProxyType(trajectories),
    )


def _canonical_session_event(
    events: list[OpenWorldEvidenceEvent],
) -> OpenWorldEvidenceEvent:
    ordered = sorted(events, key=lambda event: event.exposure_event_id.encode())
    first = ordered[0]
    if first.trajectory is None:
        raise ValueError("snapshot trajectory is missing")
    trajectory_bytes = canonical_trajectory_bytes(first.trajectory)
    for event in ordered[1:]:
        if (
            event.trajectory is None
            or canonical_trajectory_bytes(event.trajectory) != trajectory_bytes
        ):
            raise ValueError("snapshot session trajectories disagree")
    return first


def _validate_event(
    event: OpenWorldEvidenceEvent,
    *,
    expected: OpenWorldExpectedEvent,
    session: OpenWorldEvidenceSession,
    tenant_id: str,
    user_playbook_id: int,
    incumbent_semantic_digest: str,
    incumbent_full_version_fingerprint: str,
) -> None:
    # DISPLACEMENT IS NOT CORRUPTION, and the two must not report the same
    # thing. A customer can now replace an inferred outcome, and the ~10-minute
    # window between freezing evidence and finishing analysis is wide enough
    # for one to land. That changes `outcome`, `outcome_id` and
    # `outcome_revision` on a row the frozen manifest already named -- which
    # the comparison below would report as "does not match frozen manifest",
    # the same sentence it uses for genuinely corrupt evidence.
    #
    # Abort rather than continue: the frozen evidence is now known to be wrong
    # about this session, and publishing a revision derived from an outcome the
    # customer has since contradicted is the failure this whole change exists
    # to prevent. The next pass re-reads the window and gets it right.
    if (
        event.outcome_id == expected.outcome_id
        and event.outcome_revision is not None
        and expected.outcome_revision is not None
        and event.outcome_revision > expected.outcome_revision
    ) or (
        event.outcome_id != expected.outcome_id
        and event.session_id == expected.session_id
        and event.user_id == expected.user_id
        and event.exposure_event_id == expected.exposure_event_id
    ):
        raise OpenWorldEvidenceRehydrationError(
            "session outcome was displaced after evidence was frozen; "
            f"session={event.session_id!r} "
            f"frozen_revision={expected.outcome_revision!r} "
            f"current_revision={event.outcome_revision!r}"
        )
    # AN INFERRED EXTENSION IS NOT A DISPLACEMENT, and must not end the attempt.
    # A live session keeps producing reactions, and each one re-judges it
    # through a LATER cutover and refreshes the tuner's inferred outcome in
    # place: same row, same `outcome_id` and `outcome_revision`, a longer
    # prefix, and possibly a different value. The frozen evidence still holds
    # -- the trajectory below is read back through the FROZEN cutover and must
    # still digest identically -- so only the two fields that describe the
    # outcome's CURRENT prefix are allowed to differ: its value and whether its
    # digest matches the frozen trajectory. Everything else, including the
    # digest equality, is still compared. A customer's report is never this
    # case: it is not inferred, and it bumps the revision (caught above).
    # The analyst label stays the frozen one: documents read `session.outcome`.
    # Production rehydrates from the freeze-time snapshot, so this relaxation is
    # reachable only from an event built through a frozen cutover; the storage
    # reload that produced such events had no caller and is deleted.
    extended = (
        event.outcome_is_inferred
        and event.outcome_id == expected.outcome_id
        and event.outcome_revision == expected.outcome_revision
        and event.outcome_covers_trajectory_prefix
    )
    if (
        event.tenant_id != tenant_id
        or event.user_id != session.user_id
        or event.session_id != session.session_id
        or event.user_playbook_id != user_playbook_id
        or event.served_semantic_digest != incumbent_semantic_digest
        or event.served_full_version_fingerprint != incumbent_full_version_fingerprint
        or event.exposure_event_id != expected.exposure_event_id
        or event.user_id != expected.user_id
        or event.session_id != expected.session_id
        or event.user_playbook_id != expected.user_playbook_id
        or event.served_semantic_digest != expected.served_semantic_digest
        or event.served_full_version_fingerprint
        != expected.served_full_version_fingerprint
        or event.request_id != expected.request_id
        or event.correlation_reason != expected.correlation_reason
        or event.exposed_at != expected.exposed_at
        or event.trajectory_digest != expected.trajectory_digest
        or event.trajectory_byte_count != expected.trajectory_byte_count
        or event.trajectory_token_count != expected.trajectory_token_count
        or event.trajectory_through_request_id != expected.trajectory_through_request_id
        or (not extended and event.outcome != expected.outcome)
        or event.outcome_id != expected.outcome_id
        or event.outcome_revision != expected.outcome_revision
        or event.outcome_contract_digest != expected.outcome_contract_digest
        or event.governance_subject_refs != expected.governance_subject_refs
        or not event.governance_valid
        or (not extended and not event.trajectory_digest_matches)
    ):
        raise ValueError("snapshot event does not match frozen manifest")
    if event.request_id not in session.request_ids:
        raise ValueError("snapshot request is absent from frozen session")
    if event.trajectory is None:
        raise ValueError("snapshot trajectory is missing")
    trajectory_bytes = canonical_trajectory_bytes(event.trajectory)
    # THE DIGEST ALONE, and detection is IDENTICAL for it -- one false-positive
    # source lighter, not one check weaker.
    #
    # This used to recompute `len(trajectory_bytes)` and
    # `count_input_tokens(...)` against the stored counts in the same `if`. Both
    # are functions of the very bytes the digest already pins, so no byte change
    # can satisfy the digest and fail a count. What CAN is the counter changing
    # underneath the stored value: a `tiktoken` version bump re-tokenizes
    # unchanged bytes and would have failed every rehydrate in the fleet,
    # reporting a dependency upgrade as evidence corruption.
    #
    # Tampering with the counts themselves is still refused, and by the
    # comparison that can actually see it: `event.trajectory_byte_count !=
    # expected.trajectory_byte_count` above, against the FROZEN MANIFEST.
    if sha256(trajectory_bytes).hexdigest() != event.trajectory_digest:
        raise ValueError("snapshot trajectory bytes or tokens changed")
    try:
        request_ids = tuple(
            sorted(
                {
                    item["request"]["request_id"]
                    for item in event.trajectory["requests"]
                },
                key=lambda value: value.encode(),
            )
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("snapshot trajectory is invalid") from exc
    if (
        request_ids != expected.trajectory_request_ids
        or request_ids != session.request_ids
    ):
        raise ValueError("snapshot trajectory requests changed")


def _spans_for_event(
    event: OpenWorldEvidenceEvent, role: EvidenceRole, *, include_tools: bool = False
) -> tuple[EvidenceSpan, ...]:
    if event.trajectory is None:
        raise ValueError("snapshot trajectory is missing")
    if (
        event.trajectory_digest is None
        or event.tenant_id is None
        or event.user_id is None
        or event.session_id is None
    ):
        raise ValueError("snapshot event identity is missing")
    return project_trajectory_spans(
        event.trajectory,
        trajectory_digest=event.trajectory_digest,
        tenant_id=event.tenant_id,
        role=role,
        user_id=event.user_id,
        session_id=event.session_id,
        include_tools=include_tools,
    )


def project_trajectory_spans(
    trajectory: CanonicalSessionTrajectory,
    *,
    trajectory_digest: str,
    tenant_id: str,
    role: EvidenceRole,
    user_id: str,
    session_id: str,
    include_tools: bool = False,
) -> tuple[EvidenceSpan, ...]:
    """Format canonical trajectory fields; this grants no exposure/outcome authority.

    Production must first validate its frozen evidence, as the rehydrator does.
    Qualification may use synthetic scope without fabricating an exposure row.
    """
    spans: list[EvidenceSpan] = []
    for request_index, request_item in enumerate(trajectory["requests"]):
        request_id = request_item["request"]["request_id"]
        for interaction_index, interaction in enumerate(request_item["interactions"]):
            fields = [("content", interaction["content"])]
            if include_tools and interaction["tools_used"]:
                fields.append(
                    (
                        "tools_used",
                        canonical_trajectory_bytes(interaction["tools_used"]).decode(),
                    )
                )
            for field, text in fields:
                if not text:
                    continue
                field_path = f"requests[{request_index}].interactions[{interaction_index}].{field}"
                speaker = (
                    classify_turn_speaker(interaction["role"])
                    if field == "content"
                    else "other"
                )
                spans.append(
                    EvidenceSpan(
                        span_id=_span_id(
                            trajectory_digest=trajectory_digest,
                            tenant_id=tenant_id,
                            role=role,
                            user_id=user_id,
                            session_id=session_id,
                            request_id=request_id,
                            interaction_id=interaction["interaction_id"],
                            field_path=field_path,
                            immutable_text=text,
                            speaker=speaker,
                        ),
                        tenant_id=tenant_id,
                        manifest_role=role,
                        user_id=user_id,
                        session_id=session_id,
                        request_id=request_id,
                        interaction_id=interaction["interaction_id"],
                        field_path=field_path,
                        immutable_text=text,
                        speaker=speaker,
                    )
                )
    return tuple(spans)


def _span_id(
    *,
    trajectory_digest: str,
    tenant_id: str,
    role: EvidenceRole,
    user_id: str,
    session_id: str,
    request_id: str,
    interaction_id: int,
    field_path: str,
    immutable_text: str,
    speaker: TurnSpeaker,
) -> str:
    # `speaker` is part of the span's IDENTITY, not decoration hung beside it.
    # A citation names a span id, and the verifier resolves that id back to the
    # server's own copy of the text; if the id did not cover the speaker, the
    # same id would be reachable under two different attributions and a memo
    # could keep its citations while the claim about who acted silently
    # changed underneath them.
    content_digest = sha256(
        canonical_json_bytes(
            {
                "schema_version": "open-world-evidence-span-v2",
                "tenant_id": tenant_id,
                "manifest_role": role,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "interaction_id": interaction_id,
                "field_path": field_path,
                "immutable_text": immutable_text,
                "speaker": speaker,
            }
        )
    ).hexdigest()
    return f"{trajectory_digest}:{content_digest}"


__all__ = [
    "RehydratedEvidence",
    "rehydrate_frozen_evidence",
    "project_trajectory_spans",
]
