"""Refusal classes at the rehydration boundary, and empty-field projection.

``rehydrate_frozen_evidence`` must surface exactly two refusal classes: an
unsupported bundle schema, which no retry can load and so keeps its own class,
and every other mismatch, which becomes ``OpenWorldEvidenceRehydrationError``
(``REHYDRATION_FAILED``). Collapsing the first into the second is what once made
the runner report an unloadable bundle as a retryable infrastructure failure.
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import replace

import pytest

from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSchemaUnsupportedError,
    OpenWorldEvidenceSnapshot,
)
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    OpenWorldEvidenceRehydrationError,
    project_trajectory_spans,
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
)
from tests.server.services.offline_tuner.open_world.test_evidence_documents import (
    _event,
    _frozen,
)


def test_rehydration_refuses_look_alike_inputs() -> None:
    frozen, snapshot = _frozen()

    with pytest.raises(TypeError, match="frozen must be FrozenOpenWorldEvidence"):
        rehydrate_frozen_evidence(
            dataclasses.asdict(frozen),  # type: ignore[arg-type]
            snapshot,
        )
    with pytest.raises(TypeError, match="snapshot must be OpenWorldEvidenceSnapshot"):
        rehydrate_frozen_evidence(
            frozen,
            {"tenant_id": "tenant-1", "events": snapshot.events},  # type: ignore[arg-type]
        )


def test_an_unsupported_bundle_schema_keeps_its_own_refusal_class() -> None:
    frozen, snapshot = _frozen()
    stale = replace(
        frozen,
        bundle=replace(
            frozen.bundle, schema_version="offline-tuner-open-world-evidence-v1"
        ),
    )

    with pytest.raises(OpenWorldEvidenceSchemaUnsupportedError) as raised:
        rehydrate_frozen_evidence(stale, snapshot)
    # Not re-wrapped as an ordinary rehydration failure.
    assert not isinstance(raised.value, OpenWorldEvidenceRehydrationError)


def test_a_snapshot_for_another_tenant_is_a_rehydration_refusal() -> None:
    frozen, snapshot = _frozen()
    foreign = OpenWorldEvidenceSnapshot(tenant_id="tenant-2", events=snapshot.events)

    with pytest.raises(
        OpenWorldEvidenceRehydrationError,
        match="snapshot tenant does not match frozen evidence",
    ) as raised:
        rehydrate_frozen_evidence(frozen, foreign)
    assert raised.value.refusal is OpenWorldRefusal.REHYDRATION_FAILED
    # The original cause is chained, not discarded.
    assert type(raised.value.__cause__) is ValueError


def test_a_snapshot_event_without_its_trajectory_is_refused() -> None:
    frozen, snapshot = _frozen()
    events = list(snapshot.events)
    events[0] = replace(events[0], trajectory=None)

    with pytest.raises(
        OpenWorldEvidenceRehydrationError, match="snapshot trajectory is missing"
    ):
        rehydrate_frozen_evidence(
            frozen,
            OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=tuple(events)),
        )


def test_a_corrupt_bundle_derivation_is_a_rehydration_refusal() -> None:
    """A bundle whose recorded counts disagree with its own manifest is not the
    derivation of that manifest, so it cannot be rehydrated."""
    frozen, snapshot = _frozen()
    corrupt = replace(
        frozen,
        bundle=replace(
            frozen.bundle, failed_sessions=frozen.bundle.failed_sessions + 1
        ),
    )

    with pytest.raises(
        OpenWorldEvidenceRehydrationError, match="exact expected-manifest derivation"
    ):
        rehydrate_frozen_evidence(corrupt, snapshot)


def test_empty_content_and_empty_tool_lists_project_no_span() -> None:
    """An empty field carries nothing citable; projecting it would mint a span
    whose only quotable text is the empty string."""
    event = _event(session_id="session-1", outcome="failure", content="ask")
    trajectory = deepcopy(event.trajectory)
    assert trajectory is not None
    interactions = trajectory["requests"][0]["interactions"]
    interactions[0]["content"] = ""
    interactions[1]["tools_used"] = []

    spans = project_trajectory_spans(
        trajectory,
        trajectory_digest="1" * 64,
        tenant_id="tenant-1",
        role="discovery",
        user_id="user-1",
        session_id="session-1",
        include_tools=True,
    )

    assert [span.field_path for span in spans] == [
        "requests[0].interactions[1].content"
    ]
    assert spans[0].speaker == "agent"
    assert spans[0].span_id.startswith("1" * 64 + ":")


def test_span_ids_bind_role_text_and_speaker() -> None:
    """A span ID is an identity: a single changed input changes it."""
    event = _event(session_id="session-1", outcome="failure", content="ask")
    trajectory = event.trajectory
    assert trajectory is not None

    def span_ids(**changes: object) -> tuple[str, ...]:
        kwargs: dict[str, object] = {
            "trajectory_digest": "1" * 64,
            "tenant_id": "tenant-1",
            "role": "discovery",
            "user_id": "user-1",
            "session_id": "session-1",
        }
        kwargs.update(changes)
        source = kwargs.pop("trajectory", trajectory)
        return tuple(
            span.span_id
            for span in project_trajectory_spans(source, **kwargs)  # type: ignore[arg-type]
        )

    baseline = span_ids()
    assert baseline == span_ids()
    assert len(set(baseline)) == len(baseline)
    for changes in (
        {"tenant_id": "tenant-2"},
        {"role": "held_out"},
        {"user_id": "user-2"},
        {"session_id": "session-2"},
    ):
        assert set(span_ids(**changes)).isdisjoint(baseline), changes

    relabelled = deepcopy(trajectory)
    relabelled["requests"][0]["interactions"][1]["role"] = "user"
    changed = span_ids(trajectory=relabelled)
    # The user turn is untouched; the re-attributed turn gets a new identity.
    assert changed[0] == baseline[0]
    assert changed[1] != baseline[1]
