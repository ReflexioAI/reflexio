"""The agent-turn gate and the role census must walk the same interactions.

``_trajectory_has_agent_turn`` decides whether a session carries agent
behaviour worth a discovery slot. ``unattributed_role_census`` explains why a
session that carries none was refused. They are two halves of one story, and
``_trajectory_has_agent_turn``'s docstring states the coupling verbatim:

    ``unattributed_role_census`` already handles the same shape by continuing
    past the malformed entry, and the gate and the census must agree about
    which sessions have agent turns.

The failure this file exists to prevent is the two drifting on MALFORMED
INPUT. Both must skip a bad entry at the smallest well-defined scope -- one
unreadable interaction says nothing about the interaction beside it, and one
unreadable request says nothing about its sibling requests. Only an unusable
``requests`` collection is a whole-session refusal, because at that point
there is no smaller unit left.

When they disagree, nothing raises and nothing looks wrong. The gate admits a
session on the strength of a late agent turn, the census abandons the same
trajectory at the first malformed entry and reports nothing, and the abstain
returns to being silent and plausible-looking -- which is the exact state the
census was built to remove.

The property below is written so that it CANNOT pass by both halves being
broken in the same direction. Each case asserts three things: the gate reached
the probe, the census reached the probe, and the two agree. A change that made
both functions abandon the whole trajectory would satisfy the agreement leg
and still fail the two absolute legs.
"""

from __future__ import annotations

from typing import cast

import pytest

from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceEvent,
    unattributed_role_census,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    _trajectory_has_agent_turn,
)

# A slot in a request. ``OK`` is a readable interaction -- a probe position the
# walk is REQUIRED to reach. ``BAD`` is an interaction missing both ``role``
# and ``content``, which each walk must skip in place.
OK = "OK"
BAD = "BAD"

# A whole request entry that carries no ``interactions`` key at all. Each walk
# must skip it and carry on to its siblings.
BAD_REQUEST = "BAD_REQUEST"

# The role the gate must see as an agent turn, and the role the census must
# report as unattributed. Neither is plumbing, so neither is suppressed.
AGENT_ROLE = "Agent"
UNATTRIBUTED_ROLE = "wizard"
# Attributed, non-agent, non-censused: inert filler for the non-probe slots.
FILLER_ROLE = "user"


# Malformed entries at every position -- first, middle, last, only -- at the
# interaction scope and at the request scope, and interleaved across both.
SHAPES: tuple[tuple[object, ...], ...] = (
    # Interaction scope, single request.
    ((OK,),),
    ((OK, OK),),
    ((BAD, OK),),  # malformed first
    ((OK, BAD),),  # malformed last
    ((OK, BAD, OK),),  # malformed middle
    ((BAD, OK, BAD),),  # malformed first and last
    ((BAD, BAD, OK),),  # a run of malformed entries before the probe
    # Request scope.
    (BAD_REQUEST, (OK,)),  # malformed request first
    ((OK,), BAD_REQUEST),  # malformed request last
    ((OK,), BAD_REQUEST, (OK,)),  # malformed request middle
    # Both scopes at once.
    (BAD_REQUEST, (BAD, OK)),
    ((BAD,), (OK,)),
    ((OK,), (BAD, OK)),
    ((BAD, OK), BAD_REQUEST, (OK, BAD)),
)


def _probe_positions(shape: tuple[object, ...]) -> tuple[tuple[int, int], ...]:
    """Every ``(request index, slot index)`` the walk is required to reach."""
    return tuple(
        (request_index, slot_index)
        for request_index, request in enumerate(shape)
        if request != BAD_REQUEST
        for slot_index, slot in enumerate(cast("tuple[str, ...]", request))
        if slot == OK
    )


def _build(
    shape: tuple[object, ...], probe: tuple[int, int], probe_role: str
) -> OpenWorldEvidenceEvent:
    """Materialise ``shape`` with ``probe_role`` at the probe position only."""
    requests: list[dict[str, object]] = []
    for request_index, request in enumerate(shape):
        if request == BAD_REQUEST:
            # No ``interactions`` key: unreadable at the request scope.
            requests.append({"request": {"request_id": f"request-{request_index}"}})
            continue
        interactions: list[dict[str, object]] = []
        for slot_index, slot in enumerate(cast("tuple[str, ...]", request)):
            if slot == BAD:
                # Neither ``role`` nor ``content``: unreadable at the
                # interaction scope.
                interactions.append({"interaction_id": f"{request_index}-{slot_index}"})
                continue
            role = probe_role if (request_index, slot_index) == probe else FILLER_ROLE
            interactions.append({"role": role, "content": "spoken"})
        requests.append(
            {
                "request": {"request_id": f"request-{request_index}"},
                "interactions": interactions,
            }
        )
    return _event({"session_id": "session-probe", "requests": requests})


def _event(trajectory: object) -> OpenWorldEvidenceEvent:
    return OpenWorldEvidenceEvent(
        exposure_event_id="exposure-probe",
        tenant_id="tenant-1",
        user_id="user-1",
        session_id="session-probe",
        user_playbook_id=7,
        served_semantic_digest=None,
        served_full_version_fingerprint=None,
        request_id="request-0",
        correlation_reason=None,
        exposed_at=1,
        ingested_at=2,
        trajectory=cast("None", trajectory),
        trajectory_digest=None,
        trajectory_digest_matches=True,
        trajectory_byte_count=0,
        trajectory_token_count=0,
        outcome="failure",
        outcome_id=None,
        outcome_revision=None,
        outcome_contract_digest=None,
        governance_subject_refs=(),
        governance_valid=True,
    )


@pytest.mark.parametrize("shape", SHAPES, ids=[str(shape) for shape in SHAPES])
def test_gate_and_census_reach_the_same_interactions(shape: tuple[object, ...]) -> None:
    """Every readable interaction is reachable by BOTH walks, malformed
    siblings notwithstanding.

    For each readable slot the shape declares, two trajectories are built that
    differ in exactly one byte of meaning: the probe slot's ``role``. With
    ``Agent`` there, the gate must return ``True``; with an unrecognised role
    there, the census must report it. Both answers are a direct readout of
    whether that walk reached the probe, so comparing them compares the two
    skip policies and nothing else.
    """
    probes = _probe_positions(shape)
    assert probes, f"shape declares no readable slot to probe: {shape}"

    for probe in probes:
        gate_saw_probe = _trajectory_has_agent_turn(_build(shape, probe, AGENT_ROLE))
        census = unattributed_role_census([_build(shape, probe, UNATTRIBUTED_ROLE)])
        census_saw_probe = census == ((UNATTRIBUTED_ROLE, 1),)

        assert gate_saw_probe is True, (
            f"the gate abandoned the walk before the readable turn at {probe} "
            f"in {shape}, so the session is excluded `no_agent_turn` under a "
            "reason that is not true of it"
        )
        assert census_saw_probe, (
            f"the census abandoned the walk before the readable turn at "
            f"{probe} in {shape}, so it under-reports the refused roles and "
            f"the abstain goes back to being silent; got {census}"
        )
        assert gate_saw_probe == census_saw_probe, (
            "the gate and the census disagree about whether the turn at "
            f"{probe} in {shape} exists: gate={gate_saw_probe}, "
            f"census={census}"
        )


def test_an_unusable_requests_collection_refuses_in_both_walks() -> None:
    """The one scope with no smaller unit left, agreed on by both halves."""
    for trajectory in (
        {"session_id": "s"},  # no `requests` key
        {"session_id": "s", "requests": 7},  # `requests` not iterable
    ):
        event = _event(trajectory)

        assert _trajectory_has_agent_turn(event) is False
        assert unattributed_role_census([event]) == ()


def test_skipping_a_malformed_entry_is_not_admitting_it() -> None:
    """Neither walk may invent a turn out of an entry it could not read."""
    event = _build(((BAD, BAD), BAD_REQUEST, (BAD,)), (-1, -1), AGENT_ROLE)

    assert _trajectory_has_agent_turn(event) is False
    assert unattributed_role_census([event]) == ()
