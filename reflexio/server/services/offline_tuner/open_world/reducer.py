"""Deterministic reduction of persisted held-out/safety memos to one decision.

Per-session analyst labels are immutable reducer inputs. This is the sole
attempt-level authority over the held-out/safety population and never
reinterprets a ``refute`` or ``insufficient`` memo as support.

Owner ruling (Fix Round 1): membership must bind the complete frozen
manifest session identity, including its trajectory digest, and every memo
must be bound to the exact candidate under decision -- not just to the
correct ``(manifest_role, user_id, session_id)`` triple. ``PersistedSessionMemo``
carries that candidate binding; a bare ``HeldOutEnvelope`` cannot prove it
(the envelope has no candidate identity field).

ADVERSE FINDINGS VETO; UNINFORMATIVE ONES ARE COVERAGE-FLOORED. An analyst
that affirmatively reports ``refute``, ``expanded`` or ``blocker`` abstains the
attempt on its own, at any population size, in either arm -- unchanged. An
analyst that reports ``unresolved`` has said only that it could not tell, and
that is held to the shared 75 percent ``coverage_passes`` floor instead of
being an individual veto -- except where citation repair FORCED that value, for
which see the note above ``_reports_refutation``. See
``_assessments_sufficiently_resolved`` for why the distinction is load-bearing
once the population scales, and for the two properties that bound the change
(never stricter than what it replaced; identical to it for N <= 3, which is a
narrower claim than "unchanged at the eligibility floor").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceBundle,
    OpenWorldEvidenceSession,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    coverage_passes,
)

from .identity import canonical_bytes
from .models import (
    AbstainDecision,
    AcceptDecision,
    AttemptDecisionV1,
    CandidateEnvelope,
    HeldOutEnvelope,
)

_MembershipKey = tuple[str, str, str]


@dataclass(frozen=True)
class PersistedSessionMemo:
    """One durably persisted held-out/safety memo, bound to the exact
    candidate it was evaluated against."""

    envelope: HeldOutEnvelope
    candidate_digest: str

    def __post_init__(self) -> None:
        if type(self.envelope) is not HeldOutEnvelope:
            raise TypeError("envelope must be a HeldOutEnvelope")
        if (
            type(self.candidate_digest) is not str
            or len(self.candidate_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.candidate_digest)
        ):
            raise ValueError("candidate_digest must be lowercase SHA-256 hex")


def candidate_digest(candidate: CandidateEnvelope) -> str:
    """The canonical identity digest of one candidate envelope."""
    if type(candidate) is not CandidateEnvelope:
        raise TypeError("candidate must be a CandidateEnvelope")
    return sha256(canonical_bytes(candidate)).hexdigest()


def held_out_population_digest(memos: Sequence[PersistedSessionMemo]) -> str:
    """The canonical identity digest of one complete held-out/safety population."""
    ordered = sorted(
        memos,
        key=lambda m: (
            m.envelope.manifest_role,
            m.envelope.user_id,
            m.envelope.session_id,
        ),
    )
    payload = {
        "contract_type": "open_world_held_out_population",
        "schema_version": "v1",
        "payload": [
            sha256(canonical_bytes(item.envelope)).hexdigest() for item in ordered
        ],
    }
    return sha256(canonical_bytes(payload)).hexdigest()


def reduce_held_out(
    *,
    bundle: OpenWorldEvidenceBundle,
    candidate: CandidateEnvelope,
    memos: Sequence[PersistedSessionMemo],
) -> AttemptDecisionV1:
    """Reduce the complete persisted held-out/safety population to one decision.

    Args:
        bundle: The exact frozen evidence bundle the attempt is scoped to.
        candidate: The one generated candidate the memos were evaluated against.
        memos: Every persisted held-out and safety-only session memo for the
            attempt, each bound to ``candidate``. Must cover the frozen
            manifest exactly once each, including its exact trajectory
            identity.

    Returns:
        ``AcceptDecision`` when the held-out population supports the
        candidate; otherwise ``AbstainDecision`` with reason
        ``heldout_evidence_failed``.

    Raises:
        TypeError: If ``bundle``, ``candidate``, or a memo has the wrong type.
        ValueError: If the candidate or a memo does not bind the exact
            evidence bundle, a memo does not bind the exact candidate under
            decision, a memo's trajectory digest does not match the frozen
            manifest session, or ``memos`` does not cover the frozen
            held-out/safety manifest exactly once each.
    """
    if type(bundle) is not OpenWorldEvidenceBundle:
        raise TypeError("bundle must be an OpenWorldEvidenceBundle")
    if type(candidate) is not CandidateEnvelope:
        raise TypeError("candidate must be a CandidateEnvelope")
    memos = tuple(memos)
    for memo in memos:
        if type(memo) is not PersistedSessionMemo:
            raise TypeError("every memo must be a PersistedSessionMemo")
    if candidate.bundle_digest != bundle.digest:
        raise ValueError("candidate does not bind the exact evidence bundle")

    expected_candidate_digest = candidate_digest(candidate)
    expected = _expected_membership(bundle)
    _require_complete_population(
        memos,
        bundle_digest=bundle.digest,
        candidate_digest=expected_candidate_digest,
        expected=expected,
    )

    envelopes = tuple(item.envelope for item in memos)
    if any(_reports_refutation(envelope) for envelope in envelopes):
        return AbstainDecision(decision="abstain", reason="heldout_evidence_failed")
    if any(_reports_scope_expansion(envelope) for envelope in envelopes):
        return AbstainDecision(decision="abstain", reason="heldout_evidence_failed")
    if any(_reports_safety_blocker(envelope) for envelope in envelopes):
        return AbstainDecision(decision="abstain", reason="heldout_evidence_failed")
    if not _assessments_sufficiently_resolved(envelopes):
        return AbstainDecision(decision="abstain", reason="heldout_evidence_failed")

    if _has_recurrence_support(envelopes, expected):
        return AcceptDecision(decision="accept")
    return AbstainDecision(decision="abstain", reason="heldout_evidence_failed")


def _expected_membership(
    bundle: OpenWorldEvidenceBundle,
) -> dict[_MembershipKey, OpenWorldEvidenceSession]:
    expected: dict[_MembershipKey, OpenWorldEvidenceSession] = {}
    for role, manifest in (
        ("held_out", bundle.held_out),
        ("safety_only", bundle.safety_only),
    ):
        for session in manifest.sessions:
            key = (role, session.user_id, session.session_id)
            if key in expected:
                raise ValueError(
                    "evidence bundle contains a duplicate manifest session"
                )
            expected[key] = session
    return expected


def _require_complete_population(
    memos: tuple[PersistedSessionMemo, ...],
    *,
    bundle_digest: str,
    candidate_digest: str,
    expected: dict[_MembershipKey, OpenWorldEvidenceSession],
) -> None:
    seen: set[_MembershipKey] = set()
    for item in memos:
        envelope = item.envelope
        if envelope.bundle_digest != bundle_digest:
            raise ValueError("held-out memo does not bind the exact evidence bundle")
        if item.candidate_digest != candidate_digest:
            raise ValueError(
                "held-out memo does not bind the exact candidate under decision"
            )
        key = (envelope.manifest_role, envelope.user_id, envelope.session_id)
        session = expected.get(key)
        if session is None:
            raise ValueError(
                "held-out memo is not a member of the frozen held-out/safety manifest"
            )
        if key in seen:
            raise ValueError("held-out memo duplicates a manifest session")
        if envelope.trajectory_digest != session.trajectory_digest:
            raise ValueError(
                "held-out memo does not bind the exact frozen trajectory identity"
            )
        seen.add(key)
    if seen != set(expected):
        raise ValueError("held-out reduction requires the complete manifest population")


def _reports_refutation(envelope: HeldOutEnvelope) -> bool:
    """The analyst refuted the candidate, before or after citation repair."""
    return envelope.memo.control == "refute" or envelope.observed_memo.control == (
        "refute"
    )


def _reports_scope_expansion(envelope: HeldOutEnvelope) -> bool:
    """One analyst affirmatively found the candidate widened the rule's scope.

    Owner ruling (C2) is PRESERVED here: the gate still applies regardless of
    ``control``, so an ``insufficient`` memo that nonetheless reports
    ``expanded`` still abstains the whole attempt. What changed is only which
    VALUE trips it -- see ``_assessments_sufficiently_resolved``.
    """
    return (
        envelope.memo.scope_assessment == "expanded"
        or envelope.observed_memo.scope_assessment == "expanded"
    )


def _reports_safety_blocker(envelope: HeldOutEnvelope) -> bool:
    """One analyst affirmatively found a safety problem.

    Existential and control-independent, exactly as before: a single ``blocker``
    anywhere in either arm abstains the attempt at any population size. This is
    the gate that guards the automatic, terminal rewrite, and it is NOT
    thresholded -- see ``_assessments_sufficiently_resolved``.
    """
    return (
        envelope.memo.safety_assessment == "blocker"
        or envelope.observed_memo.safety_assessment == "blocker"
    )


# WHY THE ADVERSE GATES READ ``observed_memo`` TOO.
#
# ``unresolved`` has TWO provenances, and only one of them is "the analyst
# could not tell". ``build_held_out_envelope`` (``models.py``) rewrites a memo
# whose citations fail verification -- and any citation-less ``support`` or
# ``refute`` -- to ``control="insufficient"`` with BOTH assessments forced to
# ``unresolved``. So an observed ``refute``, ``expanded`` or ``blocker`` whose
# citation could not be verified arrives here looking exactly like an
# inconclusive judgement.
#
# Under the unanimous rule that erasure was harmless: the forced ``unresolved``
# vetoed anyway, so the memo stayed fail-closed by accident. Once
# ``unresolved`` is merely coverage-floored, a citation-invalid ADVERSE finding
# would become absorbable inside the 25 percent budget -- i.e. relaxing the
# floor would have silently discarded observed refutations and safety blockers.
# Reading the pre-repair observation keeps that class fail-closed.
#
# NOT the stricter alternative of vetoing on ANY validation-derived
# ``unresolved``: ungrounded citations are an ordinary LLM failure mode, so an
# existential gate over that whole class would restore the p^N decay this
# change exists to remove. The distinction drawn here is narrower and is the
# one that matters -- an adverse OBSERVATION is never discarded, while an
# ungrounded non-adverse memo stays merely uninformative.


def _assessments_sufficiently_resolved(
    envelopes: tuple[HeldOutEnvelope, ...],
) -> bool:
    """Enough of the population positively affirmed scope and safety.

    WHY THIS IS NOT ANOTHER EXISTENTIAL GATE, AND WHY THAT IS NOT A RELAXATION
    OF THE SAFETY MODEL.

    ``scope_assessment`` and ``safety_assessment`` are THREE-valued, and the two
    non-affirming values mean opposite things:

    * ``expanded`` / ``blocker`` -- ADVERSE. The analyst looked and found a
      problem. Still an absolute existential veto, at any N, above.
    * ``unresolved``             -- UNINFORMATIVE. The analyst could not tell.
      Absence of evidence, not evidence of harm.

    Gating on ``!= "preserved"`` / ``!= "clear"`` lumped the second into the
    first, which is how a binary negation of a three-valued enum fails. Note
    that ``control`` -- the third channel, written on the same memo -- already
    draws this distinction correctly: ``refute`` vetoes, ``insufficient`` does
    not. This restores the same shape to the other two fields rather than
    inventing a new policy for them.

    IT MATTERS BECAUSE BOTH ARMS SCALE. Held-out receives every session
    discovery does not take, and discovery is capped at
    ``minimum_failed_sessions + minimum_successful_sessions`` = 5; safety-only
    is the whole ``unknown`` arm, which the outcome mapping writes liberally
    (every environment failure, every unrecognised taxonomy value, every null
    verdict). So the population runs from 3 at the eligibility floor to nearly
    ``maximum_population``. Under an existential rule over an uninformative
    value, P(publish) decays as p^N: a 90-session window at a 95 percent
    per-memo affirm rate published about 1 percent of the time, against 86
    percent for a 3-session one. A better-instrumented org became LESS able to
    publish -- the incentive exactly inverted.

    The floor is the shared 75 percent ``coverage_passes`` test the selection
    layer already applies to manifest coverage, not a new knob. Two properties
    are worth stating because they bound the change:

    * It is NEVER stricter than what it replaces. A population that passed the
      old rule had zero non-affirming memos, which is 100 percent coverage.
    * It is IDENTICAL to the old rule for N <= 3, because 75 percent of 3
      rounds up to 3.

    THE SECOND IS NOT THE SAME CLAIM AS "nothing changes at the eligibility
    floor", and the difference is easy to get wrong. The minimum-eligible
    bundle (3 failed / 2 successful) leaves held-out with 3 memos, but ANY
    unknown-outcome session in the window adds a ``safety_only`` member, and at
    N = 4 ``coverage_passes(3, 4)`` is 12 >= 12 -- one uninformative memo is
    already tolerated where unanimity abstained. The reducer population is 3
    only when the window holds no unknown-outcome session at all.
    """
    total = len(envelopes)
    # JOINTLY, not two independent counters. Counting the fields separately
    # admits twice the uncertainty the 25 percent bound claims: with 20 memos,
    # five unresolved on scope ALONE and five different ones unresolved on
    # safety ALONE leave both counters at 15/20 and pass, while only 10/20
    # memos carry a complete assessment. The bound this gate advertises is
    # "at most a quarter of the population failed to assess the candidate", so
    # the unit it counts has to be the MEMO, not the field.
    fully_affirming = sum(
        1
        for envelope in envelopes
        if envelope.memo.scope_assessment == "preserved"
        and envelope.memo.safety_assessment == "clear"
    )
    return coverage_passes(fully_affirming, total)


def _has_recurrence_support(
    envelopes: tuple[HeldOutEnvelope, ...],
    expected: dict[_MembershipKey, OpenWorldEvidenceSession],
) -> bool:
    """The failed/successful held-out pair that positively carries an accept.

    THE CARRYING MEMOS MUST THEMSELVES BE FULLY RESOLVED. The population-level
    floor in ``_assessments_sufficiently_resolved`` tolerates a minority of
    ``unresolved`` assessments; it must not tolerate them in the two memos the
    acceptance actually rests on. Without this, publication could rest on two
    analysts that supported the hypothesis while being unable to say whether
    scope was preserved -- a hole the previous unanimous rule closed only as a
    side effect of abstaining on any ``unresolved`` at all.

    This costs nothing in population terms: it constrains exactly TWO memos, so
    it cannot reintroduce the p^N decay. It also keeps the whole change no
    stricter than what it replaced -- under the old rule every memo in an
    accepted population was already ``preserved``/``clear``, these two
    included.
    """
    supports_failed = False
    supports_success = False
    for envelope in envelopes:
        if envelope.manifest_role != "held_out" or envelope.memo.control != "support":
            continue
        if (
            envelope.memo.scope_assessment != "preserved"
            or envelope.memo.safety_assessment != "clear"
        ):
            continue
        outcome = expected[
            (envelope.manifest_role, envelope.user_id, envelope.session_id)
        ].outcome
        if outcome == "failure":
            supports_failed = True
        elif outcome == "success":
            supports_success = True
    return supports_failed and supports_success


__all__ = [
    "PersistedSessionMemo",
    "candidate_digest",
    "held_out_population_digest",
    "reduce_held_out",
]
