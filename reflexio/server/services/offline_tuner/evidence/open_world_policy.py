"""Fixed production policy for open-world evidence selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..retention import OPEN_WORLD_TRAJECTORY_RETENTION_SECONDS

_MAX_CANONICAL_TRAJECTORY_BYTES = 64_000
_MAX_CANONICAL_TRAJECTORY_TOKENS = 8_000

_MAX_ANALYST_CONTENT_TOKENS = 8_000
_MAX_ANALYST_CONTENT_BYTES_PER_TOKEN = 8

# Rendered-payload model. MEASURED on the live payload path -- one document
# through ``discovery_model_payload`` and ``_canonical_user_message`` -- at 0,
# 1, 10, 40, 100, 150 and 300 spans, with the production 129-character
# ``span_id``, against SHORT identifiers and against realistic UUID ones.
#
# RE-MEASURED after constant span identity was hoisted to the document.
# Single-character turns, short identifiers:
#
#     spans     0      1       10      40        100
#     before    671    1 066   4 603   16 428    40 132     ~394 per span
#     after     677      964   3 494   11 934    28 875     ~281 per span
#
# The hoist is worth ~1.5x per span, NOT the order of magnitude this comment
# previously anticipated, and the reason is worth recording so nobody re-does
# the measurement expecting the larger number: what dominates a span is now the
# 129-character ``span_id``, which alone renders ~142 tokens once base64 has
# expanded it. Only hoisting the document-constant ``trajectory_digest`` half of
# that id out of the payload would reach 3x, and that is a deliberate
# non-change -- the citation repair ladder resolves against the full id.
#
# Flat per span, as it must be: ``_spans_for_event`` emits one span per
# non-empty interaction and each still carries its own ``span_id``,
# ``request_id`` and ``field_path``. Content costs a further 0.944 tokens per
# content BYTE, because ``_canonical_user_message`` base64-encodes the whole
# payload and base64 tokenizes far worse than the prose it wraps.
#
# The constants below round those measurements UP, so the estimate is a sound
# upper bound on what the provider actually receives. A capacity gate that
# under-counts is the one failure mode that matters here, since nothing
# downstream re-checks. Verified as an upper bound across 56 shapes -- both
# identifier widths, narrow and wide ``field_path``/``request_id``, 1-byte and
# 480-byte turns, 0 to 300 spans -- with a worst-case overshoot of 1.11x.
#
# ``_ANALYST_PAYLOAD_DOCUMENT_TOKENS`` is one payload's fixed framing (bundle
# digest, outcome contract, envelope) charged once per SESSION rather than once
# per payload. That over-counts a multi-session bundle by a few hundred tokens
# per session, which is again the safe direction and keeps the estimate a
# function of the session alone -- the only unit the frozen manifest carries.
# It moved 700 -> 750 because the old value was fitted to short identifiers
# only and already under-counted a realistic UUID-identified session's framing
# (measured 727) before this change.
#
# THESE TRACK THE RENDERER AND MUST MOVE WITH IT.
# A test re-measures the live path and fails if the estimate drifts outside
# [1.0, 1.15] x measured, so the constants cannot silently go stale.
_ANALYST_PAYLOAD_DOCUMENT_TOKENS = 750
_ANALYST_PAYLOAD_SPAN_TOKENS = 300
_ANALYST_PAYLOAD_CONTENT_TOKENS_PER_BYTE = 1

# A BACKSTOP, NOT A TIGHT FIT -- and deliberately so at this rendering.
#
# NOT VERIFIED AGAINST THE PROVIDER. The route is `minimax/MiniMax-M3` and its
# real input window is not confirmed anywhere in this repository, so this
# number is not derived from one.
#
# It is sized against what the sessions this change exists to READMIT actually
# cost at today's rendering. A long session of ~150 interactions can carry
# about 5,000 conversation tokens yet render ~80,000 tokens as one call. A
# number chosen against a context window -- 64,000, say -- would refuse exactly
# that session, reintroducing the
# exclusion this change removes, one layer down. So the budget sits above the
# whole measured envelope of a legal session and bounds only what could not fit
# any plausible window: a discovery arm of five maximal sessions (~315,000) and
# a session of a thousand single-character turns (~490,000), which measures
# under 5,000 conversation tokens and passes every content meter.
#
# STILL TO BE RE-DERIVED, and by LESS than this note used to promise. Constant
# span identity has now been hoisted to the document, and it bought ~1.5x per
# span (394 -> 281 measured), not the order of magnitude anticipated above: the
# 129-character ``span_id`` is what dominates a span now, and it stays. So the
# ~80 000-token session that motivated this budget still estimates ~65 000
# tokens, and a context-window-shaped number such as 64 000 would still refuse
# it. Tightening remains affordable only alongside recording which sessions the
# budget excluded, rather than refusing the whole bundle -- a separate change.
_MAX_BUNDLE_ANALYST_PAYLOAD_TOKENS = 250_000


def trajectory_input_within_limits(*, byte_count: int, token_count: int) -> bool:
    """Whether a stored canonical trajectory fits the fixed storage-row limits.

    This measures the STORAGE ROW -- fifteen fields per interaction, including
    timestamps, identifiers and tool metadata. It is no longer what decides
    whether a session is usable as evidence; ``analyst_payload_within_limits``
    is. It survives as the stand-in for a session whose content the byte cap
    already discarded. This legacy direct-input predicate does not control the
    production loader: ``PRODUCTION_OPEN_WORLD_READ_LIMITS.max_trajectory_bytes``
    now permits 16 MiB of original source for bounded compaction.
    """
    return (
        byte_count <= _MAX_CANONICAL_TRAJECTORY_BYTES
        and token_count <= _MAX_CANONICAL_TRAJECTORY_TOKENS
    )


def analyst_payload_within_limits(
    *, content_byte_count: int, content_token_count: int
) -> bool:
    """Whether the conversation an analyst receives fits the fixed limits.

    Measured on the two fields the analyst is ever shown -- ``content`` and
    ``role`` -- rather than on the storage row. Conversation content is 20-23%
    of trajectory tokens, so applying the same 8,000-token budget to the row
    really admitted only about 1,700 tokens of conversation. The budget does
    not move; the object it is applied to does.

    THE BYTE CAP IS RETAINED, AND IT IS A WHITESPACE GUARD, NOT A SIZE CAP.
    Re-pointing the meter at content is exactly what would remove it: 60,000
    bytes of spaces measure ~470 content tokens and would sail through a
    token-only budget. It is expressed as a DENSITY, at most
    ``_MAX_ANALYST_CONTENT_BYTES_PER_TOKEN`` bytes per token, because a flat
    64,000-byte ceiling would admit that same whitespace unchanged. Measured
    bytes per token: spaces 125, newlines 31.9, prose 4.37, CJK 3.33, dense
    JSON 1.67, identifiers 1.30. Only whitespace clears 8.0, which is why the
    cap binds on nothing else. Together with the token budget the density also
    implies an absolute ceiling of 64,000 content bytes -- the same number the
    storage-row cap used, now applied to the object that has to fit.
    """
    return (
        content_token_count <= _MAX_ANALYST_CONTENT_TOKENS
        and content_byte_count
        <= _MAX_ANALYST_CONTENT_BYTES_PER_TOKEN * content_token_count
    )


def analyst_payload_render_tokens(*, span_count: int, content_byte_count: int) -> int:
    """Upper bound on the tokens one session costs in the rendered payload.

    A DIFFERENT OBJECT FROM ``analyst_payload_within_limits``, and the two are
    easy to conflate. That predicate measures conversation content and asks
    whether one session is pathological. This measures what the analyst is
    actually sent -- spans, their framing, and the base64 envelope around all
    of it -- and feeds the question of whether the whole BUNDLE fits.

    Content tokens are deliberately not an input. A session of 8 000
    single-character turns measures roughly 8 000 content tokens and passes any
    content-based meter, while rendering one framed span per turn: the span
    count, not the conversation, is what makes that case unbounded.
    """
    return (
        _ANALYST_PAYLOAD_DOCUMENT_TOKENS
        + _ANALYST_PAYLOAD_SPAN_TOKENS * span_count
        + _ANALYST_PAYLOAD_CONTENT_TOKENS_PER_BYTE * content_byte_count
    )


def bundle_within_analyst_payload_budget(*, render_token_count: int) -> bool:
    """Whether one analyst call's rendered payload fits the fixed budget.

    THE UNIT IS A CALL, NOT A SESSION AND NOT THE WHOLE BUNDLE. Discovery is
    handed its entire arm in one payload; held-out and confirmation each take a
    single document. So the caller checks the discovery arm's SUM and every
    other session on its own -- which is what a per-session cap could not do in
    either direction: it dropped one rich session while the arm had room, and
    it admitted several individually legal sessions that together did not fit.
    """
    return render_token_count <= _MAX_BUNDLE_ANALYST_PAYLOAD_TOKENS


@dataclass(frozen=True)
class _OpenWorldPolicy:
    historical_window_seconds: int
    maximum_population: int
    maximum_evidence_sessions: int
    minimum_failed_sessions: int
    minimum_successful_sessions: int
    #: How many sessions of ONE outcome class the validation arms may hold.
    #:
    #: Held-out carries two classes (failure, success) and safety-only carries
    #: one (unknown), so the worst case is three times this number of sessions
    #: -- and every one of them is a SERIAL provider call
    #: (``OpenWorldPhase3Runner._load_or_run_held_out`` loops one call per
    #: session, with no batching). Derived from what one attempt can afford:
    #:
    #:     held-out / safety-only session   = 1 serial provider call ~= 56s
    #:     attempt lease                    = 3,600s
    #:                       3,600 / 56     ~= 64 sessions, the hard ceiling
    #:
    #: 64 is where the lease EXPIRES, not a budget, and the attempt also has to
    #: pay for discovery, the rewrite and the judge inside the same lease --
    #: while holding the fleet's single serialized tick, since a tick blocks on
    #: the whole attempt. So the arms get about half the lease: 3 * 8 = 24
    #: sessions, ~1,344s, leaving ~2,256s for everything else and for provider
    #: backoff.
    #:
    #: NOTHING BOUNDED THESE ARMS BEFORE. Discovery has a session cut (at most
    #: ``minimum_failed_sessions`` + ``minimum_successful_sessions`` = 5) and a
    #: payload budget; held-out received everything discovery did not take, and
    #: safety-only received the whole ``unknown`` set, so both grew with the
    #: window. ``maximum_population`` was not a substitute: it counts EVENTS,
    #: and an org with many requests per session can turn a per-playbook
    #: population of 1,000 exposures into 80+ sessions -- already past the
    #: lease.
    #:
    #: A SECOND REASON, AND IT IS NARROWER THAN IT WAS. Publication is decided
    #: over this same population, so N used to drive P(publish) down as p**N --
    #: a better-instrumented org became LESS able to publish, the incentive
    #: exactly inverted. The two non-affirming values are now split:
    #: ``unresolved`` is now held to a 75 percent coverage floor rather than to
    #: unanimity, which removes the decay ON THAT AXIS. It does NOT remove it
    #: on the other -- ``expanded`` / ``blocker`` / ``refute`` remain "an
    #: absolute existential veto, at any N" (``reducer.py``), so a larger arm
    #: is still strictly more ways to lose.
    #:
    #: So do not read this as the load-bearing argument; the LEASE is. This one
    #: survives that split only for adverse findings, and the cap would be
    #: right on cost alone.
    #:
    #: PER OUTCOME CLASS, not per arm and not over the union, so that no class
    #: can evict another. A partial outage produces hundreds of ``unknown``
    #: sessions; under a shared budget those would starve held-out of the
    #: failure and success evidence the validation arm exists to carry.
    maximum_validation_sessions_per_outcome: int


PRODUCTION_OPEN_WORLD_POLICY = _OpenWorldPolicy(
    historical_window_seconds=OPEN_WORLD_TRAJECTORY_RETENTION_SECONDS,
    maximum_population=1_000,
    maximum_evidence_sessions=5,
    minimum_failed_sessions=3,
    minimum_successful_sessions=2,
    maximum_validation_sessions_per_outcome=8,
)


@dataclass(frozen=True)
class OpenWorldEvidenceReadLimits:
    """The numbers the storage snapshot read is bounded by, handed in by the caller.

    Storage executes the read; it does not own the policy. These are the three
    values the read needs from it, so storage imports none of the policy module
    and a test can bound a read without patching a module constant.

    Attributes:
        historical_window_seconds (int): Width of the half-open evidence window
            ending at ``frozen_at``.
        maximum_population (int): Exposure rows above which the snapshot is
            returned as ``population_limit_exceeded`` instead of being built.
        max_trajectory_bytes (int): The storage-row BYTE cap the trajectory
            loader applies mid-stream; content over it is discarded while the
            rows are still arriving.
    """

    historical_window_seconds: int
    maximum_population: int
    max_trajectory_bytes: int


#: Derived from ``PRODUCTION_OPEN_WORLD_POLICY`` and the storage-row byte cap,
#: never restated, so the read limits cannot drift from the policy they carry.
PRODUCTION_OPEN_WORLD_READ_LIMITS = OpenWorldEvidenceReadLimits(
    historical_window_seconds=PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds,
    maximum_population=PRODUCTION_OPEN_WORLD_POLICY.maximum_population,
    # Read/memory safety is separate from the rendered analyst budget.
    max_trajectory_bytes=16 * 1024 * 1024,
)


def coverage_passes(numerator: int, denominator: int) -> bool:
    """Return whether exact counts meet the fixed 75 percent requirement."""
    return denominator > 0 and numerator * 4 >= denominator * 3


def discovery_session_cut(*, available_sessions: int, minimum_sessions: int) -> int:
    """Return how many sessions of one outcome class discovery may take.

    Discovery scales with the evidence that exists instead of taking a fixed
    pair, but held-out is the validation arm, so discovery may only grow into
    sessions held-out does not need. Two derived bounds, both taken from the
    eligibility minimum this class already has -- no independent knob:

    * discovery takes at most ``minimum_sessions`` of the class, the same count
      eligibility already demands before anything may be concluded at all;
    * held-out always retains at least ``minimum_sessions - 1`` of the class.

    At the eligibility floor both terms collapse to one session per class,
    which is the fixed one-failure/one-success partition this rule supersedes.
    Below the floor the class yields zero; the bundle is ineligible anyway and
    the manifests stay complete and disjoint.

    The result depends only on counts, never on ordering, trajectory sizes, or
    a seed, so the partition stays reproducible from the manifest alone.

    RAISING AN ELIGIBILITY MINIMUM ALSO RESIZES DISCOVERY. Deriving both bounds
    from ``minimum_sessions`` is what keeps discovery from becoming an
    independent tunable, but it couples two decisions that are conceptually
    separate: "how much evidence before we tune at all" and "how much of it the
    discovery analyst sees". Raising ``minimum_failed_sessions`` to 10 to demand
    more evidence before tuning would silently grow discovery to 10 failed
    sessions and hand the analyst a much larger prompt. If you raise a minimum
    for the eligibility reason alone, decide deliberately whether discovery
    should follow it, and split the two into separately-derived quantities if
    it should not.
    """
    held_out_floor = minimum_sessions - 1
    return max(0, min(minimum_sessions, available_sessions - held_out_floor))


__all__ = [
    "PRODUCTION_OPEN_WORLD_POLICY",
    "PRODUCTION_OPEN_WORLD_READ_LIMITS",
    "OpenWorldEvidenceReadLimits",
    "coverage_passes",
    "discovery_session_cut",
]
