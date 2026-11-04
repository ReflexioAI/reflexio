"""Pure records and canonical manifests for open-world evidence selection."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from reflexio.models.api_schema.domain.entities import Sha256Digest

from ..open_world_invocations import OpenWorldRefusal, OpenWorldRefusalError
from .manifests import canonical_json_bytes
from .open_world_policy import (
    PRODUCTION_OPEN_WORLD_POLICY,
    bundle_within_analyst_payload_budget,
    coverage_passes,
    trajectory_input_within_limits,
)

# Durable reducer semantics, shared by worker lease admission and envelopes.
# Change this identity when adverse-veto, citation-repair or coverage rules change.
REDUCER_POLICY_ID = "adverse-veto-coverage-75-v1"

# Source sufficiency and exact application replace completion-label recurrence
# for the promoted rule-evidence method. The old identity remains readable only
# for historical artifacts until the native runner/SQL contract is advanced.
RULE_EVIDENCE_POLICY_ID = "evidence-grounded-exact-edit-v1"

if TYPE_CHECKING:
    from reflexio.server.services.storage.session_outcome_identity import (
        CanonicalSessionTrajectory,
    )

#: Why an exposure could not be tied to the request that produced it.
#:
#: ``session_only`` is deliberately DISTINCT from ``missing_request``, which it
#: was previously reported as. The two send an operator to opposite places. A
#: genuinely uncorrelated exposure carries neither id and is the client's to
#: fix. A session-only exposure carries enough: the exposure row holds a
#: ``session_id``, and ``requests`` records one too, so the row is correlatable
#: in principle and is refused only because the resolver's single lookup is by
#: ``request_id``. Reporting that as ``missing_request`` hides a server-side
#: gap behind what reads as a client bug, and it is the difference between
#: "your agent is not sending an id" and "we are not using the id it sent".
CorrelationReason = Literal["missing_request", "correlation_mismatch", "session_only"]

Outcome = Literal["success", "failure", "unknown"]
EvidenceRole = Literal["discovery", "held_out", "safety_only"]
TurnSpeaker = Literal["agent", "user", "other"]

#: What a session refused for its SIZE is called, wherever capacity still has
#: to read as a non-covered entry.
#:
#: ``reconstructability_reason`` deliberately does NOT return it: a session
#: that is merely large is intact, and counting it as unreconstructable made a
#: governance-clean corpus fail the 75 percent eligibility gate on size alone.
#: The string survives in the two places capacity is still an exclusion -- the
#: bundle's ``session_exclusion_reasons``, because an over-size conversation
#: cannot be rendered into an analyst payload, and the day-14 comparability
#: path, where it keeps its pre-split meaning so the confirm/revert verdict
#: does not move. ``open_world_selection.is_capacity_excluded`` is the
#: predicate that carries the fact; this is only its name.
CAPACITY_EXCLUSION_REASON = "trajectory_input_limit_exceeded"

#: An exposure whose request lies AFTER its session's outcome cutover.
#:
#: The session's outcome was inferred from a verdict over the prefix through
#: that cutover, so this exposure's turn -- the playbook it served, and
#: whatever came of it -- is not part of what the outcome describes. Its
#: reaction has not been judged yet. That is neither a failure to reconstruct
#: nor evidence: the exposure is NEUTRAL to both coverage ratios (dropped from
#: numerator and denominator alike, ``coverage_for_reasons``) and never becomes
#: a session in the bundle. Counting it as a failure would let a busy live
#: session drag a healthy target under the 75 percent gate; counting it as
#: covered would credit an outcome to a turn it never saw. A later extension
#: moves the cutover past it and it is evidence on the next read.
AWAITING_REACTION_REASON = "awaiting_reaction"

#: Reasons that leave a coverage ratio untouched. See ``AWAITING_REACTION_REASON``.
COVERAGE_NEUTRAL_REASONS: frozenset[str] = frozenset({AWAITING_REACTION_REASON})

#: The abstain for a playbook revised after every exposure in its window.
#:
#: Every persisted field feeds the full-version fingerprint, so a human edit,
#: the auto-tagger or the governance backfill re-fingerprints the playbook and
#: every earlier exposure then mismatches. Where post-edit traffic exists the
#: selection starts the window at it (``version_start_bound``); where none does
#: yet, this replaces the categorical
#: ``incumbent_full_version_fingerprint_mismatch``, which reads as a fault and
#: used to hold tuning off for as long as one pre-edit exposure stayed inside
#: the 14-day scan. The sweep records it as
#: ``ineligible_incumbent_revised_awaiting_evidence``, an ``ineligible_*`` label
#: the same-day re-attempt can re-admit once new traffic arrives.
INCUMBENT_REVISED_AWAITING_EVIDENCE_REASON = "incumbent_revised_awaiting_evidence"

#: The abstain for a coverage gate with NOTHING to count: a ``0/0`` ratio.
#:
#: ``coverage_passes`` is false for an empty denominator, so before this label
#: a ``0/0`` landed on ``insufficient_{global,target}_reconstructability`` --
#: both evidence FAULTS -- although nothing failed to reconstruct. A ``0/0``
#: has exactly two causes, and neither is a fault:
#:
#: * the publication / version-start window left no events in the manifest
#:   (``select_open_world_evidence`` narrows the scan before either ratio is
#:   computed), or no event of THIS playbook survived it; or
#: * every remaining event is ``awaiting_reaction``, which
#:   ``COVERAGE_NEUTRAL_REASONS`` drops from both sides of the ratio.
#:
#: Either way the honest answer is "wait for counted evidence". A per-event
#: failure (``correlation_mismatch``, ``governance_invalid``, a missing
#: request, ...) always lands in the denominator, so a real reconstruction
#: fault can never produce a ``0/0`` -- a NON-empty ratio below the floor keeps
#: its fault label. Deliberately NOT in ``EVIDENCE_FAULT_REASONS``: recorded
#: as ``ineligible_awaiting_counted_evidence``, it is a routine INFO tick the
#: same-day re-attempt can re-admit once a session moves.
AWAITING_COUNTED_EVIDENCE_REASON = "awaiting_counted_evidence"

#: Bundle abstains that mean the evidence DATA is broken, not merely thin.
#:
#: The categorical fences: a version fingerprint that is missing, malformed, or
#: contradicted beside a current-version exposure; exposures with no tenant,
#: timestamp or stable playbook id; a population over the policy's scan bound.
#:
#: AND the two coverage abstains. ``open_world_selection``'s reason ordering is
#: explicit that every non-``None`` per-event reason counts against coverage
#: and that size is deliberately NOT one of them, so "a corpus with no
#: integrity problem at all" cannot fail the gate. A coverage abstain is
#: therefore the aggregate of per-event integrity failures --
#: ``correlation_mismatch``, a missing or changed trajectory, a missing
#: request -- none of which is logged where it is detected. (The one routine
#: cause folded into these labels, an unregistered governance subject, takes
#: the retry-eligible ``rate_limited`` path with its own WARNING instead.)
#:
#: What stays routine: ``incumbent_revised_awaiting_evidence``,
#: ``awaiting_counted_evidence`` (a ``0/0`` ratio -- nothing to count is not a
#: failure to reconstruct), and the
#: composition's own ineligibilities (``no_outcome_contract_in_window``,
#: ``incumbent_not_found``, ...). Each member holds tuning off until someone
#: fixes the writer or the data, and the tick outcome line is the only place it
#: is reported, so ``sweep_progress.is_problem_outcome`` keeps them at WARNING
#: while routine ``ineligible_*`` answers go to INFO.
#: Every member must stay a reason the bundle derivation can return.
EVIDENCE_FAULT_REASONS: frozenset[str] = frozenset(
    {
        "evidence_population_limit_exceeded",
        "missing_full_version_fingerprint",
        "malformed_full_version_fingerprint",
        "incumbent_full_version_fingerprint_mismatch",
        "evidence_integrity_invalid",
        "missing_stable_playbook_id",
        "insufficient_global_reconstructability",
        "insufficient_target_reconstructability",
    }
)

# The two abstentions the analyst can reach BEFORE a candidate exists, and the
# reason they are one set rather than two call sites.
#
# `no_grounded_hypothesis` used to carry both of these, and they are OPPOSITE
# operator instructions: "the evidence carries no agent-side contrast" says
# stop tuning this playbook, while "the deciding session was not shown" says
# widen the corpus and try again. A reader could not tell them apart, so the
# second read as the first for as long as the string was shared.
#
# What they DO share is the structural fact three separate guards depend on:
# neither has generated anything, so neither may bind a candidate digest or a
# held-out population digest. Naming the set once is what keeps the attempt
# envelope validator, the freeze-time terminal-outcome check in
# `_open_world_evidence_freeze.py`, and the tenant authorization function from
# drifting apart -- widening one and not the others is how a legal write starts
# failing at a gate that still lists only the older member.
#
# THEY LIVE HERE, NOT IN `open_world/models.py`, for a layering reason rather
# than a modelling one: storage may import `offline_tuner/evidence/` but not the
# `open_world` package. `OutcomeContractDescriptor` lives here for the same
# reason and is re-exported from `open_world.models`; these follow it.
NO_CANDIDATE_ABSTAIN_REASONS: frozenset[str] = frozenset(
    {"no_grounded_hypothesis", "evidence_view_incomplete"}
)

# Every reason an `abstained` stage advance may carry. The mirror of the tenant
# stage-advance RPC's 'abstained' arm.
OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {*NO_CANDIDATE_ABSTAIN_REASONS, "heldout_evidence_failed"}
)


# The evidence-bundle schema version THIS image derives. The field itself has
# to spell the same string a second time -- ``Literal[...]`` admits no name --
# so `test_the_supported_schema_constant_matches_the_field_literal` pins the
# two together. It is named here because the STORAGE boundary has to recognise
# an unsupported version before Pydantic rejects it as generic malformed JSON;
# see `_require_supported_bundle_schema`.
#
# v6 -> v7 BECAUSE THE DERIVATION CHANGED, not because a field did.
# `build_open_world_evidence_bundle` now sets `reason="evidence_too_uncertain"`
# for a manifest whose outcomes are at least half `unknown`. Validation
# RE-DERIVES the bundle from its stored manifest and demands equality, so a v6
# artifact frozen by the previous image -- eligible under the old rule, and
# with the same field shape -- would be re-derived with the new reason and fail
# `bundle != derived`. That is the generic integrity refusal, which reports an
# in-flight artifact as corrupt and retryable instead of as an older schema.
# The version is what routes it to the named unsupported-schema path instead,
# and this is the same class of miss the v4 literal already recorded.
#
# v7 -> v8 FOR THE SAME REASON, ONE LAYER DOWN: the PARTITION changed.
# `_partition_expected_sessions` now caps each outcome class of the held-out
# and safety-only arms at
# `PRODUCTION_OPEN_WORLD_POLICY.maximum_validation_sessions_per_outcome`, and
# ranks the `unknown` arm by the attempt lottery instead of by identifier. A v7
# artifact frozen by the previous image with ninety held-out sessions has the
# same field shape and the same `expected` manifest, so it would re-derive to a
# DIFFERENT set of manifests and fail `bundle != derived` -- reported as a
# corrupt, retryable artifact when no retry on any host can ever load it.
#
# The per-playbook population scoping that ships alongside this does NOT need a
# version of its own: it changes what the storage READS into `expected`, and
# re-derivation runs from the stored `expected`, so a v7 artifact re-derives to
# itself under that half of the change.
#
# v8 -> v9 FOR BOTH REASONS AT ONCE. The CONTENT changed: every expected event
# now carries `trajectory_through_request_id` -- the prefix its trajectory was
# read through -- in its canonical payload, so the manifest digest moves for
# every window. And the DERIVATION changed: `awaiting_reaction` is a new
# reconstructability reason that both coverage ratios ignore. A v8 artifact in
# flight at the deploy would re-derive to a different bundle, so it is routed
# to the named, terminal `bundle_schema_unsupported` instead of the retryable
# integrity refusal; the cost is that one day's in-flight attempts end there.
#
# NOT FLAG-GATED: recruitment policy never controls lineage persistence or
# readers. Modern verdicts retain the judge's supplied boundary/count in either
# recruitment mode; historical NULL cutovers still describe the whole session.
#
# v9 -> v10 BECAUSE THE WINDOW RULE CHANGED, not a field: selection may now
# start the window at the current version's first exposure
# (`version_start_bound`), the reload check
# (`validate_expected_manifest_attribution_bound`) accepts that start, and the
# derivation names a window with no current-version exposure
# `incumbent_revised_awaiting_evidence`.
#
# THE DIRECTION THAT NEEDS THE VERSION IS NEW-TO-OLD. A v9 image -- a rollback,
# or the old half of a rolling deploy -- reloading a narrowed manifest for a
# playbook with a lifecycle row computes `max(scan_start, published_at)`, finds
# a later start, and raises `OpenWorldEvidenceIntegrityError`: the retryable
# refusal, for an artifact no retry on that image can ever load. Declaring v10
# routes it to that image's `_require_supported_bundle_schema` and the
# terminal `bundle_schema_unsupported` instead.
#
# Old-to-new would not need it on its own. Only ELIGIBLE selections are frozen,
# and an eligible v9 bundle holds no target exposure under another version, so
# v10 re-derives it identically and its v9 start is the v10 bound. The bump
# still refuses those by name -- the schema check runs inside
# `_frozen_from_artifact`, before the store's attribution check -- and that is
# the accepted cost: one deploy's in-flight attempts end
# `bundle_schema_unsupported`, terminally, rather than resuming.
#
# The MANIFEST literal (`offline-tuner-open-world-expected-v4`) is NOT moved,
# by the same division INV-2 followed: the window start was already in the
# manifest's canonical payload, so only the value selection chooses changes,
# and only for an edited playbook. Moving the literal would re-draw discovery
# for every target, edited or not.
OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA = "offline-tuner-open-world-evidence-v12"


class OpenWorldEvidenceSchemaUnsupportedError(ValueError, OpenWorldRefusalError):
    """A frozen bundle declares a schema version this image cannot derive.

    A DISTINCT class, not the bare ``ValueError`` this refusal used to raise,
    because the two converge on opposite operator instructions. Every other
    ``ValueError`` out of the derivation says "something about this bundle is
    wrong and the attempt failed" -- retryable, and reported as
    ``infrastructure_failure``. This one says the bundle was frozen by an
    image that computed a DIFFERENT manifest, so no retry on any host can ever
    load it: the row has to be drained or terminalized by hand. Reported as
    ``infrastructure_failure`` it was indistinguishable from a transient fault
    and invited exactly the retry that cannot work.

    ``ValueError`` remains the base deliberately. ``_validate_bundle_shape``
    and ``_rehydrate_frozen_evidence`` both catch ``ValueError`` at their
    boundaries; narrowing the class without keeping that base would convert a
    handled refusal into an uncaught crash at the storage boundary.
    """

    refusal = OpenWorldRefusal.BUNDLE_SCHEMA_UNSUPPORTED


# The role names this server will attribute to a speaker, case-folded.
#
# This is an ALLOWLIST built from names the product is actually OBSERVED to
# write -- never a "not-user means agent" rule. Each entry is here because a
# real writer emits it; nothing is added speculatively. ``ai``, ``bot``,
# ``model`` and similar are deliberately ABSENT: no writer in this product
# emits them, and admitting an unobserved name buys nothing while widening the
# surface on which plumbing can be mistaken for agent conduct.
#
# Agent-side provenance:
#   ``assistant`` -- the OSS CLI publishes ``InteractionData(role="assistant")``
#       (``reflexio/cli/commands/interactions.py``); demo publishers write
#       ``"Assistant"``, which folds to it.
#   ``agent`` -- the DOCUMENTED public contract. The interactions schema
#       reference states the canonical pair as
#       ``{"role": "User"}, {"role": "Agent"}``, and the public docs' examples
#       use ``"Agent"``. Stored production interactions use
#       ``User``/``Agent`` (and ``Tool``) role names -- i.e. ``Agent``, not
#       ``assistant``, is what integrators commonly store.
#
# User-side provenance:
#   ``user`` -- ``entities.py`` defaults ``InteractionData.role`` to ``"User"``,
#       which folds to this; every writer above pairs it with an agent name.
#
# ``tool`` is deliberately NOT here. It is attested in stored interactions and
# it is exactly the kind of
# turn this predicate must keep out: tool, system, retrieval and narrator turns
# are plumbing, and relabelling them as agent behaviour would invent agent
# conduct in the very field the discovery contrast is read from.
_AGENT_TURN_ROLES: frozenset[str] = frozenset({"assistant", "agent"})
_USER_TURN_ROLES: frozenset[str] = frozenset({"user"})

# Roles this server declines to attribute ON PURPOSE, case-folded.
#
# These are ``other`` for the same reason ``tool`` is kept out of
# ``_AGENT_TURN_ROLES``: they are plumbing, and reading them as agent conduct
# would invent behaviour. That makes them the CORRECT answer, not a
# misclassification -- so they are noise in the census, which exists only to
# say "these turns might have been agent turns and we could not tell". An org
# whose agent genuinely never spoke would otherwise report a ``Tool`` count
# that reads like a vocabulary bug.
#
# Narrow for the same reason the allowlist is narrow, but pointed the other
# way: every entry here SUPPRESSES a diagnostic, so an unobserved name must not
# be added -- an unfamiliar role is exactly the signal the census is for. Both
# entries are attested as stored interaction roles (storage contract tests
# write ``role="system"`` and ``role="tool"``), and ``tool`` is further
# attested in stored production interactions.
_PLUMBING_TURN_ROLES: frozenset[str] = frozenset({"tool", "system"})
TURN_SPEAKERS: frozenset[str] = frozenset({"agent", "user", "other"})


def classify_turn_speaker(role: object) -> TurnSpeaker:
    """Classify one stored interaction's ``role`` into who was speaking.

    This is the SINGLE server-side authority on speaker attribution. It is
    derived from the durable interaction row that the trajectory froze, never
    inferred from the turn's content and never supplied by the model -- an
    analyst that has to guess who spoke from the words is an analyst
    fabricating structure, which is the failure this function removes.

    The predicate is a NARROW ALLOWLIST, case-folded: ``_AGENT_TURN_ROLES`` or
    ``_USER_TURN_ROLES``. ``role`` is free text (``entities.py`` defaults it to
    ``"User"`` and caps it only at 1 000 characters), so no closed vocabulary
    can be assumed, and a broad "anything that is not a user turn is the agent"
    rule would silently relabel tool, system, retrieval or narrator turns as
    agent behaviour -- inventing agent conduct out of plumbing, in the exact
    field the discovery contrast is read from.

    Narrow is not the same as SHORT. An earlier version of this function
    recognised only ``assistant`` on the agent side, which is what the OSS CLI
    and the local replay corpora happen to write. The product's documented
    contract, its own demo publisher, and both production corpora write
    ``Agent`` -- so every production agent turn classified as ``other``, the
    discovery gate excluded every session as ``no_agent_turn``, and every
    attempt abstained ``insufficient_discovery_agent_evidence``, fleet-wide.
    The user side only worked by accident of case-folding ``"User"``. The
    lesson encoded here is that the allowlist must be justified against the
    roles the product actually WRITES, not against the corpus nearest to hand;
    see ``_AGENT_TURN_ROLES`` for per-entry provenance.

    Anything this does not recognise is therefore ``"other"``: a VISIBLE third
    category, never quietly folded into one of the two real speakers. The
    analyst is told in the prompt that ``other`` means the server could not
    attribute the turn and that it carries no agent-side claim, so an
    unrecognised role costs a contrast rather than corrupting one. That is the
    loud-failure half of the same reasoning that governs
    ``_trajectory_has_agent_turn``, which is now defined in terms of this
    function so the gate and the label cannot drift apart.

    ``.lower()`` rather than ``.casefold()`` is intentional: it is exactly what
    the agent-turn gate already applied, so adopting it here changes no
    existing partition and therefore no frozen bundle digest. The two differ
    only outside ASCII, where neither ``assistant`` nor ``user`` lives.

    Args:
        role (object): The stored interaction's ``role`` value, of any type

    Returns:
        TurnSpeaker: ``"agent"``, ``"user"``, or ``"other"``
    """
    if not isinstance(role, str):
        return "other"
    folded = role.strip().lower()
    if folded in _AGENT_TURN_ROLES:
        return "agent"
    if folded in _USER_TURN_ROLES:
        return "user"
    return "other"


_UNATTRIBUTED_ROLE_CENSUS_LIMIT = 5

# ``role`` is customer-controlled free text that ``entities.py`` caps only at
# 1 000 characters, and this census travels into an operator-facing error
# payload and a log line. It is therefore held to the same standard
# ``open_world/diagnostics.py`` holds itself to: every emitted label is
# length- and charset-clamped, and a label that fails the clamp is replaced
# WHOLESALE rather than truncated, because a truncated prefix of tenant prose
# is still tenant prose. The count survives redaction -- "seventeen turns
# carried a role we could not read" is the diagnostic signal; the bytes of the
# role are not.
_CENSUS_LABEL_MAX_LENGTH = 32
_CENSUS_SAFE_LABEL = re.compile(r"[A-Za-z0-9_-]+")
_CENSUS_REDACTED = "<redacted>"


def _census_label(role: object) -> str:
    """Clamp one stored role name to a publishable token.

    Args:
        role (object): The stored interaction's ``role`` value, of any type

    Returns:
        str: The case-folded role when it is a short plain token, else a fixed
            redaction marker that carries no tenant bytes
    """
    if not isinstance(role, str):
        return _CENSUS_REDACTED
    folded = role.strip().lower()
    if len(folded) <= _CENSUS_LABEL_MAX_LENGTH and _CENSUS_SAFE_LABEL.fullmatch(folded):
        return folded
    return _CENSUS_REDACTED


def unattributed_role_census(
    events: object, *, limit: int = _UNATTRIBUTED_ROLE_CENSUS_LIMIT
) -> tuple[tuple[str, int], ...]:
    """Count the stored role names this server could NOT attribute to a speaker.

    This exists to make one specific failure VISIBLE at runtime. A corpus whose
    agent turns carry a role name the allowlist does not know produces exactly
    the same observable as a corpus with no agent turns at all: every session is
    excluded ``no_agent_turn`` and the attempt abstains
    ``insufficient_discovery_agent_evidence``. That indistinguishability is the
    property that let a fleet-wide misclassification hide -- an operator reading
    the abstain reason had no way to tell "this org never answered anybody" from
    "3 152 turns are labelled ``Agent`` and we refused to read them".

    The census answers that question and nothing else. It is diagnostics, not
    evidence: it is deliberately NOT part of any bundle's canonical payload, so
    it cannot move a frozen digest or change a partition.

    Only roles that COULD have been agent turns are counted. ``tool`` and
    ``system`` turns are ``other`` on purpose (see ``_PLUMBING_TURN_ROLES``), so
    counting them made a silent corpus report ``Tool=239`` and read like the
    very misclassification the census exists to distinguish. Names are keyed on
    the case-folded form ``classify_turn_speaker`` itself compares, so ``Tool``
    and ``tool`` cannot occupy two slots of a five-slot list.

    Every emitted label is charset- and length-clamped by ``_census_label``:
    ``role`` is customer-controlled free text and this record reaches an
    operator-facing payload, so a non-conforming value is replaced by
    ``<redacted>`` and only its COUNT survives.

    Scope is the CALLER's responsibility. The events passed in are counted as
    given; a whole-org snapshot therefore reports another playbook's roles
    against this playbook's abstain, which is why the composition caller narrows
    to the target's own events before calling.

    Volume is bounded by construction: the caller emits it at most ONCE per
    abstained attempt, and only the ``limit`` most common names survive, so a
    corpus with thousands of distinct junk roles still yields a fixed-size
    record.

    Malformed input is skipped at the SMALLEST scope that is still well
    defined, matching ``_trajectory_has_agent_turn`` scope for scope: an
    unreadable interaction skips that interaction, an unreadable request skips
    that request, and only an unusable ``requests`` collection abandons the
    whole trajectory. That agreement is the point, not an incidental
    similarity. This census is the only thing that distinguishes "the agent
    never spoke" from "the agent spoke under a name we refused to read", so a
    census that gave up on the whole trajectory at the first malformed entry
    while the gate walked on would report nothing for sessions the gate had
    refused -- returning the starved-discovery abstain to the silent,
    plausible-looking state this record exists to remove. It is a one-way
    failure: nothing raises, and the census simply under-reports.

    Args:
        events (object): The target's expected events, each optionally
            carrying a frozen ``trajectory``. Malformed entries are skipped
            rather than raising -- a diagnostic must never be the thing that
            breaks the path it is describing.
        limit (int): Maximum distinct role names to report

    Returns:
        tuple[tuple[str, int], ...]: ``(label, count)`` pairs, most frequent
        first, ties broken by label
    """
    counts: dict[str, int] = defaultdict(int)
    try:
        iterator = iter(cast("list[object]", events))
    except TypeError:
        return ()
    for event in iterator:
        trajectory = getattr(event, "trajectory", None)
        if trajectory is None:
            continue
        try:
            request_items = list(trajectory["requests"])
        except (KeyError, TypeError):
            continue
        for request_item in request_items:
            try:
                interactions = list(request_item["interactions"])
            except (KeyError, TypeError):
                continue
            for interaction in interactions:
                try:
                    role = interaction["role"]
                    content = interaction["content"]
                except (KeyError, TypeError):
                    continue
                if not isinstance(content, str) or not content.strip():
                    continue
                if classify_turn_speaker(role) != "other":
                    continue
                if (
                    isinstance(role, str)
                    and role.strip().lower() in _PLUMBING_TURN_ROLES
                ):
                    continue
                counts[_census_label(role)] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], _utf8(item[0])))
    return tuple(ranked[: max(0, limit)])


def _utf8(value: str) -> bytes:
    return value.encode("utf-8")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")


def _require_digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


@dataclass(frozen=True)
class OpenWorldToolContract:
    name: str
    description: str

    def __post_init__(self) -> None:
        _require_text("tool name", self.name)
        _require_text("tool description", self.description)


@dataclass(frozen=True)
class OpenWorldTaskContext:
    """The historical context actually available to this evidence capture.

    Native request rows do not currently persist the agent's system prompt or
    complete tool catalog. Empty strings and an explicit missing status are
    intentional; current org configuration must not fill these fields as though
    it were historical evidence. Observed ``tools_used`` stays in the source.
    """

    agent_context: str = ""
    success_definition: str = ""
    available_tools: tuple[OpenWorldToolContract, ...] = ()
    tool_context_status: Literal["captured", "partial", "missing"] = "missing"
    provenance: str = "native-traces; historical-agent-configuration-not-captured"

    def __post_init__(self) -> None:
        if self.tool_context_status not in {"captured", "partial", "missing"}:
            raise ValueError("historical tool context status is invalid")
        _require_text("task context provenance", self.provenance)
        if self.tool_context_status == "missing" and self.available_tools:
            raise ValueError("missing historical context cannot carry tool contracts")
        tools = tuple(self.available_tools)
        if any(type(tool) is not OpenWorldToolContract for tool in tools):
            raise TypeError("historical tools must be OpenWorldToolContract records")
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError("historical tool names must be unique")
        object.__setattr__(self, "available_tools", tools)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "offline-tuner-task-context-v1",
            "agent_context": self.agent_context,
            "success_definition": self.success_definition,
            "available_tools": [
                {"name": tool.name, "description": tool.description}
                for tool in self.available_tools
            ],
            "tool_context_status": self.tool_context_status,
            "provenance": self.provenance,
        }

    @property
    def digest(self) -> str:
        return sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True)
class OpenWorldTargetIdentity:
    tenant_id: str
    user_playbook_id: int
    incumbent_semantic_digest: str
    incumbent_full_version_fingerprint: str
    success_definition_digest: str
    governance_epoch_vector: tuple[tuple[str, int], ...]
    frozen_at: int
    schema_version: Literal["offline-tuner-open-world-target-v1"] = (
        "offline-tuner-open-world-target-v1"
    )

    def __post_init__(self) -> None:
        _require_text("tenant_id", self.tenant_id)
        if type(self.user_playbook_id) is not int or self.user_playbook_id <= 0:
            raise ValueError("user_playbook_id must be positive")
        _require_digest("incumbent_semantic_digest", self.incumbent_semantic_digest)
        _require_digest(
            "incumbent_full_version_fingerprint",
            self.incumbent_full_version_fingerprint,
        )
        _require_digest("success_definition_digest", self.success_definition_digest)
        if type(self.frozen_at) is not int or self.frozen_at < 0:
            raise ValueError("frozen_at must be a nonnegative integer")
        normalized_epochs = tuple(
            sorted(self.governance_epoch_vector, key=lambda item: _utf8(item[0]))
        )
        if len({name for name, _ in normalized_epochs}) != len(normalized_epochs):
            raise ValueError("governance_epoch_vector contains duplicate subjects")
        for subject, epoch in normalized_epochs:
            _require_text("governance epoch subject", subject)
            if type(epoch) is not int or epoch < 0:
                raise ValueError("governance epochs must be nonnegative integers")
        object.__setattr__(self, "governance_epoch_vector", normalized_epochs)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "user_playbook_id": self.user_playbook_id,
            "incumbent_semantic_digest": self.incumbent_semantic_digest,
            "incumbent_full_version_fingerprint": self.incumbent_full_version_fingerprint,
            "success_definition_digest": self.success_definition_digest,
            "governance_epoch_vector": [
                {"subject": subject, "epoch": epoch}
                for subject, epoch in self.governance_epoch_vector
            ],
            "frozen_at": self.frozen_at,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())

    @property
    def digest(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class OpenWorldEvidenceEvent:
    exposure_event_id: str
    tenant_id: str | None
    user_id: str | None
    session_id: str | None
    user_playbook_id: int | None
    served_semantic_digest: str | None
    served_full_version_fingerprint: str | None
    request_id: str | None
    correlation_reason: CorrelationReason | None
    exposed_at: int | None
    ingested_at: int
    trajectory: CanonicalSessionTrajectory | None
    trajectory_digest: str | None
    trajectory_digest_matches: bool
    trajectory_byte_count: int | None
    trajectory_token_count: int | None
    outcome: Outcome | None
    outcome_id: str | None
    outcome_revision: int | None
    outcome_contract_digest: str | None
    governance_subject_refs: tuple[str, ...]
    governance_valid: bool
    #: The cutover the trajectory was read through -- the session outcome's
    #: own. None is the whole session.
    trajectory_through_request_id: str | None = None
    #: This exposure's request lies strictly after that cutover: its turn is
    #: not part of what the outcome describes yet. See
    #: ``AWAITING_REACTION_REASON``.
    awaiting_reaction: bool = False
    #: The session's outcome is one the tuner inferred (displaceable).
    outcome_is_inferred: bool = False
    #: The outcome's CURRENT cutover still covers the prefix the trajectory was
    #: read through. The only producer (the evidence snapshot) reads through
    #: the outcome's own cutover, so it is always True there; a reload through
    #: a FROZEN cutover would make it the test separating an inferred
    #: EXTENSION from a change that invalidates the frozen evidence. That
    #: storage read had no caller and is deleted.
    outcome_covers_trajectory_prefix: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "governance_subject_refs",
            tuple(sorted(set(self.governance_subject_refs), key=_utf8)),
        )


@dataclass(frozen=True)
class OpenWorldEvidenceSnapshot:
    tenant_id: str
    events: tuple[OpenWorldEvidenceEvent, ...] = field(default_factory=tuple)
    population_limit_exceeded: bool = False

    def __post_init__(self) -> None:
        _require_text("tenant_id", self.tenant_id)
        if type(self.population_limit_exceeded) is not bool:
            raise ValueError("population_limit_exceeded must be boolean")
        object.__setattr__(self, "events", tuple(self.events))
        if self.population_limit_exceeded and self.events:
            raise ValueError("an over-bound population cannot contain sampled events")


@dataclass(frozen=True)
class OpenWorldExpectedEvent:
    exposure_event_id: str
    tenant_id: str | None
    user_id: str | None
    session_id: str | None
    user_playbook_id: int | None
    served_semantic_digest: str | None
    served_full_version_fingerprint: str | None
    request_id: str | None
    trajectory_request_ids: tuple[str, ...]
    correlation_reason: CorrelationReason | None
    exposed_at: int | None
    trajectory_digest: str | None
    trajectory_byte_count: int | None
    trajectory_token_count: int | None
    outcome: Outcome | None
    outcome_id: str | None
    outcome_revision: int | None
    outcome_contract_digest: str | None
    governance_subject_refs: tuple[str, ...]
    has_agent_turn: bool
    # What this session costs in the RENDERED analyst payload -- spans, their
    # framing and the base64 envelope -- not the conversation content the
    # capacity predicate measures and not the fifteen-field storage row. It is
    # carried on the manifest because the bundle budget is derived from
    # ``expected`` alone, and by then the trajectory is long out of scope.
    analyst_render_token_count: int | None
    # Whether this session was refused for its SIZE alone. Carried beside the
    # reason rather than inside it: ``reconstructability_reason`` is ``None``
    # here, because nothing about the session is corrupt, and a bundle derived
    # from ``expected`` alone would otherwise have no way to tell an over-size
    # session from a clean one and would render it into an analyst payload.
    capacity_excluded: bool
    reconstructability_reason: str | None
    #: The prefix the trajectory -- and so ``trajectory_digest`` and
    #: ``trajectory_request_ids`` -- was read through, from the session's
    #: outcome. None is the whole session. In the canonical payload so the
    #: frozen bundle records exactly which prefix it was built from (v9).
    trajectory_through_request_id: str | None = None
    compaction_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_request_ids",
            tuple(sorted(set(self.trajectory_request_ids), key=_utf8)),
        )
        object.__setattr__(
            self,
            "governance_subject_refs",
            tuple(sorted(set(self.governance_subject_refs), key=_utf8)),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "exposure_event_id": self.exposure_event_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_playbook_id": self.user_playbook_id,
            "served_semantic_digest": self.served_semantic_digest,
            "served_full_version_fingerprint": self.served_full_version_fingerprint,
            "request_id": self.request_id,
            "trajectory_request_ids": list(self.trajectory_request_ids),
            "correlation_reason": self.correlation_reason,
            "exposed_at": self.exposed_at,
            "trajectory_digest": self.trajectory_digest,
            "trajectory_byte_count": self.trajectory_byte_count,
            "trajectory_token_count": self.trajectory_token_count,
            "outcome": self.outcome,
            "outcome_id": self.outcome_id,
            "outcome_revision": self.outcome_revision,
            "outcome_contract_digest": self.outcome_contract_digest,
            "governance_subject_refs": list(self.governance_subject_refs),
            "has_agent_turn": self.has_agent_turn,
            "analyst_render_token_count": self.analyst_render_token_count,
            "capacity_excluded": self.capacity_excluded,
            "reconstructability_reason": self.reconstructability_reason,
            "trajectory_through_request_id": self.trajectory_through_request_id,
            **({"compaction_required": True} if self.compaction_required else {}),
        }


@dataclass(frozen=True)
class OpenWorldExpectedManifest:
    tenant_id: str
    window_start: int
    window_end: int
    events: tuple[OpenWorldExpectedEvent, ...]
    population_limit_exceeded: bool = False
    integrity_invalid: bool = False
    #: NOT bumped when the manifest's CONTENT changes -- the bundle literal is.
    #: Four content changes rode the v4->v5 bundle bump (including
    #: ``OpenWorldExpectedEvent.capacity_excluded``, a field on this manifest's
    #: own events) and left this literal alone, and ``session_only`` follows
    #: them. The division is deliberate: the bundle is the artifact that
    #: crosses images, and its version is the one
    #: ``_require_supported_bundle_schema`` reads to refuse a stale one BY
    #: NAME. This literal is only an in-image consistency check, so moving it
    #: would buy no operator signal while changing ``digest`` -- and that
    #: digest ranks discovery (``_expected_discovery_rank``), so a gratuitous
    #: bump would re-draw the discovery arm for EVERY target, including the
    #: ones holding no session-only exposure at all.
    schema_version: Literal["offline-tuner-open-world-expected-v4"] = (
        "offline-tuner-open-world-expected-v4"
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "events",
            tuple(
                sorted(self.events, key=lambda event: _utf8(event.exposure_event_id))
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "population_limit_exceeded": self.population_limit_exceeded,
            "integrity_invalid": self.integrity_invalid,
            "events": [event.canonical_payload() for event in self.events],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())

    @property
    def digest(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class OpenWorldCoverageCounts:
    numerator: int
    denominator: int
    reasons: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "reasons": [
                {"reason": reason, "count": count} for reason, count in self.reasons
            ],
        }


@dataclass(frozen=True)
class OpenWorldEvidenceSession:
    user_id: str
    session_id: str
    outcome: Outcome | None
    exposure_event_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    trajectory_digest: str
    trajectory_byte_count: int
    trajectory_token_count: int
    outcome_id: str | None
    outcome_revision: int | None
    outcome_contract_digest: str | None
    governance_subject_refs: tuple[str, ...]
    analyst_render_token_count: int
    discovery_exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exposure_event_ids",
            tuple(sorted(set(self.exposure_event_ids), key=_utf8)),
        )
        object.__setattr__(
            self,
            "request_ids",
            tuple(sorted(set(self.request_ids), key=_utf8)),
        )
        object.__setattr__(
            self,
            "governance_subject_refs",
            tuple(sorted(set(self.governance_subject_refs), key=_utf8)),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "outcome": self.outcome,
            "exposure_event_ids": list(self.exposure_event_ids),
            "request_ids": list(self.request_ids),
            "trajectory_digest": self.trajectory_digest,
            "trajectory_byte_count": self.trajectory_byte_count,
            "trajectory_token_count": self.trajectory_token_count,
            "outcome_id": self.outcome_id,
            "outcome_revision": self.outcome_revision,
            "outcome_contract_digest": self.outcome_contract_digest,
            "governance_subject_refs": list(self.governance_subject_refs),
            "analyst_render_token_count": self.analyst_render_token_count,
            "discovery_exclusion_reason": self.discovery_exclusion_reason,
        }


@dataclass(frozen=True)
class OpenWorldSkippedSession:
    """One session the bundle left out, and why.

    THE RECEIPT, and it is deliberately NOT a per-session turns-shown ratio.
    Windowing and excerpting are out of scope, so every session that IS in the
    bundle is shown whole and that ratio is N-of-N by construction. The
    informative fact is the other half: which sessions never made it, which is
    what ``_build_expected_sessions`` decides and what -- until this existed --
    it discarded down to a count.

    ``reason`` is the same string ``session_exclusion_reasons`` counts, and it
    comes from the same walk. The two are one source projected twice rather
    than two independently derived facts, so they cannot disagree about how
    many sessions were excluded or for what.
    """

    user_id: str
    session_id: str
    reason: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OpenWorldRoleManifest:
    role: EvidenceRole
    sessions: tuple[OpenWorldEvidenceSession, ...] = field(default_factory=tuple)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sessions": [session.canonical_payload() for session in self.sessions],
        }


@dataclass(frozen=True)
class OpenWorldEvidenceBundle:
    target: OpenWorldTargetIdentity
    expected: OpenWorldExpectedManifest
    global_coverage: OpenWorldCoverageCounts
    target_coverage: OpenWorldCoverageCounts
    failed_sessions: int
    successful_sessions: int
    unknown_sessions: int
    session_exclusion_reasons: tuple[tuple[str, int], ...]
    discovery_exclusion_reasons: tuple[tuple[str, int], ...]
    eligible: bool
    reason: str | None
    discovery: OpenWorldRoleManifest
    held_out: OpenWorldRoleManifest
    safety_only: OpenWorldRoleManifest
    #: Every session the derivation excluded, in the order it walked them.
    #: Defaulted so the many hand-built bundles in tests keep constructing;
    #: a bundle that claims an empty receipt while its derivation skipped
    #: something is still caught, because ``validate_open_world_evidence_bundle``
    #: re-derives and compares the whole object.
    skipped_sessions: tuple[OpenWorldSkippedSession, ...] = ()
    # Bound into the frozen bundle even when missing: later org configuration
    # changes cannot silently rewrite the question the method was asked.
    task_context: OpenWorldTaskContext = field(default_factory=OpenWorldTaskContext)
    #: v5 carried FOUR changes to what this manifest holds and what its
    #: derivation computes: the size gate began measuring the analyst payload
    #: rather than the storage row, capacity exclusion became coverage rather
    #: than corruption (``OpenWorldExpectedEvent.capacity_excluded``), the
    #: budget was taken over the bundle rather than per session, and
    #: ``skipped_sessions`` joined the canonical payload.
    #:
    #: v6 is a FIFTH, and it is the smallest-looking kind: no field moved, only
    #: the values one may hold. ``correlation_reason`` gained ``session_only``,
    #: which an exposure carrying a session but no request now takes instead of
    #: ``missing_request``. That is still a derivation change -- the value is
    #: in ``OpenWorldExpectedEvent.canonical_payload``, so the manifest digest
    #: and the bundle digest both move for any window holding such a row -- and
    #: it is the direction the constant exists to catch: a v5 reader's
    #: ``Literal`` REJECTS ``session_only``, so a v6 bundle handed to one
    #: during a rolling deploy or a rollback fails the ``TypeAdapter`` and
    #: arrives as the retryable ``infrastructure_failure``, which is the
    #: opposite of the true instruction. Declaring v6 routes it instead to
    #: ``_require_supported_bundle_schema`` and the terminal, accurate
    #: ``bundle_schema_unsupported``.
    #:
    #: A bundle whose declared version is not this one is refused BY NAME
    #: rather than admitted and reported as a derivation fault.
    #:
    #: v7 is a SIXTH, and the same smallest-looking kind as v6: no field moved,
    #: only what the derivation computes. ``build_open_world_evidence_bundle``
    #: now sets ``reason="evidence_too_uncertain"`` for a manifest whose
    #: outcomes are at least half ``unknown``. Validation RE-DERIVES the bundle
    #: from its stored manifest and demands equality, so a v6 artifact frozen
    #: by the previous image -- eligible under the old rule, and with the same
    #: field shape -- would be re-derived with the new reason and fail
    #: ``bundle != derived``: the generic integrity refusal, which reports an
    #: in-flight artifact as corrupt and RETRYABLE rather than as an older
    #: schema. Declaring v7 routes it to the named, terminal
    #: ``bundle_schema_unsupported`` instead.
    #:
    #: v9 adds ``OpenWorldExpectedEvent.trajectory_through_request_id`` to the
    #: canonical payload and the coverage-neutral ``awaiting_reaction`` reason
    #: to the derivation; see ``OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA``.
    #:
    #: v10 adds the ``incumbent_revised_awaiting_evidence`` reason and the
    #: version-start reload rule; see ``OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA``.
    schema_version: Literal["offline-tuner-open-world-evidence-v12"] = (
        "offline-tuner-open-world-evidence-v12"
    )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target": self.target.canonical_payload(),
            "task_context": self.task_context.canonical_payload(),
            "expected": self.expected.canonical_payload(),
            "expected_manifest_digest": self.expected.digest,
            "global_coverage": self.global_coverage.canonical_payload(),
            "target_coverage": self.target_coverage.canonical_payload(),
            "failed_sessions": self.failed_sessions,
            "successful_sessions": self.successful_sessions,
            "unknown_sessions": self.unknown_sessions,
            "session_exclusion_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.session_exclusion_reasons
            ],
            "discovery_exclusion_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.discovery_exclusion_reasons
            ],
            "skipped_sessions": [
                session.canonical_payload() for session in self.skipped_sessions
            ],
            "eligible": self.eligible,
            "reason": self.reason,
            "discovery": self.discovery.canonical_payload(),
            "held_out": self.held_out.canonical_payload(),
            "safety_only": self.safety_only.canonical_payload(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())

    @property
    def digest(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class OpenWorldEvidenceSelection:
    bundle: OpenWorldEvidenceBundle

    @property
    def eligible(self) -> bool:
        return self.bundle.eligible

    @property
    def reason(self) -> str | None:
        return self.bundle.reason

    @property
    def digest(self) -> str:
        return self.bundle.digest

    def canonical_bytes(self) -> bytes:
        return self.bundle.canonical_bytes()


def _reason_counts(reasons: list[str]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return tuple(sorted(counts.items(), key=lambda item: _utf8(item[0])))


def coverage_for_reasons(reasons: list[str | None]) -> OpenWorldCoverageCounts:
    """Count a reason list into one coverage ratio, keeping every entry.

    ``reasons`` is one entry per EXPECTED event, ``None`` meaning the event
    reconstructed. A failure lowers the numerator and leaves the denominator
    alone, so an evidence pipeline that has stopped joining sessions reads as
    poor coverage instead of disappearing.

    The ONE exception is a coverage-neutral reason (``COVERAGE_NEUTRAL_REASONS``:
    an exposure awaiting its reaction), dropped from both sides. It is not a
    failure of the pipeline and not evidence yet, so it must move neither the
    numerator nor the denominator.
    """
    counted = [reason for reason in reasons if reason not in COVERAGE_NEUTRAL_REASONS]
    failures = [reason for reason in counted if reason is not None]
    return OpenWorldCoverageCounts(
        numerator=len(counted) - len(failures),
        denominator=len(counted),
        reasons=_reason_counts(failures),
    )


def expected_target_reason(
    event: OpenWorldExpectedEvent,
    target: OpenWorldTargetIdentity,
) -> str | None:
    """Classify one expected event against ONE playbook identity.

    The comparison is identity-agnostic even though the ``target`` field names
    say "incumbent": it checks the served fingerprint, semantic digest and
    success-definition digest against whichever identity the caller supplies.
    The evidence target supplies the incumbent.
    """
    fingerprint = event.served_full_version_fingerprint
    if fingerprint is None:
        return "missing_full_version_fingerprint"
    if len(fingerprint) != 64 or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        return "malformed_full_version_fingerprint"
    if fingerprint != target.incumbent_full_version_fingerprint:
        return "incumbent_full_version_fingerprint_mismatch"
    if event.reconstructability_reason is not None:
        return event.reconstructability_reason
    if event.served_semantic_digest != target.incumbent_semantic_digest:
        return "incumbent_digest_mismatch"
    return None


# Private aliases kept for existing call sites. These are the SAME function
# objects, not copies, so a future reimplementation under either name diverges
# from the public one rather than shadowing it.
_coverage_for_reasons = coverage_for_reasons
_expected_target_reason = expected_target_reason


class _ServedExposure(Protocol):
    """The four fields the version start reads -- carried identically by the
    evidence event the freeze selects from and the expected event a reload
    re-checks, which is what makes the two computations the SAME one."""

    @property
    def tenant_id(self) -> str | None: ...

    @property
    def user_playbook_id(self) -> int | None: ...

    @property
    def served_full_version_fingerprint(self) -> str | None: ...

    @property
    def exposed_at(self) -> int | None: ...


def _target_exposures(
    events: Iterable[_ServedExposure], target: OpenWorldTargetIdentity
) -> list[_ServedExposure]:
    # The same scope ``build_open_world_evidence_bundle`` computes the target
    # ratio over: an untenanted or foreign row is never the target's.
    return [
        event
        for event in events
        if event.tenant_id == target.tenant_id
        and event.user_playbook_id == target.user_playbook_id
    ]


def version_started_at(
    events: Iterable[_ServedExposure], target: OpenWorldTargetIdentity
) -> int | None:
    """When the version being tuned was first served, among ``events``.

    The earliest timestamped target exposure carrying the incumbent's full
    version fingerprint, or ``None`` if none does. Derived from the evidence
    rather than from a write log because every mutation path -- the API, the
    dashboard, the auto-tagger, the governance backfill, an upsert -- changes
    the fingerprint, and only the exposures record when the new one began
    serving.
    """
    served = [
        event.exposed_at
        for event in _target_exposures(events, target)
        if event.served_full_version_fingerprint
        == target.incumbent_full_version_fingerprint
        and event.exposed_at is not None
    ]
    return min(served) if served else None


def version_start_bound(
    events: Iterable[_ServedExposure], target: OpenWorldTargetIdentity
) -> int | None:
    """The version start that narrows the window, or ``None`` if it may not.

    ``events`` is everything inside ``[max(scan_start, published_at),
    frozen_at)``. Returns ``version_started_at`` only when BOTH hold:

    * at least one target exposure was served under ANOTHER version (a
      different, missing or malformed fingerprint) -- without one there is
      nothing to drop, and narrowing anyway would move the manifest digest,
      and with it the discovery draw, for every untouched playbook; and
    * every such exposure is timestamped strictly BEFORE the version started.

    The second clause is the fence. An old version served at or after the new
    one's first exposure -- interleaved writers, a reverted edit, a
    fingerprint algorithm that moved -- is not a stale prefix, it is a fault,
    and the categorical ``incumbent_full_version_fingerprint_mismatch`` it
    produces is kept. An untimed one cannot be shown to precede anything, so
    it keeps the fence too.

    THE FREEZE AND THE RELOAD BOTH CALL THIS, over the same events: the
    selection hands it the windowed evidence events, the reload the manifest's
    expected events, which carry these four fields verbatim.
    """
    targeted = _target_exposures(events, target)
    started = version_started_at(targeted, target)
    if started is None:
        return None
    others = [
        event.exposed_at
        for event in targeted
        if event.served_full_version_fingerprint
        != target.incumbent_full_version_fingerprint
    ]
    if not others or any(
        exposed_at is None or exposed_at >= started for exposed_at in others
    ):
        return None
    return started


def _validate_expected_manifest_identity(
    target: OpenWorldTargetIdentity,
    expected: OpenWorldExpectedManifest,
) -> None:
    if expected.schema_version != "offline-tuner-open-world-expected-v4":
        raise ValueError("expected manifest schema is unsupported")
    if expected.tenant_id != target.tenant_id:
        raise ValueError("target and expected manifest tenants differ")
    if expected.window_end != target.frozen_at:
        raise ValueError("target and expected manifest boundaries differ")
    # A RANGE, not an equality, and the weakening is deliberate. The window's
    # lower bound is
    # ``max(published_at, window_end - historical_window_seconds)`` (INV-2, see
    # ``select_open_world_evidence``), possibly narrowed further to the current
    # version's first exposure, and ``published_at`` is a per-playbook
    # fact this validator is not given: the bundle must stay re-derivable from
    # ``(target, expected)`` alone, which is what ``build_open_world_evidence_bundle``
    # is re-run on when a frozen manifest is reloaded. So what is checkable here
    # is that the manifest never reaches FURTHER BACK than the fixed policy
    # scan -- the direction that would admit evidence nobody selected -- and
    # never inverts. A narrower window is legitimate and carries its own
    # justification in the lifecycle row.
    scan_start = (
        expected.window_end - PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds
    )
    if not scan_start <= expected.window_start <= expected.window_end:
        raise ValueError("expected manifest window does not match fixed policy")
    if type(expected.population_limit_exceeded) is not bool:
        raise ValueError("population_limit_exceeded must be boolean")
    if type(expected.integrity_invalid) is not bool:
        raise ValueError("integrity_invalid must be boolean")
    if expected.population_limit_exceeded and (
        expected.events or expected.integrity_invalid
    ):
        raise ValueError("over-bound population contains partial manifest state")
    if len(expected.events) > PRODUCTION_OPEN_WORLD_POLICY.maximum_population:
        raise ValueError("expected manifest exceeds the fixed population limit")
    event_ids = [event.exposure_event_id for event in expected.events]
    if any(not isinstance(event_id, str) or not event_id for event_id in event_ids):
        raise ValueError("expected manifest exposure identities must be non-empty")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("expected manifest contains duplicate exposure events")


def _validate_expected_event_metadata(
    event: OpenWorldExpectedEvent,
    target: OpenWorldTargetIdentity,
) -> None:
    for name, value in (
        ("user_id", event.user_id),
        ("session_id", event.session_id),
        ("request_id", event.request_id),
    ):
        _require_text(name, value)
    if event.tenant_id != target.tenant_id:
        raise ValueError("selected evidence tenant does not match target")
    if event.user_playbook_id != target.user_playbook_id:
        raise ValueError("selected evidence playbook does not match target")
    if type(event.exposed_at) is not int:
        raise ValueError("selected evidence timestamp must be an integer")
    if event.correlation_reason is not None:
        raise ValueError("selected evidence has an unresolved correlation")
    _require_digest("trajectory_digest", event.trajectory_digest)
    if type(event.trajectory_byte_count) is not int or event.trajectory_byte_count < 0:
        raise ValueError("trajectory_byte_count must be a nonnegative integer")
    if (
        type(event.trajectory_token_count) is not int
        or event.trajectory_token_count < 0
    ):
        raise ValueError("trajectory_token_count must be a nonnegative integer")
    # NO PER-SESSION SIZE CAP HERE. This used to refuse a selected event whose
    # fifteen-field STORAGE ROW exceeded the fixed limits, which is neither the
    # object selection now admits sessions on (conversation content) nor the
    # object that has to fit (the rendered payload). Once selection was
    # re-pointed at content, this raise fired on exactly the sessions the
    # change exists to admit. The capacity invariant is now per-bundle and
    # lives in ``build_open_world_evidence_bundle``; what remains here is the
    # INTEGRITY half -- the count must be a nonnegative integer, because the
    # bundle total is summed from it.
    if type(event.compaction_required) is not bool:
        raise ValueError("compaction_required must be boolean")
    if event.compaction_required and (
        event.capacity_excluded or event.analyst_render_token_count != 32_000
    ):
        raise ValueError("compaction reservation is invalid")
    if event.outcome is not None:
        if event.outcome not in {"success", "failure", "unknown"}:
            raise ValueError("selected evidence outcome is invalid")
        _require_text("outcome_id", event.outcome_id)
        if type(event.outcome_revision) is not int or event.outcome_revision <= 0:
            raise ValueError("outcome_revision must be a positive integer")
        _require_digest("outcome_contract_digest", event.outcome_contract_digest)
    elif any(
        value is not None
        for value in (
            event.outcome_id,
            event.outcome_revision,
            event.outcome_contract_digest,
        )
    ):
        raise ValueError("absent outcome must not carry a partial label identity")
    if not event.governance_subject_refs:
        raise ValueError("selected evidence governance subjects are empty")
    for subject_ref in event.governance_subject_refs:
        _require_text("governance subject", subject_ref)


def _validate_selectable_expected_event(
    event: OpenWorldExpectedEvent,
    target: OpenWorldTargetIdentity,
) -> None:
    _validate_expected_event_metadata(event, target)
    if not event.trajectory_request_ids or event.request_id not in (
        event.trajectory_request_ids
    ):
        raise ValueError("selected evidence trajectory requests are incomplete")
    for request_id in event.trajectory_request_ids:
        _require_text("trajectory request_id", request_id)
    if (
        type(event.analyst_render_token_count) is not int
        or event.analyst_render_token_count < 0
    ):
        raise ValueError("analyst_render_token_count must be a nonnegative integer")


def _is_discarded_capacity_event(event: OpenWorldExpectedEvent) -> bool:
    # The streaming loader retains the digest/byte count, but drops the
    # payload and uses zero tokens after its byte cap is exceeded.
    return (
        event.capacity_excluded is True
        and event.trajectory_request_ids == ()
        and event.analyst_render_token_count is None
        and type(event.trajectory_byte_count) is int
        and event.trajectory_token_count == 0
        and not trajectory_input_within_limits(
            byte_count=cast(int, event.trajectory_byte_count), token_count=0
        )
    )


def _expected_session_exclusion_reason(
    events: list[OpenWorldExpectedEvent],
) -> str | None:
    if len({event.outcome for event in events}) != 1:
        return "conflicting_session_outcome"
    if len({event.trajectory_digest for event in events}) != 1:
        return "conflicting_session_trajectory"
    if len({(event.outcome_id, event.outcome_revision) for event in events}) != 1:
        return "conflicting_session_outcome_identity"
    if (
        len(
            {
                (
                    event.trajectory_byte_count,
                    event.trajectory_token_count,
                    event.analyst_render_token_count,
                )
                for event in events
            }
        )
        != 1
    ):
        return "conflicting_session_trajectory_counts"
    if len({event.trajectory_request_ids for event in events}) != 1:
        return "conflicting_session_requests"
    if len({event.outcome_contract_digest for event in events}) != 1:
        return "conflicting_session_outcome_contract"
    # LAST, so genuine corruption always wins the label. Capacity is not an
    # integrity failure -- it no longer lowers either coverage ratio -- but it
    # is still an exclusion: the session is too large to render into an analyst
    # payload, and admitting it here would put a 9,001-token conversation in
    # front of the model, which is the failure the content meter exists to
    # prevent arriving one layer down.
    if any(event.capacity_excluded for event in events):
        return CAPACITY_EXCLUSION_REASON
    return None


def _session_discovery_exclusion_reason(
    events: list[OpenWorldExpectedEvent],
) -> str | None:
    """Why this session may not fill a DISCOVERY slot, or None if it may.

    Scoped to discovery on purpose; the session stays in the bundle and keeps
    its held-out or safety-only role. See ``_partition_expected_sessions``.
    """
    if not any(event.has_agent_turn for event in events):
        return "no_agent_turn"
    return None


def _build_expected_sessions(
    events: list[OpenWorldExpectedEvent],
    target: OpenWorldTargetIdentity,
) -> tuple[tuple[OpenWorldEvidenceSession, ...], tuple[OpenWorldSkippedSession, ...]]:
    """The sessions the bundle carries, and the receipt of the ones it does not.

    Returns the SKIPPED SET rather than a count-by-reason. The counts the
    bundle publishes as ``session_exclusion_reasons`` are folded from this same
    tuple at the one call site, so a session cannot be in the receipt without
    being in the counts, or the reverse.
    """
    grouped: dict[tuple[str, str], list[OpenWorldExpectedEvent]] = defaultdict(list)
    for event in events:
        if _is_discarded_capacity_event(event):
            _validate_expected_event_metadata(event, target)
        else:
            _validate_selectable_expected_event(event, target)
        grouped[(cast(str, event.user_id), cast(str, event.session_id))].append(event)

    sessions: list[OpenWorldEvidenceSession] = []
    skipped: list[OpenWorldSkippedSession] = []
    for (user_id, session_id), session_events in sorted(
        grouped.items(), key=lambda item: (_utf8(item[0][0]), _utf8(item[0][1]))
    ):
        exclusion = _expected_session_exclusion_reason(session_events)
        if exclusion is not None:
            skipped.append(
                OpenWorldSkippedSession(
                    user_id=user_id, session_id=session_id, reason=exclusion
                )
            )
            continue
        first = session_events[0]
        sessions.append(
            OpenWorldEvidenceSession(
                user_id=user_id,
                session_id=session_id,
                outcome=first.outcome,
                exposure_event_ids=tuple(
                    event.exposure_event_id for event in session_events
                ),
                request_ids=first.trajectory_request_ids,
                trajectory_digest=cast(str, first.trajectory_digest),
                trajectory_byte_count=cast(int, first.trajectory_byte_count),
                trajectory_token_count=cast(int, first.trajectory_token_count),
                outcome_id=first.outcome_id,
                outcome_revision=first.outcome_revision,
                outcome_contract_digest=first.outcome_contract_digest,
                governance_subject_refs=tuple(
                    subject_ref
                    for event in session_events
                    for subject_ref in event.governance_subject_refs
                ),
                analyst_render_token_count=cast(int, first.analyst_render_token_count),
                discovery_exclusion_reason=_session_discovery_exclusion_reason(
                    session_events
                ),
            )
        )
    return tuple(sessions), tuple(skipped)


def _expected_discovery_rank(
    session: OpenWorldEvidenceSession,
    *,
    role: str,
    target: OpenWorldTargetIdentity,
    expected: OpenWorldExpectedManifest,
) -> tuple[bytes, bytes, bytes]:
    rank = sha256(
        canonical_json_bytes(
            {
                "target_digest": target.digest,
                "expected_manifest_digest": expected.digest,
                "role": role,
                "user_id": session.user_id,
                "session_id": session.session_id,
            }
        )
    ).digest()
    return rank, _utf8(session.user_id), _utf8(session.session_id)


def _partition_expected_sessions(
    sessions: tuple[OpenWorldEvidenceSession, ...],
    *,
    target: OpenWorldTargetIdentity,
    expected: OpenWorldExpectedManifest,
) -> tuple[OpenWorldRoleManifest, OpenWorldRoleManifest, OpenWorldRoleManifest]:
    # No outcome lottery: a useful diagnosed gap need not have a completion
    # label, and an unknown label must not exile a source to a safety-only arm.
    # Ranking is fixed before diagnosis and independent of model outputs.
    usable = sorted(
        (session for session in sessions if session.discovery_exclusion_reason is None),
        key=lambda session: _expected_discovery_rank(
            session, role="source", target=target, expected=expected
        ),
    )
    return (
        OpenWorldRoleManifest(
            role="discovery",
            sessions=tuple(
                usable[: PRODUCTION_OPEN_WORLD_POLICY.maximum_evidence_sessions]
            ),
        ),
        OpenWorldRoleManifest(role="held_out"),
        OpenWorldRoleManifest(role="safety_only"),
    )


def _validate_analyst_payload_budget(
    discovery: OpenWorldRoleManifest,
    held_out: OpenWorldRoleManifest,
    safety_only: OpenWorldRoleManifest,
) -> None:
    """Every analyst call this bundle implies must fit the fixed budget.

    THE CAPACITY INVARIANT, MOVED OFF THE STORAGE ROW. What it replaced
    rejected a selected event whose fifteen-field storage row was large, which
    is neither what selection admits sessions on nor what has to fit -- so once
    selection was re-pointed at conversation content, that raise fired on
    exactly the sessions the change exists to admit.

    THE UNIT IS ONE ANALYST CALL, NOT THE UNION OF THE BUNDLE. The discovery
    analyst is handed the whole discovery arm in a single payload, and is the
    only call that carries more than one session; ``held_out_model_input`` and
    ``confirmation_model_input`` each take exactly ONE ``EvidenceDocument``. So
    a bundle holding ninety held-out sessions is not too large -- it is ninety
    small calls -- while five discovery sessions that individually fit and
    together do not is precisely the failure the per-session cap could not see.

    Measured on the RENDERED payload rather than on the content meter that
    decides whether a session is pathological. ``_spans_for_event`` emits one
    framed span per non-empty interaction, so a session of thousands of
    single-character turns measures a few thousand content tokens -- legal
    under any content meter -- and renders one span of framing each.
    ``field_path`` is positional and cannot be hoisted, so per-span overhead
    has a floor, and only a budget on the rendered object bounds that case.
    """
    calls = [sum(session.analyst_render_token_count for session in discovery.sessions)]
    calls.extend(
        session.analyst_render_token_count
        for manifest in (held_out, safety_only)
        for session in manifest.sessions
    )
    if any(
        not bundle_within_analyst_payload_budget(render_token_count=render_tokens)
        for render_tokens in calls
    ):
        raise ValueError(
            "selected evidence bundle exceeds the fixed analyst payload budget"
        )


def build_open_world_evidence_bundle(
    target: OpenWorldTargetIdentity,
    expected: OpenWorldExpectedManifest,
    *,
    task_context: OpenWorldTaskContext | None = None,
) -> OpenWorldEvidenceBundle:
    """Derive the only valid bundle for a target and expected manifest."""
    task_context = task_context or OpenWorldTaskContext()
    if target.success_definition_digest != task_context.digest:
        raise ValueError("target must bind the actual native task context")
    _validate_expected_manifest_identity(target, expected)
    global_coverage = _coverage_for_reasons(
        [event.reconstructability_reason for event in expected.events]
    )
    target_events = [
        event
        for event in expected.events
        if event.tenant_id == target.tenant_id
        and event.user_playbook_id == target.user_playbook_id
    ]
    target_reasons = [_expected_target_reason(event, target) for event in target_events]
    target_coverage = _coverage_for_reasons(target_reasons)

    reason: str | None = None
    if expected.population_limit_exceeded:
        reason = "evidence_population_limit_exceeded"
    else:
        for categorical_reason in (
            "missing_full_version_fingerprint",
            "malformed_full_version_fingerprint",
            "incumbent_full_version_fingerprint_mismatch",
        ):
            if categorical_reason in target_reasons:
                reason = categorical_reason
                break
        if reason == "incumbent_full_version_fingerprint_mismatch" and not any(
            event.served_full_version_fingerprint
            == target.incumbent_full_version_fingerprint
            for event in target_events
        ):
            # Every target exposure was served by a version that has since
            # changed, and the current one has served nobody yet. That is not
            # the fault the mismatch fence exists for -- it is an edit waiting
            # for traffic -- so it gets its own honest label. Still an
            # abstain, and still day-spending, but as `ineligible_<reason>`
            # the same-day re-attempt re-admits it once a session that
            # exposed the playbook moves. A mismatch BESIDE a current-version
            # exposure that `select_open_world_evidence` could not drop as a
            # stale prefix (see `version_start_bound`) stays categorical.
            reason = INCUMBENT_REVISED_AWAITING_EVIDENCE_REASON
        if reason is None and (
            expected.integrity_invalid
            or any(
                event.reconstructability_reason
                in {"missing_tenant", "missing_timestamp"}
                for event in expected.events
            )
        ):
            reason = "evidence_integrity_invalid"
        elif reason is None and not coverage_passes(
            global_coverage.numerator, global_coverage.denominator
        ):
            # A ``0/0`` is a wait, not a fault: see
            # ``AWAITING_COUNTED_EVIDENCE_REASON``.
            reason = (
                AWAITING_COUNTED_EVIDENCE_REASON
                if global_coverage.denominator == 0
                else "insufficient_global_reconstructability"
            )
        elif reason is None and any(
            event.tenant_id == target.tenant_id and event.user_playbook_id is None
            for event in expected.events
        ):
            reason = "missing_stable_playbook_id"
        elif reason is None and not coverage_passes(
            target_coverage.numerator, target_coverage.denominator
        ):
            reason = (
                AWAITING_COUNTED_EVIDENCE_REASON
                if target_coverage.denominator == 0
                else "insufficient_target_reconstructability"
            )

    sessions: tuple[OpenWorldEvidenceSession, ...] = ()
    skipped_sessions: tuple[OpenWorldSkippedSession, ...] = ()
    if reason is None:
        sessions, skipped_sessions = _build_expected_sessions(
            [
                event
                for event, target_reason in zip(
                    target_events, target_reasons, strict=True
                )
                if target_reason is None
            ],
            target,
        )
    failed_sessions = sum(session.outcome == "failure" for session in sessions)
    successful_sessions = sum(session.outcome == "success" for session in sessions)
    unknown_sessions = sum(session.outcome == "unknown" for session in sessions)
    if reason is None and not any(
        session.discovery_exclusion_reason is None for session in sessions
    ):
        reason = "insufficient_discovery_agent_evidence"

    discovery, held_out, safety_only = _partition_expected_sessions(
        sessions, target=target, expected=expected
    )
    _validate_analyst_payload_budget(discovery, held_out, safety_only)
    return OpenWorldEvidenceBundle(
        target=target,
        expected=expected,
        global_coverage=global_coverage,
        target_coverage=target_coverage,
        failed_sessions=failed_sessions,
        successful_sessions=successful_sessions,
        unknown_sessions=unknown_sessions,
        session_exclusion_reasons=_reason_counts(
            [session.reason for session in skipped_sessions]
        ),
        skipped_sessions=skipped_sessions,
        task_context=task_context,
        discovery_exclusion_reasons=_reason_counts(
            [
                session.discovery_exclusion_reason
                for session in sessions
                if session.discovery_exclusion_reason is not None
            ]
        ),
        eligible=reason is None,
        reason=reason,
        discovery=discovery,
        held_out=held_out,
        safety_only=safety_only,
    )


def validate_expected_manifest_attribution_bound(
    bundle: OpenWorldEvidenceBundle, published_at: int | None
) -> None:
    """Pin a reloaded manifest's lower bound to the version that produced it.

    ``_validate_expected_manifest_identity`` can only check a RANGE. It is
    re-run on ``(target, expected)`` alone when a frozen manifest is reloaded,
    and ``published_at`` is a per-playbook fact that pair does not carry -- so
    the strongest it can say is "no further back than the fixed policy scan,
    and not inverted". That admits any start inside a 14-day band, which means
    a manifest re-derived against the WRONG ``published_at`` still validates.

    This is the other half, for the callers that DO hold the lifecycle row. The
    bound selection applies is ``max(base, version_started_at)`` where
    ``base = max(window_end - historical_window_seconds, published_at)``
    (INV-2) and the version start counts only when ``version_start_bound``
    admits it (see ``select_open_world_evidence``). Two shapes are legal:

    * ``window_start == base``: the manifest holds everything selection saw,
      so ``version_start_bound`` is re-run over it -- the SAME function over
      the same four fields the freeze read -- and must decline to narrow.
      A pre-change manifest that kept a droppable prefix fails here.
    * ``window_start > base``: the manifest holds only ``[version start,
      frozen_at)``. Its earliest current-version exposure must sit EXACTLY at
      the start, and no exposure under any other version may remain -- every
      one preceded the start, or selection would not have narrowed.

    Plus, either way, no recorded exposure may predate the start.

    NOT RECORDED, DERIVED. ``version_started_at`` is not a manifest field:
    adding one would move the manifest digest -- and with it the discovery
    draw -- for every target, edited or not, which is the cost the v4 literal's
    note warns about. And a recorded value would prove nothing a forger could
    not rewrite alongside the start. What remains invisible to any check over
    the manifest alone is the same class INV-2 already accepted: a start moved
    back to ``base`` with the dropped prefix also deleted, or forward onto a
    first exposure with nothing before it. Neither can re-admit an exposure
    served under another version; the bundle derivation's categorical fence
    still sees every one that is present.

    ``published_at is None`` means the playbook was never published by the
    tuner, or its lifecycle row has been reclaimed -- and the retention sweep
    will not reclaim one published within ``historical_window_seconds`` of now
    -- so the scan floor is ``base``. The version rule still applies.

    Args:
        bundle (OpenWorldEvidenceBundle): The reloaded bundle.
        published_at (int | None): When the tuner published the version this
            bundle targets, from its lifecycle row.

    Raises:
        ValueError: If the recorded window start is not the attribution bound
            that ``published_at`` and the recorded exposures imply.
    """
    expected = bundle.expected
    scan_start = (
        expected.window_end - PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds
    )
    base = scan_start if published_at is None else max(scan_start, published_at)
    start = expected.window_start
    if any(
        event.exposed_at is not None and event.exposed_at < start
        for event in expected.events
    ):
        raise ValueError(
            f"expected manifest window start {start} is not the attribution "
            "bound: it records an exposure before its own start"
        )
    if start == base:
        narrowed = version_start_bound(expected.events, bundle.target)
        if narrowed is None:
            return
        raise ValueError(
            f"expected manifest window start {start} is not the attribution "
            f"bound {narrowed}: every exposure under another version precedes "
            "the current version's first exposure, so the window starts there"
        )
    started = version_started_at(expected.events, bundle.target)
    stale = any(
        event.served_full_version_fingerprint
        != bundle.target.incumbent_full_version_fingerprint
        for event in _target_exposures(expected.events, bundle.target)
    )
    if start > base and started == start and not stale:
        return
    raise ValueError(
        f"expected manifest window start {start} is not the attribution bound "
        f"{base} implied by published_at {published_at}, nor the first "
        "exposure of the version being tuned"
    )


def validate_open_world_evidence_bundle(bundle: OpenWorldEvidenceBundle) -> None:
    """Reject any bundle that is not the fixed derivation of its manifest."""
    if type(bundle) is not OpenWorldEvidenceBundle:
        raise TypeError("bundle must be an OpenWorldEvidenceBundle")
    if bundle.schema_version != OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA:
        raise OpenWorldEvidenceSchemaUnsupportedError("bundle schema is unsupported")
    derived = build_open_world_evidence_bundle(
        bundle.target, bundle.expected, task_context=bundle.task_context
    )
    if bundle != derived:
        raise ValueError(
            "bundle roles, dependencies, coverage, or policy result are not the "
            "exact expected-manifest derivation"
        )


def validate_open_world_evidence_selection(
    selection: OpenWorldEvidenceSelection,
) -> None:
    """Validate the semantic contract exposed by a selection wrapper."""
    if type(selection) is not OpenWorldEvidenceSelection:
        raise TypeError("selection must be an OpenWorldEvidenceSelection")
    validate_open_world_evidence_bundle(selection.bundle)


__all__ = [
    "AWAITING_REACTION_REASON",
    "CAPACITY_EXCLUSION_REASON",
    "COVERAGE_NEUTRAL_REASONS",
    "INCUMBENT_REVISED_AWAITING_EVIDENCE_REASON",
    "NO_CANDIDATE_ABSTAIN_REASONS",
    "OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES",
    "TURN_SPEAKERS",
    "TurnSpeaker",
    "classify_turn_speaker",
    "OpenWorldCoverageCounts",
    "OpenWorldEvidenceBundle",
    "OpenWorldEvidenceEvent",
    "OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA",
    "OpenWorldEvidenceSchemaUnsupportedError",
    "OpenWorldEvidenceSelection",
    "OpenWorldEvidenceSession",
    "OpenWorldEvidenceSnapshot",
    "OpenWorldExpectedEvent",
    "OpenWorldExpectedManifest",
    "OpenWorldRoleManifest",
    "OpenWorldSkippedSession",
    "OpenWorldTargetIdentity",
    "build_open_world_evidence_bundle",
    "validate_expected_manifest_attribution_bound",
    "version_start_bound",
    "version_started_at",
    "coverage_for_reasons",
    "expected_target_reason",
    "unattributed_role_census",
    "validate_open_world_evidence_bundle",
    "validate_open_world_evidence_selection",
]


class _StrictFrozenModel(BaseModel):
    """Strict, frozen, extra-forbidding base for the descriptors defined here.

    Restated rather than imported from ``open_world.models``: this module is
    imported by storage, which may not import the open-world package.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# This module and ``open_world.models`` each grew a ``_require_text`` /
# ``_require_digest`` pair, and they are NOT the same function:
#
#     this module     not isinstance(value, str) or not value
#     open_world      type(value) is not str    or not value.strip()
#
# so the open-world pair additionally rejects whitespace-only text and str
# SUBCLASSES. ``OutcomeContractDescriptor`` was written against the strict pair,
# and moving the class here silently rebound its validators to the loose one --
# ``schema_version="   "`` started validating. Restated here so the move is
# behaviour-preserving, and kept local rather than tightening this module's
# shared pair, which the records above are written against.
def _require_strict_text(name: str, value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_strict_digest(name: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


class OutcomeContractDescriptor(_StrictFrozenModel):
    digest: Sha256Digest
    kind: Literal["success_definition", "structured_outcome_source"]
    success_definition: str | None
    source_id: str | None
    schema_version: str
    allowed_values: tuple[Literal["success", "failure", "unknown"], ...]

    @field_validator("digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        _require_strict_digest("digest", value)
        return value

    @field_validator("schema_version")
    @classmethod
    def require_schema_version(cls, value: str) -> str:
        _require_strict_text("schema_version", value)
        return value

    @model_validator(mode="after")
    def validate_kind_fields(self) -> Self:
        if self.kind == "success_definition":
            _require_strict_text("success_definition", self.success_definition)
            if self.source_id is not None:
                raise ValueError(
                    "success_definition descriptor cannot include source_id"
                )
        else:
            _require_strict_text("source_id", self.source_id)
            if self.success_definition is not None:
                raise ValueError(
                    "structured_outcome_source descriptor cannot include success_definition"
                )
        if self.allowed_values != ("success", "failure", "unknown"):
            raise ValueError("allowed_values must be success, failure, unknown")
        return self
