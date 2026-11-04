"""Frozen data contracts for the open-world analysis path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reflexio.models.api_schema.domain import OptimizationJobClaim
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    NO_CANDIDATE_ABSTAIN_REASONS,
    OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES,
    REDUCER_POLICY_ID,
    TURN_SPEAKERS,
    OutcomeContractDescriptor,
)
from reflexio.server.services.playbook.publication import QualificationAuthorityRef

# Re-exported, not restated. Both sets are DEFINED in the evidence package
# because the storage layer needs them and may not import this package;
# naming them here keeps every
# open-world importer unchanged. Same pattern as `OutcomeContractDescriptor`.
__all_abstain_reexports__ = (
    NO_CANDIDATE_ABSTAIN_REASONS,
    OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES,
)

_ANALYST_ROLES = frozenset({"discovery", "candidate", "held_out", "confirmation"})
_MANIFEST_ROLES = frozenset({"discovery", "held_out", "safety_only"})
# Deliberately imported from the evidence package rather than restated, so the
# label the analyst reads cannot drift from the classifier that derives it.
_TURN_SPEAKERS = TURN_SPEAKERS
_HELD_OUT_MANIFEST_ROLES = frozenset({"held_out", "safety_only"})
_SESSION_OUTCOMES = frozenset({"success", "failure", "unknown"})
_MUTABLE_MODEL_REVISIONS = frozenset({"latest", "stable", "default", "current"})
_CANDIDATE_DIFF_FACTORY_TOKEN = object()
_CANDIDATE_ENVELOPE_FACTORY_TOKEN = object()
_VERIFIED_DISCOVERY_ENVELOPE_FACTORY_TOKEN = object()
_HELD_OUT_ENVELOPE_FACTORY_TOKEN = object()
_CONFIRMATION_ENVELOPE_FACTORY_TOKEN = object()
_PINNED_GENERATOR_AUTHORIZATION_FACTORY_TOKEN = object()
_PREPARED_INVOCATION_FACTORY_TOKEN = object()
_QUALIFIED_ANALYST_FACTORY_TOKEN = object()


def _require_text(name: str, value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_digest(name: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _require_role(name: str, value: object, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} is invalid")


def _require_nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class _DuplicateUsageKeyError(ValueError):
    pass


def _usage_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateUsageKeyError(key)
        value[key] = item
    return value


def _canonical_usage_json(value: str | None) -> str | None:
    if value is None:
        return None
    _require_text("usage_json", value)
    try:
        parsed = json.loads(value, object_pairs_hook=_usage_object)
    except _DuplicateUsageKeyError as exc:
        raise ValueError("usage_json must not contain duplicate object keys") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("usage_json must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("usage_json must be a JSON object")
    try:
        return canonical_json_bytes(parsed).decode()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "usage_json must contain canonical JSON-compatible values"
        ) from exc


def _require_immutable_model_revision(name: str, value: object) -> None:
    _require_text(name, value)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be non-empty text")
    if value.strip().casefold() in _MUTABLE_MODEL_REVISIONS:
        raise ValueError(f"{name} cannot be a mutable alias")


def _freeze_tuple(name: str, value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{name} must be an immutable tuple or list")


class _StrictOutput(StrictStructuredOutput):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_json_arrays_for_tuple_fields(cls, data: object) -> object:
        """Coerce JSON arrays into tuples before strict validation.

        JSON has no tuple type -- every provider returns an array -- and
        ``strict=True`` refuses to coerce one into a ``tuple``. Without this,
        every ``tuple[...]`` field on these models is UNSATISFIABLE by any
        model, forever: a live run returned a well-formed, correctly-named
        abstention and still failed with ``observations:tuple_type``,
        ``citations:tuple_type`` and three siblings.

        The coercion is deliberately narrow. It runs only where the field's
        own annotation is a tuple, so a field genuinely typed ``list`` still
        rejects a tuple and vice versa -- strictness is preserved everywhere
        except the one place JSON cannot express the distinction. Tuples also
        keep these models hashable, which ``frozen=True`` depends on, so
        widening the annotations to ``list`` was not an option.

        Args:
            data (object): The raw payload, normally a mapping decoded from
                the provider's JSON. Anything else is passed through
                untouched for pydantic to reject.

        Returns:
            object: The payload with array values converted to tuples for
                tuple-annotated fields.
        """
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        for name, info in cls.model_fields.items():
            if get_origin(info.annotation) is not tuple:
                continue
            value = coerced.get(name)
            if type(value) is list:
                coerced[name] = tuple(value)
        return coerced


class EvidenceCitation(_StrictOutput):
    span_id: str
    exact_quote: str

    @field_validator("span_id", "exact_quote")
    @classmethod
    def require_text(cls, value: str) -> str:
        _require_text(cls.__name__, value)
        return value


class DiscoveryMemo(_StrictOutput):
    """One discovery analyst's memo, which may be a grounded hypothesis or an abstention.

    ``hypothesis`` is nullable and NOT defaulted, exactly like
    ``OutcomeContractDescriptor.success_definition``: the key stays in the
    emitted schema's ``required`` list, so a strict structured-output endpoint
    still obliges the model to answer the field, and the model answers an
    abstention by writing ``null`` rather than by omitting the key.
    """

    decision: Literal["grounded_hypothesis", "abstain"]
    hypothesis: str | None
    observations: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    relevance_to_success: str
    alternative_explanations: tuple[str, ...]
    counterevidence: tuple[str, ...]
    scope: str
    uncertainties: tuple[str, ...]

    @field_validator("relevance_to_success", "scope")
    @classmethod
    def require_text(cls, value: str) -> str:
        _require_text(cls.__name__, value)
        return value

    @model_validator(mode="after")
    def validate_decision_fields(self) -> Self:
        """Require hypothesis text only under ``grounded_hypothesis``.

        An analyst that reads the evidence correctly and finds no grounded
        behavior contrast owes us an abstention. Requiring ``hypothesis``
        unconditionally made that abstention unrepresentable -- the memo the
        model was right to produce was rejected by its own schema -- so a
        correct analyst and a broken one failed identically.

        ``abstain`` therefore admits ``hypothesis=None``. It deliberately does
        NOT admit blank text: ``""`` is neither a hypothesis nor a declared
        absence, and admitting it would put a value in the field that every
        reader has to special-case. ``None`` is the only way to say "no
        hypothesis".

        The other required fields stay required under BOTH decisions on
        purpose. An abstention still has to say what it examined
        (``relevance_to_success``) and over what range (``scope``) -- the
        prompt asks for scope unconditionally, and "I could not ground a
        contrast" is an answer those two fields can carry. Making them
        conditional would let an abstention arrive with no stated reason.

        Text is TOLERATED, not required, alongside ``abstain``: the packaged
        qualification fixtures carry one, and ``_verified_discovery_envelope``
        downgrades a grounded memo to ``abstain`` via ``model_copy`` -- which
        does not re-run validators -- so forbidding it would mint objects that
        fail their own schema on the next round-trip.
        """
        # Required under 'grounded_hypothesis'; and whenever a value IS present
        # it must be real text, so 'abstain' can say None but never "".
        if self.decision == "grounded_hypothesis" or self.hypothesis is not None:
            _require_text("hypothesis", self.hypothesis)
        return self


class CandidateRevisionV1(_StrictOutput):
    revised_content: str
    change_rationale: str
    declared_scope: str
    supporting_citation_ids: tuple[str, ...]

    @field_validator("revised_content", "change_rationale", "declared_scope")
    @classmethod
    def require_text(cls, value: str) -> str:
        _require_text(cls.__name__, value)
        return value

    @field_validator("supporting_citation_ids")
    @classmethod
    def require_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("supporting_citation_ids must not be empty")
        for citation_id in value:
            _require_text("supporting_citation_ids", citation_id)
        if len(set(value)) != len(value):
            raise ValueError("supporting_citation_ids must be unique")
        return value


class HeldOutSessionMemo(_StrictOutput):
    control: Literal["support", "refute", "insufficient"]
    scope_assessment: Literal["preserved", "expanded", "unresolved"]
    safety_assessment: Literal["clear", "blocker", "unresolved"]
    observations: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    alternative_explanations: tuple[str, ...]
    uncertainties: tuple[str, ...]


class ConfirmationSessionMemo(_StrictOutput):
    """One online session's three-valued verdict on a live provisional successor.

    Deliberately ``HeldOutSessionMemo`` minus ``scope_assessment`` and
    ``safety_assessment``. Those two exist because the held-out analyst is
    judging WHETHER TO PUBLISH; the confirmation analyst is judging a successor
    that is already live and whose scope was settled at publication. There is
    no field here that can hold a hypothesis, a scope, or a reason to keep the
    candidate: ``_StrictOutput`` forbids extras, so an invented key is a parse
    failure. The analyst has exactly one three-valued channel into the decision.
    """

    control: Literal["support", "refute", "insufficient"]
    observations: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    alternative_explanations: tuple[str, ...]
    uncertainties: tuple[str, ...]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AcceptDecision(_StrictFrozenModel):
    decision: Literal["accept"]
    reason: None = None


class AbstainDecision(_StrictFrozenModel):
    decision: Literal["abstain"]
    reason: Literal[
        "no_grounded_hypothesis",
        "evidence_view_incomplete",
        "heldout_evidence_failed",
    ]


AttemptDecisionV1 = Annotated[
    AcceptDecision | AbstainDecision,
    Field(discriminator="decision"),
]


# 'regeneration_fenced' is a refusal, not a fault: the attempt-start fence
# found a restored lifecycle burned on this exact evidence, so regenerating
# would buy a candidate that has already been pulled back once. It sits beside
# 'stale_incumbent' -- the other reason meaning "the world says do not spend on
# this" -- and so terminalizes through the same 'failed' stage rather than
# through an abstain decision. An abstain would have to write a durable
# AttemptDecisionEnvelope, and that artifact's authorization function
# (20260820010000) requires any abstain reason other than
# 'no_grounded_hypothesis' to bind an exact durable candidate -- which a fenced
# attempt, having generated nothing, does not have.
# 'invocation_slot_pinned' means NO PROVIDER CALL WAS MADE. The durable row
# identity this question resolves to is already owned by a different
# optimization job, so the attempt failed while claiming its slot rather than
# while asking anything.
#
# That collision is structural, not transient. `analyst_input_identity` is
# derived WITHOUT `job_id` and the discovery bundle is quantized per UTC day, so
# two jobs for the same playbook on the same day compute the SAME identity, and
# `prepare_offline_tuner_open_world_invocation` refuses to rebind it. The
# identity cannot be advanced past the collision: the tenant authorization
# functions recompute that exact preimage before admitting a completed artifact
# (see the DELIBERATELY ABSENT note in `identity`), so an advanced row is one
# the SQL can never authorize -- and advancing it anyway billed for an answer
# that was then discarded.
#
# It is separated from 'infrastructure_failure' because the two were previously
# indistinguishable in every report, and that is exactly what hid the identity
# burn: a provider fault stranded a `prepared` row, the next job then died at
# prepare without calling anything, and both surfaced as the same reason. A
# reader could only tell them apart by querying the invocations table by hand.
# Splitting the reason is the whole remedy -- the burn is now VISIBLE rather
# than avoided, and it clears on its own at the next UTC day boundary, when the
# bundle digest changes and no collision exists.
#
# It is a FAILURE and not an abstention: nothing was judged, so there is no
# decision to record, and it terminalizes through the same 'failed' stage as its
# siblings. (Contrast the mechanical citation abstention, which is carried out
# of band in job metadata precisely BECAUSE the attempt did converge on a
# decision -- see diagnostics.CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY.)
# 'bundle_schema_unsupported' means THIS BUNDLE CAN NEVER BE LOADED, by any
# host, on any retry. The frozen artifact declares an evidence-bundle schema
# version that predates this image, and the bundle is validated as the EXACT
# derivation of its manifest -- so an older bundle is not degraded, it is
# unloadable. Split out of 'infrastructure_failure' for the same reason
# 'invocation_slot_pinned' was: that reason promises "transient, we will try
# again", which is the opposite of the true instruction here (drain the row, or
# terminalize it deliberately, before the new image serves). A stranded job read
# exactly like a flaky provider, and the only way to tell them apart was to read
# the artifact's schema_version by hand.
#
# It is a FAILURE and not an abstention: the refusal happens BEFORE derivation,
# so no analyst was called and there is no decision artifact to write.
# Unidentified v1 decision envelopes also retire through this existing outcome:
# their old bytes remain intact, and the legacy-policy diagnostic plus
# policy_transition flag distinguishes this from an unloadable evidence bundle.
AttemptTerminalFailureReason = Literal[
    "analyst_unqualified",
    "stale_incumbent",
    "governance_invalidated",
    "infrastructure_failure",
    "invocation_slot_pinned",
    "regeneration_fenced",
    "bundle_schema_unsupported",
    # Two sweep ticks collided on one playbook's frozen evidence and the loser
    # was refused at the manifest-membership check, having analysed nothing.
    #
    # A FAILURE and not an abstention, for the same reason as its siblings
    # above: nothing was judged, so there is no decision artifact to write and
    # it terminalizes through the 'failed' stage. Split out of
    # `infrastructure_failure`, which reads as "a provider fault, try again" --
    # the opposite of what this is. Like `invocation_slot_pinned` it clears on
    # its own at the next UTC day boundary, when the bundle digest moves.
    #
    # Recorded rather than skipped silently, because the conflict IS the
    # concurrency signal: a silent skip erases the only evidence that two ticks
    # ever collided.
    "evidence_manifest_conflict",
]

# Derived, never restated. This vocabulary was hand-typed twice -- once as the
# `Literal` above and once as an inline `frozenset` in the validator below --
# and the two were kept in step by hand, which is how a value admitted by the
# type but rejected at runtime (or the reverse) gets in. Every other role-set in
# this module is already one named module-level constant; this makes the last
# one match. A test asserts the two expressions still agree.
_ATTEMPT_TERMINAL_FAILURE_REASONS: frozenset[str] = frozenset(
    get_args(AttemptTerminalFailureReason)
)


@dataclass(frozen=True)
class AttemptTerminalFailure:
    #: The persisted terminal outcome. One of the migrated vocabulary above,
    #: and nothing else -- ``advance_playbook_optimization_stage`` returns
    #: ``false`` rather than raising on an unlisted value, so a reason the
    #: tenant function has never heard of leaves the row silently non-terminal.
    reason: AttemptTerminalFailureReason
    #: Whether the provider RATE-LIMITED this attempt after
    #: ``analyst.py``'s backoff budget was spent.
    #:
    #: NOT a terminal outcome, and deliberately not one. A rate limit is the
    #: one failure that says nothing whatever about this playbook -- the
    #: question was never asked -- so the day's sweep-progress record has to
    #: stay RETRY-ELIGIBLE rather than be spent on it. Widening
    #: ``AttemptTerminalFailureReason`` to say so would mean a second
    #: ``CREATE OR REPLACE FUNCTION`` migration of the stage-advance
    #: vocabulary, and two independently-authored ones do not merge: the later
    #: silently drops the earlier's values, which is what ``20260904000000``
    #: exists to repair. A boolean beside the reason buys the signal the sweep
    #: needs without touching a persisted contract.
    #:
    #: The job-level attribution is not lost by keeping it out of the reason:
    #: the terminal diagnostic records the exception class chain, so
    #: ``OpenWorldProviderRateLimitedError`` is named on the durable row.
    rate_limited: bool = False
    #: An unidentified v1 decision was retired without judging or publishing it.
    #: A fresh attempt may bind the current policy; the old artifact is retained.
    policy_transition: bool = False
    #: The provider returned a COMPLETE reply that the method's schema rejected,
    #: on the client's corrective retry as well -- the model answered, and the
    #: answer was invalid. Not a provider or platform fault, so the sweep label
    #: must not read ``infrastructure_failure``.
    #:
    #: A flag beside the reason for the same cause as ``rate_limited``: the
    #: persisted vocabulary cannot widen without a stage-advance migration, so
    #: the durable row keeps ``infrastructure_failure`` and its terminal
    #: diagnostic records the exception chain and failing field paths. Only the
    #: sweep's progress label (``_sweep_outcome_label``) reads this. Unlike a
    #: rate limit it still SPENDS the day: a re-ask on the same frozen evidence
    #: would be paid for again with no new information.
    output_rejected: bool = False
    #: Value-free attribution for the outcome LOG line (refusal name, raised
    #: class and location, deepest cause): see
    #: ``diagnostics.diagnostic_log_fragment``. Never persisted -- the durable
    #: copy is the terminal diagnostic on the job row -- and ``None`` when the
    #: attempt did not fail on an exception. Excluded from equality: two
    #: failures with the same persisted outcome are the same outcome.
    failure_detail: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _require_role("reason", self.reason, _ATTEMPT_TERMINAL_FAILURE_REASONS)


class OpenWorldCapabilityUnavailableError(ValueError):
    """A qualified component cannot provide the trusted invocation capability."""


OptimizationAttemptReason = Literal[
    "no_grounded_hypothesis",
    "evidence_view_incomplete",
    "heldout_evidence_failed",
    "analyst_unqualified",
    "stale_incumbent",
    "governance_invalidated",
    "infrastructure_failure",
    "invocation_slot_pinned",
    "regeneration_fenced",
    "bundle_schema_unsupported",
]


@dataclass(frozen=True)
class EvidenceSpan:
    """One citable run of frozen text, and the server's record of who said it.

    ``manifest_role`` is the ARM this span was drawn into -- discovery,
    held_out or safety_only -- and says nothing about the speaker. ``speaker``
    is the speaker, and the two were conflated for exactly as long as the
    analyst had no way to tell them apart: shown only ``manifest_role`` and a
    ``field_path`` of ``requests[0].interactions[N].content``, it could not see
    which turns were the agent's, and said so unprompted --

        "the spans shown ... do not identify the speaker of each turn, so any
         contrast that could be drawn is on the user-input side rather than on
         what the agent did, declined to do, or failed to do."

    while the v1.1.0 discovery prompt simultaneously demanded an agent-side
    contrast and forbade user-side hypotheses. Those two facts were
    incompatible, and the observed workaround was worse than the gap: a sibling
    run GUESSED at speakers from content, spotting ``Generating ...`` echoes.

    ``speaker`` is derived server-side by ``classify_turn_speaker`` from the
    stored interaction's ``role`` column. It is never inferred from the text and
    never supplied by the model.
    """

    span_id: str
    tenant_id: str
    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    request_id: str
    interaction_id: int
    field_path: str
    immutable_text: str
    speaker: Literal["agent", "user", "other"]

    def __post_init__(self) -> None:
        for name in (
            "span_id",
            "tenant_id",
            "user_id",
            "session_id",
            "request_id",
            "field_path",
            "immutable_text",
        ):
            _require_text(name, getattr(self, name))
        _require_role("manifest_role", self.manifest_role, _MANIFEST_ROLES)
        _require_role("speaker", self.speaker, _TURN_SPEAKERS)
        _require_nonnegative_int("interaction_id", self.interaction_id)


@dataclass(frozen=True)
class EvidenceDocument:
    """One frozen session's citable text, plus the label the analyst needs.

    ``session_outcome`` is the session's adjudicated outcome under the frozen
    outcome contract, carried through from
    ``OpenWorldEvidenceSession.outcome``. It is present on DISCOVERY documents
    only, and that asymmetry is the point:

    * The discovery analyst is asked to contrast failed against successful
      sessions. Without the label the contrast is not expressible from the
      payload, so the only correct answer is to abstain.
    * The held-out and safety-only analysts judge ONE unseen session against a
      hypothesis that is already fixed. Handing them the ground-truth label
      would let a memo be rationalised from the label instead of from the
      cited spans -- validation-set leakage, in exactly the place the
      generalization check is supposed to be blind.

    An online confirmation session borrows ``manifest_role="held_out"`` and
    belongs to no frozen manifest, so it carries no label either.
    """

    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str
    spans: tuple[EvidenceSpan, ...]
    session_outcome: Literal["success", "failure", "unknown"] | None = None
    compaction_summary: str | None = None

    def __post_init__(self) -> None:
        _require_role("manifest_role", self.manifest_role, _MANIFEST_ROLES)
        if self.manifest_role == "discovery" and self.session_outcome is not None:
            _require_role("session_outcome", self.session_outcome, _SESSION_OUTCOMES)
        elif self.session_outcome is not None:
            raise ValueError(
                "only discovery evidence documents carry a session_outcome"
            )
        _require_text("user_id", self.user_id)
        _require_text("session_id", self.session_id)
        _require_digest("trajectory_digest", self.trajectory_digest)
        object.__setattr__(self, "spans", _freeze_tuple("spans", self.spans))
        for span in self.spans:
            if type(span) is not EvidenceSpan:
                raise TypeError("spans must contain EvidenceSpan values")
            if (
                span.manifest_role != self.manifest_role
                or span.user_id != self.user_id
                or span.session_id != self.session_id
            ):
                raise ValueError("evidence span does not match its document")


@dataclass(frozen=True, init=False)
class CandidateContentDiff:
    incumbent_content_digest: str
    candidate_content_digest: str
    unified_diff: str

    def __init__(
        self,
        incumbent_content_digest: str,
        candidate_content_digest: str,
        unified_diff: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CANDIDATE_DIFF_FACTORY_TOKEN:
            raise TypeError(
                "CandidateContentDiff must be created by the candidate factory"
            )
        _require_digest("incumbent_content_digest", incumbent_content_digest)
        _require_digest("candidate_content_digest", candidate_content_digest)
        _require_text("unified_diff", unified_diff)
        object.__setattr__(self, "incumbent_content_digest", incumbent_content_digest)
        object.__setattr__(self, "candidate_content_digest", candidate_content_digest)
        object.__setattr__(self, "unified_diff", unified_diff)


def _candidate_content_diff(
    *,
    incumbent_content_digest: str,
    candidate_content_digest: str,
    unified_diff: str,
) -> CandidateContentDiff:
    return CandidateContentDiff(
        incumbent_content_digest,
        candidate_content_digest,
        unified_diff,
        _factory_token=_CANDIDATE_DIFF_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class VerifiedCitation:
    citation_id: str
    span_id: str
    exact_quote: str
    start_offset: int
    end_offset: int
    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str

    def __post_init__(self) -> None:
        for name in ("citation_id", "span_id", "exact_quote", "user_id", "session_id"):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("start_offset", self.start_offset)
        _require_nonnegative_int("end_offset", self.end_offset)
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must follow start_offset")
        _require_role("manifest_role", self.manifest_role, _MANIFEST_ROLES)
        _require_digest("trajectory_digest", self.trajectory_digest)


@dataclass(frozen=True)
class ComponentIdentity:
    component_role: Literal["discovery", "candidate", "held_out", "confirmation"]
    provider: str
    immutable_model_revision: str
    prompt_id: str
    prompt_version: str
    prompt_template_digest: str
    output_schema_id: str
    output_schema_version: str
    output_schema_digest: str
    generation_settings_digest: str
    qualification_suite_version: str

    def __post_init__(self) -> None:
        _require_role("component_role", self.component_role, _ANALYST_ROLES)
        for name in (
            "provider",
            "prompt_id",
            "prompt_version",
            "output_schema_id",
            "output_schema_version",
            "qualification_suite_version",
        ):
            _require_text(name, getattr(self, name))
        _require_immutable_model_revision(
            "immutable_model_revision", self.immutable_model_revision
        )
        for name in (
            "prompt_template_digest",
            "output_schema_digest",
            "generation_settings_digest",
        ):
            _require_digest(name, getattr(self, name))

    @property
    def digest(self) -> str:
        from .identity import component_identity_digest

        return component_identity_digest(self)


@dataclass(frozen=True, init=False)
class PinnedGeneratorAuthorization:
    """The canonical authorization binding for the pinned candidate generator."""

    generator_identity: ComponentIdentity
    digest: str

    def __init__(
        self,
        generator_identity: ComponentIdentity,
        digest: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _PINNED_GENERATOR_AUTHORIZATION_FACTORY_TOKEN:
            raise TypeError(
                "PinnedGeneratorAuthorization must be created by the generator factory"
            )
        _require_identity_role("generator_identity", generator_identity, "candidate")
        _require_digest("digest", digest)
        object.__setattr__(self, "generator_identity", generator_identity)
        object.__setattr__(self, "digest", digest)


def _pinned_generator_authorization(
    *, generator_identity: ComponentIdentity, digest: str
) -> PinnedGeneratorAuthorization:
    return PinnedGeneratorAuthorization(
        generator_identity,
        digest,
        _factory_token=_PINNED_GENERATOR_AUTHORIZATION_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class AnalystInputIdentity:
    """The question a paid analyst call asks, and the durable row's identity.

    ``attempt`` scopes the identity so a STRANDED row can be retried. A run
    that reaches the provider and then fails to persist -- a non-terminating
    completion, a parse failure -- leaves its row in ``prepared``, and because
    the row identity is the question identity, every later attempt at the same
    question that day collided with it and terminated as
    ``invocation_slot_pinned``. The pin cleared only at the next UTC day
    boundary, so one flake burned a playbook's whole day.

    Attempt 1 is byte-identical to the pre-attempt identity: the preimage omits
    ``attempt`` entirely and keeps ``...-v1``. That is deliberate and is what
    makes this safe to deploy against a populated table -- every row already
    written keeps its digest, and the SQL authorizer's v1 branch keeps
    authorizing it. Only attempt >= 2 takes the ``...-v2`` preimage.

    A storage backend that recomputes this digest before admitting a
    completed artifact MUST move with it; a Python-only advance prepares a row,
    calls the analyst, gets an answer and then has it refused at persist. See
    the preimage note in ``identity.py``.
    """

    component_identity_digest: str
    qualification_result_digest: str
    rendered_input_digest: str
    outcome_contract_digest: str
    role: Literal["discovery", "candidate", "held_out", "confirmation"]
    attempt: int = 1

    def __post_init__(self) -> None:
        for name in (
            "component_identity_digest",
            "qualification_result_digest",
            "rendered_input_digest",
            "outcome_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        _require_role("role", self.role, _ANALYST_ROLES)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")

    @property
    def digest(self) -> str:
        from .identity import analyst_input_identity_digest

        return analyst_input_identity_digest(self)


@dataclass(frozen=True)
class InvocationIntent:
    job_id: int
    analyst_input_identity: str
    component_identity_digest: str
    input_digest: str
    role: Literal["discovery", "candidate", "held_out", "confirmation"]
    expected_provider: str
    expected_model_revision: str
    manifest_role: Literal["held_out", "safety_only"] | None
    user_id: str | None
    session_id: str | None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        for name in (
            "analyst_input_identity",
            "component_identity_digest",
            "input_digest",
        ):
            _require_digest(name, getattr(self, name))
        _require_role("role", self.role, _ANALYST_ROLES)
        _require_text("expected_provider", self.expected_provider)
        _require_immutable_model_revision(
            "expected_model_revision", self.expected_model_revision
        )
        if self.role == "held_out":
            _require_role("manifest_role", self.manifest_role, _HELD_OUT_MANIFEST_ROLES)
            _require_text("user_id", self.user_id)
            _require_text("session_id", self.session_id)
        elif self.role == "confirmation":
            # An ONLINE session identifies a user and a session but belongs to
            # no frozen manifest -- it did not exist when the manifest was
            # frozen. manifest_role IS NULL is therefore asserted here, not
            # merely permitted, and it is what distinguishes the two shapes.
            if self.manifest_role is not None:
                raise ValueError(
                    "confirmation invocations may not bind a manifest role"
                )
            _require_text("user_id", self.user_id)
            _require_text("session_id", self.session_id)
        elif any(
            value is not None
            for value in (self.manifest_role, self.user_id, self.session_id)
        ):
            raise ValueError(
                "only held_out invocations may identify a manifest session"
            )


@dataclass(frozen=True)
class HeldOutCandidateBinding:
    """The durable singleton candidate one held-out call is bound to."""

    artifact_digest: str
    analyst_input_identity: str

    def __post_init__(self) -> None:
        _require_digest("artifact_digest", self.artifact_digest)
        _require_digest("analyst_input_identity", self.analyst_input_identity)


@dataclass(frozen=True, init=False)
class PreparedInvocation:
    """One exact, durable preparation for a provider-idempotent model call.

    Factory-gated: only ``identity.py``'s prepare builders can mint one, and
    each of those requires a factory-authorized ``QualifiedAnalyst`` or
    ``PinnedGeneratorAuthorization``. A caller cannot hand-assemble a
    self-consistent instance around a fabricated qualification digest.
    """

    job_id: int
    bundle_digest: str
    component_identity: ComponentIdentity
    qualification: AnalystQualificationResultRef | PinnedGeneratorAuthorization
    qualification_result_digest: str
    rendered_input_digest: str
    outcome_contract_digest: str
    role: Literal["discovery", "candidate", "held_out", "confirmation"]
    manifest_role: Literal["held_out", "safety_only"] | None
    user_id: str | None
    session_id: str | None
    trajectory_digest: str | None
    candidate_binding: HeldOutCandidateBinding | None
    analyst_input_identity: AnalystInputIdentity
    intent: InvocationIntent
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        job_id: int,
        bundle_digest: str,
        component_identity: ComponentIdentity,
        qualification: AnalystQualificationResultRef | PinnedGeneratorAuthorization,
        qualification_result_digest: str,
        rendered_input_digest: str,
        outcome_contract_digest: str,
        role: Literal["discovery", "candidate", "held_out", "confirmation"],
        manifest_role: Literal["held_out", "safety_only"] | None,
        user_id: str | None,
        session_id: str | None,
        trajectory_digest: str | None,
        candidate_binding: HeldOutCandidateBinding | None,
        analyst_input_identity: AnalystInputIdentity,
        intent: InvocationIntent,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _PREPARED_INVOCATION_FACTORY_TOKEN:
            raise TypeError(
                "PreparedInvocation must be created by the prepare-request factory"
            )
        for name, value in {
            "job_id": job_id,
            "bundle_digest": bundle_digest,
            "component_identity": component_identity,
            "qualification": qualification,
            "qualification_result_digest": qualification_result_digest,
            "rendered_input_digest": rendered_input_digest,
            "outcome_contract_digest": outcome_contract_digest,
            "role": role,
            "manifest_role": manifest_role,
            "user_id": user_id,
            "session_id": session_id,
            "trajectory_digest": trajectory_digest,
            "candidate_binding": candidate_binding,
            "analyst_input_identity": analyst_input_identity,
            "intent": intent,
            "_factory_token": _factory_token,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        _require_digest("bundle_digest", self.bundle_digest)
        self._require_authorization()
        _require_identity_role("component_identity", self.component_identity, self.role)
        for name in (
            "qualification_result_digest",
            "rendered_input_digest",
            "outcome_contract_digest",
        ):
            _require_digest(name, getattr(self, name))
        _require_role("role", self.role, _ANALYST_ROLES)
        expected_input = AnalystInputIdentity(
            component_identity_digest=self.component_identity.digest,
            qualification_result_digest=self.qualification_result_digest,
            rendered_input_digest=self.rendered_input_digest,
            outcome_contract_digest=self.outcome_contract_digest,
            role=self.role,
            # `attempt` is carried over rather than re-derived, because unlike
            # every other field it is not a function of the request: it is the
            # caller's retry ordinal, and there is nothing here to derive it
            # from. Each remaining field is still checked, so a stale identity
            # is still caught -- what this permits is precisely a second
            # attempt at the SAME question, which is the point.
            attempt=self.analyst_input_identity.attempt,
        )
        if self.analyst_input_identity != expected_input:
            raise ValueError("prepared analyst input identity is stale")
        expected_intent = InvocationIntent(
            job_id=self.job_id,
            analyst_input_identity=expected_input.digest,
            component_identity_digest=self.component_identity.digest,
            input_digest=self.rendered_input_digest,
            role=self.role,
            expected_provider=self.component_identity.provider,
            expected_model_revision=self.component_identity.immutable_model_revision,
            manifest_role=self.manifest_role,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        if self.intent != expected_intent:
            raise ValueError("prepared invocation intent is stale")

    def _require_authorization(self) -> None:
        """Bind the exact authorization artifact this call was minted from."""
        if self.role == "candidate":
            if type(self.qualification) is not PinnedGeneratorAuthorization:
                raise TypeError(
                    "candidate calls require a PinnedGeneratorAuthorization"
                )
            authorized_digest = self.qualification.digest
            authorized_identity = self.qualification.generator_identity.digest
        else:
            if type(self.qualification) is not AnalystQualificationResultRef:
                raise TypeError(
                    "analyst calls require an AnalystQualificationResultRef"
                )
            if not self.qualification.passed:
                raise ValueError("prepared invocation analyst is not qualified")
            authorized_digest = self.qualification.result_digest
            authorized_identity = self.qualification.component_identity_digest
        if self.component_identity.digest != authorized_identity:
            raise ValueError("prepared invocation authorization does not bind identity")
        if self.qualification_result_digest != authorized_digest:
            raise ValueError("prepared invocation qualification is stale")
        if self.role in ("held_out", "confirmation"):
            # A confirmation call binds the exact online trajectory it judged
            # and the exact durable candidate that was serving, on the same
            # terms as a held-out call -- the same pair the durable invocation
            # row's binding CHECK requires of both roles.
            _require_digest("trajectory_digest", self.trajectory_digest)
            if type(self.candidate_binding) is not HeldOutCandidateBinding:
                raise TypeError(
                    "held-out and confirmation calls require a HeldOutCandidateBinding"
                )
        elif self.trajectory_digest is not None or self.candidate_binding is not None:
            raise ValueError(
                "only held_out invocations bind a trajectory and durable candidate"
            )

    @property
    def idempotency_key(self) -> str:
        """The durable row identity, which is also the base provider call key.

        This is the QUESTION's own identity, and it is deliberately NOT advanced
        per attempt: the tenant authorization functions recompute exactly this
        preimage before admitting a completed artifact, so any Python-side
        advancement produces a row the SQL can never authorize. See the
        DELIBERATELY ABSENT note in ``identity``.

        ``repair_attempt_idempotency_key`` layers on top of this value, so the
        provider key for call ``m`` within one durable attempt is distinct
        without either derivation knowing about the other.
        """
        return self.analyst_input_identity.digest


def _prepared_invocation(**values: object) -> PreparedInvocation:
    return PreparedInvocation(
        **values,  # type: ignore[arg-type]
        _factory_token=_PREPARED_INVOCATION_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class InvocationRecord:
    analyst_input_identity: str
    component_identity_digest: str
    output_digest: str
    observed_provider: str
    observed_model_revision: str
    called_at: int
    latency_ms: int
    usage_json: str | None

    def __post_init__(self) -> None:
        for name in (
            "analyst_input_identity",
            "component_identity_digest",
            "output_digest",
        ):
            _require_digest(name, getattr(self, name))
        _require_text("observed_provider", self.observed_provider)
        _require_immutable_model_revision(
            "observed_model_revision", self.observed_model_revision
        )
        _require_nonnegative_int("called_at", self.called_at)
        _require_nonnegative_int("latency_ms", self.latency_ms)
        object.__setattr__(self, "usage_json", _canonical_usage_json(self.usage_json))


@dataclass(frozen=True)
class ObservedStructuredCompletion:
    value: StrictStructuredOutput
    observed_provider: str
    observed_model_revision: str
    called_at: int
    latency_ms: int
    usage_json: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.value, StrictStructuredOutput):
            raise TypeError("value must be StrictStructuredOutput")
        _require_text("observed_provider", self.observed_provider)
        _require_immutable_model_revision(
            "observed_model_revision", self.observed_model_revision
        )
        _require_nonnegative_int("called_at", self.called_at)
        _require_nonnegative_int("latency_ms", self.latency_ms)
        object.__setattr__(self, "usage_json", _canonical_usage_json(self.usage_json))


@dataclass(frozen=True)
class AnalystQualificationResultRef:
    component_identity_digest: str
    suite_digest: str
    result_digest: str
    passed: bool

    def __post_init__(self) -> None:
        for name in ("component_identity_digest", "suite_digest", "result_digest"):
            _require_digest(name, getattr(self, name))
        if type(self.passed) is not bool:
            raise ValueError("passed must be boolean")


@dataclass(frozen=True, init=False)
class VerifiedDiscoveryEnvelope:
    bundle_digest: str
    memo: DiscoveryMemo
    observed_memo: DiscoveryMemo
    verified_citations: tuple[VerifiedCitation, ...]
    analyst_identity: ComponentIdentity
    qualification_result_digest: str
    invocation: InvocationRecord
    prepared_invocation: PreparedInvocation
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        bundle_digest: str,
        memo: DiscoveryMemo,
        observed_memo: DiscoveryMemo | None = None,
        verified_citations: tuple[VerifiedCitation, ...],
        analyst_identity: ComponentIdentity,
        qualification_result_digest: str,
        invocation: InvocationRecord,
        prepared_invocation: PreparedInvocation,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _VERIFIED_DISCOVERY_ENVELOPE_FACTORY_TOKEN:
            raise TypeError(
                "VerifiedDiscoveryEnvelope must be created by the discovery factory"
            )
        raw_memo = memo if observed_memo is None else observed_memo
        for name, value in {
            "bundle_digest": bundle_digest,
            "memo": memo,
            "observed_memo": raw_memo,
            "verified_citations": verified_citations,
            "analyst_identity": analyst_identity,
            "qualification_result_digest": qualification_result_digest,
            "invocation": invocation,
            "prepared_invocation": prepared_invocation,
            "_factory_token": _factory_token,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_digest("bundle_digest", self.bundle_digest)
        object.__setattr__(
            self,
            "verified_citations",
            _freeze_tuple("verified_citations", self.verified_citations),
        )
        if type(self.memo) is not DiscoveryMemo:
            raise TypeError("memo must be DiscoveryMemo")
        if type(self.observed_memo) is not DiscoveryMemo:
            raise TypeError("observed_memo must be DiscoveryMemo")
        _require_identity_role("analyst_identity", self.analyst_identity, "discovery")
        _require_digest("qualification_result_digest", self.qualification_result_digest)
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst_identity,
            qualification_result_digest=self.qualification_result_digest,
            role="discovery",
        )
        if self.prepared_invocation.bundle_digest != self.bundle_digest:
            raise ValueError("prepared invocation bundle is stale")
        _require_invocation(
            "invocation",
            self.invocation,
            self.prepared_invocation,
            self.observed_memo,
        )
        _require_citation_projection(
            self.memo.citations,
            self.verified_citations,
            role="discovery",
            require_nonempty=self.memo.decision == "grounded_hypothesis",
        )


def _projected_citations(
    verified: tuple[VerifiedCitation, ...],
) -> tuple[EvidenceCitation, ...]:
    """Restate verified citations in the shape a memo carries.

    ``_require_citation_projection`` demands that a memo's citations equal its
    verified citations on ``(span_id, exact_quote)``. When the verifier repairs
    a mistyped span ID by resolving the citation on its quote, the RESOLVED ID
    is the one that was actually verified, so the memo has to carry it or the
    envelope fails its own invariant.

    Rewriting the memo does not lose what the model said: ``observed_memo`` is
    retained verbatim on every envelope, so a repaired transcription stays
    auditable as the difference between the two.

    Args:
        verified (tuple[VerifiedCitation, ...]): Citations the verifier resolved

    Returns:
        tuple[EvidenceCitation, ...]: The same citations as memo-shaped values
    """
    return tuple(
        EvidenceCitation(span_id=item.span_id, exact_quote=item.exact_quote)
        for item in verified
    )


def _verified_discovery_envelope(
    *,
    request: DiscoveryRequest,
    observed_memo: DiscoveryMemo,
    invocation: InvocationRecord,
) -> VerifiedDiscoveryEnvelope:
    from .citations import (
        CitationVerificationError,
        verify_grounded_discovery_citations,
    )

    if type(request) is not DiscoveryRequest:
        raise TypeError("request must be DiscoveryRequest")
    if type(observed_memo) is not DiscoveryMemo:
        raise TypeError("observed_memo must be DiscoveryMemo")
    allowed_spans = tuple(
        span for document in request.discovery_sessions for span in document.spans
    )
    try:
        # NOT bare `verify_citations`. The wrapper adds the one clause of the
        # discovery contract that was stated only in prose: a
        # `grounded_hypothesis` must rest on at least one span the AGENT spoke.
        # Every other clause -- allowlist, byte-exact quoting, ambiguity,
        # duplication -- was already mechanical, so a memo citing only the
        # customer's own turns verified cleanly and was accepted. It raises
        # `no_agent_side_citation`, a category distinct from every quote-level
        # one, so the two causes stay apart in the diagnostic record.
        verified_citations = verify_grounded_discovery_citations(
            observed_memo.citations,
            allowed_spans=allowed_spans,
            decision=observed_memo.decision,
        )
    # NARROW on purpose. `CitationVerificationError` is the only failure the
    # MODEL is answerable for, and downgrading the memo to an abstention is the
    # right answer to it. `verify_citations` also raises bare `ValueError` for
    # defects in the spans the SERVER supplied -- "allowed spans contain
    # duplicate span IDs", "allowed span ID is not a rehydrated evidence span"
    # -- and catching those here charged the analyst for a span-construction
    # bug, producing an abstention indistinguishable from a real judgement.
    # Server defects must escape as errors, not be laundered into evidence.
    except CitationVerificationError:
        verified_citations = ()
        memo = observed_memo.model_copy(update={"decision": "abstain", "citations": ()})
    else:
        memo = observed_memo
        projected = _projected_citations(verified_citations)
        if projected != tuple(observed_memo.citations):
            memo = memo.model_copy(update={"citations": projected})
        if memo.decision == "grounded_hypothesis" and not verified_citations:
            memo = memo.model_copy(update={"decision": "abstain", "citations": ()})

    return VerifiedDiscoveryEnvelope(
        bundle_digest=request.bundle_digest,
        memo=memo,
        observed_memo=observed_memo,
        verified_citations=verified_citations,
        analyst_identity=request.analyst.identity,
        qualification_result_digest=request.analyst.qualification.result_digest,
        invocation=invocation,
        prepared_invocation=request.prepared_invocation,
        _factory_token=_VERIFIED_DISCOVERY_ENVELOPE_FACTORY_TOKEN,
    )


@dataclass(frozen=True, init=False)
class CandidateEnvelope:
    bundle_digest: str
    incumbent_full_version_fingerprint: str
    revision: CandidateRevisionV1
    content_diff: CandidateContentDiff
    verified_discovery_citation_ids: tuple[str, ...]
    generator_identity: ComponentIdentity
    invocation: InvocationRecord
    prepared_invocation: PreparedInvocation
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        bundle_digest: str,
        incumbent_full_version_fingerprint: str,
        revision: CandidateRevisionV1,
        content_diff: CandidateContentDiff,
        verified_discovery_citation_ids: tuple[str, ...],
        generator_identity: ComponentIdentity,
        invocation: InvocationRecord,
        prepared_invocation: PreparedInvocation,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CANDIDATE_ENVELOPE_FACTORY_TOKEN:
            raise TypeError(
                "CandidateEnvelope must be created by the candidate factory"
            )
        for name, value in {
            "bundle_digest": bundle_digest,
            "incumbent_full_version_fingerprint": incumbent_full_version_fingerprint,
            "revision": revision,
            "content_diff": content_diff,
            "verified_discovery_citation_ids": verified_discovery_citation_ids,
            "generator_identity": generator_identity,
            "invocation": invocation,
            "prepared_invocation": prepared_invocation,
            "_factory_token": _factory_token,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_digest("bundle_digest", self.bundle_digest)
        object.__setattr__(
            self,
            "verified_discovery_citation_ids",
            _freeze_tuple(
                "verified_discovery_citation_ids",
                self.verified_discovery_citation_ids,
            ),
        )
        _require_digest(
            "incumbent_full_version_fingerprint",
            self.incumbent_full_version_fingerprint,
        )
        if type(self.revision) is not CandidateRevisionV1:
            raise TypeError("revision must be CandidateRevisionV1")
        if type(self.content_diff) is not CandidateContentDiff:
            raise TypeError("content_diff must be CandidateContentDiff")
        if (
            self.content_diff.candidate_content_digest
            != hashlib.sha256(self.revision.revised_content.encode()).hexdigest()
        ):
            raise ValueError("content_diff does not bind revised content")
        if (
            self.content_diff.incumbent_content_digest
            == self.content_diff.candidate_content_digest
        ):
            raise ValueError("candidate content is unchanged")
        if not self.verified_discovery_citation_ids:
            raise ValueError("verified_discovery_citation_ids must not be empty")
        for citation_id in self.verified_discovery_citation_ids:
            _require_text("verified_discovery_citation_ids", citation_id)
        if len(set(self.verified_discovery_citation_ids)) != len(
            self.verified_discovery_citation_ids
        ):
            raise ValueError("verified_discovery_citation_ids must be unique")
        if set(self.revision.supporting_citation_ids) != set(
            self.verified_discovery_citation_ids
        ):
            raise ValueError("revision contains an unverified discovery citation")
        _require_identity_role(
            "generator_identity", self.generator_identity, "candidate"
        )
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.generator_identity,
            qualification_result_digest=self.prepared_invocation.qualification_result_digest,
            role="candidate",
        )
        if self.prepared_invocation.bundle_digest != self.bundle_digest:
            raise ValueError("prepared invocation bundle is stale")
        _require_invocation(
            "invocation", self.invocation, self.prepared_invocation, self.revision
        )


def _candidate_envelope(**values: object) -> CandidateEnvelope:
    return CandidateEnvelope(
        bundle_digest=cast(str, values["bundle_digest"]),
        incumbent_full_version_fingerprint=cast(
            str, values["incumbent_full_version_fingerprint"]
        ),
        revision=cast(CandidateRevisionV1, values["revision"]),
        content_diff=cast(CandidateContentDiff, values["content_diff"]),
        verified_discovery_citation_ids=cast(
            tuple[str, ...], values["verified_discovery_citation_ids"]
        ),
        generator_identity=cast(ComponentIdentity, values["generator_identity"]),
        invocation=cast(InvocationRecord, values["invocation"]),
        prepared_invocation=cast(PreparedInvocation, values["prepared_invocation"]),
        _factory_token=_CANDIDATE_ENVELOPE_FACTORY_TOKEN,
    )


@dataclass(frozen=True, init=False)
class HeldOutEnvelope:
    bundle_digest: str
    manifest_role: Literal["held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str
    memo: HeldOutSessionMemo
    observed_memo: HeldOutSessionMemo
    verified_citations: tuple[VerifiedCitation, ...]
    analyst_identity: ComponentIdentity
    qualification_result_digest: str
    invocation: InvocationRecord
    prepared_invocation: PreparedInvocation
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        bundle_digest: str,
        manifest_role: Literal["held_out", "safety_only"],
        user_id: str,
        session_id: str,
        trajectory_digest: str,
        memo: HeldOutSessionMemo,
        observed_memo: HeldOutSessionMemo,
        verified_citations: tuple[VerifiedCitation, ...],
        analyst_identity: ComponentIdentity,
        qualification_result_digest: str,
        invocation: InvocationRecord,
        prepared_invocation: PreparedInvocation,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _HELD_OUT_ENVELOPE_FACTORY_TOKEN:
            raise TypeError("HeldOutEnvelope must be created by the held-out factory")
        for name, value in {
            "bundle_digest": bundle_digest,
            "manifest_role": manifest_role,
            "user_id": user_id,
            "session_id": session_id,
            "trajectory_digest": trajectory_digest,
            "memo": memo,
            "observed_memo": observed_memo,
            "verified_citations": verified_citations,
            "analyst_identity": analyst_identity,
            "qualification_result_digest": qualification_result_digest,
            "invocation": invocation,
            "prepared_invocation": prepared_invocation,
            "_factory_token": _factory_token,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_digest("bundle_digest", self.bundle_digest)
        object.__setattr__(
            self,
            "verified_citations",
            _freeze_tuple("verified_citations", self.verified_citations),
        )
        _require_role("manifest_role", self.manifest_role, _HELD_OUT_MANIFEST_ROLES)
        _require_text("user_id", self.user_id)
        _require_text("session_id", self.session_id)
        _require_digest("trajectory_digest", self.trajectory_digest)
        if type(self.memo) is not HeldOutSessionMemo:
            raise TypeError("memo must be HeldOutSessionMemo")
        if type(self.observed_memo) is not HeldOutSessionMemo:
            raise TypeError("observed_memo must be HeldOutSessionMemo")
        _require_identity_role("analyst_identity", self.analyst_identity, "held_out")
        _require_digest("qualification_result_digest", self.qualification_result_digest)
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst_identity,
            qualification_result_digest=self.qualification_result_digest,
            role="held_out",
            manifest_role=self.manifest_role,
            user_id=self.user_id,
            session_id=self.session_id,
            trajectory_digest=self.trajectory_digest,
        )
        if self.prepared_invocation.bundle_digest != self.bundle_digest:
            raise ValueError("prepared invocation bundle is stale")
        _require_invocation(
            "invocation",
            self.invocation,
            self.prepared_invocation,
            self.observed_memo,
        )
        _require_citation_projection(
            self.memo.citations,
            self.verified_citations,
            role=self.manifest_role,
            user_id=self.user_id,
            session_id=self.session_id,
            trajectory_digest=self.trajectory_digest,
            require_nonempty=self.memo.control in {"support", "refute"},
        )


def _held_out_envelope(
    *,
    request: HeldOutSessionRequest,
    observed_memo: HeldOutSessionMemo,
    invocation: InvocationRecord,
) -> HeldOutEnvelope:
    from .citations import CitationVerificationError, verify_citations

    if type(request) is not HeldOutSessionRequest:
        raise TypeError("request must be HeldOutSessionRequest")
    if type(observed_memo) is not HeldOutSessionMemo:
        raise TypeError("observed_memo must be HeldOutSessionMemo")
    try:
        verified_citations = verify_citations(
            observed_memo.citations,
            allowed_spans=request.evidence.spans,
        )
    # Narrow for the reason given in `_verified_discovery_envelope`: only a
    # model-attributable citation failure may become an "insufficient" verdict.
    except CitationVerificationError:
        verified_citations = ()
        memo = observed_memo.model_copy(
            update={
                "control": "insufficient",
                "scope_assessment": "unresolved",
                "safety_assessment": "unresolved",
                "citations": (),
            }
        )
    else:
        memo = observed_memo
        projected = _projected_citations(verified_citations)
        if projected != tuple(observed_memo.citations):
            memo = memo.model_copy(update={"citations": projected})
        if memo.control in {"support", "refute"} and not verified_citations:
            memo = memo.model_copy(
                update={
                    "control": "insufficient",
                    "scope_assessment": "unresolved",
                    "safety_assessment": "unresolved",
                    "citations": (),
                }
            )

    return HeldOutEnvelope(
        bundle_digest=request.bundle_digest,
        manifest_role=request.manifest_role,
        user_id=request.evidence.user_id,
        session_id=request.evidence.session_id,
        trajectory_digest=request.evidence.trajectory_digest,
        memo=memo,
        observed_memo=observed_memo,
        verified_citations=verified_citations,
        analyst_identity=request.analyst.identity,
        qualification_result_digest=request.analyst.qualification.result_digest,
        invocation=invocation,
        prepared_invocation=request.prepared_invocation,
        _factory_token=_HELD_OUT_ENVELOPE_FACTORY_TOKEN,
    )


class UnsupportedReducerPolicyError(RuntimeError):
    """Leave an attempt resumable for a worker implementing its persisted policy."""


class LegacyReducerPolicyError(UnsupportedReducerPolicyError):
    """An unidentified v1 decision must retire without inferred policy authority."""


@dataclass(frozen=True)
class AttemptDecisionEnvelope:
    bundle_digest: str
    candidate_digest: str | None
    held_out_population_digest: str | None
    decision: AttemptDecisionV1
    reducer_policy_id: str = REDUCER_POLICY_ID

    def __post_init__(self) -> None:
        if self.reducer_policy_id != REDUCER_POLICY_ID:
            raise UnsupportedReducerPolicyError(
                f"unsupported reducer policy: {self.reducer_policy_id}"
            )
        _require_digest("bundle_digest", self.bundle_digest)
        if type(self.decision) not in (AcceptDecision, AbstainDecision):
            raise TypeError("decision must be an AttemptDecisionV1")
        if (
            type(self.decision) is AbstainDecision
            and self.decision.reason in NO_CANDIDATE_ABSTAIN_REASONS
        ):
            if (
                self.candidate_digest is not None
                or self.held_out_population_digest is not None
            ):
                raise ValueError(
                    "an early abstention must not fabricate candidate identities"
                )
            return
        for name in ("candidate_digest", "held_out_population_digest"):
            _require_digest(name, getattr(self, name))


def _publication_context_authority_payload(
    authority: QualificationAuthorityRef,
) -> dict[str, object]:
    return {
        "qualification_authority": {
            "epoch": authority.epoch,
            "authority_digest": authority.authority_digest,
        },
        "discovery": {
            "component_identity_digest": (
                authority.discovery_component_identity_digest
            ),
            "qualification_suite_digest": (
                authority.discovery_qualification_suite_digest
            ),
            "qualification_result_digest": (
                authority.discovery_qualification_result_digest
            ),
        },
        "held_out": {
            "component_identity_digest": authority.held_out_component_identity_digest,
            "qualification_suite_digest": (
                authority.held_out_qualification_suite_digest
            ),
            "qualification_result_digest": (
                authority.held_out_qualification_result_digest
            ),
        },
        "candidate": {
            "generator_identity_digest": (
                authority.candidate_generator_identity_digest
            ),
            "authorization_digest": (
                authority.candidate_generator_authorization_digest
            ),
        },
    }


def _validate_publication_context_identity(
    *,
    context_json: str,
    context_digest: str,
    bundle_digest: str,
    outcome_contract_digest: str,
    authority: QualificationAuthorityRef,
) -> dict[str, object]:
    _require_digest("attempt_context_digest", context_digest)
    if hashlib.sha256(context_json.encode()).hexdigest() != context_digest:
        raise ValueError("attempt context digest does not match its JSON")
    try:
        payload = json.loads(context_json)
    except json.JSONDecodeError as exc:
        raise ValueError("attempt context is not valid JSON") from exc
    expected_identity = {
        "contract_type": "open_world_attempt_context",
        "schema_version": "v2",
        "bundle_digest": bundle_digest,
        "outcome_contract_digest": outcome_contract_digest,
        **_publication_context_authority_payload(authority),
    }
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("manifest"), list)
        or any(payload.get(name) != value for name, value in expected_identity.items())
        or set(payload) != {*expected_identity, "manifest"}
    ):
        raise ValueError("attempt context does not bind the durable publication input")
    return payload


@dataclass(frozen=True)
class OpenWorldAcceptedPublicationInputs:
    """Exact durable records authorized to enter provisional publication."""

    job_id: int
    attempt_key: str
    incumbent_user_playbook_id: int
    incumbent_full_version_fingerprint: str
    evidence_bundle_digest: str
    outcome_contract_digest: str
    candidate: CandidateEnvelope
    candidate_digest: str
    decision: AttemptDecisionEnvelope
    decision_digest: str
    attempt_context_json: str
    attempt_context_digest: str
    subject_epochs: tuple[tuple[str, int], ...]
    qualification_authority: QualificationAuthorityRef
    claim: OptimizationJobClaim

    def __post_init__(self) -> None:
        from .identity import canonical_bytes
        from .reducer import candidate_digest

        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("publication input job_id must be positive")
        _require_text("publication input attempt_key", self.attempt_key)
        if (
            type(self.incumbent_user_playbook_id) is not int
            or self.incumbent_user_playbook_id <= 0
        ):
            raise ValueError("publication input incumbent ID must be positive")
        for name in (
            "incumbent_full_version_fingerprint",
            "evidence_bundle_digest",
            "outcome_contract_digest",
            "candidate_digest",
            "decision_digest",
            "attempt_context_digest",
        ):
            _require_digest(name, getattr(self, name))
        if type(self.candidate) is not CandidateEnvelope:
            raise TypeError("publication input candidate must be a CandidateEnvelope")
        if (
            type(self.decision) is not AttemptDecisionEnvelope
            or type(self.decision.decision) is not AcceptDecision
        ):
            raise ValueError("publication input decision must be accepted")
        if type(self.qualification_authority) is not QualificationAuthorityRef:
            raise TypeError("publication input authority must be typed")
        if (
            type(self.claim) is not OptimizationJobClaim
            or self.claim.job_id != self.job_id
        ):
            raise ValueError("publication input claim does not authorize the job")
        if self.candidate_digest != candidate_digest(self.candidate):
            raise ValueError("publication input candidate digest is stale")
        if (
            self.decision_digest
            != hashlib.sha256(canonical_bytes(self.decision)).hexdigest()
        ):
            raise ValueError("publication input decision digest is stale")
        if (
            self.candidate.prepared_invocation.job_id != self.job_id
            or self.candidate.bundle_digest != self.evidence_bundle_digest
            or self.candidate.prepared_invocation.bundle_digest
            != self.evidence_bundle_digest
            or self.candidate.prepared_invocation.outcome_contract_digest
            != self.outcome_contract_digest
            or self.candidate.incumbent_full_version_fingerprint
            != self.incumbent_full_version_fingerprint
            or self.decision.bundle_digest != self.evidence_bundle_digest
            or self.decision.candidate_digest != self.candidate_digest
        ):
            raise ValueError("publication input artifacts do not cross-bind")
        authority = self.qualification_authority
        if (
            self.candidate.generator_identity.digest
            != authority.candidate_generator_identity_digest
            or self.candidate.prepared_invocation.qualification_result_digest
            != authority.candidate_generator_authorization_digest
        ):
            raise ValueError("publication input candidate authority is stale")
        _validate_publication_context_identity(
            context_json=self.attempt_context_json,
            context_digest=self.attempt_context_digest,
            bundle_digest=self.evidence_bundle_digest,
            outcome_contract_digest=self.outcome_contract_digest,
            authority=authority,
        )
        if not isinstance(self.subject_epochs, tuple) or not self.subject_epochs:
            raise ValueError("publication input governance subjects must be non-empty")
        for entry in self.subject_epochs:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or type(entry[0]) is not str
                or not entry[0]
                or type(entry[1]) is not int
                or entry[1] < 0
            ):
                raise ValueError("publication input governance subject is invalid")
        normalized = tuple(
            sorted(self.subject_epochs, key=lambda item: item[0].encode())
        )
        if normalized != self.subject_epochs or len(
            {ref for ref, _ in normalized}
        ) != len(normalized):
            raise ValueError("publication input governance subjects must be canonical")


@dataclass(frozen=True, init=False)
class QualifiedAnalyst:
    """One analyst proven to have passed the canonical qualification suite.

    The only mint is ``qualification.seal_qualified_analyst``, which requires a
    reducer-emitted ``AnalystQualificationResult`` and derives the reference
    from it. That sealed result is retained (``_qualification_result``) so the
    seal stays checkable at every authorization boundary: a caller-authored
    ``AnalystQualificationResultRef`` cannot become authorization here.
    """

    identity: ComponentIdentity
    qualification: AnalystQualificationResultRef
    _qualification_result: object = field(init=False, repr=False, compare=False)
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        identity: ComponentIdentity,
        qualification: AnalystQualificationResultRef,
        _qualification_result: object = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _QUALIFIED_ANALYST_FACTORY_TOKEN:
            raise TypeError(
                "QualifiedAnalyst must be created by a qualification factory"
            )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "qualification", qualification)
        object.__setattr__(self, "_qualification_result", _qualification_result)
        object.__setattr__(self, "_factory_token", _factory_token)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_analyst_reference(self.identity, self.qualification)
        _require_sealed_authorization(self)


@dataclass(frozen=True)
class ProvisionalCaseAnalyst:
    """A non-authorizing analyst reference used only to render suite cases.

    Rendering a qualification case into a production request needs an analyst
    reference before any qualification result exists. This carries the same
    identity/reference shape a request serializes, but it is a distinct type
    with no sealed result, so it can never enter a ``QualifiedAnalystSet`` and
    therefore can never authorize an attempt.
    """

    identity: ComponentIdentity
    qualification: AnalystQualificationResultRef

    def __post_init__(self) -> None:
        _require_analyst_reference(self.identity, self.qualification)


type AttemptAnalystRef = QualifiedAnalyst | ProvisionalCaseAnalyst


@dataclass(frozen=True, init=False)
class QualifiedAnalystSet:
    """The complete authorized analyst/generator set for one attempt."""

    discovery: QualifiedAnalyst
    held_out: QualifiedAnalyst
    generator_identity: ComponentIdentity
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        discovery: QualifiedAnalyst,
        held_out: QualifiedAnalyst,
        generator_identity: ComponentIdentity,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _QUALIFIED_ANALYST_FACTORY_TOKEN:
            raise TypeError(
                "QualifiedAnalystSet must be created by a qualification factory"
            )
        object.__setattr__(self, "discovery", discovery)
        object.__setattr__(self, "held_out", held_out)
        object.__setattr__(self, "generator_identity", generator_identity)
        object.__setattr__(self, "_factory_token", _factory_token)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_qualified_role("discovery", self.discovery, "discovery")
        _require_qualified_role("held_out", self.held_out, "held_out")
        _require_identity_role(
            "generator_identity", self.generator_identity, "candidate"
        )


QualificationAuthorityState = Literal["active", "unavailable"]


@dataclass(frozen=True)
class ActiveOpenWorldQualificationAuthority:
    epoch: int
    state: Literal["active"]
    authority_digest: str
    qualification: QualificationAuthorityRef
    unavailable_reason: None
    updated_at: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("qualification authority epoch must be positive")
        if self.state != "active":
            raise ValueError("active qualification authority state is invalid")
        _require_digest("authority_digest", self.authority_digest)
        if type(self.qualification) is not QualificationAuthorityRef:
            raise TypeError("qualification must be QualificationAuthorityRef")
        if (
            self.qualification.epoch != self.epoch
            or self.qualification.authority_digest != self.authority_digest
        ):
            raise ValueError("qualification authority reference is stale")
        if self.unavailable_reason is not None:
            raise ValueError("active qualification authority has no unavailable reason")
        _require_nonnegative_int("updated_at", self.updated_at)


@dataclass(frozen=True)
class UnavailableOpenWorldQualificationAuthority:
    epoch: int
    state: Literal["unavailable"]
    authority_digest: str
    qualification: None
    unavailable_reason: str
    updated_at: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("qualification authority epoch must be positive")
        if self.state != "unavailable":
            raise ValueError("unavailable qualification authority state is invalid")
        _require_digest("authority_digest", self.authority_digest)
        if self.qualification is not None:
            raise ValueError("unavailable qualification authority has no qualification")
        _require_text("unavailable_reason", self.unavailable_reason)
        _require_nonnegative_int("updated_at", self.updated_at)


type OpenWorldQualificationAuthority = (
    ActiveOpenWorldQualificationAuthority | UnavailableOpenWorldQualificationAuthority
)


def _qualified_analyst_set(
    *,
    discovery: QualifiedAnalyst,
    held_out: QualifiedAnalyst,
    generator_identity: ComponentIdentity,
) -> QualifiedAnalystSet:
    return QualifiedAnalystSet(
        discovery=discovery,
        held_out=held_out,
        generator_identity=generator_identity,
        _factory_token=_QUALIFIED_ANALYST_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class RejectedQuote:
    """One quote the citation verifier could not find, echoed back to its author.

    ``exact_quote`` is the MODEL's own string, never a span's text. That is the
    whole point: a repair prompt that quoted the evidence back would hand the
    analyst text it had failed to cite, which is indistinguishable from feeding
    it new evidence.
    """

    exact_quote: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("exact_quote", "reason"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CitationRepairRequest:
    """The server's account of why the previous discovery answer was rejected.

    This travels INSIDE the canonical untrusted-data payload rather than as an
    extra chat turn. That is deliberate: ``guard_exact_request`` refuses a
    request whose message list differs from the qualified one, because a
    free-form repair turn shows the model a prompt no qualification case ever
    attested. A schema'd field of the discovery input keeps the message shape
    the qualified two -- system prompt plus one framed payload -- while still
    telling the analyst what to fix.

    It carries no span text, no span IDs the model did not already write, and no
    offsets. Everything in it is either the model's own prior output or a fixed
    category string, which is what makes a repair attempt unable to widen the
    evidence the analyst can see.
    """

    attempt: int
    rejected_quotes: tuple[RejectedQuote, ...]

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt < 2:
            raise ValueError("attempt must be an integer of 2 or more")
        object.__setattr__(
            self,
            "rejected_quotes",
            _freeze_tuple("rejected_quotes", self.rejected_quotes),
        )
        if not self.rejected_quotes:
            raise ValueError("rejected_quotes must not be empty")
        for quote in self.rejected_quotes:
            if type(quote) is not RejectedQuote:
                raise TypeError("rejected_quotes must contain RejectedQuote values")


@dataclass(frozen=True)
class RejectedCitationId:
    """One supporting citation ID the candidate named that does not exist.

    ``citation_id`` is the MODEL's own string, echoed back so the generator can
    see what it actually wrote. It is never one of the server's verified IDs:
    the verified set is already in the candidate's input as
    ``verified_discovery_citation_ids``, so repeating it here would add nothing,
    and echoing a REAL ID next to a rejected one risks the model copying the
    wrong neighbour.
    """

    citation_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("citation_id", "reason"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CandidateCitationRepairRequest:
    """The server's account of why the previous candidate revision was rejected.

    The discovery analyst's sibling of this class repairs QUOTES; this one
    repairs IDS, and the difference is not cosmetic. A discovery citation
    carries two coordinates into the evidence -- a span ID and a quote -- so a
    mistyped ID can be recovered from the quote, which is the coordinate models
    transcribe reliably. A supporting citation ID has no second coordinate.
    Nothing corroborates that a 63-character string was *meant* to be a
    particular 64-character one, so the ID is never repaired by resemblance;
    the generator is asked again, and equality stays exact.

    Like its sibling it travels INSIDE the canonical payload rather than as an
    extra chat turn, because ``guard_exact_request`` refuses a message list that
    differs from the qualified one.

    It carries only the model's own prior output and fixed reason strings, so a
    repair attempt cannot widen the evidence the generator can see.
    """

    attempt: int
    rejected_citation_ids: tuple[RejectedCitationId, ...]

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt < 2:
            raise ValueError("attempt must be an integer of 2 or more")
        object.__setattr__(
            self,
            "rejected_citation_ids",
            _freeze_tuple("rejected_citation_ids", self.rejected_citation_ids),
        )
        if not self.rejected_citation_ids:
            raise ValueError("rejected_citation_ids must not be empty")
        for rejected in self.rejected_citation_ids:
            if type(rejected) is not RejectedCitationId:
                raise TypeError(
                    "rejected_citation_ids must contain RejectedCitationId values"
                )


class CandidateCitationRejectionError(ValueError):
    """A candidate named supporting citations outside the verified set.

    Subclasses ``ValueError`` so that an exhausted repair still converges
    exactly where an unrepaired rejection always did. It exists to carry the
    offending IDs to the repair turn: the runner cannot tell the generator what
    to fix from a bare message string, and re-deriving the difference at the
    catch site would be a second expression of the rule
    ``build_candidate_envelope`` already applies.
    """

    def __init__(self, rejected_citation_ids: tuple[str, ...]) -> None:
        super().__init__("candidate contains an unverified discovery citation")
        self.rejected_citation_ids = rejected_citation_ids


@dataclass(frozen=True)
class DiscoveryRequest:
    job_id: int
    bundle_digest: str
    discovery_sessions: tuple[EvidenceDocument, ...]
    outcome_contract: OutcomeContractDescriptor
    analyst: AttemptAnalystRef
    rendered_input_digest: str
    prepared_invocation: PreparedInvocation
    citation_repair: CitationRepairRequest | None = None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        if self.citation_repair is not None and (
            type(self.citation_repair) is not CitationRepairRequest
        ):
            raise TypeError("citation_repair must be a CitationRepairRequest")
        _require_digest("bundle_digest", self.bundle_digest)
        object.__setattr__(
            self,
            "discovery_sessions",
            _freeze_tuple("discovery_sessions", self.discovery_sessions),
        )
        for document in self.discovery_sessions:
            if (
                type(document) is not EvidenceDocument
                or document.manifest_role != "discovery"
            ):
                raise ValueError("discovery_sessions must contain discovery documents")
        _require_outcome_contract(self.outcome_contract)
        _require_attempt_analyst_role("analyst", self.analyst, "discovery")
        _require_digest("rendered_input_digest", self.rendered_input_digest)
        from .identity import discovery_request_input_digest

        # `citation_repair` is deliberately absent here. It is not part of the
        # question's identity -- see `discovery_model_input` -- so a repair
        # attempt carries the SAME rendered input digest as the attempt it
        # repairs, and occupies the same durable invocation slot.
        if self.rendered_input_digest != discovery_request_input_digest(
            bundle_digest=self.bundle_digest,
            discovery_sessions=self.discovery_sessions,
            outcome_contract=self.outcome_contract,
            analyst=self.analyst,
        ):
            raise ValueError("request input digest is stale")
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst.identity,
            qualification_result_digest=self.analyst.qualification.result_digest,
            role="discovery",
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            rendered_input_digest=self.rendered_input_digest,
            outcome_contract_digest=self.outcome_contract.digest,
        )

    @property
    def idempotency_key(self) -> str:
        return self.prepared_invocation.idempotency_key


@dataclass(frozen=True)
class CandidateRequest:
    job_id: int
    bundle_digest: str
    incumbent_content: str
    incumbent_full_version_fingerprint: str
    discovery: VerifiedDiscoveryEnvelope
    outcome_contract: OutcomeContractDescriptor
    generator_authorization: PinnedGeneratorAuthorization
    rendered_input_digest: str
    prepared_invocation: PreparedInvocation
    # Excluded from `candidate_request_input_digest`, and therefore from
    # `analyst_input_identity`, for the reason the discovery sibling states: a
    # repair does not ask a second question, it re-asks this one. The durable
    # row and the one-candidate-per-job slot are shared across attempts.
    citation_repair: CandidateCitationRepairRequest | None = None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        if self.citation_repair is not None and (
            type(self.citation_repair) is not CandidateCitationRepairRequest
        ):
            raise TypeError("citation_repair must be a CandidateCitationRepairRequest")
        _require_digest("bundle_digest", self.bundle_digest)
        _require_text("incumbent_content", self.incumbent_content)
        _require_digest(
            "incumbent_full_version_fingerprint",
            self.incumbent_full_version_fingerprint,
        )
        if type(self.discovery) is not VerifiedDiscoveryEnvelope:
            raise TypeError("discovery must be VerifiedDiscoveryEnvelope")
        if self.discovery.bundle_digest != self.bundle_digest:
            raise ValueError("discovery does not bind bundle")
        _require_outcome_contract(self.outcome_contract)
        if type(self.generator_authorization) is not PinnedGeneratorAuthorization:
            raise TypeError(
                "generator_authorization must be PinnedGeneratorAuthorization"
            )
        _require_digest("rendered_input_digest", self.rendered_input_digest)
        from .identity import candidate_request_input_digest

        if self.rendered_input_digest != candidate_request_input_digest(
            bundle_digest=self.bundle_digest,
            incumbent_content=self.incumbent_content,
            incumbent_full_version_fingerprint=self.incumbent_full_version_fingerprint,
            discovery=self.discovery,
            outcome_contract=self.outcome_contract,
            generator_authorization=self.generator_authorization,
        ):
            raise ValueError("request input digest is stale")
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.generator_authorization.generator_identity,
            qualification_result_digest=self.generator_authorization.digest,
            role="candidate",
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            rendered_input_digest=self.rendered_input_digest,
            outcome_contract_digest=self.outcome_contract.digest,
        )

    @property
    def idempotency_key(self) -> str:
        return self.prepared_invocation.idempotency_key

    @property
    def generator_identity(self) -> ComponentIdentity:
        return self.generator_authorization.generator_identity


@dataclass(frozen=True)
class HeldOutSessionRequest:
    job_id: int
    bundle_digest: str
    manifest_role: Literal["held_out", "safety_only"]
    evidence: EvidenceDocument
    discovery: VerifiedDiscoveryEnvelope
    candidate: CandidateEnvelope
    outcome_contract: OutcomeContractDescriptor
    analyst: AttemptAnalystRef
    rendered_input_digest: str
    prepared_invocation: PreparedInvocation

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        _require_digest("bundle_digest", self.bundle_digest)
        _require_role("manifest_role", self.manifest_role, _HELD_OUT_MANIFEST_ROLES)
        if type(self.evidence) is not EvidenceDocument:
            raise TypeError("evidence must be EvidenceDocument")
        if self.evidence.manifest_role != self.manifest_role:
            raise ValueError("manifest_role must match evidence")
        if type(self.discovery) is not VerifiedDiscoveryEnvelope:
            raise TypeError("discovery must be VerifiedDiscoveryEnvelope")
        if type(self.candidate) is not CandidateEnvelope:
            raise TypeError("candidate must be CandidateEnvelope")
        if (
            self.discovery.bundle_digest != self.bundle_digest
            or self.candidate.bundle_digest != self.bundle_digest
        ):
            raise ValueError("request artifacts do not bind bundle")
        _require_outcome_contract(self.outcome_contract)
        _require_attempt_analyst_role("analyst", self.analyst, "held_out")
        _require_digest("rendered_input_digest", self.rendered_input_digest)
        from .identity import held_out_request_input_digest

        if self.rendered_input_digest != held_out_request_input_digest(
            bundle_digest=self.bundle_digest,
            manifest_role=self.manifest_role,
            evidence=self.evidence,
            discovery=self.discovery,
            candidate=self.candidate,
            outcome_contract=self.outcome_contract,
            analyst=self.analyst,
        ):
            raise ValueError("request input digest is stale")
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst.identity,
            qualification_result_digest=self.analyst.qualification.result_digest,
            role="held_out",
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            rendered_input_digest=self.rendered_input_digest,
            outcome_contract_digest=self.outcome_contract.digest,
            manifest_role=self.manifest_role,
            user_id=self.evidence.user_id,
            session_id=self.evidence.session_id,
            trajectory_digest=self.evidence.trajectory_digest,
            candidate_binding=held_out_candidate_binding(self.candidate),
        )

    @property
    def idempotency_key(self) -> str:
        return self.prepared_invocation.idempotency_key


@dataclass(frozen=True)
class ConfirmationSessionRequest:
    """One online session put to the confirmation analyst.

    ``HeldOutSessionRequest`` minus ``manifest_role``: an online session belongs
    to no frozen manifest, because it did not exist when the manifest was
    frozen. Its evidence document still carries the ``held_out`` manifest role,
    which is asserted rather than derived -- that pin is the second wall behind
    L5, because it makes a discovery-corpus document unusable here even before
    citation verification narrows the addressable spans to this one trajectory.

    L2 lives in ``__post_init__``: ``rendered_input_digest`` is RECOMPUTED from
    exactly the frozen artifacts this request carries, so a request that
    substituted a different hypothesis, candidate, evidence document or outcome
    contract cannot be constructed at all.
    """

    job_id: int
    bundle_digest: str
    evidence: EvidenceDocument
    discovery: VerifiedDiscoveryEnvelope
    candidate: CandidateEnvelope
    outcome_contract: OutcomeContractDescriptor
    analyst: AttemptAnalystRef
    rendered_input_digest: str
    prepared_invocation: PreparedInvocation

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job_id must be positive")
        _require_digest("bundle_digest", self.bundle_digest)
        if type(self.evidence) is not EvidenceDocument:
            raise TypeError("evidence must be EvidenceDocument")
        if self.evidence.manifest_role != "held_out":
            raise ValueError("confirmation evidence must carry the held_out role")
        if type(self.discovery) is not VerifiedDiscoveryEnvelope:
            raise TypeError("discovery must be VerifiedDiscoveryEnvelope")
        if type(self.candidate) is not CandidateEnvelope:
            raise TypeError("candidate must be CandidateEnvelope")
        if (
            self.discovery.bundle_digest != self.bundle_digest
            or self.candidate.bundle_digest != self.bundle_digest
        ):
            raise ValueError("request artifacts do not bind bundle")
        _require_outcome_contract(self.outcome_contract)
        _require_attempt_analyst_role("analyst", self.analyst, "confirmation")
        _require_digest("rendered_input_digest", self.rendered_input_digest)
        from .identity import confirmation_request_input_digest

        if self.rendered_input_digest != confirmation_request_input_digest(
            bundle_digest=self.bundle_digest,
            evidence=self.evidence,
            discovery=self.discovery,
            candidate=self.candidate,
            outcome_contract=self.outcome_contract,
            analyst=self.analyst,
        ):
            raise ValueError("request input digest is stale")
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst.identity,
            qualification_result_digest=self.analyst.qualification.result_digest,
            role="confirmation",
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            rendered_input_digest=self.rendered_input_digest,
            outcome_contract_digest=self.outcome_contract.digest,
            manifest_role=None,
            user_id=self.evidence.user_id,
            session_id=self.evidence.session_id,
            trajectory_digest=self.evidence.trajectory_digest,
            candidate_binding=held_out_candidate_binding(self.candidate),
        )
        if self.prepared_invocation.manifest_role is not None:
            raise ValueError("confirmation invocation cannot bind a manifest role")

    @property
    def idempotency_key(self) -> str:
        return self.prepared_invocation.idempotency_key


@dataclass(frozen=True, init=False)
class ConfirmationEnvelope:
    """One completed confirmation call, sealed by its factory.

    ``HeldOutEnvelope`` minus ``manifest_role``, carrying a
    ``ConfirmationSessionMemo``.
    """

    bundle_digest: str
    user_id: str
    session_id: str
    trajectory_digest: str
    memo: ConfirmationSessionMemo
    observed_memo: ConfirmationSessionMemo
    verified_citations: tuple[VerifiedCitation, ...]
    analyst_identity: ComponentIdentity
    qualification_result_digest: str
    invocation: InvocationRecord
    prepared_invocation: PreparedInvocation
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        bundle_digest: str,
        user_id: str,
        session_id: str,
        trajectory_digest: str,
        memo: ConfirmationSessionMemo,
        observed_memo: ConfirmationSessionMemo,
        verified_citations: tuple[VerifiedCitation, ...],
        analyst_identity: ComponentIdentity,
        qualification_result_digest: str,
        invocation: InvocationRecord,
        prepared_invocation: PreparedInvocation,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CONFIRMATION_ENVELOPE_FACTORY_TOKEN:
            raise TypeError(
                "ConfirmationEnvelope must be created by the confirmation factory"
            )
        for name, value in {
            "bundle_digest": bundle_digest,
            "user_id": user_id,
            "session_id": session_id,
            "trajectory_digest": trajectory_digest,
            "memo": memo,
            "observed_memo": observed_memo,
            "verified_citations": verified_citations,
            "analyst_identity": analyst_identity,
            "qualification_result_digest": qualification_result_digest,
            "invocation": invocation,
            "prepared_invocation": prepared_invocation,
            "_factory_token": _factory_token,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        _require_digest("bundle_digest", self.bundle_digest)
        object.__setattr__(
            self,
            "verified_citations",
            _freeze_tuple("verified_citations", self.verified_citations),
        )
        _require_text("user_id", self.user_id)
        _require_text("session_id", self.session_id)
        _require_digest("trajectory_digest", self.trajectory_digest)
        if type(self.memo) is not ConfirmationSessionMemo:
            raise TypeError("memo must be ConfirmationSessionMemo")
        if type(self.observed_memo) is not ConfirmationSessionMemo:
            raise TypeError("observed_memo must be ConfirmationSessionMemo")
        _require_identity_role(
            "analyst_identity", self.analyst_identity, "confirmation"
        )
        _require_digest("qualification_result_digest", self.qualification_result_digest)
        _require_prepared_invocation(
            self.prepared_invocation,
            identity=self.analyst_identity,
            qualification_result_digest=self.qualification_result_digest,
            role="confirmation",
            manifest_role=None,
            user_id=self.user_id,
            session_id=self.session_id,
            trajectory_digest=self.trajectory_digest,
        )
        if self.prepared_invocation.manifest_role is not None:
            raise ValueError("confirmation invocation cannot bind a manifest role")
        if self.prepared_invocation.bundle_digest != self.bundle_digest:
            raise ValueError("prepared invocation bundle is stale")
        _require_invocation(
            "invocation",
            self.invocation,
            self.prepared_invocation,
            self.observed_memo,
        )
        _require_citation_projection(
            self.memo.citations,
            self.verified_citations,
            role="held_out",
            user_id=self.user_id,
            session_id=self.session_id,
            trajectory_digest=self.trajectory_digest,
            require_nonempty=self.memo.control in {"support", "refute"},
        )


def _confirmation_envelope(
    *,
    request: ConfirmationSessionRequest,
    observed_memo: ConfirmationSessionMemo,
    invocation: InvocationRecord,
) -> ConfirmationEnvelope:
    from .citations import CitationVerificationError, verify_citations

    if type(request) is not ConfirmationSessionRequest:
        raise TypeError("request must be ConfirmationSessionRequest")
    if type(observed_memo) is not ConfirmationSessionMemo:
        raise TypeError("observed_memo must be ConfirmationSessionMemo")
    try:
        verified_citations = verify_citations(
            observed_memo.citations,
            allowed_spans=request.evidence.spans,
        )
    # Narrow for the reason given in `_verified_discovery_envelope`: only a
    # model-attributable citation failure may become an "insufficient" verdict.
    except CitationVerificationError:
        verified_citations = ()
        memo = observed_memo.model_copy(
            update={"control": "insufficient", "citations": ()}
        )
    else:
        memo = observed_memo
        projected = _projected_citations(verified_citations)
        if projected != tuple(observed_memo.citations):
            memo = memo.model_copy(update={"citations": projected})
        if memo.control in {"support", "refute"} and not verified_citations:
            memo = memo.model_copy(update={"control": "insufficient", "citations": ()})

    return ConfirmationEnvelope(
        bundle_digest=request.bundle_digest,
        user_id=request.evidence.user_id,
        session_id=request.evidence.session_id,
        trajectory_digest=request.evidence.trajectory_digest,
        memo=memo,
        observed_memo=observed_memo,
        verified_citations=verified_citations,
        analyst_identity=request.analyst.identity,
        qualification_result_digest=request.analyst.qualification.result_digest,
        invocation=invocation,
        prepared_invocation=request.prepared_invocation,
        _factory_token=_CONFIRMATION_ENVELOPE_FACTORY_TOKEN,
    )


Phase3RunResult = AttemptDecisionEnvelope | AttemptTerminalFailure


def held_out_candidate_binding(candidate: CandidateEnvelope) -> HeldOutCandidateBinding:
    """Derive the durable candidate identity one held-out call must bind."""
    from .identity import canonical_bytes

    if type(candidate) is not CandidateEnvelope:
        raise TypeError("candidate must be CandidateEnvelope")
    return HeldOutCandidateBinding(
        artifact_digest=hashlib.sha256(canonical_bytes(candidate)).hexdigest(),
        analyst_input_identity=candidate.invocation.analyst_input_identity,
    )


def _require_identity_role(name: str, value: object, role: str) -> None:
    if type(value) is not ComponentIdentity:
        raise TypeError(f"{name} must be ComponentIdentity")
    if value.component_role != role:
        raise ValueError(f"{name} must have {role} role")


def _require_invocation(
    name: str,
    value: object,
    prepared: PreparedInvocation,
    output: StrictStructuredOutput,
) -> None:
    if type(value) is not InvocationRecord:
        raise TypeError(f"{name} must be InvocationRecord")
    if (
        value.analyst_input_identity != prepared.idempotency_key
        or value.component_identity_digest != prepared.component_identity.digest
    ):
        raise ValueError(f"{name} does not bind the prepared analyst input")
    if (
        value.observed_provider != prepared.component_identity.provider
        or value.observed_model_revision
        != prepared.component_identity.immutable_model_revision
    ):
        raise ValueError(f"{name} observed provenance does not match prepared identity")
    from .identity import canonical_bytes

    if value.output_digest != hashlib.sha256(canonical_bytes(output)).hexdigest():
        raise ValueError(f"{name} output digest does not match structured output")


def _require_prepared_invocation(
    value: object,
    *,
    identity: ComponentIdentity,
    qualification_result_digest: str,
    role: str,
    job_id: int | None = None,
    bundle_digest: str | None = None,
    rendered_input_digest: str | None = None,
    outcome_contract_digest: str | None = None,
    manifest_role: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trajectory_digest: str | None = None,
    candidate_binding: HeldOutCandidateBinding | None = None,
) -> None:
    if type(value) is not PreparedInvocation:
        raise TypeError("prepared_invocation must be PreparedInvocation")
    expected = {
        "component_identity": identity,
        "qualification_result_digest": qualification_result_digest,
        "role": role,
    }
    optional_expected = {
        "job_id": job_id,
        "bundle_digest": bundle_digest,
        "rendered_input_digest": rendered_input_digest,
        "outcome_contract_digest": outcome_contract_digest,
        "manifest_role": manifest_role,
        "user_id": user_id,
        "session_id": session_id,
        "trajectory_digest": trajectory_digest,
        "candidate_binding": candidate_binding,
    }
    expected.update(
        {name: item for name, item in optional_expected.items() if item is not None}
    )
    for name, item in expected.items():
        if getattr(value, name) != item:
            raise ValueError(f"prepared invocation {name} is stale")


def _require_citation_projection(
    model_citations: tuple[EvidenceCitation, ...],
    verified_citations: tuple[VerifiedCitation, ...],
    *,
    role: str,
    require_nonempty: bool,
    user_id: str | None = None,
    session_id: str | None = None,
    trajectory_digest: str | None = None,
) -> None:
    if require_nonempty and not model_citations:
        raise ValueError("evidence citation is required for this control")
    if len({citation.citation_id for citation in verified_citations}) != len(
        verified_citations
    ):
        raise ValueError("verified citation ids must be unique")
    model_projection = [
        (citation.span_id, citation.exact_quote) for citation in model_citations
    ]
    verified_projection = [
        (citation.span_id, citation.exact_quote) for citation in verified_citations
    ]
    if len(set(model_projection)) != len(model_projection):
        raise ValueError("model citations must be unique")
    if len(set(verified_projection)) != len(verified_projection):
        raise ValueError("verified citations must be unique")
    if set(model_projection) != set(verified_projection):
        raise ValueError("model citations do not match verified citations")
    for citation in verified_citations:
        if type(citation) is not VerifiedCitation:
            raise TypeError("verified_citations must contain VerifiedCitation values")
        if citation.manifest_role != role:
            raise ValueError("verified citation has the wrong manifest role")
        if user_id is not None and citation.user_id != user_id:
            raise ValueError("verified citation has the wrong user session binding")
        if session_id is not None and citation.session_id != session_id:
            raise ValueError("verified citation has the wrong user session binding")
        if (
            trajectory_digest is not None
            and citation.trajectory_digest != trajectory_digest
        ):
            raise ValueError("verified citation has the wrong trajectory binding")


def _require_analyst_reference(identity: object, qualification: object) -> None:
    if type(identity) is not ComponentIdentity:
        raise TypeError("identity must be ComponentIdentity")
    if identity.component_role not in {"discovery", "held_out", "confirmation"}:
        raise ValueError(
            "analyst references must be discovery, held_out or confirmation"
        )
    if type(qualification) is not AnalystQualificationResultRef:
        raise TypeError("qualification must be AnalystQualificationResultRef")
    if qualification.component_identity_digest != identity.digest:
        raise ValueError("qualification does not bind identity")
    if not qualification.passed:
        raise ValueError("qualified analyst must have passed")


def _require_sealed_authorization(analyst: QualifiedAnalyst) -> None:
    """Require ``analyst`` to carry the reducer seal that authorized it."""
    from .qualification import analyst_authorization_is_sealed

    if not analyst_authorization_is_sealed(analyst):
        raise ValueError(
            "qualified analyst is not sealed by a reducer qualification result"
        )


def _require_qualified_role(name: str, value: object, role: str) -> None:
    if type(value) is not QualifiedAnalyst:
        raise TypeError(f"{name} must be QualifiedAnalyst")
    if value.identity.component_role != role:
        raise ValueError(f"{name} must have {role} role")
    _require_sealed_authorization(value)


def _require_attempt_analyst_role(name: str, value: object, role: str) -> None:
    """Require a request-shaped analyst reference: qualified or provisional.

    A request is not authorization -- ``bind_attempt_context`` is -- so suite
    rendering may use a ``ProvisionalCaseAnalyst`` here.
    """
    if type(value) not in (QualifiedAnalyst, ProvisionalCaseAnalyst):
        raise TypeError(f"{name} must be QualifiedAnalyst or ProvisionalCaseAnalyst")
    analyst = cast(AttemptAnalystRef, value)
    if analyst.identity.component_role != role:
        raise ValueError(f"{name} must have {role} role")


def _require_outcome_contract(value: object) -> None:
    if type(value) is not OutcomeContractDescriptor:
        raise TypeError("outcome_contract must be OutcomeContractDescriptor")
