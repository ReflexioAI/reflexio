"""Deterministic, model-free selection of open-world tuning evidence."""

from __future__ import annotations

from contextvars import ContextVar
from typing import NamedTuple, cast

from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)

from .compaction import (
    DEFAULT_COMPACTION_BATCH_BYTES,
    MAX_COMPACTION_RENDER_TOKENS,
    CompactionError,
    compaction_batch_count,
    compaction_source,
)
from .open_world_models import (
    AWAITING_REACTION_REASON,
    OpenWorldEvidenceEvent,
    OpenWorldEvidenceSelection,
    OpenWorldEvidenceSnapshot,
    OpenWorldExpectedEvent,
    OpenWorldExpectedManifest,
    OpenWorldTargetIdentity,
    OpenWorldTaskContext,
    build_open_world_evidence_bundle,
    classify_turn_speaker,
    validate_open_world_evidence_selection,
    version_start_bound,
)
from .open_world_policy import (
    PRODUCTION_OPEN_WORLD_POLICY,
    analyst_payload_render_tokens,
    analyst_payload_within_limits,
    trajectory_input_within_limits,
)


def _utf8(value: str) -> bytes:
    return value.encode("utf-8")


def _is_full_version_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _trajectory_request_ids(event: OpenWorldEvidenceEvent) -> tuple[str, ...] | None:
    if event.trajectory is None:
        return None
    try:
        request_ids = {
            item["request"]["request_id"] for item in event.trajectory["requests"]
        }
        if not request_ids or any(
            not isinstance(request_id, str) or not request_id
            for request_id in request_ids
        ):
            return None
        return tuple(sorted(request_ids, key=_utf8))
    except (KeyError, TypeError):
        return None


def _trajectory_has_agent_turn(event: OpenWorldEvidenceEvent) -> bool:
    """Whether this trajectory records at least one non-empty agent turn.

    Discovery asks the analyst to contrast the AGENT's behaviour between a
    failed and a successful session. A session the agent never answered carries
    no such behaviour: it can only ever yield user-side spans, and a user-side
    difference is context rather than a finding, so the memo abstains while the
    session still consumes a discovery slot.

    The predicate is a deliberately NARROW allowlist of case-folded role names
    -- because ``role`` is free text (``entities.py`` defaults it to ``"User"``
    and caps it only at 1 000 characters), so no closed vocabulary can be
    assumed. A role name this does not recognise excludes the session LOUDLY:
    it is counted as a ``no_agent_turn`` discovery exclusion and, if that
    starves discovery, abstains the attempt with a categorical reason. Widening
    the predicate to "anything that is not a user turn" would instead admit
    tool or system turns silently, which is the failure this gate exists to
    remove.

    "Loudly" is only true if the operator can tell WHICH failure they are
    looking at. An exclusion here is indistinguishable, on its own, from a
    corpus in which the agent genuinely never spoke -- so when this gate starves
    discovery, the composition layer attaches ``unattributed_role_census`` to
    the abstain, naming the role values that were refused. That is what turns a
    fleet-wide misclassification from a silent, plausible-looking abstain into a
    reported one.

    That narrowness is now expressed ONCE, in ``classify_turn_speaker``, which
    also labels the spans the analyst reads. Sharing one function is the point:
    a gate that admitted a session and a label that then called its turns
    ``other`` would hand the analyst a discovery slot with no attributable
    agent behaviour in it, which is precisely the state this gate exists to
    prevent.

    Emptiness is judged after stripping, one notch stricter than the span
    builder's bare falsiness check, so a whitespace-only agent turn -- which
    would reach the analyst as a blank span -- does not satisfy the gate.

    Malformed input is skipped at the SMALLEST scope that is still well
    defined, never at the scope of the whole trajectory. One interaction
    missing ``role`` or ``content`` says nothing about the interactions beside
    it, so swallowing the entire nested walk on the first ``KeyError`` would
    exclude a session ``no_agent_turn`` that in fact holds a perfectly good
    agent turn in a later request -- a silent eligibility loss reported under a
    reason that is not true of the session. ``unattributed_role_census``
    already handles the same shape by continuing past the malformed entry, and
    the gate and the census must agree about which sessions have agent turns.
    Only a trajectory whose ``requests`` collection is itself unusable is a
    whole-session refusal, because at that point there is no smaller unit left.
    """
    if event.trajectory is None:
        return False
    try:
        request_items = list(event.trajectory["requests"])
    except (KeyError, TypeError):
        return False
    for request_item in request_items:
        try:
            interactions = list(request_item["interactions"])
        except (KeyError, TypeError):
            continue
        for interaction in interactions:
            try:
                speaker = classify_turn_speaker(interaction["role"])
                content = interaction["content"]
            except (KeyError, TypeError):
                continue
            if speaker == "agent" and isinstance(content, str) and content.strip():
                return True
    return False


class _AnalystPayloadCounts(NamedTuple):
    """One walk of a trajectory, measured for two different questions.

    ``content_byte_count`` and ``content_token_count`` answer the CAPACITY
    question about the session -- is this conversation pathological. The extra
    ``span_count`` answers a question about the rendered BUNDLE -- how many
    framed spans this session contributes. They share one walk because they
    must share one skip rule: ``_spans_for_event`` emits nothing for an empty
    ``content``, so a turn counted here that it would skip, or skipped here
    that it would emit, is the same class of measurement error either way.
    """

    content_byte_count: int
    content_token_count: int
    span_count: int


_SESSION_COUNTS: ContextVar[dict[int, _AnalystPayloadCounts | None] | None] = (
    ContextVar("selection_session_counts", default=None)
)
_TOOL_COUNTS: ContextVar[dict[int, bool] | None] = ContextVar(
    "selection_tool_counts", default=None
)
_COMPACTION_COUNTS: ContextVar[dict[tuple[int, int], bool] | None] = ContextVar(
    "selection_compaction_counts", default=None
)


def _analyst_payload_counts(
    event: OpenWorldEvidenceEvent,
) -> _AnalystPayloadCounts | None:
    measurements = _SESSION_COUNTS.get()
    identity = id(event.trajectory)
    if measurements is None:
        return _measure_analyst_payload(event)
    if identity not in measurements:
        measurements[identity] = _measure_analyst_payload(event)
    return measurements[identity]


def _measure_analyst_payload(
    event: OpenWorldEvidenceEvent,
) -> _AnalystPayloadCounts | None:
    """Bytes, cl100k tokens and span count of the analyst's payload.

    The analyst is shown two of the storage row's fifteen fields per
    interaction -- ``content`` and ``role`` -- so those are the two this
    measures. An interaction whose ``content`` is empty is skipped, because
    ``_spans_for_event`` skips it too and it therefore never reaches the
    model; measuring a turn the analyst will not be shown would re-introduce
    the same class of error this change exists to remove, just smaller.

    The raw stored ``role`` is measured rather than the ``classify_turn_speaker``
    label the span actually carries. That over-counts slightly, which is the
    safe direction for a capacity gate, and ``role`` is free text capped at
    1,000 characters, so the over-count is bounded.

    Returns ``None`` when there is nothing to measure -- no trajectory, or a
    ``requests`` collection that is not usable at all. Malformed input below
    that is skipped at the smallest well-defined scope, matching
    ``_trajectory_has_agent_turn``: one interaction missing ``role`` or
    ``content`` says nothing about the interactions beside it.
    """
    if event.trajectory is None:
        return None
    try:
        request_items = list(event.trajectory["requests"])
    except (KeyError, TypeError):
        return None
    parts: list[str] = []
    span_count = 0
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
            if not isinstance(content, str) or not content:
                continue
            parts.append(role if isinstance(role, str) else "")
            parts.append(content)
            span_count += 1
    payload = "\n".join(parts)
    return _AnalystPayloadCounts(
        content_byte_count=len(payload.encode("utf-8")),
        content_token_count=count_input_tokens(payload),
        span_count=span_count,
    )


def _analyst_payload_limit_exceeded(event: OpenWorldEvidenceEvent) -> bool:
    """Whether this session is too large for the analyst to be shown.

    Callers must have established that both stored counts are nonnegative
    ``int``s before calling: that is an INTEGRITY question and it is answered
    separately from this CAPACITY one.

    When the content cannot be measured the storage-row limits stand in. The
    loader discards retained content for the BYTE cap alone, so this branch is
    reached by whitespace-pathological sessions, and standing in keeps them
    classified as a capacity exclusion instead of degrading them to
    ``missing_trajectory``.
    """
    counts = _analyst_payload_counts(event)
    if counts is None:
        return not trajectory_input_within_limits(
            byte_count=cast(int, event.trajectory_byte_count),
            token_count=cast(int, event.trajectory_token_count),
        )
    return not analyst_payload_within_limits(
        content_byte_count=counts.content_byte_count,
        content_token_count=counts.content_token_count,
    )


def _analyst_render_token_count(event: OpenWorldEvidenceEvent) -> int | None:
    """What this session would cost in the rendered analyst payload.

    ``None`` when the trajectory cannot be measured at all -- the same
    condition under which ``_analyst_payload_counts`` returns ``None``. Such an
    event is never selectable (it is already ``missing_trajectory`` or a
    capacity exclusion), so no bundle total is ever built from a ``None``.
    """
    counts = _analyst_payload_counts(event)
    if counts is None:
        return None
    return analyst_payload_render_tokens(
        span_count=counts.span_count,
        content_byte_count=counts.content_byte_count,
    )


def _malformed_trajectory_counts(event: OpenWorldEvidenceEvent) -> bool:
    """Whether the stored counts are unusable as numbers.

    An INTEGRITY question, and the one thing the two capacity return sites did
    NOT share: site 1 gated on this being false and answered only the capacity
    question, while site 2 answered both through a single reason string.
    Splitting them is what stops a negative or non-``int`` count -- real
    corruption -- being reclassified as "merely large" and counted as covered.
    """
    return (
        type(event.trajectory_byte_count) is not int
        or event.trajectory_byte_count < 0
        or type(event.trajectory_token_count) is not int
        or event.trajectory_token_count < 0
    )


def _valid_trajectory_input_limit_exceeded(event: OpenWorldEvidenceEvent) -> bool:
    return not _malformed_trajectory_counts(event) and _analyst_payload_limit_exceeded(
        event
    )


def _trajectory_state_reason(event: OpenWorldEvidenceEvent) -> str | None:
    if event.trajectory_digest is None:
        return "missing_trajectory"
    if not event.trajectory_digest_matches:
        return "trajectory_changed"
    # A retained payload must carry usable request identities before it can
    # be treated as capacity-excluded or awaiting a later reaction.
    if event.trajectory is not None and _trajectory_request_ids(event) is None:
        return "correlation_mismatch"
    if _valid_trajectory_input_limit_exceeded(event):
        # SKIPPED, not returned, and this is where capacity stopped being an
        # integrity answer. It still has to be tested HERE rather than simply
        # deleted, because it guards the arm below: the loader's byte cap
        # discards content mid-stream, so a whitespace-pathological session
        # arrives with no trajectory at all, and falling through would report a
        # SIZE refusal as a LOST one. What excludes it now is
        # ``is_capacity_excluded``.
        return None
    if event.trajectory is None:
        return "missing_trajectory"
    return None


def reconstructability_reason(event: OpenWorldEvidenceEvent) -> str | None:
    """Why this event could not be RECONSTRUCTED, or ``None`` if it was.

    The fixed exhaustive reason ordering, and SIZE IS NOT ONE OF ITS ANSWERS. A
    session refused for being large has a matching digest, an intact
    correlation and valid governance; nothing about it failed to reconstruct.
    Classifying it beside ``missing_trajectory`` and ``trajectory_changed``
    made a corpus with no integrity problem at all fail the 75 percent
    eligibility gate -- 17 over-size sessions out of 74 read as 77 percent, two
    points from the whole tuner going ineligible.

    ``None`` is the ONLY value that moves that ratio: ``coverage_for_reasons``
    counts every non-``None`` entry as a failure, so a new reason string would
    have changed the label and nothing else.

    THE CAPACITY FACT IS NOT LOST -- it moves to :func:`is_capacity_excluded`.
    Reading only this function would tell a caller that an over-size session is
    fine to hand an analyst, which it is not: the bundle still excludes it, and
    the day-14 comparability path still subtracts it.
    """
    if event.tenant_id is None:
        return "missing_tenant"
    if event.exposed_at is None:
        return "missing_timestamp"
    if event.user_playbook_id is None:
        return "missing_playbook_id"
    if event.served_semantic_digest is None:
        return "missing_playbook_digest"
    if event.served_full_version_fingerprint is None:
        return "missing_full_version_fingerprint"
    if not _is_full_version_fingerprint(event.served_full_version_fingerprint):
        return "malformed_full_version_fingerprint"
    if event.correlation_reason is not None:
        return event.correlation_reason
    if event.request_id is None:
        return "missing_request"
    if event.session_id is None:
        return "missing_session"
    if event.user_id is None:
        return "missing_user"
    trajectory_state_reason = _trajectory_state_reason(event)
    if trajectory_state_reason is not None:
        return trajectory_state_reason
    # BEFORE the correlation test below, which it would otherwise fail: the
    # trajectory is the prefix through the outcome's cutover, so a request
    # after the cutover is legitimately absent from it. That is a turn whose
    # reaction has not been judged, not a mis-correlated exposure -- and the
    # reason is coverage-neutral (see `AWAITING_REACTION_REASON`). Tested
    # AFTER the trajectory state, so a prefix that is itself missing or
    # changed still reports that integrity failure first.
    if event.awaiting_reaction:
        return AWAITING_REACTION_REASON
    trajectory_request_ids = _trajectory_request_ids(event)
    if (
        trajectory_request_ids is not None
        and event.request_id not in trajectory_request_ids
    ):
        return "correlation_mismatch"
    # THE INTEGRITY HALF OF SITE 2, AND ONLY THAT HALF. This clause used to
    # answer the capacity question too, through the same reason string, so a
    # blanket reclassification here would have reported a negative or
    # non-``int`` count as "merely large" and let corruption read as a covered
    # session -- the exact inversion the split exists to prevent. Site 1 never
    # conflated the two: it GATES on these counts being well formed.
    if _malformed_trajectory_counts(event):
        return "malformed_trajectory_counts"
    label_reason = _optional_outcome_reason(event)
    if label_reason is not None:
        return label_reason
    if not event.governance_valid:
        return "governance_invalid"
    return None


def _optional_outcome_reason(event: OpenWorldEvidenceEvent) -> str | None:
    # Completion labels are optional context for the evidence-driven tuner.
    # An absent label must not prevent reconstruction of an otherwise intact
    # exposure/trajectory. A supplied label still needs its complete identity.
    if event.outcome is not None:
        if event.outcome not in {"success", "failure", "unknown"}:
            return "invalid_outcome_identity"
        if event.outcome_contract_digest is None:
            return "missing_success_definition"
        if (
            not isinstance(event.outcome_id, str)
            or not event.outcome_id.strip()
            or type(event.outcome_revision) is not int
            or event.outcome_revision <= 0
        ):
            return "invalid_outcome_identity"
    elif any(
        value is not None
        for value in (
            event.outcome_id,
            event.outcome_revision,
            event.outcome_contract_digest,
        )
    ):
        return "invalid_outcome_identity"
    return None


def is_capacity_excluded(event: OpenWorldEvidenceEvent) -> bool:
    """Whether SIZE is the only thing standing between this session and use.

    The other half of the split, carried as its own predicate rather than as a
    reason string -- a string could not be both "this reconstructed" for the
    eligibility ratio and "do not judge this" for the day-14 gate.

    Conjunctive on purpose. A session that is over-size AND corrupt is CORRUPT:
    it answers ``False`` here and keeps its integrity reason, so the fixed
    ordering is not quietly rearranged by asking the size question first.
    ``_analyst_payload_limit_exceeded`` may be called only with well-formed
    counts, which a ``None`` reason has already established.
    """
    return reconstructability_reason(event) is None and _analyst_payload_limit_exceeded(
        event
    )


def _in_window(
    event: OpenWorldEvidenceEvent, *, window_start: int, window_end: int
) -> bool:
    timestamp = event.exposed_at if event.exposed_at is not None else event.ingested_at
    return window_start <= timestamp < window_end


def _tool_evidence_requires_compaction(event: OpenWorldEvidenceEvent) -> bool:
    measurements = _TOOL_COUNTS.get()
    identity = id(event.trajectory)
    if measurements is None:
        return _measure_tool_evidence(event)
    if identity not in measurements:
        measurements[identity] = _measure_tool_evidence(event)
    return measurements[identity]


def _measure_tool_evidence(event: OpenWorldEvidenceEvent) -> bool:
    """Admit large tool evidence even when the conversation itself is short."""
    if event.trajectory is None:
        return False
    tool_bytes = 0
    tool_spans = 0
    for request in event.trajectory["requests"]:
        for interaction in request["interactions"]:
            tools = interaction.get("tools_used")
            if tools:
                tool_bytes += len(canonical_trajectory_bytes(tools))
                tool_spans += 1
    if not tool_spans:
        return False
    counts = _analyst_payload_counts(event)
    if counts is None:
        return False
    return (
        analyst_payload_render_tokens(
            span_count=counts.span_count + tool_spans,
            content_byte_count=counts.content_byte_count + tool_bytes,
        )
        > MAX_COMPACTION_RENDER_TOKENS
    )


def expected_event_from_evidence_event(
    event: OpenWorldEvidenceEvent,
    reason: str | None,
    *,
    allow_compaction: bool = False,
    compaction_batch_bytes: int = DEFAULT_COMPACTION_BATCH_BYTES,
) -> OpenWorldExpectedEvent:
    """Project one loaded evidence event onto its EXPECTED-event record.

    Window-agnostic: it copies identity, correlation and outcome fields and
    attaches the classification ``reason`` it is handed; evidence selection
    hands it ``reconstructability_reason`` over the historical window.
    """
    render_count = _analyst_render_token_count(event)
    compact = (
        allow_compaction
        and reason is None
        and event.trajectory is not None
        and (
            is_capacity_excluded(event)
            or _tool_evidence_requires_compaction(event)
            or (
                render_count is not None and render_count > MAX_COMPACTION_RENDER_TOKENS
            )
        )
    )
    budget_excluded = compact and _compaction_budget_excluded(
        event, max_batch_bytes=compaction_batch_bytes
    )
    compact = compact and not budget_excluded
    return OpenWorldExpectedEvent(
        exposure_event_id=event.exposure_event_id,
        tenant_id=event.tenant_id,
        user_id=event.user_id,
        session_id=event.session_id,
        user_playbook_id=event.user_playbook_id,
        served_semantic_digest=event.served_semantic_digest,
        served_full_version_fingerprint=event.served_full_version_fingerprint,
        request_id=event.request_id,
        trajectory_request_ids=_trajectory_request_ids(event) or (),
        correlation_reason=event.correlation_reason,
        exposed_at=event.exposed_at,
        trajectory_digest=event.trajectory_digest,
        trajectory_byte_count=event.trajectory_byte_count,
        trajectory_token_count=event.trajectory_token_count,
        outcome=event.outcome,
        outcome_id=event.outcome_id,
        outcome_revision=event.outcome_revision,
        outcome_contract_digest=event.outcome_contract_digest,
        governance_subject_refs=event.governance_subject_refs,
        has_agent_turn=_trajectory_has_agent_turn(event),
        analyst_render_token_count=MAX_COMPACTION_RENDER_TOKENS
        if compact
        else render_count,
        capacity_excluded=(is_capacity_excluded(event) and not compact)
        or budget_excluded,
        compaction_required=compact,
        reconstructability_reason=reason,
        trajectory_through_request_id=event.trajectory_through_request_id,
    )


# Private alias kept for existing call sites. Same function object, not a
# copy.
_expected_event = expected_event_from_evidence_event


def _compaction_budget_excluded(
    event: OpenWorldEvidenceEvent, *, max_batch_bytes: int
) -> bool:
    measurements = _COMPACTION_COUNTS.get()
    key = (id(event.trajectory), max_batch_bytes)
    if measurements is None:
        return _measure_compaction_budget(event, max_batch_bytes=max_batch_bytes)
    if key not in measurements:
        measurements[key] = _measure_compaction_budget(
            event, max_batch_bytes=max_batch_bytes
        )
    return measurements[key]


def _measure_compaction_budget(
    event: OpenWorldEvidenceEvent, *, max_batch_bytes: int
) -> bool:
    """Use the exact projected fields and shared source/framing identities."""
    if event.trajectory is None:
        return True
    sources = []
    for request_index, request_item in enumerate(event.trajectory["requests"]):
        request_id = request_item["request"]["request_id"]
        for interaction_index, interaction in enumerate(request_item["interactions"]):
            fields = [("content", interaction["content"])]
            tools = interaction.get("tools_used")
            if tools:
                fields.append(
                    ("tools_used", canonical_trajectory_bytes(tools).decode())
                )
            for field, text in fields:
                if not text:
                    continue
                sources.append(
                    compaction_source(
                        request_id=request_id,
                        interaction_id=interaction["interaction_id"],
                        field_path=f"requests[{request_index}].interactions[{interaction_index}].{field}",
                        speaker=classify_turn_speaker(interaction["role"])
                        if field == "content"
                        else "other",
                        text=text,
                    )
                )
    try:
        compaction_batch_count(sources, max_batch_bytes=max_batch_bytes)
    except CompactionError:
        return True
    return False


def select_open_world_evidence(
    snapshot: OpenWorldEvidenceSnapshot,
    target: OpenWorldTargetIdentity,
    *,
    published_at: int | None = None,
    allow_compaction: bool = False,
    compaction_batch_bytes: int = DEFAULT_COMPACTION_BATCH_BYTES,
    task_context: OpenWorldTaskContext | None = None,
) -> OpenWorldEvidenceSelection:
    """Select evidence, measuring each shared immutable trajectory only once.

    One source prefix may back thousands of exposures. The dictionaries hold
    only counts, live for this synchronous call, and are discarded on every
    exit. No customer content or measurement survives to another attempt.
    """
    counts_token = _SESSION_COUNTS.set({})
    tools_token = _TOOL_COUNTS.set({})
    compaction_token = _COMPACTION_COUNTS.set({})
    try:
        return _select_open_world_evidence(
            snapshot,
            target,
            published_at=published_at,
            allow_compaction=allow_compaction,
            compaction_batch_bytes=compaction_batch_bytes,
            task_context=task_context,
        )
    finally:
        _COMPACTION_COUNTS.reset(compaction_token)
        _TOOL_COUNTS.reset(tools_token)
        _SESSION_COUNTS.reset(counts_token)


def _select_open_world_evidence(
    snapshot: OpenWorldEvidenceSnapshot,
    target: OpenWorldTargetIdentity,
    *,
    published_at: int | None = None,
    allow_compaction: bool = False,
    compaction_batch_bytes: int = DEFAULT_COMPACTION_BATCH_BYTES,
    task_context: OpenWorldTaskContext | None = None,
) -> OpenWorldEvidenceSelection:
    """Select one canonical evidence bundle without I/O or model calls.

    Args:
        snapshot (OpenWorldEvidenceSnapshot): The window read this selection
            runs over. It is a SUPERSET read -- the storage scan is bounded by
            the policy window alone -- so narrowing happens here, not there.
        target (OpenWorldTargetIdentity): The playbook identity being tuned.
        published_at (int | None): When the incumbent was published BY THE
            TUNER, from its terminal lifecycle row; ``None`` when no lifecycle
            row exists (never tuned, or the row has since been reclaimed).

    Returns:
        OpenWorldEvidenceSelection: The validated selection.

    INV-2 -- evidence is attributed to the version that produced it. Sessions
    that ran before ``published_at`` were served by the PREDECESSOR, so letting
    them into the window judges a brand-new successor on the rule it replaced.
    They do not merely fail to support a revision: every one of them lands in
    the coverage DENOMINATOR carrying
    ``incumbent_full_version_fingerprint_mismatch``, so a fresh publication
    drags global coverage under the fixed 75 percent and the tuner abstains for
    a fortnight on evidence it should never have read.

    The bound is ``max(published_at, frozen_at - historical_window_seconds)``,
    not ``published_at`` alone, and the ``max`` is load-bearing in BOTH
    directions:

    * A publication INSIDE the scan window has a live row, and then
      ``published_at`` is the tighter bound and INV-2 holds exactly.
    * Outside it, the row may be gone -- reclaimed with its optimization job by
      ``_retention_gc_retired_optimization_jobs`` -- and ``scan_start`` is then
      the tighter bound anyway, so nothing is lost.

    That second bullet is only true because the GC is bounded to make it true.
    It USED to be argued from ``lineage_gc.tombstone_grace_window_days``
    defaulting to 90 days, "far outside a 14-day scan". That argument was
    invalid on two counts. The 90 is a DEFAULT, not a floor
    (``config_schema.py``: ``Field(default=90, gt=0)``), so an org may legally
    run a one-day grace; and keep-unless-revised made the reclaim REACHABLE
    where it previously was not, because the two unreleased retention holds
    that used to pin the row against GC are no longer written at all. A 7-day
    grace would have dropped the anchor on day eight and attributed five days
    of PREDECESSOR-served sessions to the successor.

    So the guarantee is carried by the GC instead of asserted about it:
    ``_retention_gc_retired_optimization_jobs`` refuses any job whose lifecycle
    was published within ``historical_window_seconds`` of the database clock.
    ``None`` therefore MEANS "published longer ago than this scan can reach",
    and ``scan_start`` is the correct answer rather than a tolerable one.

    ``None`` leaves the scan floor as the publication half of the bound: a
    playbook the tuner never published has no lifecycle boundary.

    THE VERSION START EXTENDS INV-2 TO EVERY OTHER WAY A VERSION CHANGES.
    ``published_at`` exists only for the tuner's own publications, but every
    persisted field feeds the full-version fingerprint, so a human edit, the
    auto-tagger or the governance backfill re-fingerprints the playbook with no
    lifecycle row -- and every earlier exposure then carried the old
    fingerprint and vetoed tuning categorically until it aged out of the scan,
    up to fourteen days. So the bound is
    ``max(scan_start, published_at, version_started_at)``, where the version
    start is the first exposure served under the incumbent fingerprint and
    counts ONLY when every exposure served under another version precedes it
    (``version_start_bound``). That is derived from the evidence itself, so it
    covers every mutation path with no schema change, and the pre-edit prefix
    leaves BOTH coverage denominators -- which also retires the storage's
    digest nulling of that prefix (``missing_playbook_digest``) as a cause.

    An old version served at or after the new one's first exposure keeps the
    window and the categorical ``incumbent_full_version_fingerprint_mismatch``:
    interleaved versions, a reverted edit or an algorithm drift is a fault, not
    a stale prefix. A revision with no current-version exposure yet abstains as
    ``incumbent_revised_awaiting_evidence`` (decided in
    ``build_open_world_evidence_bundle``). The reload check re-runs the same
    rule, ``validate_expected_manifest_attribution_bound``.
    """
    if snapshot.tenant_id != target.tenant_id:
        raise ValueError("snapshot tenant does not match target tenant")

    window_end = target.frozen_at
    scan_start = window_end - PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds
    window_start = scan_start if published_at is None else max(scan_start, published_at)
    if not snapshot.population_limit_exceeded:
        version_start = version_start_bound(
            (
                event
                for event in snapshot.events
                if _in_window(event, window_start=window_start, window_end=window_end)
            ),
            target,
        )
        if version_start is not None:
            window_start = version_start
    if snapshot.population_limit_exceeded:
        expected = OpenWorldExpectedManifest(
            tenant_id=target.tenant_id,
            window_start=window_start,
            window_end=window_end,
            events=(),
            population_limit_exceeded=True,
        )
        selection = OpenWorldEvidenceSelection(
            bundle=build_open_world_evidence_bundle(
                target, expected, task_context=task_context
            )
        )
        validate_open_world_evidence_selection(selection)
        return selection
    window_events = [
        event
        for event in snapshot.events
        if _in_window(event, window_start=window_start, window_end=window_end)
    ]
    tenant_mismatch = any(
        event.tenant_id is not None and event.tenant_id != target.tenant_id
        for event in window_events
    )
    events = [
        event
        for event in window_events
        if event.tenant_id is None or event.tenant_id == target.tenant_id
    ]
    event_reasons = [(event, reconstructability_reason(event)) for event in events]
    expected = OpenWorldExpectedManifest(
        tenant_id=target.tenant_id,
        window_start=window_start,
        window_end=window_end,
        events=tuple(
            _expected_event(
                event,
                reason,
                allow_compaction=allow_compaction,
                compaction_batch_bytes=compaction_batch_bytes,
            )
            for event, reason in event_reasons
        ),
        integrity_invalid=tenant_mismatch,
    )
    selection = OpenWorldEvidenceSelection(
        bundle=build_open_world_evidence_bundle(
            target, expected, task_context=task_context
        )
    )
    validate_open_world_evidence_selection(selection)
    return selection


__all__ = [
    "expected_event_from_evidence_event",
    "is_capacity_excluded",
    "reconstructability_reason",
    "select_open_world_evidence",
]
