"""Pure assembly of an open-world evidence snapshot from rows storage has read.

The storage backend's snapshot read owns the transaction, the keyset walks,
the SQL and the cursor. Everything it does BETWEEN two queries -- deciding
which exposure correlates with which request, which cutover a session is read
through, which ``(request, cutover)`` pairs to ask about, how a stream of joined request/interaction rows becomes one
canonical trajectory per session, and how all of that becomes events -- lives
here, as functions of plain rows.

THIS MODULE IMPORTS NO STORAGE BACKEND AND NO DATABASE DRIVER. Policy numbers
arrive as arguments (``OpenWorldEvidenceReadLimits``), never by import, so a
read can be bounded by a test without patching a module constant.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, cast

from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.playbook.publication import (
    incumbent_user_playbook_semantic_digest,
)
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
    CanonicalTrajectoryDigestAccumulator,
    canonical_session_trajectory,
    canonical_trajectory_bytes,
)

from .open_world_models import (
    OpenWorldEvidenceEvent,
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
    Outcome,
    OutcomeContractDescriptor,
)

SessionKey = tuple[str, str]

_DIGEST_CHARS = frozenset("0123456789abcdef")
#: The exact tuple ``OutcomeContractDescriptor`` requires, IN THIS ORDER.
#:
#: ``tenant.outcome_contracts.allowed_values`` stores the same three values but
#: is written ``sorted(...)`` -- ``["failure", "success", "unknown"]``. The
#: column is a SET; the descriptor field is an ORDERED tuple. Compare the
#: column as a frozenset and emit this constant. Passing the column through
#: positionally builds a descriptor the model validator rejects.
_CANONICAL_OUTCOME_ALLOWED_VALUES: tuple[
    Literal["success", "failure", "unknown"], ...
] = ("success", "failure", "unknown")
#: What the digest accumulator and the materialisation raise on a row they
#: cannot canonicalise. Any of them invalidates the session, never the read.
_ROW_ERRORS = (KeyError, OverflowError, RuntimeError, TypeError, ValueError)


def text_or_none(value: object) -> str | None:
    """A stripped non-empty string, or None for anything else."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def digest_or_none(value: object) -> str | None:
    """A lowercase sha256 hex digest, or None for anything else."""
    normalized = text_or_none(value)
    if (
        normalized is None
        or len(normalized) != 64
        or not set(normalized) <= _DIGEST_CHARS
    ):
        return None
    return normalized


def positive_int_or_none(value: object) -> int | None:
    """A strictly positive ``int`` (``bool`` excluded), or None."""
    if type(value) is not int or value <= 0:
        return None
    return value


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedTrajectory:
    """One session's trajectory as the snapshot read reconstructed it."""

    trajectory: CanonicalSessionTrajectory | None
    digest: str | None
    byte_count: int | None
    token_count: int | None
    governance_subject_refs: tuple[str, ...]
    governance_complete: bool
    correlation_mismatch: bool


#: A session whose trajectory was not loaded at all -- no key, over the
#: joined-row cap, or absent from the stream.
ABSENT_TRAJECTORY = LoadedTrajectory(None, None, None, None, (), True, False)


@dataclass
class TrajectoryAccumulator:
    """Fold one session's joined request/interaction rows into a trajectory.

    Rows arrive in the trajectory query's order -- request, then its
    interactions -- and are consumed ONE AT A TIME, so the byte cap is applied
    while the rows are still streaming: content over ``max_trajectory_bytes``
    is cleared as soon as the running canonical size passes it, and never
    retained again. The digest keeps accumulating, so a byte-over session
    still reports its digest and size; only its content is gone.
    """

    key: SessionKey
    max_trajectory_bytes: int
    governance_subject_refs: set[str] = field(default_factory=set)
    governance_complete: bool = True
    retained_request_rows: list[dict[str, Any]] = field(default_factory=list)
    retained_interactions_by_request: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    invalid: bool = False
    correlation_mismatch: bool = False
    byte_limit_exceeded: bool = False
    _active_request_id: str | None = field(init=False, default=None)
    _digest_accumulator: CanonicalTrajectoryDigestAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self._digest_accumulator = CanonicalTrajectoryDigestAccumulator(self.key[1])

    def consume_joined_row(self, row: Mapping[str, Any]) -> None:
        """Fold one joined row in; a row from another session invalidates."""
        if self.invalid:
            return
        request_id = text_or_none(row.get("request_id"))
        request_user_id = text_or_none(row.get("request_user_id"))
        request_session_id = text_or_none(row.get("request_session_id"))
        if (
            request_id is None
            or request_user_id != self.key[0]
            or request_session_id != self.key[1]
        ):
            self._invalidate()
            return
        if request_id != self._active_request_id:
            self._start_request(request_id, row)
        if self.invalid or row.get("interaction_id") is None:
            self._drop_content_if_byte_limit_exceeded()
            return
        if text_or_none(row.get("interaction_user_id")) != self.key[0]:
            self.correlation_mismatch = True
            return
        interaction = self._interaction_row(row)
        subject_ref = text_or_none(row.get("interaction_governance_subject_ref"))
        self._record_governance_subject(subject_ref)
        try:
            self._digest_accumulator.add_interaction(interaction)
        except _ROW_ERRORS:
            self._invalidate()
            return
        if not self.byte_limit_exceeded:
            self.retained_interactions_by_request[request_id].append(interaction)
        self._drop_content_if_byte_limit_exceeded()

    def _start_request(self, request_id: str, row: Mapping[str, Any]) -> None:
        request = self._request_row(row)
        request_subject_ref = text_or_none(row.get("request_governance_subject_ref"))
        self._record_governance_subject(request_subject_ref)
        try:
            if self._active_request_id is not None:
                self._digest_accumulator.finish_request()
            self._digest_accumulator.start_request(request)
        except _ROW_ERRORS:
            self._invalidate()
            return
        self._active_request_id = request_id
        if not self.byte_limit_exceeded:
            self.retained_request_rows.append(request)
        self._drop_content_if_byte_limit_exceeded()

    def _drop_content_if_byte_limit_exceeded(self) -> None:
        if self.invalid or self.byte_limit_exceeded:
            return
        try:
            byte_count = self._digest_accumulator.byte_count_if_finalized()
        except _ROW_ERRORS:
            self._invalidate()
            return
        if byte_count > self.max_trajectory_bytes:
            self.byte_limit_exceeded = True
            self._clear_retained_content()

    @staticmethod
    def _request_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "request_id": row["request_id"],
            "user_id": row["request_user_id"],
            "created_at": row["request_created_at"],
            "source": row["request_source"],
            "agent_version": row["request_agent_version"],
            "session_id": row["request_session_id"],
            "evaluation_only": row["request_evaluation_only"],
            "retrieval_experiment_id": row["request_retrieval_experiment_id"],
            "retrieval_experiment_arm": row["request_retrieval_experiment_arm"],
            "governance_subject_ref": row["request_governance_subject_ref"],
        }

    @staticmethod
    def _interaction_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "interaction_id": row["interaction_id"],
            "user_id": row["interaction_user_id"],
            "request_id": row["interaction_request_id"],
            "created_at": row["interaction_created_at"],
            "content": row["interaction_content"],
            "role": row["interaction_role"],
            "token_count": row["interaction_token_count"],
            "user_action": row["interaction_user_action"],
            "user_action_description": row["interaction_user_action_description"],
            "interacted_image_url": row["interaction_interacted_image_url"],
            "image_encoding": row["interaction_image_encoding"],
            "shadow_content": row["interaction_shadow_content"],
            "expert_content": row["interaction_expert_content"],
            "tools_used": row["interaction_tools_used"],
            "citations": row["interaction_citations"],
            "retrieved_learnings": row["interaction_retrieved_learnings"],
            "governance_subject_ref": row["interaction_governance_subject_ref"],
        }

    def _record_governance_subject(self, subject_ref: str | None) -> None:
        if subject_ref is None:
            self.governance_complete = False
        else:
            self.governance_subject_refs.add(subject_ref)

    def _invalidate(self) -> None:
        self.invalid = True
        self._clear_retained_content()

    def _clear_retained_content(self) -> None:
        self.retained_request_rows.clear()
        self.retained_interactions_by_request.clear()

    def finish(self) -> LoadedTrajectory:
        """The session's trajectory, digest, size and governance refs."""
        digest, byte_count = self._finish_digest_and_bytes()
        if self.invalid:
            trajectory = None
            digest = None
            byte_count = None
            token_count = None
        elif self.byte_limit_exceeded:
            trajectory = None
            token_count = 0
        else:
            # A token-over trajectory is RETAINED. The token cap is a capacity
            # judgement made against the storage row, not against the content
            # the analyst reads; discarding here would leave selection nothing
            # to measure and would reclassify an intact session as
            # ``missing_trajectory``. Capacity is decided at selection.
            # The BYTE cap above keeps its discard: that helper clears content
            # mid-stream, so byte-over content is not recoverable here.
            trajectory, token_count = self._materialize_eligible_trajectory()
            if trajectory is None:
                digest = None
                byte_count = None
                token_count = None
        return self._loaded(
            trajectory=trajectory,
            digest=digest,
            byte_count=byte_count,
            token_count=token_count,
        )

    def _finish_digest_and_bytes(self) -> tuple[str | None, int | None]:
        if self.invalid or self._active_request_id is None:
            return None, None
        try:
            self._digest_accumulator.finish_request()
            byte_count = self._digest_accumulator.byte_count_if_finalized()
            return self._digest_accumulator.hexdigest(), byte_count
        except _ROW_ERRORS:
            self._invalidate()
            return None, None

    def _materialize_eligible_trajectory(
        self,
    ) -> tuple[CanonicalSessionTrajectory | None, int | None]:
        try:
            trajectory = canonical_session_trajectory(
                self.key[1],
                self.retained_request_rows,
                self.retained_interactions_by_request,
            )
            trajectory_bytes = canonical_trajectory_bytes(trajectory)
            return trajectory, count_input_tokens(trajectory_bytes.decode("utf-8"))
        except (KeyError, OverflowError, TypeError, UnicodeDecodeError, ValueError):
            self._invalidate()
            return None, None

    def _loaded(
        self,
        *,
        trajectory: CanonicalSessionTrajectory | None,
        digest: str | None,
        byte_count: int | None,
        token_count: int | None,
    ) -> LoadedTrajectory:
        return LoadedTrajectory(
            trajectory=trajectory,
            digest=digest,
            byte_count=byte_count,
            token_count=token_count,
            governance_subject_refs=tuple(
                sorted(
                    self.governance_subject_refs,
                    key=lambda value: value.encode("utf-8"),
                )
            ),
            governance_complete=self.governance_complete,
            correlation_mismatch=self.correlation_mismatch,
        )


def assemble_trajectories(
    rows: Iterable[Mapping[str, Any]], *, max_trajectory_bytes: int
) -> Iterator[tuple[SessionKey, LoadedTrajectory]]:
    """Stream joined rows into one ``LoadedTrajectory`` per session.

    LAZY, deliberately: ``rows`` is the server-side named cursor, and this
    pulls one row at a time, yielding a session the moment the next session's
    first row arrives. Nothing but the active session is ever held, and that
    session's content is dropped mid-stream once it passes
    ``max_trajectory_bytes`` -- so a materialised list here would defeat both
    the named cursor and the byte cap.

    Rows must arrive grouped by ``(request_user_id, request_session_id)``, as
    the trajectory query's ``ORDER BY`` delivers them.

    Args:
        rows (Iterable[Mapping[str, Any]]): Joined request/interaction rows.
        max_trajectory_bytes (int): The storage-row byte cap.

    Yields:
        tuple[SessionKey, LoadedTrajectory]: Each session's key and trajectory.
    """
    active_key: SessionKey | None = None
    accumulator: TrajectoryAccumulator | None = None
    for raw_row in rows:
        row = dict(raw_row)
        key = (str(row["request_user_id"]), str(row["request_session_id"]))
        if key != active_key:
            if accumulator is not None and active_key is not None:
                yield active_key, accumulator.finish()
            active_key = key
            accumulator = TrajectoryAccumulator(
                key=key, max_trajectory_bytes=max_trajectory_bytes
            )
        if accumulator is not None:
            accumulator.consume_joined_row(row)
    if accumulator is not None and active_key is not None:
        yield active_key, accumulator.finish()


# --------------------------------------------------------------------------
# The outcome contract
# --------------------------------------------------------------------------


class OutcomeContractRebuildError(ValueError):
    """The persisted rows do not rebuild into an outcome-contract descriptor.

    Storage re-raises this as ``OpenWorldOutcomeContractUnavailableError`` (a
    ``StorageError``), which the runner maps to ``infrastructure_failure``.
    This module may not import that class, so the translation is storage's.
    """


def rebuild_outcome_contract_descriptor(
    *,
    outcome_contract_digest: str,
    contract_row: Mapping[str, Any] | None,
    source_rows: Sequence[Mapping[str, Any]],
) -> OutcomeContractDescriptor:
    """Rebuild the descriptor from the two rows that persist it, or refuse.

    ``tenant.outcome_contracts`` stores ``sha256(source)`` and not ``source``,
    so ``source_id`` cannot come from the contract row alone. It is recovered
    from ``tenant.session_outcomes.source`` and then PROVED to be the exact
    preimage the contract commits to. That proof is also what licenses
    ``kind="structured_outcome_source"``: the row is shown to have come from
    the session-outcome writer, which is the only writer that registers
    contracts. A future ``success_definition``-kind contract would have no
    session-outcome binding and would fail here loudly rather than be
    mislabeled. No default kind is ever emitted.

    Raises:
        OutcomeContractRebuildError: On every refusal.
    """
    if contract_row is None:
        raise OutcomeContractRebuildError(
            f"no outcome contract row for digest {outcome_contract_digest}"
        )
    if len(source_rows) != 1:
        raise OutcomeContractRebuildError(
            "expected exactly one distinct session outcome source for outcome "
            f"contract {outcome_contract_digest}, found {len(source_rows)}"
        )
    # The preimage is the RAW stored value, byte for byte. The writer
    # (``_session_outcomes.py::record_session_outcome``) hashes
    # ``str(first["source"] or "")`` with no normalization at all, so
    # normalizing here before hashing would express the same identity two
    # different ways -- and a source carrying surrounding whitespace would then
    # fail the equality below and take the whole evidence read down with
    # ``OpenWorldOutcomeContractUnavailableError``. Persisted sources are NOT
    # guaranteed to be pattern-clean: ``add_request`` validates new input
    # against ``SESSION_OUTCOME_SOURCE_PATTERN``, but the read model is
    # ``PersistedSessionOutcomeSource = str``, documented as "returned exactly
    # as stored, without new-input validation".
    #
    # Emptiness is still judged on the STRIPPED form, exactly as before: a
    # whitespace-only source is refused rather than admitted with a blank
    # ``source_id``. Only the hashed and emitted VALUE changed.
    source = source_rows[0].get("source")
    if not isinstance(source, str) or not source.strip():
        raise OutcomeContractRebuildError(
            "session outcome source is empty for outcome contract "
            f"{outcome_contract_digest}"
        )
    source_digest = digest_or_none(contract_row.get("source_digest"))
    if source_digest is None:
        raise OutcomeContractRebuildError(
            "stored source digest is not a sha256 digest for outcome contract "
            f"{outcome_contract_digest}"
        )
    # Mirrors ``_session_outcomes.py::_outcome_source_digest`` exactly.
    if sha256(source.encode("utf-8")).hexdigest() != source_digest:
        raise OutcomeContractRebuildError(
            "session outcome source does not hash to the stored source digest "
            f"for outcome contract {outcome_contract_digest}"
        )
    allowed_values = contract_row.get("allowed_values")
    if not isinstance(allowed_values, list) or not all(
        isinstance(value, str) for value in allowed_values
    ):
        raise OutcomeContractRebuildError(
            "stored allowed_values is not an array of strings for outcome "
            f"contract {outcome_contract_digest}"
        )
    # A SET comparison, deliberately: the column is sorted and the descriptor
    # is ordered. Emit the canonical tuple; never the column's order.
    if frozenset(allowed_values) != frozenset(_CANONICAL_OUTCOME_ALLOWED_VALUES):
        raise OutcomeContractRebuildError(
            "stored allowed_values is not the canonical success/failure/"
            f"unknown set for outcome contract {outcome_contract_digest}"
        )
    schema_version = text_or_none(contract_row.get("schema_version"))
    if schema_version is None:
        raise OutcomeContractRebuildError(
            "stored schema_version is empty for outcome contract "
            f"{outcome_contract_digest}"
        )
    return OutcomeContractDescriptor(
        digest=outcome_contract_digest,
        kind="structured_outcome_source",
        success_definition=None,
        source_id=source,
        schema_version=schema_version,
        allowed_values=_CANONICAL_OUTCOME_ALLOWED_VALUES,
    )


# --------------------------------------------------------------------------
# Correlation, cutovers and prefixes
# --------------------------------------------------------------------------


def dependency_key(row: Mapping[str, Any]) -> SessionKey | None:
    """The ``(user_id, session_id)`` a resolved exposure row depends on."""
    key = row.get("_dependency_key")
    if not isinstance(key, tuple) or len(key) != 2:
        return None
    user_id = text_or_none(key[0])
    session_id = text_or_none(key[1])
    if user_id is None or session_id is None:
        return None
    return user_id, session_id


def session_keys(rows: Iterable[Mapping[str, Any]]) -> set[SessionKey]:
    """Every session the resolved rows depend on."""
    return {key for row in rows if (key := dependency_key(row)) is not None}


def session_only_exposure_ids(exposure_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Exposures that carry a session but no request: the ones to bind."""
    return sorted(
        str(row["exposure_event_id"])
        for row in exposure_rows
        if text_or_none(row.get("request_id")) is None
        and text_or_none(row.get("session_id")) is not None
    )


def correlated_request_ids(
    exposure_rows: Iterable[Mapping[str, Any]], bound_requests: Mapping[str, str]
) -> list[str]:
    """The requests the resolver must read: sent ids plus retrieval-log bindings."""
    return sorted(
        {
            request_id
            for row in exposure_rows
            if (
                request_id := text_or_none(row.get("request_id"))
                or bound_requests.get(str(row.get("exposure_event_id")))
            )
            is not None
        }
    )


def resolve_exposure_correlations(
    exposure_rows: Iterable[Mapping[str, Any]],
    *,
    bound_requests: Mapping[str, str],
    retained_requests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Correlate each exposure with its request, or name why it cannot be.

    Each returned row is a copy of the exposure carrying ``request_id`` (the
    sent id, else the retrieval-log binding), ``correlation_reason`` (None when
    correlated) and ``_dependency_key`` (the request's own ``(user, session)``
    when the request is retained and names both).

    Args:
        exposure_rows (Iterable[Mapping[str, Any]]): The exposure population.
        bound_requests (Mapping[str, str]): ``exposure_event_id`` -> bound
            request, for the session-only rows the retrieval log binds.
        retained_requests (Mapping[str, Mapping[str, Any]]): ``request_id`` ->
            the request's current ``user_id`` and ``session_id``.

    Returns:
        list[dict[str, Any]]: The resolved rows, in input order.
    """
    resolved_rows: list[dict[str, Any]] = []
    for exposure_row in exposure_rows:
        row = dict(exposure_row)
        # A session-only row the retrieval log binds takes its request here
        # and then passes EXACTLY the checks a sent id does -- the request
        # must be retained and must agree on user and session.
        request_id = text_or_none(row.get("request_id")) or bound_requests.get(
            str(row.get("exposure_event_id"))
        )
        row["request_id"] = request_id
        row["correlation_reason"] = None
        row["_dependency_key"] = None
        if request_id is None:
            # Name WHICH kind of uncorrelated this is. Both end in the same
            # exclusion, so this changes no evidence -- it changes where the
            # refusal sends whoever reads it.
            #
            # A session-only exposure reaching here carries a `session_id`
            # but could not be bound through the retrieval log: the publish
            # never cited the playbook (or has not happened yet), or the
            # pair is ambiguous -- see `_session_only_bound_request`. It is
            # still not a client that sent nothing, so it keeps its own
            # name rather than `missing_request`; reported as that it reads
            # as "your agent is not sending an id" when the truth is "its
            # publish never named this playbook".
            row["correlation_reason"] = (
                "session_only"
                if text_or_none(row.get("session_id")) is not None
                else "missing_request"
            )
            resolved_rows.append(row)
            continue
        request_row = retained_requests.get(request_id)
        if request_row is None:
            row["correlation_reason"] = "missing_request"
            resolved_rows.append(row)
            continue
        request_user_id = text_or_none(request_row.get("user_id"))
        request_session_id = text_or_none(request_row.get("session_id"))
        if request_user_id is None or request_session_id is None:
            row["correlation_reason"] = "correlation_mismatch"
            resolved_rows.append(row)
            continue
        row["_dependency_key"] = (request_user_id, request_session_id)
        for column, request_value in (
            ("user_id", request_user_id),
            ("session_id", request_session_id),
        ):
            exposure_value = text_or_none(row.get(column))
            if exposure_value is None:
                row[column] = request_value
            elif exposure_value != request_value:
                row["correlation_reason"] = "correlation_mismatch"
        resolved_rows.append(row)
    return resolved_rows


def trajectory_cutovers(
    resolved_rows: Iterable[Mapping[str, Any]],
    outcomes: Mapping[SessionKey, Mapping[str, Any]],
) -> dict[SessionKey, str | None]:
    """The cutover each session's trajectory is read through.

    The outcome's own ``trajectory_through_request_id`` (None when the
    session has no outcome or the outcome is whole-session).
    """
    return {
        key: (
            text_or_none(outcomes[key].get("trajectory_through_request_id"))
            if key in outcomes
            else None
        )
        for key in session_keys(resolved_rows)
    }


def prefix_pairs(
    resolved_rows: Iterable[Mapping[str, Any]],
    cutovers: Mapping[SessionKey, str | None],
) -> list[tuple[str, str]]:
    """``(request, cutover)`` pairs to ask the prefix predicate about, sorted.

    Asked in BOTH directions for each exposure's request against its
    session's cutover (strictly-after needs both). Storage asks them in SQL
    through the shared predicate; nothing here compares timestamps.
    """
    pairs: set[tuple[str, str]] = set()
    for row in resolved_rows:
        key = dependency_key(row)
        request_id = text_or_none(row.get("request_id"))
        cutover = cutovers.get(key) if key is not None else None
        if cutover is None or request_id is None:
            continue
        pairs.add((request_id, cutover))
        pairs.add((cutover, request_id))
    return sorted(pairs)


def validated_playbook_identity(
    exposure_row: Mapping[str, Any],
    *,
    raw_playbook_id: int | None,
    raw_served_digest: str | None,
    playbooks: Mapping[int, Mapping[str, Any]],
) -> tuple[int | None, str | None]:
    """The exposure's playbook id, and its served digest only if still current.

    The digest survives only when the playbook row exists, its owner matches
    the exposure's, it has content, and the digest the exposure recorded is
    the playbook's CURRENT semantic digest.
    """
    if raw_playbook_id is None:
        return None, None
    playbook = playbooks.get(raw_playbook_id)
    if playbook is None:
        return raw_playbook_id, None
    owner_user_id = text_or_none(exposure_row.get("playbook_owner_user_id"))
    if owner_user_id is None or text_or_none(playbook.get("user_id")) != owner_user_id:
        return raw_playbook_id, None
    content = playbook.get("content")
    trigger = playbook.get("trigger")
    if not isinstance(content, str) or not content:
        return raw_playbook_id, None
    current_digest = incumbent_user_playbook_semantic_digest(
        content_digest=sha256(content.encode()).hexdigest(),
        trigger=str(trigger) if trigger is not None else None,
    )
    if raw_served_digest != current_digest:
        return raw_playbook_id, None
    return raw_playbook_id, raw_served_digest


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EventDependency:
    """What one resolved exposure row's event depends on, before governance."""

    row: Mapping[str, Any]
    outcome_row: Mapping[str, Any] | None
    trajectory: LoadedTrajectory
    governance_subject_refs: tuple[str, ...]
    governance_complete: bool


def event_dependencies(
    resolved_rows: Iterable[Mapping[str, Any]],
    *,
    outcomes: Mapping[SessionKey, Mapping[str, Any]],
    trajectories: Mapping[SessionKey, LoadedTrajectory],
    target_owner_subject_ref: str | None,
) -> list[EventDependency]:
    """Gather each row's outcome, trajectory and governance subject refs.

    ``governance_complete`` is False when any ref an event must carry -- the
    target owner's, the exposure's, the playbook owner's, every trajectory
    row's and the outcome's -- is missing.
    """
    dependencies: list[EventDependency] = []
    for row in resolved_rows:
        key = dependency_key(row)
        outcome_row = outcomes.get(key) if key is not None else None
        loaded_trajectory = (
            trajectories.get(key, ABSENT_TRAJECTORY)
            if key is not None
            else ABSENT_TRAJECTORY
        )
        raw_refs = [
            target_owner_subject_ref,
            text_or_none(row.get("governance_subject_ref")),
            text_or_none(row.get("playbook_owner_governance_subject_ref")),
            *loaded_trajectory.governance_subject_refs,
        ]
        governance_complete = (
            all(subject_ref is not None for subject_ref in raw_refs[:3])
            and loaded_trajectory.governance_complete
        )
        if outcome_row is not None:
            outcome_subject_ref = text_or_none(
                outcome_row.get("governance_subject_ref")
            )
            raw_refs.append(outcome_subject_ref)
            governance_complete = (
                governance_complete and outcome_subject_ref is not None
            )
        governance_subject_refs = tuple(
            sorted(
                {subject_ref for subject_ref in raw_refs if subject_ref is not None},
                key=lambda value: value.encode("utf-8"),
            )
        )
        dependencies.append(
            EventDependency(
                row=row,
                outcome_row=outcome_row,
                trajectory=loaded_trajectory,
                governance_subject_refs=governance_subject_refs,
                governance_complete=governance_complete,
            )
        )
    return dependencies


def governance_subject_refs_to_read(
    target: OpenWorldTargetIdentity, dependencies: Iterable[EventDependency]
) -> set[str]:
    """Every subject the governance read must answer for: declared and touched."""
    refs = {subject_ref for subject_ref, _epoch in target.governance_epoch_vector}
    for dependency in dependencies:
        refs.update(dependency.governance_subject_refs)
    return refs


def _governance_is_valid(
    governance_subject_refs: tuple[str, ...],
    *,
    governance_complete: bool,
    target_epochs: Mapping[str, int],
    governance_subjects: Mapping[str, tuple[int, bool]],
    blocked_subjects: set[str],
) -> bool:
    if not governance_complete:
        return False
    for subject_ref in governance_subject_refs:
        current = governance_subjects.get(subject_ref)
        if (
            current is None
            or target_epochs.get(subject_ref) != current[0]
            or not current[1]
            or subject_ref in blocked_subjects
        ):
            return False
    return True


def assemble_snapshot(
    dependencies: Iterable[EventDependency],
    *,
    target: OpenWorldTargetIdentity,
    playbooks: Mapping[int, Mapping[str, Any]],
    contracts: set[str],
    cutovers: Mapping[SessionKey, str | None],
    in_prefix: set[tuple[str, str]],
    governance_subjects: Mapping[str, tuple[int, bool]],
    blocked_subjects: set[str],
) -> OpenWorldEvidenceSnapshot:
    """One event per dependency, as the target's snapshot.

    ``target.governance_epoch_vector`` declares which epochs count as valid:
    the vector the target was selected against, so "valid" means "unchanged
    since then". An empty vector marks every event governance-invalid,
    because ``_governance_is_valid`` requires an epoch entry for every subject.

    Args:
        dependencies (Iterable[EventDependency]): From ``event_dependencies``.
        target (OpenWorldTargetIdentity): The attempt's target.
        playbooks (Mapping[int, Mapping[str, Any]]): Playbook rows by id.
        contracts (set[str]): Outcome-contract digests that exist.
        cutovers (Mapping[SessionKey, str | None]): From
            ``trajectory_cutovers``.
        in_prefix (set[tuple[str, str]]): The ``prefix_pairs`` the prefix
            predicate admitted.
        governance_subjects (Mapping[str, tuple[int, bool]]): Subject ->
            ``(erasure_epoch, writable)``.
        blocked_subjects (set[str]): Subjects behind an erasure barrier.

    Returns:
        OpenWorldEvidenceSnapshot: The snapshot, events in dependency order.
    """
    target_epochs = dict(target.governance_epoch_vector)
    events: list[OpenWorldEvidenceEvent] = []
    for dependency in dependencies:
        row = dependency.row
        outcome_row = dependency.outcome_row
        loaded_trajectory = dependency.trajectory
        playbook_id, served_digest = validated_playbook_identity(
            row,
            raw_playbook_id=positive_int_or_none(row.get("user_playbook_id")),
            raw_served_digest=digest_or_none(row.get("served_semantic_digest")),
            playbooks=playbooks,
        )
        finalized_digest = (
            digest_or_none(outcome_row.get("finalized_trajectory_digest"))
            if outcome_row is not None
            else None
        )
        outcome_value = (
            str(outcome_row["outcome"])
            if outcome_row is not None
            and str(outcome_row.get("outcome")) in {"success", "failure", "unknown"}
            else None
        )
        raw_contract_digest = (
            digest_or_none(outcome_row.get("outcome_contract_digest"))
            if outcome_row is not None
            else None
        )
        contract_digest = (
            raw_contract_digest if raw_contract_digest in contracts else None
        )
        correlation_reason = row.get("correlation_reason")
        if correlation_reason is None and loaded_trajectory.correlation_mismatch:
            correlation_reason = "correlation_mismatch"
        key = dependency_key(row)
        cutover = cutovers.get(key) if key is not None else None
        request_id = text_or_none(row.get("request_id"))
        events.append(
            OpenWorldEvidenceEvent(
                exposure_event_id=str(row["exposure_event_id"]),
                tenant_id=target.tenant_id,
                user_id=text_or_none(row.get("user_id")),
                session_id=text_or_none(row.get("session_id")),
                user_playbook_id=playbook_id,
                served_semantic_digest=served_digest,
                served_full_version_fingerprint=row.get(
                    "served_full_version_fingerprint"
                ),
                request_id=request_id,
                correlation_reason=correlation_reason,
                exposed_at=(
                    int(row["exposed_at"])
                    if row.get("exposed_at") is not None
                    else None
                ),
                ingested_at=int(row["ingested_at"]),
                trajectory=loaded_trajectory.trajectory,
                trajectory_digest=loaded_trajectory.digest,
                trajectory_digest_matches=(
                    loaded_trajectory.digest is not None
                    and (
                        finalized_digest is None
                        or loaded_trajectory.digest == finalized_digest
                    )
                ),
                trajectory_byte_count=loaded_trajectory.byte_count,
                trajectory_token_count=loaded_trajectory.token_count,
                outcome=cast(Outcome | None, outcome_value),
                outcome_id=(
                    str(outcome_row["outcome_id"])
                    if outcome_row is not None
                    and outcome_row.get("outcome_id") is not None
                    else None
                ),
                outcome_revision=(
                    int(outcome_row["outcome_revision"])
                    if outcome_row is not None
                    and outcome_row.get("outcome_revision") is not None
                    else None
                ),
                outcome_contract_digest=contract_digest,
                governance_valid=_governance_is_valid(
                    dependency.governance_subject_refs,
                    governance_complete=dependency.governance_complete,
                    target_epochs=target_epochs,
                    governance_subjects=governance_subjects,
                    blocked_subjects=blocked_subjects,
                ),
                governance_subject_refs=dependency.governance_subject_refs,
                trajectory_through_request_id=cutover,
                # Strictly AFTER the cutover: the cutover is not in the
                # prefix through the request, and the request is not in
                # the prefix through the cutover.
                awaiting_reaction=(
                    cutover is not None
                    and request_id is not None
                    and (cutover, request_id) in in_prefix
                    and (request_id, cutover) not in in_prefix
                ),
                outcome_is_inferred=(
                    outcome_row is not None and bool(outcome_row.get("is_inferred"))
                ),
                # ``outcome_covers_trajectory_prefix`` keeps its default
                # (True): the trajectory is read through the outcome's own
                # current cutover, which trivially covers itself.
            )
        )
    return OpenWorldEvidenceSnapshot(tenant_id=target.tenant_id, events=tuple(events))


__all__ = [
    "ABSENT_TRAJECTORY",
    "EventDependency",
    "LoadedTrajectory",
    "OutcomeContractRebuildError",
    "SessionKey",
    "TrajectoryAccumulator",
    "assemble_snapshot",
    "assemble_trajectories",
    "correlated_request_ids",
    "dependency_key",
    "digest_or_none",
    "event_dependencies",
    "governance_subject_refs_to_read",
    "positive_int_or_none",
    "prefix_pairs",
    "rebuild_outcome_contract_descriptor",
    "resolve_exposure_correlations",
    "session_keys",
    "session_only_exposure_ids",
    "text_or_none",
    "trajectory_cutovers",
    "validated_playbook_identity",
]
