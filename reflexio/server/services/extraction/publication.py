# ruff: noqa: S608 -- SQL identifiers come from the fixed storage table registry.
"""One owned transaction for an ordinary retained output's publication.

Preparation is outside the writer; dispatch and financial delivery follow commit.
Initial inline publication and resumed outputs share the same original preparation
and input proofs. Window adoption and autonomous financial discovery remain separate
integration gates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.metering_events import (
    emit_learnings_generated_records_strict,
    record_window_extraction_finances,
)
from reflexio.server.services.deferred_learning_plan import (
    ExtractorBookmarkAdvance,
    ExtractorBookmarkPosition,
    FinalizationResult,
    PlaybookWritePlan,
    ProfileWritePlan,
)
from reflexio.server.services.durable_learning.user_lease import (
    _require_user_work_owner,
    owned_agent_run_write,
)
from reflexio.server.services.extraction.answer_inputs import (
    require_publication_answer_inputs,
)
from reflexio.server.services.extraction.history_inputs import (
    HistoricalInputs,
    historical_input_scope,
    history_from_output,
    merge_historical_inputs,
)
from reflexio.server.services.extraction.overflow_answers import require_current_answer
from reflexio.server.services.extraction.preparation_attempt import (
    preparation_attempt_scope,
)
from reflexio.server.services.extraction.publication_preparation import (
    SavedPublicationPreparation,
    bookmark_from_output,
)
from reflexio.server.services.extraction.publication_receipts import (
    PublicationReceipt,
    output_publication_origin,
    require_output_id,
)
from reflexio.server.services.storage.error import PostCommitStorageError
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
    LeaseLostError,
    PendingToolCallStatus,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    _OWNER_PROOF_PAGE_SIZE,
    StreamSQL,
    _run_revision_time,
    decoded,
)
from reflexio.server.work_scope import current_project_id

logger = logging.getLogger(__name__)


def _failure_detail(error: BaseException) -> str:
    from reflexio.server.services.durable_learning.worker import exception_detail

    return exception_detail(error)


def settle_finalization_failure(
    storage: BaseStorage,
    run: AgentRunRecord,
    error: Exception,
    *,
    max_attempts: int,
    last_error: str,
) -> AgentRunRecord:
    """One allowance/deferral rule for initial and resumed publication owners.

    Input contention, unavailable readiness and a provider outage (the charging
    invariant in ``provider_circuit``) defer without spending an attempt; every
    other failure is charged against ``max_attempts``.
    """
    from reflexio.server.llm.compute_admission import ComputeAdmissionError
    from reflexio.server.llm.provider_circuit import provider_wait_seconds
    from reflexio.server.services.storage.error import (
        ReadinessUnavailableError,
        execution_input_is_busy,
    )

    reason = error.reason if isinstance(error, ComputeAdmissionError) else error
    if isinstance(reason, LeaseLostError):
        raise reason
    with owned_agent_run_write(storage, run):
        busy = execution_input_is_busy(reason)
        # Do not settle or replace a prepared result on input contention.
        entry = None if busy else storage.pending_agent_preparation_attempt(run)
        readiness = isinstance(reason, ReadinessUnavailableError) and entry is None
        # Preparation calls the model; a route that is down for everyone is
        # not this run's failure, whether or not preparation had started.
        outage_wait = None if busy else provider_wait_seconds(reason)
        charge = not (busy or readiness or outage_wait is not None)
        count = run.finalization_attempts + int(charge)
        now = datetime.now(UTC)
        if busy:
            next_at, last_error = now, "ExecutionInputBusyError"
        elif readiness:
            next_at, last_error = (
                now + timedelta(seconds=30),
                "ReadinessUnavailableError",
            )
        elif outage_wait is not None:
            next_at = now + timedelta(seconds=outage_wait)
            last_error = "provider_unavailable"
        else:
            next_at = now + timedelta(seconds=min(300, max(1, 2 ** max(0, count - 1))))
        updated = storage.update_agent_run_status(
            run.id,
            AgentRunStatus.FAILED
            if charge and count >= max_attempts
            else AgentRunStatus.FINALIZATION_FAILED,
            next_resume_at=next_at,
            last_error=last_error,
            increment_finalization_attempts=charge,
            expected_statuses=(run.status,),
            expected_claimed_by=run.claimed_by,
            expected_claimed_at=run.claimed_at,
        )
        if updated is None:
            raise LeaseLostError("Finalization failure lost its retained execution")
        if entry is not None:
            storage.settle_agent_preparation_failure(
                updated, entry.identity, charged=charge
            )
        return updated


@dataclass
class PreparedPublication:
    """Actual service and its already resolved, embedded business write plan."""

    service: Any
    write_plan: ProfileWritePlan | PlaybookWritePlan | None
    bookmark: ExtractorBookmarkPosition | None = None

    def persist(self) -> list[str]:
        if self.write_plan is None:
            return []
        self.service._persist_write_plan(self.write_plan)
        if isinstance(self.write_plan, ProfileWritePlan):
            return [
                str(row.profile_id)
                for row in self.write_plan.new_profiles
                if row.profile_id
            ]
        return [
            str(row.user_playbook_id)
            for row in self.write_plan.new_playbooks
            if row.user_playbook_id
        ]

    def dispatch(self) -> None:
        if isinstance(self.write_plan, PlaybookWritePlan):
            self.service._dispatch_playbook_schedulers(self.write_plan)


@dataclass(frozen=True)
class _RemainingAnswer:
    identifier: str
    resolved: bool
    live_until: datetime | None


def _remaining_work(
    storage: BaseStorage, db: StreamSQL, run: AgentRunRecord
) -> list[_RemainingAnswer]:
    """Reconcile the live unconsumed set under R and nonwaiting A/D locks."""
    remaining: list[_RemainingAnswer] = []
    after = ""
    expected_scope = human_feedback_scope(storage.org_id)
    while True:
        dependencies = db.query(
            f"SELECT d.pending_tool_call_id,d.resolved_at FROM {db.table('_run_tool_dependencies')} d "
            f"JOIN {db.table('_agent_runs')} r ON r.id=d.run_id "
            f"WHERE d.run_id=? AND r.org_id=? AND {db.project_filter()} "
            "AND d.consumed_at IS NULL AND d.pending_tool_call_id>? "
            f"ORDER BY d.pending_tool_call_id LIMIT {_OWNER_PROOF_PAGE_SIZE}"
            + db.validation_lock(alias="d"),
            (run.id, storage.org_id, current_project_id() or "", after),
        )
        if not dependencies:
            break
        ids = [row["pending_tool_call_id"] for row in dependencies]
        placeholders = ",".join("?" for _ in ids)
        answers = db.query(
            "SELECT r.id,r.status,r.scope,r.scope_hash,r.tool_name,r.superseded_by,"
            "r.resolved_at,r.valid_until,r.expires_at "
            f"FROM {db.table('_pending_tool_calls')} r WHERE r.org_id=? AND {db.project_filter()} "
            f"AND r.id IN ({placeholders}) ORDER BY r.id" + db.validation_lock(),
            (storage.org_id, current_project_id() or "", *ids),
        )
        if {row["id"] for row in answers} != set(ids):
            raise InputBudgetConfigurationError("Remaining answer dependency changed")
        resolutions = {
            row["pending_tool_call_id"]: row["resolved_at"] for row in dependencies
        }
        now = datetime.fromtimestamp(db.now(), UTC)
        for row in answers:
            if (
                row["scope_hash"] != build_scope_hash(expected_scope)
                or decoded(row["scope"]) != expected_scope
                or row["tool_name"] != "ask_human"
            ):
                raise InputBudgetConfigurationError("Remaining answer scope changed")
            state = PendingToolCallStatus(row["status"])
            if row["superseded_by"] is not None:
                continue
            if state == PendingToolCallStatus.RESOLVED:
                resolved_at = _run_revision_time(row["resolved_at"])
                valid_until = _run_revision_time(row["valid_until"])
                if resolved_at is None or (
                    valid_until is not None and valid_until <= now
                ):
                    continue
                expected = _run_revision_time(resolutions[row["id"]])
                if expected is not None and expected != resolved_at:
                    raise InputBudgetConfigurationError(
                        "Remaining answer resolution changed"
                    )
                if expected is None:
                    # Resolution can precede attachment. Preserve the actual
                    # row's time so the next model turn can load this answer.
                    db.query(
                        f"UPDATE {db.table('_run_tool_dependencies')} SET resolved_at=? "
                        "WHERE run_id=? AND pending_tool_call_id=? AND resolved_at IS NULL",
                        (resolved_at.isoformat(), run.id, row["id"]),
                    )
                remaining.append(_RemainingAnswer(row["id"], True, valid_until))
            elif state == PendingToolCallStatus.PENDING:
                expires_at = _run_revision_time(row["expires_at"])
                if expires_at is None or expires_at > now:
                    remaining.append(_RemainingAnswer(row["id"], False, expires_at))
        after = ids[-1]
    return remaining


def _remaining_status(
    remaining: list[_RemainingAnswer], now: datetime
) -> tuple[AgentRunStatus, list[str]]:
    # All these rows are still locked. Use one final clock for every page,
    # including after status/hint writes that may have waited for other locks.
    live = [row for row in remaining if row.live_until is None or row.live_until > now]
    status = AgentRunStatus.FINALIZED
    if live:
        status = (
            AgentRunStatus.RESUME_READY
            if any(row.resolved for row in live)
            else AgentRunStatus.FINALIZED_PENDING_TOOL
        )
    return status, [row.identifier for row in live]


def finalize_window_execution(
    storage: BaseStorage, run: AgentRunRecord
) -> AgentRunRecord:
    """Terminalize a compatibility window from its locked live dependencies.

    The window already owns learning/effects. This shares ordinary publication's
    dependency reconciliation without creating another receipt or financial owner.
    """
    with owned_agent_run_write(storage, run), storage._stream_sql() as db:
        if run.status in (
            AgentRunStatus.FINALIZED,
            AgentRunStatus.FINALIZED_PENDING_TOOL,
        ):
            return run
        if run.status not in (
            AgentRunStatus.AGENT_COMPLETED,
            AgentRunStatus.FINALIZING,
        ):
            raise LeaseLostError("Execution is not eligible for window finalization")
        remaining = _remaining_work(storage, db, run)
        witness = run
        observed_at = datetime.fromtimestamp(db.now(), UTC)
        while True:
            status, pending = _remaining_status(remaining, observed_at)
            updated = storage.update_agent_run_status(
                run.id,
                status,
                pending_tool_call_ids=pending,
                next_resume_at=observed_at
                if status == AgentRunStatus.RESUME_READY
                else None,
                expected_statuses=(witness.status,),
                expected_claimed_by=witness.claimed_by,
                expected_claimed_at=witness.claimed_at,
            )
            if updated is None:
                raise LeaseLostError("Window finalization lost its execution revision")
            observed_at = max(observed_at, datetime.fromtimestamp(db.now(), UTC))
            if (status, pending) == _remaining_status(remaining, observed_at):
                return updated
            witness = updated


def _validate_final_answers(
    storage: BaseStorage, db: StreamSQL, run: AgentRunRecord
) -> None:
    """Check captured answers before preparation and after final writer waits."""
    _, inputs = require_publication_answer_inputs(run.committed_output)
    observed = storage._lock_extraction_answers(db, sorted(inputs))
    if {key: build_scope_hash(value) for key, value in observed.items()} != inputs:
        raise InputBudgetConfigurationError("Extraction human answer changed")
    now = datetime.fromtimestamp(db.now(), UTC)
    for witness in observed.values():
        require_current_answer(witness, storage.org_id, now)


def _deliver(storage: BaseStorage, receipt: PublicationReceipt) -> None:
    if receipt.delivery_state != "pending":
        return
    if not storage._publication_scope_matches(receipt.org_id, receipt.project_id):
        raise ValueError("Publication financial scope changed")
    storage._require_independent_writer()
    facts = receipt.financial_facts
    if facts is None:
        raise ValueError("Pending publication financial facts unavailable")
    if receipt.row_kind == "window_handoff":
        if receipt.delivery_key is None:
            raise ValueError("Window financial identity is unavailable")
        record_window_extraction_finances(
            org_id=receipt.org_id,
            window_id=receipt.delivery_key,
            pipeline=facts["pipeline"],
            billing={"ids": list(receipt.learning_ids), **facts},
        )
    else:
        emit_learnings_generated_records_strict(
            org_id=receipt.org_id,
            learning_ids=list(receipt.learning_ids),
            source=facts["source"],
            pipeline=facts["pipeline"],
            entity_type=receipt.entity_type,
            created_at=facts["created_at"],
        )
    # A recorder must not leave us inside another transaction or work scope.
    # In particular, absence on a different project's APP view is not completion.
    if not storage._publication_scope_matches(receipt.org_id, receipt.project_id):
        raise ValueError("Publication financial scope changed")
    storage._require_independent_writer()
    if storage._ack_publication_receipt(receipt):
        return
    # Only this owner, after ALL of its strict events were accepted, may classify
    # peer cleanup as complete. The narrow ACK primitive keeps absence=False.
    if (
        storage._get_publication_receipt(
            receipt.receipt_id, project_id=receipt.project_id
        )
        is None
    ):
        return
    raise RuntimeError("Publication acknowledgement changed")


def _prepare_publication(
    storage: BaseStorage,
    retained: AgentRunRecord,
    candidate_history: HistoricalInputs | None,
    prepare: Callable[
        [AgentRunRecord, SavedPublicationPreparation | None], PreparedPublication
    ],
) -> tuple[PreparedPublication, HistoricalInputs]:
    if candidate_history is None:
        raise ValueError("Original candidate history unavailable")
    saved_data = storage.agent_output_preparation(retained)
    if saved_data is not None:
        saved = SavedPublicationPreparation.from_payload(retained, saved_data)
        history = saved.history_inputs
        # Never replace saved secondary evidence with a fresh historical read.
        storage.validate_agent_history_inputs(
            retained, history.profiles, history.playbooks
        )
        return prepare(retained, saved), history
    with (
        preparation_attempt_scope(storage, retained) as attempt,
        historical_input_scope(storage, retained) as capture,
    ):
        prepared = prepare(retained, None)
        history = capture.finish()
    history = merge_historical_inputs(candidate_history, history)
    from reflexio.server.services.storage.error import execution_input_is_busy

    # Never persist raw completion without strict input validation. Retain the
    # completed payload in this actual owner while waiting OUTSIDE its writer;
    # only persistence is retried, never provider/preparation work.
    payload = SavedPublicationPreparation(
        prepared.write_plan, history, prepared.bookmark
    ).payload(retained)
    wait = False
    while True:
        try:
            if wait:
                storage.wait_for_agent_preparation_inputs(
                    retained, history.profiles, history.playbooks
                )
            storage.agent_output_preparation(
                retained,
                save=payload,
                attempt_id=attempt.identity,
            )
            return prepared, history
        except Exception as error:
            if not execution_input_is_busy(error):
                raise
            wait = True


def _publish_agent_output(
    storage: BaseStorage,
    run: AgentRunRecord,
    *,
    prepare: Callable[
        [AgentRunRecord, SavedPublicationPreparation | None], PreparedPublication
    ],
    on_published: Callable[[AgentRunRecord, PreparedPublication | None], None],
    origin: Literal["initial_inline", "resume"],
) -> tuple[AgentRunRecord, FinalizationResult, PreparedPublication | None]:
    """Own replay/preparation, atomic publication and postcommit obligations."""
    if storage.in_commit_scope:
        raise RuntimeError("Agent output publication requires its own outer writer")
    retained = deepcopy(run)
    output = retained.committed_output
    if not isinstance(output, dict):
        raise InputBudgetConfigurationError("Retained publication output unavailable")
    if output_publication_origin(output) != origin:
        raise InputBudgetConfigurationError(
            "Publication adapter changed its original origin"
        )
    publication = output.get("publication")
    output_id = require_output_id(
        publication.get("id") if isinstance(publication, dict) else None
    )
    candidate_history: HistoricalInputs | None = None
    with owned_agent_run_write(storage, retained), storage._stream_sql() as db:
        storage._require_current_agent_output(db, retained, output)
        existing = storage._get_output_publication_receipt(db, retained.id, output_id)
        if existing is None:
            # Refuse a lost owner or changed input before secondary compute;
            # recheck again in the final writer after computation releases locks.
            storage.validate_agent_output_sources(retained)
            _validate_final_answers(storage, db, retained)
            candidate_history = history_from_output(output)
            storage.validate_agent_history_inputs(
                retained, candidate_history.profiles, candidate_history.playbooks
            )
    # Cache lookup precedes all source reconstruction and secondary model work.
    prepared = None
    history = None
    if existing is None:
        prepared, history = _prepare_publication(
            storage, retained, candidate_history, prepare
        )
    won = False
    updated: AgentRunRecord | None = None
    receipt: PublicationReceipt | None = None
    history_deadline: int | None = None
    try:
        with owned_agent_run_write(storage, retained), storage._stream_sql() as db:
            storage._require_current_agent_output(db, retained, output)
            receipt = storage._get_output_publication_receipt(
                db, retained.id, output_id
            )
            if receipt is None:
                if prepared is None or history is None or candidate_history is None:
                    raise ValueError(
                        "Publication preparation disappeared before retained writer"
                    )
                owner = _require_user_work_owner(storage)
                storage.lock_extraction_stream(owner.user_id)
                storage.validate_agent_output_sources(retained)
                history = merge_historical_inputs(candidate_history, history)
                history_deadline = storage.validate_agent_history_inputs(
                    retained, history.profiles, history.playbooks
                )
                ids = prepared.persist()
                if origin == "initial_inline" and prepared.bookmark is not None:
                    prepared.service._apply_bookmark_advance(prepared.bookmark)
                receipt, won = storage._create_agent_output_receipt(
                    retained, learning_ids=ids, origin=origin
                )
                storage.consume_agent_output_dependencies(retained)
            remaining = _remaining_work(storage, db, retained)
            witness = retained
            observed_at = datetime.fromtimestamp(db.now(), UTC)
            while True:
                status, pending = _remaining_status(remaining, observed_at)
                updated = storage.update_agent_run_status(
                    retained.id,
                    status,
                    pending_tool_call_ids=pending,
                    next_resume_at=datetime.fromtimestamp(db.now(), UTC)
                    if status == AgentRunStatus.RESUME_READY
                    else None,
                    expected_statuses=(witness.status,),
                    expected_claimed_by=witness.claimed_by,
                    expected_claimed_at=witness.claimed_at,
                )
                if updated is None:
                    raise LeaseLostError("Publication lost retained execution revision")
                if won:
                    _validate_final_answers(storage, db, retained)
                # A wall-clock adjustment cannot revive an expiry already
                # observed within this same locked transaction.
                observed_at = max(observed_at, datetime.fromtimestamp(db.now(), UTC))
                if (
                    won
                    and history_deadline is not None
                    and int(observed_at.timestamp()) > history_deadline
                ):
                    raise InputBudgetConfigurationError(
                        "Historical profile expired during publication"
                    )
                if (status, pending) == _remaining_status(remaining, observed_at):
                    break
                # Only expiry can change this locked, closed set. Each extra
                # status write removes at least one remaining answer; it never
                # repeats preparation, learning, consumption or provider work.
                witness = updated
    except PostCommitStorageError as error:
        # The primary transaction committed. Preserve its actual run/receipt,
        # report the derived index failure and still perform postcommit duties.
        if updated is None or receipt is None:
            raise
        logger.warning(
            "event=publication_index_pending run_id=%s %s",
            retained.id,
            _failure_detail(error),
        )
    if updated is None or receipt is None:
        raise RuntimeError("Publication did not retain its business outcome")
    # Never turn a postcommit outage into a failed model/publication attempt.
    if won:
        try:
            on_published(updated, prepared)
        except Exception as error:
            logger.warning(
                "event=publication_derived_dispatch_failed run_id=%s %s",
                retained.id,
                _failure_detail(error),
            )
    try:
        _deliver(storage, receipt)
    except Exception as error:
        logger.warning(
            "event=publication_delivery_pending receipt_id=%s %s",
            receipt.receipt_id,
            _failure_detail(error),
        )
    return (
        updated,
        FinalizationResult(list(receipt.learning_ids), won_receipt=won),
        prepared,
    )


def prepare_initial_publication(
    storage: BaseStorage,
    run: AgentRunRecord,
    service: Any,
    items: Any,
    bookmark_advance: ExtractorBookmarkAdvance | None,
) -> PreparedPublication:
    """Retain the original inline plan before its learning publication writer."""
    retained = deepcopy(run)
    with owned_agent_run_write(storage, retained), storage._stream_sql() as db:
        output = retained.committed_output
        if not isinstance(output, dict):
            raise InputBudgetConfigurationError(
                "Retained publication output unavailable"
            )
        publication = output.get("publication")
        output_id = require_output_id(
            publication.get("id") if isinstance(publication, dict) else None
        )
        storage._require_current_agent_output(db, retained, output)
        if (
            storage._get_output_publication_receipt(db, retained.id, output_id)
            is not None
        ):
            return PreparedPublication(service, None)
        storage.validate_agent_output_sources(retained)
        _validate_final_answers(storage, db, retained)
        candidate = history_from_output(output)
        storage.validate_agent_history_inputs(
            retained, candidate.profiles, candidate.playbooks
        )

    def prepare(
        _run: AgentRunRecord, saved: SavedPublicationPreparation | None
    ) -> PreparedPublication:
        if saved is not None:
            return PreparedPublication(service, saved.write_plan, saved.bookmark)
        bookmark = bookmark_from_output(retained) if items else None
        if bookmark is not None and (
            bookmark_advance is None
            or bookmark != ExtractorBookmarkPosition.from_advance(bookmark_advance)
        ):
            raise InputBudgetConfigurationError(
                "Initial bookmark changed before preparation"
            )
        write_plan = service._resolve_write_plan([items]) if items else None
        return PreparedPublication(service, write_plan, bookmark)

    prepared, _ = _prepare_publication(storage, retained, candidate, prepare)
    return prepared


def publish_initial_output(
    storage: BaseStorage,
    run: AgentRunRecord,
    service: Any,
) -> tuple[AgentRunRecord, FinalizationResult, PreparedPublication | None]:
    """Inline adapter; existing emit owns winner-only token and derived effects."""

    def prepare(
        _run: AgentRunRecord, saved: SavedPublicationPreparation | None
    ) -> PreparedPublication:
        if saved is None:
            raise InputBudgetConfigurationError(
                "Initial output preparation is unavailable"
            )
        return PreparedPublication(service, saved.write_plan, saved.bookmark)

    return _publish_agent_output(
        storage,
        run,
        prepare=prepare,
        on_published=lambda _run, _prepared: None,
        origin="initial_inline",
    )


def publish_resumed_output(
    storage: BaseStorage,
    run: AgentRunRecord,
    *,
    prepare: Callable[
        [AgentRunRecord, SavedPublicationPreparation | None], PreparedPublication
    ],
    on_published: Callable[[AgentRunRecord], None],
) -> AgentRunRecord:
    """Worker adapter, including retry of an initial output's unchanged obligation."""
    origin = output_publication_origin(run.committed_output)
    if origin == "initial_window":
        raise InputBudgetConfigurationError(
            "Initial window publication requires its window owner"
        )

    def dispatch(updated: AgentRunRecord, prepared: PreparedPublication | None) -> None:
        try:
            if prepared is not None:
                prepared.dispatch()
        except Exception as error:
            logger.warning(
                "event=publication_derived_dispatch_failed run_id=%s %s",
                run.id,
                _failure_detail(error),
            )
        on_published(updated)

    updated, _, _ = _publish_agent_output(
        storage,
        run,
        prepare=prepare,
        on_published=dispatch,
        origin=origin,
    )
    return updated
