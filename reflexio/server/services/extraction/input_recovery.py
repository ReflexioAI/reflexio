# ruff: noqa: S608 -- Identifiers come only from the fixed storage table registry.
"""Read-only recovery receipts and explicit, fenced operator application.

This is not an automatic retry policy. Missing parents cannot establish original
quota authority and remain held. Callers provide an already project-scoped store.
A receipt's ``action`` selects what applying it does: ``rearm`` (the default)
re-arms a held cursor, ``reset_legacy_policy`` restarts a legacy checkpoint held
by a policy mismatch, and ``retire`` closes a parked window without extracting
it so the user's later inputs can proceed.
"""

import hashlib
import json
from typing import Any, Literal

from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LEGACY_POLICY_HOLDS,
    LeaseLostError,
    OverflowCheckpointConflictError,
    Window,
    decoded,
    is_retry_exhausted,
    validate_extraction_overflow_manifest,
)
from reflexio.server.work_scope import current_project_id

# Only parked holds (``retry_at=inf``) qualify; a configuration failure that is
# still retrying on backoff is not held and needs no re-arm. A cursor parked by
# the retry ceiling (``is_retry_exhausted``) qualifies too, as hold
# ``retry_exhausted``.
RECOVERABLE_HOLDS = frozenset(
    {"input_budget_exceeded", "input_budget_configuration_error"}
)

RecoveryAction = Literal["rearm", "reset_legacy_policy", "retire"]


def _legacy_policy_reason(
    last_error: Any, committed: Any, checkpoint: dict[str, Any]
) -> str:
    """Whether a held checkpoint is one the legacy policy reset may restart.

    A checkpoint carrying the instruction witness already adopts a context-only
    policy change by itself, and a changed instruction stays held by design.
    """
    if last_error not in LEGACY_POLICY_HOLDS:
        return "not_configuration_hold"
    if committed:
        return "committed_output"
    if (
        "extraction_instruction_contract" in checkpoint
        or "extraction_policy" not in checkpoint
    ):
        return "not_legacy_policy_checkpoint"
    if checkpoint.get("active_child") or checkpoint.get("waiting_child"):
        return "active_continuation"
    return "ready"


def _inspect(
    storage: BaseStorage, window_id: str, *, action: RecoveryAction = "rearm"
) -> tuple[Window, dict[str, Any]]:
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            f"SELECT window_id,user_id,kind,project_id,predecessor,end_seq,manifest,policy,"
            f"force,skip_aggregation,outcome,invalidated,completed FROM {db.table('extraction_windows')} "
            "WHERE window_id=? AND project_id=? LIMIT 1",
            (window_id, current_project_id() or ""),
        )
        if not rows:
            raise OverflowCheckpointConflictError(
                "Recovery window unavailable in project"
            )
        window = Window.from_row(rows[0])
        cursors = db.query(
            f"SELECT window_id,attempts,retry_at,last_error FROM {db.table('extraction_cursors')} "
            "WHERE user_id=? AND kind=? AND project_id=? LIMIT 1",
            (window.user_id, window.kind, window.project_id),
        )
        parents = db.query(
            f"SELECT r.status,r.generation_request_snapshot,r.updated_at,"
            f"r.committed_output IS NOT NULL AS committed FROM {db.table('_agent_runs')} r "
            f"JOIN {db.table('extraction_windows')} w ON r.id='window:'||w.window_id "
            "AND r.user_id=w.user_id AND r.extractor_kind=w.kind "
            "WHERE w.window_id=? AND r.org_id=? AND w.user_id=? AND w.project_id=? LIMIT 1",
            (
                window.window_id,
                storage.org_id,
                window.user_id,
                window.project_id,
            ),
        )
    cursor = cursors[0] if cursors else {}
    parent = parents[0] if parents else {}
    state = {"window": rows[0], "cursor": cursor, "parent": parent}
    fingerprint = hashlib.sha256(
        json.dumps(state, sort_keys=True, default=str).encode()
    ).hexdigest()
    reason = "ready"
    manifest: dict[str, Any] = {}
    exhausted = is_retry_exhausted(cursor.get("last_error"))
    if (
        window.invalidated
        or rows[0]["completed"]
        or cursor.get("window_id") != window_id
        or cursor.get("retry_at") != float("inf")
        or not cursor.get("last_error")
    ):
        reason = "not_held"
    elif action == "retire":
        # Retiring learns nothing and spends nothing, so any parked hold
        # qualifies, whatever its parent or journal state.
        reason = "ready"
    elif not (exhausted or cursor.get("last_error") in RECOVERABLE_HOLDS):
        reason = "not_held"
    elif exhausted and action == "rearm":
        # The retry ceiling is not a quota decision: no parent or journal
        # authority is needed to grant one more attempt, and any journaled
        # provider quota is still enforced by the extraction itself.
        reason = "ready"
    elif not parent:
        reason = "missing_parent_unknown_history"
    elif parent["status"] != "failed":
        reason = "parent_not_failed"
    else:
        snapshot = decoded(parent["generation_request_snapshot"])
        try:
            manifest = validate_extraction_overflow_manifest(snapshot.get("overflow"))
        except OverflowCheckpointConflictError:
            reason = "invalid_or_unknown_journal"
        if manifest and len(manifest["attempts"]) >= manifest["attempt_limit"]:
            reason = "quota_exhausted"
        elif manifest and action == "reset_legacy_policy":
            reason = _legacy_policy_reason(
                cursor.get("last_error"), parent["committed"], manifest["checkpoint"]
            )
    return window, {
        "version": 1,
        "action": action,
        "org_id": storage.org_id,
        "project_id": window.project_id,
        "window_id": window_id,
        "state_sha256": fingerprint,
        "recoverable": reason == "ready",
        "reason": reason,
        "hold": "retry_exhausted" if exhausted else cursor.get("last_error"),
        # ``<ExceptionClass>@<locator>`` (or a fixed reason word): coordinates,
        # never a message, so safe to put on a receipt.
        "last_error": cursor.get("last_error"),
        "cursor_attempts": cursor.get("attempts"),
        "parent_revision": manifest.get("revision"),
        "source_digest": manifest.get("source_digest"),
        "attempts_used": len(manifest["attempts"]) if manifest else None,
        "input_protocol": manifest.get("input_protocol"),
    }


def plan_input_budget_recovery(
    storage: BaseStorage, window_id: str, *, action: RecoveryAction = "rearm"
) -> dict[str, Any]:
    """Return a metadata-only receipt; no leases, model calls or writes.

    ``reset_legacy_policy`` asks whether the window's legacy checkpoint can
    restart under the current policy: it reruns every chunk, so it spends
    attempts from the same budget. ``retire`` asks whether the window is parked
    and can be closed without being extracted; an overlapping policy still
    shows the next window its last inputs as context, as after any completion.
    An operator chooses either per window.
    """
    return _inspect(storage, window_id, action=action)[1]


def apply_input_budget_recovery(storage: BaseStorage, receipt: dict[str, Any]) -> None:
    """Apply a reviewed receipt once; preserve failed history until completion."""
    action = receipt.get("action", "rearm")
    window, current = _inspect(storage, receipt["window_id"], action=action)
    if current != receipt or not current["recoverable"]:
        raise OverflowCheckpointConflictError(
            "Recovery receipt changed or is not eligible"
        )
    token = storage.claim_user_extraction(
        window.user_id, "operator-input-recovery", 300
    )
    if token is None:
        raise LeaseLostError("Recovery user has an active lease")
    try:
        if _inspect(storage, window.window_id, action=action)[1] != receipt:
            raise OverflowCheckpointConflictError(
                "Recovery changed before lease acquisition"
            )
        if action == "retire":
            storage.retire_extraction_window(
                window,
                token,
                expected_attempts=current["cursor_attempts"],
                expected_last_error=current["last_error"],
            )
            return
        if action == "reset_legacy_policy":
            storage.reset_legacy_extraction_policy(
                window,
                token,
                expected_revision=current["parent_revision"],
                source_digest=current["source_digest"],
                expected_attempts=current["cursor_attempts"],
                expected_last_error=current["last_error"],
                expected_input_protocol=current["input_protocol"],
            )
            return
        if current["hold"] == "retry_exhausted":
            storage.rearm_exhausted_extraction(
                window,
                token,
                expected_attempts=current["cursor_attempts"],
                expected_last_error=current["last_error"],
            )
            return
        storage.recover_extraction_input_budget(
            window,
            token,
            expected_revision=current["parent_revision"],
            source_digest=current["source_digest"],
            expected_attempts=current["cursor_attempts"],
            expected_input_protocol=current["input_protocol"],
        )
    finally:
        storage.release_user_extraction(window.user_id, token)
