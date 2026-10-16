# ruff: noqa: S608 -- Identifiers come only from the fixed storage table registry.
"""Read-only recovery receipts and explicit, fenced operator application.

This is not an automatic retry policy. Missing parents cannot establish original
quota authority and remain held. Callers provide an already project-scoped store.
"""

import hashlib
import json
from typing import Any

from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
    OverflowCheckpointConflictError,
    Window,
    decoded,
    validate_extraction_overflow_manifest,
)
from reflexio.server.work_scope import current_project_id


def _inspect(storage: BaseStorage, window_id: str) -> tuple[Window, dict[str, Any]]:
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
            f"SELECT r.status,r.generation_request_snapshot,r.updated_at FROM {db.table('_agent_runs')} r "
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
    if (
        window.invalidated
        or rows[0]["completed"]
        or cursor.get("window_id") != window_id
        or cursor.get("last_error") != "input_budget_exceeded"
        or cursor.get("retry_at") != float("inf")
    ):
        reason = "not_held"
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
    return window, {
        "version": 1,
        "org_id": storage.org_id,
        "project_id": window.project_id,
        "window_id": window_id,
        "state_sha256": fingerprint,
        "recoverable": reason == "ready",
        "reason": reason,
        "cursor_attempts": cursor.get("attempts"),
        "parent_revision": manifest.get("revision"),
        "source_digest": manifest.get("source_digest"),
        "attempts_used": len(manifest["attempts"]) if manifest else None,
        "input_protocol": manifest.get("input_protocol"),
    }


def plan_input_budget_recovery(storage: BaseStorage, window_id: str) -> dict[str, Any]:
    """Return a metadata-only receipt; no leases, model calls or writes."""
    return _inspect(storage, window_id)[1]


def apply_input_budget_recovery(storage: BaseStorage, receipt: dict[str, Any]) -> None:
    """Apply a reviewed receipt once; preserve failed history until completion."""
    window, current = _inspect(storage, receipt["window_id"])
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
        if _inspect(storage, window.window_id)[1] != receipt:
            raise OverflowCheckpointConflictError(
                "Recovery changed before lease acquisition"
            )
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
