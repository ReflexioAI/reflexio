"""Bounded, content-free diagnostics and previews for operator-selected recovery.

Inspection never claims work or obtains a locking finalization receipt. A
preview is evidence, not authority: apply must re-read it under the run scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.resumable_agent import (
    decode_committed_output,
    pending_tool_calls_disabled_reason,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredExtractedPlaybookList,
    StructuredPlaybookList,
    StructuredReferencedExtractedPlaybookList,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    build_scope_hash,
)


class RecoveryRefusedError(RuntimeError):
    """Safe reason code only; never include customer data in this exception."""


class RecoverySelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    project_id: str | None
    updated_at: datetime
    fingerprint: str
    payload_fingerprint: str
    config_fingerprint: str


class RecoveryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    org_id: str
    project_id: str | None
    selections: list[RecoverySelection] = Field(min_length=1, max_length=10)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def payload_fingerprint(run: AgentRunRecord) -> str:
    return _digest(
        {
            "binding": asdict(run.binding),
            "project_id": run.project_id,
            "output": run.committed_output,
            "generation": run.generation_request_snapshot,
            "service": run.service_config_snapshot,
            "context": run.agent_context_snapshot,
            "pending": run.pending_tool_call_ids,
        }
    )


def _validate_saved_output(model: type[BaseModel], output: dict[str, Any]) -> None:
    collection = "profiles" if model is StructuredProfilesOutput else "playbooks"
    if not isinstance(output.get(collection), list):
        raise ValueError("Saved output must explicitly contain its collection")
    _require_preserved_output_fields(output, model.model_validate(output))


def _require_preserved_output_fields(raw: Any, parsed: Any) -> None:
    """Refuse fields/candidates a permissive historical schema would discard."""
    if isinstance(parsed, BaseModel):
        parsed = {name: getattr(parsed, name) for name in type(parsed).model_fields}
    if isinstance(raw, dict):
        if not isinstance(parsed, dict) or not raw.keys() <= parsed.keys():
            raise ValueError("Unrecognized saved output fields")
        for key, value in raw.items():
            _require_preserved_output_fields(value, parsed[key])
    elif isinstance(raw, list):
        if not isinstance(parsed, list) or len(raw) != len(parsed):
            raise ValueError("Saved output candidates were discarded")
        for value, item in zip(raw, parsed, strict=True):
            _require_preserved_output_fields(value, item)


def inspect_run(
    context: RequestContext,
    run: AgentRunRecord,
    *,
    now: datetime | None = None,
    owned_claim: bool = False,
) -> dict[str, Any]:
    """Report explicit reasons without exposing content or raw exceptions."""
    if run.binding.org_id != context.org_id or context.storage is None:
        raise RecoveryRefusedError("organization_mismatch")
    current = now or datetime.now(UTC)
    config = context.configurator.get_config()
    ttl = config.pending_tool_call_config.resume_claim_ttl_seconds
    stale = current - timedelta(seconds=ttl)
    facts = context.storage.get_agent_run_recovery_facts(run.id)
    if not facts:
        raise RecoveryRefusedError("diagnostic_evidence_unavailable")
    blockers: list[str] = (
        ["receipt_results_missing"] if facts["receipt_missing_results"] else []
    )
    if run.status not in (
        AgentRunStatus.AGENT_COMPLETED,
        AgentRunStatus.FINALIZATION_FAILED,
        AgentRunStatus.FINALIZING,
    ):
        blockers.append("not_finalization_candidate")
    if run.updated_at is None:
        blockers.append("missing_version")
    if not owned_claim:
        if run.claimed_by and (run.claimed_at is None or run.claimed_at >= stale):
            blockers.append("live_or_unknown_claim")
        if run.status == AgentRunStatus.AGENT_COMPLETED and (
            run.updated_at is None or run.updated_at >= stale
        ):
            blockers.append("publish_finalization_may_be_active")
        if run.status == AgentRunStatus.FINALIZING and (
            run.claimed_at is None or run.claimed_at >= stale
        ):
            blockers.append("finalization_may_be_active")
    if run.next_resume_at and run.next_resume_at > current:
        blockers.append("retry_not_due")
    if (
        run.finalization_attempts
        >= config.pending_tool_call_config.max_finalization_attempts
    ):
        blockers.append("retry_limit_reached")
    if not facts["window_ready"]:
        blockers.append("window_not_ready")
    if run.pending_tool_call_ids or facts["dependencies"]:
        blockers.append("pending_dependencies")
    sources = set(run.binding.source_interaction_ids)
    if not sources or facts["valid_source_count"] != len(sources):
        blockers.append("source_missing_or_scope_mismatch")
    extractor = {
        "profile": config.profile_extractor_config,
        "playbook": config.user_playbook_extractor_config,
    }.get(run.binding.extractor_kind)
    config_hash = (
        build_scope_hash(extractor.model_dump(mode="json")) if extractor else None
    )
    if not config_hash or config_hash != run.binding.extractor_config_hash:
        blockers.append("configuration_changed")
    schemas = {
        model.__name__: model
        for model in (
            StructuredProfilesOutput,
            StructuredPlaybookList,
            StructuredExtractedPlaybookList,
            StructuredReferencedExtractedPlaybookList,
        )
    }
    name = run.generation_request_snapshot.get("output_schema_name")
    model = schemas.get(name) if isinstance(name, str) else None
    if run.binding.extractor_kind == "profile" and name is None:
        # Profile output has one stable contract; playbooks have multiple contracts.
        model = StructuredProfilesOutput
    if model is None or (
        (model is StructuredProfilesOutput) != (run.binding.extractor_kind == "profile")
    ):
        blockers.append("missing_or_incompatible_schema")
    elif run.committed_output is not None:
        try:
            output, _ = decode_committed_output(run.committed_output)
            _validate_saved_output(model, output)
        except (ValueError, TypeError, ValidationError):
            blockers.append("invalid_saved_output")
    if run.committed_output is None:
        blockers.append("saved_output_missing")
    return {
        "run_id": run.id,
        "org_id": run.binding.org_id,
        "project_id": run.project_id,
        "status": run.status.value,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "progress_stage": run.progress_stage,
        "last_progress_at": run.last_progress_at.isoformat()
        if run.last_progress_at
        else None,
        "progress_age_seconds": max(0, (current - run.last_progress_at).total_seconds())
        if run.last_progress_at
        else None,
        "claim_present": run.claimed_by is not None,
        "claim_expires_at": (run.claimed_at + timedelta(seconds=ttl)).isoformat()
        if run.claimed_at
        else None,
        "finalization_attempts": run.finalization_attempts,
        "resume_disabled_reason": pending_tool_calls_disabled_reason(context),
        "saved_output_present": run.committed_output is not None,
        "receipt_exists": bool(facts["receipt_exists"]),
        "receipt_count": facts["receipt_count"],
        "receipt_missing_results": facts["receipt_missing_results"],
        "source_count": len(sources),
        "valid_source_count": facts["valid_source_count"],
        "blockers": blockers,
        "eligible": not blockers,
        "fingerprint": _digest(asdict(run)),
        "payload_fingerprint": payload_fingerprint(run),
        "config_fingerprint": build_scope_hash(config.model_dump(mode="json")),
        "recovery_operation_id": run.recovery_operation_id,
        "recovery_history": run.recovery_history,
    }


def selection_from_report(report: dict[str, Any]) -> RecoverySelection:
    if not report["eligible"]:
        raise RecoveryRefusedError("preview_has_blockers")
    return RecoverySelection.model_validate(
        {key: report[key] for key in RecoverySelection.model_fields}
    )


def validate_selection(
    context: RequestContext,
    run: AgentRunRecord,
    selection: RecoverySelection,
    *,
    owned_claim: bool = False,
) -> None:
    report = inspect_run(context, run, owned_claim=owned_claim)
    if (
        run.id != selection.run_id
        or run.binding.org_id != selection.org_id
        or run.project_id != selection.project_id
    ):
        raise RecoveryRefusedError("selection_scope_changed")
    if not report["eligible"]:
        raise RecoveryRefusedError("eligibility_changed")
    if (
        report["config_fingerprint"] != selection.config_fingerprint
        or report["payload_fingerprint"] != selection.payload_fingerprint
    ):
        raise RecoveryRefusedError("preview_changed")
    if not owned_claim and report["fingerprint"] != selection.fingerprint:
        raise RecoveryRefusedError("preview_changed")
