from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Literal, NoReturn, Protocol, cast, get_args

from reflexio.models.api_schema.domain import AgentPlaybook, AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.governance import (
    AuditActorType,
    AuditEntityType,
    AuditEvent,
    AuditOperation,
    AuditStatus,
    PurgeOperation,
    PurgeOperationTarget,
    PurgeOperationType,
    PurgeScopeType,
    PurgeTargetStatus,
)
from reflexio.models.config_schema import GovernanceRetentionConfig

_LEGACY_AUDIT_REQUEST_REF = "reqref_v1_legacy_unknown"

_PURGE_OPERATION_TARGETS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS purge_operation_targets (
    org_id TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_ref TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    detail TEXT,
    deleted_count INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT,
    started_at INTEGER,
    completed_at INTEGER,
    PRIMARY KEY (org_id, purge_id, target_name, target_ref, phase),
    FOREIGN KEY (org_id, purge_id) REFERENCES purge_operations(org_id, purge_id) ON DELETE CASCADE
);
"""

GOVERNANCE_DDL = f"""
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_ref TEXT,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    subject_ref TEXT,
    request_ref TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    detail TEXT,
    created_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_events_org_idem
    ON audit_events(org_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_events_subject_created
    ON audit_events(org_id, subject_ref, created_at, event_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_org_created
    ON audit_events(org_id, created_at, event_id);

CREATE TABLE IF NOT EXISTS purge_operations (
    org_id TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    subject_ref TEXT,
    request_ref TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_code TEXT,
    error_detail TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER,
    PRIMARY KEY (org_id, purge_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_purge_operations_org_idem
    ON purge_operations(org_id, idempotency_key);

{_PURGE_OPERATION_TARGETS_TABLE_DDL}
CREATE INDEX IF NOT EXISTS idx_purge_targets_purge_phase
    ON purge_operation_targets(org_id, purge_id, phase, status);
"""

_PREPARE_PHASE = "prepare_targets"
_SNAPSHOT_TARGET_NAME = "target_snapshot"
_CANONICAL_DELETE_TARGET_NAMES = (
    "request",
    "interaction",
    "profile",
    "user_playbook",
    "agent_success_evaluation_result",
    "profile_purge",
    "user_playbook_purge",
)
_ALLOWED_AUDIT_ACTOR_TYPES = frozenset(get_args(AuditActorType))
_ALLOWED_AUDIT_OPERATIONS = frozenset(get_args(AuditOperation))
_ALLOWED_AUDIT_ENTITY_TYPES = frozenset(get_args(AuditEntityType))
_ALLOWED_AUDIT_STATUSES = frozenset(get_args(AuditStatus))
_ALLOWED_PURGE_OPERATION_TYPES = frozenset(get_args(PurgeOperationType))
_ALLOWED_PURGE_SCOPE_TYPES = frozenset(get_args(PurgeScopeType))
_ALLOWED_PURGE_TARGET_STATUSES = frozenset(get_args(PurgeTargetStatus))
_ALLOWED_PURGE_TARGET_NAMES = frozenset(
    {
        _SNAPSHOT_TARGET_NAME,
        "request",
        "interaction",
        "profile",
        "user_playbook",
        "agent_success_evaluation_result",
        "agent_playbook",
        "profile_purge",
        "user_playbook_purge",
    }
)
_ALLOWED_PURGE_TARGET_PHASES = frozenset(
    {
        _PREPARE_PHASE,
        "delete",
        "hide_for_rebuild",
        "rebuild_without_erased_sources",
    }
)
_ALLOWED_AUDIT_DETAIL_KEYS = frozenset(
    {
        "agent_playbook_id",
        "count",
        "deleted_counts",
        "deleted_count",
        "rebuilt_agent_playbook_ids",
        "route",
        "status",
    }
)
_ALLOWED_PURGE_TARGET_DETAIL_KEYS = frozenset(
    {
        "affected_agent_playbook_ids",
        "agent_playbook_id",
        "count",
        "deleted_counts",
        "deleted_count",
        "erased_source_ids",
        "owned_user_playbook_ids",
        "original_source_windows",
        "previous_lifecycle_status",
        "prepared",
        "rebuilt_agent_playbook_ids",
        "remaining_source_windows",
        "route",
        "source_interaction_ids",
        "status",
        "user_playbook_id",
    }
)
_DISALLOWED_DETAIL_KEYS = frozenset(
    {
        "content",
        "email",
        "prompt",
        "request_id",
        "request_ref",
        "user_id",
    }
)
_EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
_REQUEST_ID_RE = re.compile(
    r"\b(?:reqref_(?!v1_)|request[_-]|req[_-])[A-Za-z0-9_-]*\b",
    re.IGNORECASE,
)
_TOKEN_NAME_RE = re.compile(
    r"\b(?:api[-_ ]?token|token[-_ ]?name|bearer|secret[-_ ]?key)\b",
    re.IGNORECASE,
)
_RAW_EXCEPTION_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\s*:")
_SAFE_INTERNAL_ID_RE = re.compile(r"^[0-9]+$")
_USER_LIKE_TARGET_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_CODE_SHAPED_VALUE_RE = re.compile(r"^[A-Za-z0-9]+(?:[_.:-][A-Za-z0-9]+)+$")
_IDENTIFIERISH_CODE_VALUE_RE = re.compile(
    r"^(?:user|subject|actor)[_.:-][A-Za-z0-9]+(?:[_.:-][A-Za-z0-9]+)*$",
    re.IGNORECASE,
)
_SAFE_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_IDENTIFIERISH_ERROR_CODE_RE = re.compile(
    r"^(?:user|subject|request|req|actor|email)[-_.:]?[A-Za-z0-9_.:-]+$",
    re.IGNORECASE,
)
_ALLOWED_DETAIL_STATUS_VALUES = frozenset(
    {
        "archive_in_progress",
        "complete",
        "error",
        "failed",
        "ok",
        "pending",
        "running",
    }
)
_ALLOWED_PREVIOUS_LIFECYCLE_STATUS_VALUES = frozenset(
    status.value for status in Status if status.value is not None
)
_ALLOWED_DETAIL_ROUTE_VALUES = frozenset(
    {
        "prepare_targets",
        "delete",
        "hide_for_rebuild",
        "rebuild_without_erased_sources",
    }
)
_ALLOWED_DELETED_COUNTS_KEYS = frozenset(
    {
        "interactions",
        "user_playbooks",
        "profiles",
        "requests",
        "agent_success_evaluation_results",
        "purged_profiles",
        "purged_user_playbooks",
    }
)


def init_governance_tables(conn: sqlite3.Connection) -> None:
    _upgrade_legacy_purge_operation_targets_table(conn)
    conn.executescript(GOVERNANCE_DDL)
    _enforce_audit_request_ref_not_null(conn)


def _epoch_now() -> int:
    return int(datetime.now(UTC).timestamp())


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _json_loads(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


def _raise_governance_validation_error(field_name: str, reason: str) -> NoReturn:
    raise ValueError(f"Unsafe governance {field_name}: {reason}")


def _validate_governance_string(field_name: str, value: str) -> None:
    if _EMAIL_RE.search(value):
        _raise_governance_validation_error(field_name, "email")
    if _REQUEST_ID_RE.search(value):
        _raise_governance_validation_error(field_name, "request_id")
    if _TOKEN_NAME_RE.search(value):
        _raise_governance_validation_error(field_name, "token")
    if _RAW_EXCEPTION_RE.search(value):
        _raise_governance_validation_error(field_name, "raw exception text")


def _validate_governance_prose_string(field_name: str, value: str) -> None:
    _validate_governance_string(field_name, value)
    lowered = value.lower()
    if "prompt" in lowered or "content" in lowered:
        _raise_governance_validation_error(field_name, "prompt/content")


def _validate_governance_prefixed_ref(
    field_name: str, value: str | None, *, prefix: str
) -> None:
    if value is None:
        return
    if re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{32}}", value) is None:
        _raise_governance_validation_error(
            field_name, f"must match {prefix}<32 lowercase hex chars>"
        )


def _validate_governance_code_shaped(
    field_name: str,
    value: str,
    *,
    allow_minimized_ref: bool,
) -> str:
    if not value:
        _raise_governance_validation_error(field_name, "required")
    _validate_governance_string(field_name, value)
    if allow_minimized_ref and any(
        re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{32}}", value)
        for prefix in ("subref_v1_", "reqref_v1_", "actref_v1_")
    ):
        return value
    if value.startswith(("subref_v1_", "reqref_v1_", "actref_v1_")):
        _raise_governance_validation_error(field_name, "identifier")
    if _IDENTIFIERISH_CODE_VALUE_RE.fullmatch(value):
        _raise_governance_validation_error(field_name, "identifier")
    if _SAFE_INTERNAL_ID_RE.fullmatch(value):
        return value
    if _CODE_SHAPED_VALUE_RE.fullmatch(value):
        return value
    if _USER_LIKE_TARGET_REF_RE.fullmatch(value):
        _raise_governance_validation_error(field_name, "user-like identifier")
    _raise_governance_validation_error(
        field_name, "must be minimized, internal, or code-shaped"
    )
    raise AssertionError("unreachable")


def _validate_governance_idempotency_key(
    field_name: str, value: str | None
) -> str | None:
    if value is None:
        return None
    if _SAFE_INTERNAL_ID_RE.fullmatch(value):
        _raise_governance_validation_error(field_name, "numeric identifier")
    return _validate_governance_code_shaped(
        field_name,
        value,
        allow_minimized_ref=False,
    )


def _validate_governance_detail_enum(
    field_name: str, value: Any, *, allowed_values: frozenset[str]
) -> str:
    if not isinstance(value, str):
        _raise_governance_validation_error(field_name, "expected str")
    _validate_governance_prose_string(field_name, value)
    if value not in allowed_values:
        _raise_governance_validation_error(field_name, "must be canonical")
    return value


def _validate_governance_purge_id(field_name: str, value: str) -> str:
    if not value:
        _raise_governance_validation_error(field_name, "required")
    _validate_governance_string(field_name, value)
    if value.startswith(("subref_v1_", "reqref_v1_", "actref_v1_")):
        _raise_governance_validation_error(field_name, "identifier")
    if not value.startswith("purge_"):
        _raise_governance_validation_error(field_name, "must start with purge_")
    if _CODE_SHAPED_VALUE_RE.fullmatch(value) is None:
        _raise_governance_validation_error(field_name, "must be code-shaped")
    suffix = value[len("purge_") :]
    if _IDENTIFIERISH_CODE_VALUE_RE.fullmatch(suffix):
        _raise_governance_validation_error(field_name, "identifier")
    if suffix.isdecimal():
        _raise_governance_validation_error(field_name, "numeric identifier")
    return value


def _validate_governance_int(field_name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_governance_validation_error(field_name, "expected int")


def _validate_governance_nonnegative_int(field_name: str, value: Any) -> int:
    _validate_governance_int(field_name, value)
    if value < 0:
        _raise_governance_validation_error(field_name, "must be nonnegative")
    return cast(int, value)


def _validate_governance_deleted_count(value: Any) -> int:
    return _validate_governance_nonnegative_int("deleted_count", value)


def _validate_governance_int_list(field_name: str, value: Any) -> list[int]:
    if not isinstance(value, list):
        _raise_governance_validation_error(field_name, "expected list[int]")
    normalized_items: list[int] = []
    for item in value:
        _validate_governance_int(field_name, item)
        normalized_items.append(cast(int, item))
    return normalized_items


def _validate_governance_deleted_counts(field_name: str, value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        _raise_governance_validation_error(field_name, "expected dict[str, int]")
    normalized_counts: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        if key in normalized_counts:
            _raise_governance_validation_error(field_name, f"duplicate key {key}")
        if key not in _ALLOWED_DELETED_COUNTS_KEYS:
            _raise_governance_validation_error(field_name, key)
        normalized_counts[key] = _validate_governance_deleted_count(raw_value)
    return normalized_counts


def _normalize_governance_window_item(
    field_name: str, index: int, item: object
) -> dict[str, Any]:
    if not isinstance(item, dict):
        _raise_governance_validation_error(
            f"{field_name}[{index}]", "expected window dict"
        )
    window_item = cast(dict[Any, Any], item)
    normalized_item: dict[str, Any] = {}
    for raw_key, raw_value in window_item.items():
        normalized_key = str(raw_key).strip().lower()
        if normalized_key in normalized_item:
            _raise_governance_validation_error(
                f"{field_name}[{index}]", f"duplicate key {normalized_key}"
            )
        normalized_item[normalized_key] = raw_value
    return normalized_item


def _validate_governance_window_list(
    field_name: str, value: Any
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        _raise_governance_validation_error(field_name, "expected list[window]")
    normalized_windows: list[dict[str, object]] = []
    for index, item in enumerate(value):
        normalized_item = _normalize_governance_window_item(field_name, index, item)
        normalized_keys = set(normalized_item)
        unexpected_keys = normalized_keys - {
            "user_playbook_id",
            "source_interaction_ids",
        }
        if unexpected_keys:
            _raise_governance_validation_error(
                f"{field_name}[{index}]", sorted(unexpected_keys)[0]
            )
        if "user_playbook_id" not in normalized_item:
            _raise_governance_validation_error(
                f"{field_name}[{index}].user_playbook_id", "required"
            )
        _validate_governance_int(
            f"{field_name}[{index}].user_playbook_id",
            normalized_item["user_playbook_id"],
        )
        canonical_item: dict[str, object] = {
            "user_playbook_id": cast(int, normalized_item["user_playbook_id"])
        }
        if "source_interaction_ids" in normalized_item:
            canonical_item["source_interaction_ids"] = _validate_governance_int_list(
                f"{field_name}[{index}].source_interaction_ids",
                normalized_item["source_interaction_ids"],
            )
        normalized_windows.append(canonical_item)
    return normalized_windows


def _parse_governance_window_list(
    field_name: str, value: list[dict[str, object]]
) -> list[AgentPlaybookSourceWindow]:
    windows: list[AgentPlaybookSourceWindow] = []
    for normalized_item in _validate_governance_window_list(field_name, value):
        user_playbook_id = cast(int, normalized_item["user_playbook_id"])
        source_ids = cast(
            list[int], normalized_item.get("source_interaction_ids") or []
        )
        windows.append(
            AgentPlaybookSourceWindow(
                user_playbook_id=user_playbook_id,
                source_interaction_ids=[int(source_id) for source_id in source_ids],
            )
        )
    return windows


def _canonicalize_governance_windows(
    field_name: str, value: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [
        window.model_dump()
        for window in _parse_governance_window_list(field_name, value)
    ]


def _build_agent_playbook_source_window_rows(
    agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
) -> list[tuple[int, int, str]]:
    by_id: dict[int, list[int]] = {}
    for window in windows:
        ids = by_id.setdefault(window.user_playbook_id, [])
        seen = set(ids)
        for source_id in window.source_interaction_ids:
            if source_id not in seen:
                ids.append(source_id)
                seen.add(source_id)
    return [
        (
            agent_playbook_id,
            user_playbook_id,
            _json_dumps(source_interaction_ids) or "[]",
        )
        for user_playbook_id, source_interaction_ids in by_id.items()
    ]


def _validate_governance_target_ref(
    *, target_name: str, phase: str, target_ref: str
) -> str:
    if target_name == _SNAPSHOT_TARGET_NAME:
        if phase != _PREPARE_PHASE:
            _raise_governance_validation_error(
                _SNAPSHOT_TARGET_NAME, "must use prepare_targets phase"
            )
        if target_ref != "all":
            _raise_governance_validation_error("target_ref", "must be all")
        return target_ref
    if target_name in {
        "request",
        "interaction",
        "profile",
        "user_playbook",
        "profile_purge",
        "user_playbook_purge",
    }:
        if phase != "delete":
            _raise_governance_validation_error(
                phase,
                f"{target_name} targets must use delete phase",
            )
        if target_ref != "all":
            _raise_governance_validation_error("target_ref", "must be all")
        return target_ref
    if target_name == "agent_playbook":
        if phase not in {"hide_for_rebuild", "rebuild_without_erased_sources"}:
            _raise_governance_validation_error(
                phase,
                "agent_playbook targets must use hide_for_rebuild or "
                "rebuild_without_erased_sources",
            )
        if _SAFE_INTERNAL_ID_RE.fullmatch(target_ref):
            return target_ref
        _raise_governance_validation_error(
            "target_ref", "must be a numeric internal id"
        )
    if target_ref in {"", "all"}:
        return target_ref
    if _SAFE_INTERNAL_ID_RE.fullmatch(target_ref):
        return target_ref
    for prefix in ("reqref_v1_", "subref_v1_", "actref_v1_"):
        if re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{32}}", target_ref):
            return target_ref
        if target_ref.startswith(prefix):
            _raise_governance_validation_error(
                "target_ref", f"must match {prefix}<32 lowercase hex chars>"
            )
    _validate_governance_string("target_ref", target_ref)
    if _USER_LIKE_TARGET_REF_RE.fullmatch(target_ref):
        _raise_governance_validation_error("target_ref", "user-like identifier")
    _raise_governance_validation_error("target_ref", "must be minimized or internal")
    raise AssertionError("unreachable")


def _validate_governance_detail_entry(
    field_name: str,
    key: str,
    value: Any,
    *,
    allowed_keys: frozenset[str],
) -> object:
    if key in _DISALLOWED_DETAIL_KEYS:
        _raise_governance_validation_error(field_name, key)
    if key not in allowed_keys:
        _raise_governance_validation_error(field_name, key)
    if key in {"count", "deleted_count"}:
        return _validate_governance_nonnegative_int(field_name, value)
    if key in {"agent_playbook_id", "user_playbook_id"}:
        _validate_governance_int(field_name, value)
        return cast(int, value)
    if key == "deleted_counts":
        return _validate_governance_deleted_counts(field_name, value)
    if key in {
        "affected_agent_playbook_ids",
        "erased_source_ids",
        "owned_user_playbook_ids",
        "rebuilt_agent_playbook_ids",
        "source_interaction_ids",
    }:
        return _validate_governance_int_list(field_name, value)
    if key in {"original_source_windows", "remaining_source_windows"}:
        return _validate_governance_window_list(field_name, value)
    if key == "previous_lifecycle_status":
        if value is None:
            return None
        return _validate_governance_detail_enum(
            field_name,
            value,
            allowed_values=_ALLOWED_PREVIOUS_LIFECYCLE_STATUS_VALUES,
        )
    if key == "prepared":
        if not isinstance(value, bool):
            _raise_governance_validation_error(field_name, "expected bool")
        return value
    if key == "route":
        return _validate_governance_detail_enum(
            field_name,
            value,
            allowed_values=_ALLOWED_DETAIL_ROUTE_VALUES,
        )
    if key == "status":
        return _validate_governance_detail_enum(
            field_name,
            value,
            allowed_values=_ALLOWED_DETAIL_STATUS_VALUES,
        )
    _raise_governance_validation_error(field_name, key)


def _validate_governance_detail(
    field_name: str,
    detail: dict[str, object] | None,
    *,
    allowed_keys: frozenset[str],
) -> dict[str, object] | None:
    if detail is None:
        return None
    if not isinstance(detail, dict):
        _raise_governance_validation_error(field_name, "expected dict")
    normalized_detail: dict[str, object] = {}
    for key, value in detail.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in normalized_detail:
            _raise_governance_validation_error(
                field_name, f"duplicate key {normalized_key}"
            )
        normalized_detail[normalized_key] = _validate_governance_detail_entry(
            f"{field_name}.{normalized_key}",
            normalized_key,
            value,
            allowed_keys=allowed_keys,
        )
    return normalized_detail


def _validate_governance_code_like(field_name: str, value: str) -> str:
    if not value:
        _raise_governance_validation_error(field_name, "required")
    _validate_governance_string(field_name, value)
    if value.startswith(("subref_v1_", "reqref_v1_", "actref_v1_")):
        _raise_governance_validation_error(field_name, "identifier")
    if _IDENTIFIERISH_ERROR_CODE_RE.fullmatch(value):
        _raise_governance_validation_error(field_name, "identifier")
    if not _SAFE_ERROR_CODE_RE.fullmatch(value):
        _raise_governance_validation_error(
            field_name, "must be a stable diagnostic code"
        )
    return value


def _validate_governance_error_detail(error_detail: str | None) -> str | None:
    if error_detail is None:
        return None
    return _validate_governance_code_like("error_detail", error_detail)


def _validate_governance_error_code(error_code: str) -> str:
    return _validate_governance_code_like("error_code", error_code)


def _validate_governance_enum(
    field_name: str, value: str, *, allowed: frozenset[str]
) -> str:
    if value not in allowed:
        _raise_governance_validation_error(
            field_name,
            f"must be one of {', '.join(sorted(allowed))}",
        )
    return value


def _normalize_governance_detail_for_identity(
    detail: dict[str, object] | None,
) -> str | None:
    if detail is None:
        return None
    normalized_detail: dict[str, object] = {}
    for key, value in detail.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in {"original_source_windows", "remaining_source_windows"}:
            normalized_windows = [
                _normalize_governance_window_item(normalized_key, index, item)
                for index, item in enumerate(cast(list[object], value))
            ]
            normalized_detail[normalized_key] = normalized_windows
            continue
        normalized_detail[normalized_key] = value
    return json.dumps(normalized_detail, sort_keys=True, separators=(",", ":"))


def _validate_audit_event_for_persistence(event: AuditEvent) -> None:
    _validate_governance_enum(
        "actor_type",
        event.actor_type,
        allowed=_ALLOWED_AUDIT_ACTOR_TYPES,
    )
    _validate_governance_enum(
        "operation",
        event.operation,
        allowed=_ALLOWED_AUDIT_OPERATIONS,
    )
    _validate_governance_enum(
        "entity_type",
        event.entity_type,
        allowed=_ALLOWED_AUDIT_ENTITY_TYPES,
    )
    _validate_governance_enum(
        "status",
        event.status,
        allowed=_ALLOWED_AUDIT_STATUSES,
    )
    _validate_governance_prefixed_ref("actor_ref", event.actor_ref, prefix="actref_v1_")
    _validate_governance_prefixed_ref(
        "subject_ref", event.subject_ref, prefix="subref_v1_"
    )
    if event.request_ref is None:
        _raise_governance_validation_error("request_ref", "required")
    _validate_governance_prefixed_ref(
        "request_ref", event.request_ref, prefix="reqref_v1_"
    )
    if event.entity_id is not None:
        _validate_governance_code_shaped(
            "entity_id",
            event.entity_id,
            allow_minimized_ref=True,
        )
    _validate_governance_idempotency_key("idempotency_key", event.idempotency_key)
    _validate_governance_detail(
        "audit_event.detail",
        event.detail,
        allowed_keys=_ALLOWED_AUDIT_DETAIL_KEYS,
    )


def _canonicalize_audit_event_for_persistence(event: AuditEvent) -> AuditEvent:
    _validate_audit_event_for_persistence(event)
    return event.model_copy(
        update={
            "detail": _validate_governance_detail(
                "audit_event.detail",
                event.detail,
                allowed_keys=_ALLOWED_AUDIT_DETAIL_KEYS,
            )
        }
    )


def _upgrade_legacy_purge_operation_targets_table(conn: sqlite3.Connection) -> None:
    target_columns = [
        row[1] for row in conn.execute("PRAGMA table_info(purge_operation_targets)")
    ]
    if not target_columns or "org_id" in target_columns:
        return
    conn.execute(
        "ALTER TABLE purge_operation_targets RENAME TO purge_operation_targets_legacy"
    )
    conn.executescript(_PURGE_OPERATION_TARGETS_TABLE_DDL)
    conn.execute(
        """INSERT INTO purge_operation_targets (
               org_id, purge_id, target_name, target_ref, phase, status, detail,
               deleted_count, error_detail, started_at, completed_at
           )
           SELECT uniquely_mapped_purges.org_id, legacy.purge_id, legacy.target_name,
                  legacy.target_ref, legacy.phase, legacy.status, legacy.detail,
                  legacy.deleted_count, legacy.error_detail, legacy.started_at,
                  legacy.completed_at
           FROM purge_operation_targets_legacy AS legacy
           JOIN (
               SELECT MIN(org_id) AS org_id, purge_id
               FROM purge_operations
               GROUP BY purge_id
               HAVING COUNT(*) = 1
           ) AS uniquely_mapped_purges
             ON uniquely_mapped_purges.purge_id = legacy.purge_id"""
    )
    conn.execute("DROP TABLE purge_operation_targets_legacy")


def _enforce_audit_request_ref_not_null(conn: sqlite3.Connection) -> None:
    audit_columns = [row[1] for row in conn.execute("PRAGMA table_info(audit_events)")]
    if not audit_columns:
        return
    conn.execute(
        "UPDATE audit_events SET request_ref = ? WHERE request_ref IS NULL",
        (_LEGACY_AUDIT_REQUEST_REF,),
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS audit_events_request_ref_not_null
        BEFORE INSERT ON audit_events
        WHEN NEW.request_ref IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'audit_events.request_ref is required');
        END
        """
    )


def _is_successful_erase_event(
    event: AuditEvent, *, purge_id: str | None = None
) -> bool:
    if event.operation != "ERASE" or event.status != "ok":
        return False
    if purge_id is not None:
        return event.idempotency_key == purge_id
    return True


def _successful_erase_identity(
    event: AuditEvent,
) -> tuple[
    str,
    str,
    str | None,
    str,
    str,
    str | None,
    str | None,
    str | None,
    str,
    str | None,
    str | None,
]:
    return (
        event.org_id,
        event.actor_type,
        event.actor_ref,
        event.operation,
        event.entity_type,
        event.entity_id,
        event.subject_ref,
        event.request_ref,
        event.status,
        event.idempotency_key,
        _normalize_governance_detail_for_identity(
            cast(dict[str, object] | None, event.detail)
        ),
    )


def _row_to_audit_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        org_id=row["org_id"],
        actor_type=row["actor_type"],
        actor_ref=row["actor_ref"],
        operation=row["operation"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        subject_ref=row["subject_ref"],
        request_ref=row["request_ref"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        detail=_json_loads(row["detail"]),
        created_at=row["created_at"],
    )


def _row_to_purge_operation(row: sqlite3.Row) -> PurgeOperation:
    return PurgeOperation(
        purge_id=row["purge_id"],
        org_id=row["org_id"],
        operation_type=row["operation_type"],
        scope_type=row["scope_type"],
        subject_ref=row["subject_ref"],
        request_ref=row["request_ref"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        error_code=row["error_code"],
        error_detail=row["error_detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _row_to_purge_target(row: sqlite3.Row) -> PurgeOperationTarget:
    return PurgeOperationTarget(
        purge_id=row["purge_id"],
        target_name=row["target_name"],
        target_ref=row["target_ref"],
        phase=row["phase"],
        status=row["status"],
        detail=_json_loads(row["detail"]),
        deleted_count=row["deleted_count"],
        error_detail=row["error_detail"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


class _SQLiteGovernanceDeps(Protocol):
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _has_sqlite_vec: bool

    def _fetchall(
        self, sql: str, params: list[Any] | tuple[Any, ...]
    ) -> list[sqlite3.Row]: ...

    def _fetchone(
        self, sql: str, params: list[Any] | tuple[Any, ...]
    ) -> sqlite3.Row | None: ...

    def _partition_purge_vs_delete(
        self, entity_type: Literal["profile", "user_playbook"], ids: list[str]
    ) -> tuple[list[str], list[str]]: ...

    def _delete_in_chunks(
        self, table_name: str, column_name: str, values: list[Any]
    ) -> None: ...

    def _delete_source_windows_for_user_playbook_ids(
        self, user_playbook_ids: list[int]
    ) -> None: ...

    def _get_embedding(self, text: str) -> list[float]: ...

    def set_source_windows_for_agent_playbook(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None: ...

    def get_source_windows_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[AgentPlaybookSourceWindow]: ...

    def get_agent_playbook_by_id(
        self,
        agent_playbook_id: int,
        *,
        include_tombstones: bool = False,
    ) -> AgentPlaybook | None: ...

    def _index_agent_playbook_fts_vec(self, ap: AgentPlaybook) -> None: ...


class SQLiteGovernanceMixin:
    """SQLite governance storage primitives."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _deps(self) -> _SQLiteGovernanceDeps:
        return cast(_SQLiteGovernanceDeps, self)

    def _replace_agent_playbook_source_windows_locked(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )
        source_window_rows = _build_agent_playbook_source_window_rows(
            agent_playbook_id, windows
        )
        if source_window_rows:
            self.conn.executemany(
                """INSERT OR IGNORE INTO agent_playbook_source_user_playbooks
                   (agent_playbook_id, user_playbook_id, source_interaction_ids)
                   VALUES (?, ?, ?)""",
                source_window_rows,
            )

    def _delete_agent_playbook_search_rows_locked(self, agent_playbook_id: int) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbooks_fts WHERE rowid = ?",
            (agent_playbook_id,),
        )
        if self._deps()._has_sqlite_vec:
            self.conn.execute(
                "DELETE FROM agent_playbooks_vec WHERE rowid = ?",
                (agent_playbook_id,),
            )

    def _upsert_agent_playbook_search_rows_locked(
        self,
        *,
        agent_playbook_id: int,
        trigger: str | None,
        content: str,
        expanded_terms: str | None,
        embedding: list[float],
    ) -> None:
        self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
        fts_parts = [trigger or "", content]
        if expanded_terms:
            fts_parts.append(expanded_terms)
        self.conn.execute(
            "INSERT INTO agent_playbooks_fts(rowid, search_text) VALUES (?, ?)",
            (
                agent_playbook_id,
                " ".join(part for part in fts_parts if part) or "",
            ),
        )
        if self._deps()._has_sqlite_vec and embedding:
            self.conn.execute(
                "INSERT INTO agent_playbooks_vec(rowid, embedding) VALUES (?, ?)",
                (agent_playbook_id, json.dumps(embedding)),
            )

    def _validate_prepared_delete_target_matrix_locked(self, purge_id: str) -> None:
        snapshot = self.conn.execute(
            """SELECT 1 FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        ).fetchone()
        if snapshot is None:
            raise ValueError("Cannot delete user data without target snapshot marker")
        delete_rows = self.conn.execute(
            """SELECT target_name, status FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                 AND target_ref = 'all'""",
            (self.org_id, purge_id),
        ).fetchall()
        delete_statuses = {
            str(row["target_name"]): str(row["status"]) for row in delete_rows
        }
        missing_delete_targets = [
            target_name
            for target_name in _CANONICAL_DELETE_TARGET_NAMES
            if delete_statuses.get(target_name) not in {"pending", "complete"}
        ]
        if missing_delete_targets:
            raise ValueError(
                "Cannot delete user data without complete delete target matrix: "
                + ", ".join(missing_delete_targets)
            )

    def _validate_hide_for_rebuild_targets_locked(self, purge_id: str) -> None:
        rebuild_rows = self.conn.execute(
            """SELECT DISTINCT target_ref
               FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                 AND phase = 'rebuild_without_erased_sources' AND target_ref != ''
               ORDER BY target_ref ASC""",
            (self.org_id, purge_id),
        ).fetchall()
        if not rebuild_rows:
            return
        hidden_refs = {
            str(row["target_ref"])
            for row in self.conn.execute(
                """SELECT target_ref
                   FROM purge_operation_targets
                   WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                     AND phase = 'hide_for_rebuild' AND status = 'complete'""",
                (self.org_id, purge_id),
            ).fetchall()
        }
        missing_hidden_refs = [
            str(row["target_ref"])
            for row in rebuild_rows
            if str(row["target_ref"]) not in hidden_refs
        ]
        if missing_hidden_refs:
            raise ValueError(
                "Cannot delete user data before hide_for_rebuild completes for "
                f"planned agent_playbooks: {', '.join(missing_hidden_refs)}"
            )

    def _planned_governance_delete_counts(
        self, user_id: str, owned_user_playbook_ids: set[int]
    ) -> dict[str, int]:
        request_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM requests WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        interaction_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM interactions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        eval_result_row = self.conn.execute(
            """SELECT COUNT(*) AS cnt
               FROM agent_success_evaluation_result
               WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        profile_rows = self.conn.execute(
            "SELECT profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        if request_row is None or interaction_row is None or eval_result_row is None:
            raise ValueError("Missing governance count rows")
        profile_ids = [str(row["profile_id"]) for row in profile_rows]
        purge_profile_ids, delete_profile_ids = self._deps()._partition_purge_vs_delete(
            "profile",
            profile_ids,
        )
        playbook_ids = [
            str(user_playbook_id)
            for user_playbook_id in sorted(owned_user_playbook_ids)
        ]
        purge_playbook_ids, delete_playbook_ids = (
            self._deps()._partition_purge_vs_delete(
                "user_playbook",
                playbook_ids,
            )
        )
        return {
            "request": int(request_row["cnt"]),
            "interaction": int(interaction_row["cnt"]),
            "profile": len(delete_profile_ids),
            "profile_purge": len(purge_profile_ids),
            "user_playbook": len(delete_playbook_ids),
            "agent_success_evaluation_result": int(eval_result_row["cnt"]),
            "user_playbook_purge": len(purge_playbook_ids),
        }

    def _owned_user_playbook_ids_locked(self, user_id: str) -> set[int]:
        return {
            int(row["user_playbook_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        }

    def _prepared_owned_user_playbook_ids_locked(self, purge_id: str) -> set[int]:
        row = self.conn.execute(
            """SELECT detail FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ?
                 AND target_ref = 'all' AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        ).fetchone()
        if row is None:
            raise ValueError("Prepared target snapshot is missing")
        detail = _json_loads(row["detail"])
        if not isinstance(detail, dict):
            raise ValueError("Prepared target snapshot detail is missing")
        return set(
            _validate_governance_int_list(
                "owned_user_playbook_ids",
                detail.get("owned_user_playbook_ids"),
            )
        )

    def _append_audit_event_with_cursor(
        self, cur: sqlite3.Connection | sqlite3.Cursor, event: AuditEvent
    ) -> bool:
        inserted = cur.execute(
            """INSERT OR IGNORE INTO audit_events (
                   org_id, actor_type, actor_ref, operation, entity_type, entity_id,
                   subject_ref, request_ref, idempotency_key, status, detail, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.org_id,
                event.actor_type,
                event.actor_ref,
                event.operation,
                event.entity_type,
                event.entity_id,
                event.subject_ref,
                event.request_ref,
                event.idempotency_key,
                event.status,
                _json_dumps(event.detail),
                event.created_at,
            ),
        )
        return inserted.rowcount > 0

    def _record_purge_target_locked(
        self,
        *,
        purge_id: str,
        target_name: str,
        target_ref: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        detail: dict[str, object] | None,
        deleted_count: int,
        error_detail: str | None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        _validate_governance_enum(
            "target_name",
            target_name,
            allowed=_ALLOWED_PURGE_TARGET_NAMES,
        )
        _validate_governance_enum(
            "phase",
            phase,
            allowed=_ALLOWED_PURGE_TARGET_PHASES,
        )
        _validate_governance_enum(
            "status",
            status,
            allowed=_ALLOWED_PURGE_TARGET_STATUSES,
        )
        detail = _validate_governance_detail(
            "detail",
            detail,
            allowed_keys=_ALLOWED_PURGE_TARGET_DETAIL_KEYS,
        )
        error_detail = _validate_governance_error_detail(error_detail)
        target_ref = _validate_governance_target_ref(
            target_name=target_name,
            phase=phase,
            target_ref=target_ref,
        )
        deleted_count = _validate_governance_deleted_count(deleted_count)
        now = _epoch_now()
        existing = self.conn.execute(
            """SELECT started_at, completed_at
               FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = ? AND phase = ?""",
            (self.org_id, purge_id, target_name, target_ref, phase),
        ).fetchone()
        started_at = existing["started_at"] if existing else None
        completed_at = existing["completed_at"] if existing else None
        if started_at is None and status in {"running", "failed", "complete"}:
            started_at = now
        if status in {"failed", "complete"}:
            completed_at = now
        self.conn.execute(
            """INSERT INTO purge_operation_targets (
                   org_id, purge_id, target_name, target_ref, phase, status, detail,
                   deleted_count, error_detail, started_at, completed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(org_id, purge_id, target_name, target_ref, phase) DO UPDATE SET
                   status = excluded.status,
                   detail = COALESCE(excluded.detail, purge_operation_targets.detail),
                   deleted_count = excluded.deleted_count,
                   error_detail = excluded.error_detail,
                   started_at = COALESCE(purge_operation_targets.started_at, excluded.started_at),
                   completed_at = excluded.completed_at""",
            (
                self.org_id,
                purge_id,
                target_name,
                target_ref,
                phase,
                status,
                _json_dumps(detail),
                deleted_count,
                error_detail,
                started_at,
                completed_at,
            ),
        )
        self.conn.execute(
            """UPDATE purge_operations
               SET status = CASE
                   WHEN status IN ('complete', 'failed') THEN status
                   WHEN ? IN ('running', 'complete') THEN 'running'
                   ELSE status
               END,
                   updated_at = ?
               WHERE purge_id = ? AND org_id = ?""",
            (status, now, purge_id, self.org_id),
        )

    def append_audit_event(self, event: AuditEvent) -> bool:
        if _is_successful_erase_event(event):
            raise ValueError(
                "Successful ERASE audit rows may only be written by "
                "complete_purge_operation_with_audit()"
            )
        if event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        event = _canonicalize_audit_event_for_persistence(event)
        with self._lock:
            inserted = self._append_audit_event_with_cursor(self.conn, event)
            self.conn.commit()
            return inserted

    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        deps = self._deps()
        if org_id is not None and org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        sql = "SELECT * FROM audit_events WHERE org_id = ?"
        params: list[Any] = [self.org_id]
        if subject_ref is not None:
            sql += " AND subject_ref = ?"
            params.append(subject_ref)
        sql += " ORDER BY created_at ASC, event_id ASC"
        rows = deps._fetchall(sql, params)
        return [_row_to_audit_event(row) for row in rows]

    def _purge_governance_entity_content_locked(
        self,
        *,
        entity_type: Literal["profile", "user_playbook"],
        entity_id: str,
        rowid: int,
    ) -> bool:
        from ._lineage import _PURGE_SQL, _append_event_stmt

        sql = _PURGE_SQL[entity_type]
        cur = self.conn.execute(sql, (entity_id,))
        if cur.rowcount <= 0:
            return False
        _append_event_stmt(
            self.conn,
            org_id=self.org_id,
            entity_type=entity_type,
            entity_id=entity_id,
            op="purge",
            prov="wasPurged",
            source_ids=[],
            actor="erasure",
            request_id=f"purge_{entity_id}",
            reason="content_purge",
        )
        if entity_type == "profile":
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?",
                (entity_id,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM profiles_vec WHERE rowid = ?",
                    (rowid,),
                )
        else:
            self.conn.execute(
                "DELETE FROM user_playbooks_fts WHERE rowid = ?",
                (rowid,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM user_playbooks_vec WHERE rowid = ?",
                    (rowid,),
                )
        return True

    def _clear_user_data_for_governance_locked(
        self,
        user_id: str,
        *,
        expected_user_playbook_ids: set[int] | None = None,
    ) -> dict[str, int]:
        deps = self._deps()
        interaction_ids = [
            int(row["interaction_id"])
            for row in self.conn.execute(
                "SELECT interaction_id FROM interactions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        raw_upb_ids = [
            int(row["user_playbook_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        if (
            expected_user_playbook_ids is not None
            and set(raw_upb_ids) != expected_user_playbook_ids
        ):
            raise ValueError(
                "Current user playbooks no longer match prepared purge snapshot"
            )
        request_ids = [
            str(row["request_id"])
            for row in self.conn.execute(
                "SELECT request_id FROM requests WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        profile_rows = self.conn.execute(
            "SELECT rowid, profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        profile_rowid_by_id = {
            str(row["profile_id"]): int(row["rowid"]) for row in profile_rows
        }
        all_profile_ids = list(profile_rowid_by_id)

        purge_profile_ids, delete_profile_ids = deps._partition_purge_vs_delete(
            "profile",
            all_profile_ids,
        )
        purge_upb_str_ids, delete_upb_str_ids = deps._partition_purge_vs_delete(
            "user_playbook",
            [str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        )
        purge_upb_ids = [int(entity_id) for entity_id in purge_upb_str_ids]
        delete_upb_ids = [int(entity_id) for entity_id in delete_upb_str_ids]
        erased_entity_ids = [
            *request_ids,
            *[str(interaction_id) for interaction_id in interaction_ids],
            *all_profile_ids,
            *[str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        ]
        if erased_entity_ids:
            erased_entity_id_set = set(erased_entity_ids)
            lineage_source_event_ids: list[int] = []
            for row in self.conn.execute(
                "SELECT event_id, source_ids FROM lineage_event WHERE org_id = ?",
                (self.org_id,),
            ).fetchall():
                try:
                    source_ids = json.loads(str(row["source_ids"] or "[]"))
                except json.JSONDecodeError:
                    source_ids = []
                if any(
                    str(source_id) in erased_entity_id_set for source_id in source_ids
                ):
                    lineage_source_event_ids.append(int(row["event_id"]))
            deps._delete_in_chunks(
                "lineage_event", "event_id", lineage_source_event_ids
            )
            placeholders = ",".join("?" for _ in erased_entity_ids)
            self.conn.execute(
                f"""DELETE FROM lineage_event
                    WHERE org_id = ?
                      AND (
                        request_id IN ({placeholders})
                        OR entity_id IN ({placeholders})
                      )""",  # noqa: S608
                [self.org_id, *erased_entity_ids, *erased_entity_ids],
            )
        delete_profile_rowids = [
            profile_rowid_by_id[profile_id]
            for profile_id in delete_profile_ids
            if profile_id in profile_rowid_by_id
        ]

        deps._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
        deps._delete_in_chunks("user_playbooks_fts", "rowid", delete_upb_ids)
        deps._delete_in_chunks("profiles_fts", "profile_id", delete_profile_ids)
        if deps._has_sqlite_vec:
            deps._delete_in_chunks("interactions_vec", "rowid", interaction_ids)
            deps._delete_in_chunks("user_playbooks_vec", "rowid", delete_upb_ids)
            deps._delete_in_chunks("profiles_vec", "rowid", delete_profile_rowids)

        interactions_cur = self.conn.execute(
            "DELETE FROM interactions WHERE user_id = ?",
            (user_id,),
        )
        eval_results_cur = self.conn.execute(
            """DELETE FROM agent_success_evaluation_result
               WHERE user_id = ?""",
            (user_id,),
        )
        requests_cur = self.conn.execute(
            "DELETE FROM requests WHERE user_id = ?",
            (user_id,),
        )
        if delete_upb_ids:
            deps._delete_source_windows_for_user_playbook_ids(delete_upb_ids)
            deps._delete_in_chunks("user_playbooks", "user_playbook_id", delete_upb_ids)
        if delete_profile_ids:
            deps._delete_in_chunks("profiles", "profile_id", delete_profile_ids)

        purged_profiles = 0
        for profile_id in purge_profile_ids:
            rowid = profile_rowid_by_id.get(profile_id)
            if rowid is None:
                continue
            purged_profiles += int(
                self._purge_governance_entity_content_locked(
                    entity_type="profile",
                    entity_id=profile_id,
                    rowid=rowid,
                )
            )

        purged_user_playbooks = 0
        for user_playbook_id in purge_upb_ids:
            purged_user_playbooks += int(
                self._purge_governance_entity_content_locked(
                    entity_type="user_playbook",
                    entity_id=str(user_playbook_id),
                    rowid=user_playbook_id,
                )
            )

        return {
            "interactions": interactions_cur.rowcount,
            "user_playbooks": len(delete_upb_ids),
            "profiles": len(delete_profile_ids),
            "requests": requests_cur.rowcount,
            "agent_success_evaluation_results": eval_results_cur.rowcount,
            "purged_profiles": purged_profiles,
            "purged_user_playbooks": purged_user_playbooks,
        }

    def begin_purge_operation(
        self,
        purge_id: str,
        idempotency_key: str,
        operation_type: Literal["user_erasure", "org_purge"],
        scope_type: Literal["user", "org"],
        subject_ref: str | None,
        request_ref: str,
    ) -> PurgeOperation:
        _validate_governance_enum(
            "operation_type",
            operation_type,
            allowed=_ALLOWED_PURGE_OPERATION_TYPES,
        )
        _validate_governance_enum(
            "scope_type",
            scope_type,
            allowed=_ALLOWED_PURGE_SCOPE_TYPES,
        )
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        _validate_governance_prefixed_ref(
            "request_ref", request_ref, prefix="reqref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_idempotency_key = cast(
            str,
            _validate_governance_idempotency_key("idempotency_key", idempotency_key),
        )
        now = _epoch_now()
        with self._lock:
            existing = self.conn.execute(
                """SELECT * FROM purge_operations
                   WHERE org_id = ? AND idempotency_key = ?""",
                (self.org_id, validated_idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_operation = _row_to_purge_operation(existing)
                expected_identity = {
                    "purge_id": validated_purge_id,
                    "operation_type": operation_type,
                    "scope_type": scope_type,
                    "subject_ref": subject_ref,
                    "request_ref": request_ref,
                }
                for field_name, expected_value in expected_identity.items():
                    if getattr(existing_operation, field_name) != expected_value:
                        raise ValueError(
                            "Existing purge operation for idempotency_key has "
                            f"mismatched {field_name}"
                        )
                return _row_to_purge_operation(existing)
            self.conn.execute(
                """INSERT INTO purge_operations (
                       purge_id, org_id, operation_type, scope_type, subject_ref,
                       request_ref, idempotency_key, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    validated_purge_id,
                    self.org_id,
                    operation_type,
                    scope_type,
                    subject_ref,
                    request_ref,
                    validated_idempotency_key,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_purge_operation(validated_purge_id)

    def record_purge_target(
        self,
        purge_id: str,
        target_name: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        target_ref: str = "",
        detail: dict[str, object] | None = None,
        deleted_count: int = 0,
        error_detail: str | None = None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        _validate_governance_enum(
            "target_name",
            target_name,
            allowed=_ALLOWED_PURGE_TARGET_NAMES,
        )
        _validate_governance_enum(
            "phase",
            phase,
            allowed=_ALLOWED_PURGE_TARGET_PHASES,
        )
        _validate_governance_enum(
            "status",
            status,
            allowed=_ALLOWED_PURGE_TARGET_STATUSES,
        )
        with self._lock:
            self._record_purge_target_locked(
                purge_id=purge_id,
                target_name=target_name,
                target_ref=target_ref,
                phase=phase,
                status=status,
                detail=detail,
                deleted_count=deleted_count,
                error_detail=error_detail,
            )
            self.conn.commit()

    def list_purge_targets(
        self, purge_id: str, phase: str | None = None
    ) -> list[PurgeOperationTarget]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        deps = self._deps()
        sql = "SELECT * FROM purge_operation_targets WHERE org_id = ? AND purge_id = ?"
        params: list[Any] = [self.org_id, purge_id]
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY phase ASC, target_name ASC, target_ref ASC"
        rows = deps._fetchall(sql, params)
        return [_row_to_purge_target(row) for row in rows]

    def purge_targets_prepared(self, purge_id: str) -> bool:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        row = self._deps()._fetchone(
            """SELECT 1 FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        )
        return row is not None

    def prepare_governance_erase_targets(
        self,
        purge_id: str,
        user_id: str,
        owned_user_playbook_ids: set[int] | None = None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        with self._lock:
            if self.purge_targets_prepared(purge_id):
                return
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                owned_user_playbook_ids = (
                    set(owned_user_playbook_ids)
                    if owned_user_playbook_ids is not None
                    else self._owned_user_playbook_ids_locked(user_id)
                )
                targets = self._planned_governance_delete_counts(
                    user_id,
                    owned_user_playbook_ids,
                )
                affected_agent_playbook_ids: list[int] = []
                rebuild_details_by_agent_playbook_id: dict[int, dict[str, object]] = {}
                if owned_user_playbook_ids:
                    placeholders = ",".join("?" for _ in owned_user_playbook_ids)
                    rows = self.conn.execute(
                        f"""SELECT DISTINCT apsup.agent_playbook_id
                            FROM agent_playbook_source_user_playbooks
                            AS apsup
                            JOIN agent_playbooks ap
                              ON ap.agent_playbook_id = apsup.agent_playbook_id
                            WHERE user_playbook_id IN ({placeholders})
                            ORDER BY apsup.agent_playbook_id ASC""",
                        sorted(owned_user_playbook_ids),
                    ).fetchall()
                    affected_agent_playbook_ids = [
                        int(row["agent_playbook_id"]) for row in rows
                    ]
                    for agent_playbook_id in affected_agent_playbook_ids:
                        status_row = self.conn.execute(
                            """SELECT status
                               FROM agent_playbooks
                               WHERE agent_playbook_id = ?""",
                            (agent_playbook_id,),
                        ).fetchone()
                        if status_row is None:
                            raise ValueError(
                                f"Agent playbook with ID {agent_playbook_id} not found"
                            )
                        window_rows = self.conn.execute(
                            """SELECT user_playbook_id, source_interaction_ids
                               FROM agent_playbook_source_user_playbooks
                               WHERE agent_playbook_id = ?
                               ORDER BY user_playbook_id ASC""",
                            (agent_playbook_id,),
                        ).fetchall()
                        original_window_dicts = [
                            {
                                "user_playbook_id": int(row["user_playbook_id"]),
                                "source_interaction_ids": [
                                    int(source_id)
                                    for source_id in (
                                        _json_loads(row["source_interaction_ids"]) or []
                                    )
                                ],
                            }
                            for row in window_rows
                        ]
                        remaining_window_dicts = [
                            window
                            for window in original_window_dicts
                            if int(window["user_playbook_id"])
                            not in owned_user_playbook_ids
                        ]
                        rebuild_details_by_agent_playbook_id[agent_playbook_id] = {
                            "original_source_windows": original_window_dicts,
                            "previous_lifecycle_status": status_row["status"],
                            "remaining_source_windows": remaining_window_dicts,
                        }
                for target_name, count in targets.items():
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name=target_name,
                        target_ref="all",
                        phase="delete",
                        status="pending",
                        detail={"count": count},
                        deleted_count=0,
                        error_detail=None,
                    )
                for agent_playbook_id in affected_agent_playbook_ids:
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="rebuild_without_erased_sources",
                        status="pending",
                        detail=rebuild_details_by_agent_playbook_id[agent_playbook_id],
                        deleted_count=0,
                        error_detail=None,
                    )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name=_SNAPSHOT_TARGET_NAME,
                    target_ref="all",
                    phase=_PREPARE_PHASE,
                    status="complete",
                    detail={
                        "owned_user_playbook_ids": sorted(owned_user_playbook_ids),
                        "affected_agent_playbook_ids": affected_agent_playbook_ids,
                    },
                    deleted_count=0,
                    error_detail=None,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def hide_governance_agent_playbooks_for_rebuild(self, purge_id: str) -> list[int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                target_rows = self.conn.execute(
                    """SELECT target_ref
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ?
                         AND target_name = 'agent_playbook'
                         AND phase = 'rebuild_without_erased_sources'
                         AND target_ref != ''
                         AND status != 'complete'
                       ORDER BY CAST(target_ref AS INTEGER) ASC""",
                    (self.org_id, purge_id),
                ).fetchall()
                agent_playbook_ids = [int(row["target_ref"]) for row in target_rows]
                if not agent_playbook_ids:
                    self.conn.commit()
                    return []
                placeholders = ",".join("?" for _ in agent_playbook_ids)
                self.conn.execute(
                    f"""UPDATE agent_playbooks
                        SET status = ?
                        WHERE agent_playbook_id IN ({placeholders})""",
                    [Status.ARCHIVE_IN_PROGRESS.value, *agent_playbook_ids],
                )
                for agent_playbook_id in agent_playbook_ids:
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="hide_for_rebuild",
                        status="complete",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="rebuild_without_erased_sources",
                        status="running",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return agent_playbook_ids

    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        name_map = {
            "interactions": "interaction",
            "user_playbooks": "user_playbook",
            "profiles": "profile",
            "requests": "request",
            "agent_success_evaluation_results": "agent_success_evaluation_result",
            "purged_profiles": "profile_purge",
            "purged_user_playbooks": "user_playbook_purge",
        }
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self._validate_prepared_delete_target_matrix_locked(purge_id)
                self._validate_hide_for_rebuild_targets_locked(purge_id)
                expected_user_playbook_ids = (
                    self._prepared_owned_user_playbook_ids_locked(purge_id)
                )
                counts = self._clear_user_data_for_governance_locked(
                    user_id,
                    expected_user_playbook_ids=expected_user_playbook_ids,
                )
                for key, value in counts.items():
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name=name_map.get(key, key),
                        target_ref="all",
                        phase="delete",
                        status="complete",
                        detail={"count": int(value)},
                        deleted_count=int(value),
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return counts

    def apply_governance_agent_playbook_rebuild(
        self,
        purge_id: str,
        agent_playbook_id: int,
        remaining_source_windows: list[dict[str, object]],
        content: str | None,
        trigger: str | None,
        rationale: str | None,
        blocking_issue: dict[str, object] | None,
        expanded_terms: str | None,
        tags: list[str] | None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        windows = _parse_governance_window_list(
            "remaining_source_windows", remaining_source_windows
        )
        canonical_remaining_windows = [window.model_dump() for window in windows]
        content_value = content or ""
        trigger_value = trigger or None
        embedding_text = trigger_value or content_value
        embedding = (
            self._deps()._get_embedding(embedding_text) if embedding_text else []
        )
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                rebuild_target_row = self.conn.execute(
                    """SELECT status, detail
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'rebuild_without_erased_sources'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if rebuild_target_row is None:
                    raise ValueError("planned rebuild target does not exist")
                if rebuild_target_row["status"] == "complete":
                    raise ValueError("planned rebuild target is already complete")
                rebuild_detail = _json_loads(rebuild_target_row["detail"])
                if not isinstance(rebuild_detail, dict) or not {
                    "original_source_windows",
                    "previous_lifecycle_status",
                    "remaining_source_windows",
                }.issubset(rebuild_detail):
                    raise ValueError(
                        "planned rebuild target is missing source window detail"
                    )
                planned_remaining_windows = _canonicalize_governance_windows(
                    "planned remaining_source_windows",
                    cast(
                        list[dict[str, object]],
                        rebuild_detail["remaining_source_windows"],
                    ),
                )
                if planned_remaining_windows != canonical_remaining_windows:
                    raise ValueError(
                        "remaining_source_windows must match the planned rebuild target"
                    )
                previous_lifecycle_status = cast(
                    str | None, rebuild_detail["previous_lifecycle_status"]
                )
                hide_target_row = self.conn.execute(
                    """SELECT status
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'hide_for_rebuild'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if hide_target_row is None or hide_target_row["status"] != "complete":
                    raise ValueError("hide_for_rebuild target must be complete")
                if windows:
                    cur = self.conn.execute(
                        """UPDATE agent_playbooks
                           SET content = ?, trigger = ?, rationale = ?, blocking_issue = ?,
                               embedding = ?, expanded_terms = ?, tags = ?, status = ?
                           WHERE agent_playbook_id = ?""",
                        (
                            content_value,
                            trigger_value,
                            rationale,
                            json.dumps(blocking_issue)
                            if blocking_issue is not None
                            else None,
                            _json_dumps(embedding),
                            expanded_terms,
                            _json_dumps(tags),
                            previous_lifecycle_status,
                            agent_playbook_id,
                        ),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    self._replace_agent_playbook_source_windows_locked(
                        agent_playbook_id, windows
                    )
                    self._upsert_agent_playbook_search_rows_locked(
                        agent_playbook_id=agent_playbook_id,
                        trigger=trigger_value,
                        content=content_value,
                        expanded_terms=expanded_terms,
                        embedding=embedding,
                    )
                else:
                    from ._playbook import _emit_hard_delete_playbook

                    self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
                    self.conn.execute(
                        "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    cur = self.conn.execute(
                        "DELETE FROM agent_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    _emit_hard_delete_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(agent_playbook_id),
                        request_id=purge_id,
                    )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name="agent_playbook",
                    target_ref=str(agent_playbook_id),
                    phase="rebuild_without_erased_sources",
                    status="complete",
                    detail=None,
                    deleted_count=0,
                    error_detail=None,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        if audit_event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        if audit_event.idempotency_key != purge_id:
            raise ValueError("Audit event idempotency key must match purge_id")
        if not _is_successful_erase_event(audit_event, purge_id=purge_id):
            raise ValueError(
                "Completion requires a successful ERASE audit event for this purge"
            )
        audit_event = _canonicalize_audit_event_for_persistence(audit_event)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                row = self.conn.execute(
                    "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (purge_id, self.org_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                purge_operation = _row_to_purge_operation(row)
                if purge_operation.subject_ref != audit_event.subject_ref:
                    raise ValueError(
                        "Audit event subject_ref must match purge operation subject_ref"
                    )
                if purge_operation.request_ref != audit_event.request_ref:
                    raise ValueError(
                        "Audit event request_ref must match purge operation request_ref"
                    )
                snapshot = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        "Cannot complete purge without target snapshot marker"
                    )
                delete_rows = self.conn.execute(
                    """SELECT target_name, status FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                         AND target_ref = 'all'""",
                    (self.org_id, purge_id),
                ).fetchall()
                delete_statuses = {
                    str(row["target_name"]): str(row["status"]) for row in delete_rows
                }
                missing_delete_targets = [
                    target_name
                    for target_name in _CANONICAL_DELETE_TARGET_NAMES
                    if delete_statuses.get(target_name) != "complete"
                ]
                if missing_delete_targets:
                    raise ValueError(
                        "Cannot complete purge without complete delete target matrix: "
                        + ", ".join(missing_delete_targets)
                    )
                incomplete = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND status != 'complete'
                       LIMIT 1""",
                    (self.org_id, purge_id),
                ).fetchone()
                if incomplete is not None:
                    raise ValueError("Cannot complete purge with incomplete targets")
                existing_audit_row = self.conn.execute(
                    """SELECT * FROM audit_events
                       WHERE org_id = ? AND idempotency_key = ?""",
                    (self.org_id, purge_id),
                ).fetchone()
                if existing_audit_row is not None:
                    existing_event = _row_to_audit_event(existing_audit_row)
                    if not _is_successful_erase_event(
                        existing_event, purge_id=purge_id
                    ):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                    if _successful_erase_identity(
                        existing_event
                    ) != _successful_erase_identity(audit_event):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                else:
                    self._append_audit_event_with_cursor(self.conn, audit_event)
                    existing_audit_row = self.conn.execute(
                        """SELECT * FROM audit_events
                           WHERE org_id = ? AND idempotency_key = ?""",
                        (self.org_id, purge_id),
                    ).fetchone()
                if existing_audit_row is None:
                    raise ValueError(
                        "Completion requires exactly one successful ERASE audit row "
                        "for the purge_id"
                    )
                existing_event = _row_to_audit_event(existing_audit_row)
                if not _is_successful_erase_event(existing_event, purge_id=purge_id):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                if _successful_erase_identity(
                    existing_event
                ) != _successful_erase_identity(audit_event):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = ?
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def fail_purge_operation(
        self, purge_id: str, error_code: str, error_detail: str
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_error_code = _validate_governance_error_code(error_code)
        validated_error_detail = _validate_governance_error_detail(error_detail)
        now = _epoch_now()
        with self._lock:
            cur = self.conn.execute(
                """UPDATE purge_operations
                   SET status = 'failed', error_code = ?, error_detail = ?,
                   updated_at = ?, completed_at = ?
                   WHERE purge_id = ? AND org_id = ?""",
                (
                    validated_error_code,
                    validated_error_detail,
                    now,
                    now,
                    purge_id,
                    self.org_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Purge operation {purge_id!r} not found")
            self.conn.commit()
        return self.get_purge_operation(purge_id)

    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        row = self._deps()._fetchone(
            "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
            (purge_id, self.org_id),
        )
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        return _row_to_purge_operation(row)

    def gc_governance_retention(self, *, config: GovernanceRetentionConfig) -> int:
        if not config.audit_events_retention_enabled:
            return 0
        cutoff_epoch = _epoch_now() - config.audit_events_retention_days * 24 * 60 * 60
        with self._lock:
            cur = self.conn.execute(
                """DELETE FROM audit_events
                   WHERE event_id IN (
                       SELECT event_id
                       FROM audit_events
                       WHERE org_id = ? AND created_at < ?
                       ORDER BY created_at ASC, event_id ASC
                       LIMIT ?
                   )""",
                (
                    self.org_id,
                    cutoff_epoch,
                    config.audit_events_delete_batch_limit,
                ),
            )
            deleted = int(cur.rowcount or 0)
            self.conn.commit()
        return deleted
