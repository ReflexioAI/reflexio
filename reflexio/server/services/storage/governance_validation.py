"""Pure governance validation and canonicalization helpers.

These are stateless functions and constants (no sqlite3/storage-class dependency)
that validate and canonicalize governance schema objects.  Both the SQLite and
Supabase storage backends import from this module so that neither needs to reach
into a private sibling package.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, NoReturn, cast, get_args

from reflexio.models.api_schema.domain import AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.governance import (
    AuditActorType,
    AuditEntityType,
    AuditEvent,
    AuditOperation,
    AuditStatus,
    PurgeOperationType,
    PurgeScopeType,
    PurgeTargetStatus,
)

_PREPARE_PHASE = "prepare_targets"
_SNAPSHOT_TARGET_NAME = "target_snapshot"
_CANONICAL_DELETE_TARGET_NAMES = (
    "session_outcome",
    "request",
    "interaction",
    "profile",
    "user_playbook",
    "agent_success_evaluation_result",
    "retrieved_learning_evaluation_result",
    "evaluation_operation_state",
    "offline_tuner_reward_label",
    "offline_tuner_reward_label_target_by_target_owner",
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
        "session_outcome",
        "interaction",
        "profile",
        "user_playbook",
        "agent_success_evaluation_result",
        "retrieved_learning_evaluation_result",
        "evaluation_operation_state",
        "offline_tuner_reward_label",
        "offline_tuner_reward_label_target_by_target_owner",
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
        "retrieved_learning_evaluation_results",
        "evaluation_operation_states",
        "offline_tuner_reward_labels",
        "offline_tuner_reward_label_targets_by_target_owner",
        "purged_profiles",
        "purged_user_playbooks",
    }
)


def _epoch_now() -> int:
    return int(datetime.now(UTC).timestamp())


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
    if target_name in _CANONICAL_DELETE_TARGET_NAMES:
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
