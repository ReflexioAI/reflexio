"""Explicit historical configuration conversion for the canonical runtime.

Current schemas own canonical validation. Historical rewrite rules live here so
explicit upgrades remain usable after runtime compatibility hooks are removed.
"""

from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError

from reflexio.config_env import (
    ConfigurationMigrationError as ConfigurationMigrationError,
)
from reflexio.config_env import StoredConfigurationError as StoredConfigurationError

_LEGACY_OPTIMIZER_BOOL = TypeAdapter(bool)
_LEGACY_MERGE_LIMIT = TypeAdapter(Annotated[int, Field(ge=0)])
_LEGACY_RETRIEVAL_POOL = TypeAdapter(Annotated[int, Field(gt=0)])
_LEGACY_AUDIT_RETENTION_DAYS = TypeAdapter(Annotated[int, Field(gt=0)])
_LEGACY_STOP_SCORE = TypeAdapter(
    Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
)


def strip_retired_optimizer_stop_score(data: Any) -> Any:
    """Validate and retire historical policy only during explicit conversion."""
    if not isinstance(data, dict) or "early_stop_score" not in data:
        return data
    try:
        _LEGACY_STOP_SCORE.validate_python(data["early_stop_score"])
        _LEGACY_OPTIMIZER_BOOL.validate_python(data.get("enabled", False))
    except ValidationError:
        raise ConfigurationMigrationError(
            "Invalid obsolete optimizer early_stop_score or enabled setting."
        ) from None
    return {key: value for key, value in data.items() if key != "early_stop_score"}


class OptimizerSchedulingMigrationError(ConfigurationMigrationError):
    """Invalid saved scheduling policy must not trigger a storage-default fallback."""


def canonicalize_retrieval_floor_policy(data: Any) -> Any:
    """Preserve the old effective gate and capacity during explicit conversion."""
    if not isinstance(data, dict) or not {"enabled", "pool_size"}.intersection(data):
        return data
    try:
        enabled = _LEGACY_OPTIMIZER_BOOL.validate_python(data.get("enabled", False))
        pool_size = _LEGACY_RETRIEVAL_POOL.validate_python(data.get("pool_size", 30))
    except ValidationError:
        raise ConfigurationMigrationError(
            "Invalid saved retrieval floor enabled or pool_size setting"
        ) from None
    normalized = {
        key: value for key, value in data.items() if key not in {"enabled", "pool_size"}
    }
    normalized.setdefault("candidate_pool_size", pool_size if enabled else 0)
    return normalized


def canonicalize_audit_retention_policy(data: Any) -> Any:
    """Preserve historical deletion consent, including disabled custom ages."""
    old_fields = {"audit_events_retention_enabled", "audit_events_retention_days"}
    if not isinstance(data, dict) or not old_fields.intersection(data):
        return data
    if "audit_events_max_age_days" in data:
        raise ConfigurationMigrationError(
            "Conflicting current and historical audit retention fields; "
            "explicitly select one policy before conversion"
        )
    try:
        enabled = _LEGACY_OPTIMIZER_BOOL.validate_python(
            data.get("audit_events_retention_enabled", False)
        )
        # Historical default only; current runtime defaults to indefinite retention.
        days = _LEGACY_AUDIT_RETENTION_DAYS.validate_python(
            data.get("audit_events_retention_days", 365)
        )
    except ValidationError:
        raise ConfigurationMigrationError(
            "Invalid saved audit retention enabled or days setting"
        ) from None
    return {
        **{key: value for key, value in data.items() if key not in old_fields},
        "audit_events_max_age_days": days if enabled else 0,
    }


def canonicalize_agent_optimizer_policy(data: Any) -> Any:
    """Convert only the formerly canonical agent gate pair during an upgrade.

    Other unsupported historical fields are deliberately left for strict
    validation; local startup must not accidentally admit old storage formats.
    """
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    legacy_agent_keys = (
        "optimize_agent_playbooks",
        "auto_update_pending_agent_playbooks",
    )
    if any(key in data for key in legacy_agent_keys):
        # The old adoption default was true, unlike the old user target.
        # Preserve the complete effective opt-in, including an explicit veto.
        try:
            optimize = _LEGACY_OPTIMIZER_BOOL.validate_python(
                data.get(legacy_agent_keys[0], False)
            )
            adopt = _LEGACY_OPTIMIZER_BOOL.validate_python(
                data.get(legacy_agent_keys[1], True)
            )
        except ValidationError:
            raise ConfigurationMigrationError(
                "Invalid saved agent optimizer switches; correct optimize_agent_playbooks "
                "and auto_update_pending_agent_playbooks before retrying"
            ) from None
        normalized.setdefault("agent_playbooks_enabled", optimize and adopt)
        for key in legacy_agent_keys:
            normalized.pop(key, None)
    return normalized


def _normalize_optimizer_policy(data: Any) -> Any:
    """Canonicalize retired controls without enabling previously disabled work."""
    if not isinstance(data, dict):
        return data
    # These are current canonical policy validation/defaults, not legacy decoders.
    from reflexio.models.config_schema import (
        DEFAULT_MAX_MERGE_INVOCATIONS,
        OptimizerSchedulingPolicy,
    )

    normalized = dict(data)
    scheduling = {
        key: data[key] for key in OptimizerSchedulingPolicy.model_fields if key in data
    }
    if scheduling:
        try:
            OptimizerSchedulingPolicy.model_validate(scheduling)
        except ValidationError as error:
            fields = ", ".join(sorted({str(item["loc"][0]) for item in error.errors()}))
            raise OptimizerSchedulingMigrationError(
                f"Invalid optimizer scheduling setting: {fields}"
            ) from None
    legacy_user_keys = ("optimize_user_playbooks", "auto_update_user_playbooks")
    if any(key in data for key in legacy_user_keys):
        # Validate both: even an inactive policy must not hide malformed input.
        optimize = _LEGACY_OPTIMIZER_BOOL.validate_python(
            data.get(legacy_user_keys[0], False)
        )
        adopt = _LEGACY_OPTIMIZER_BOOL.validate_python(
            data.get(legacy_user_keys[1], False)
        )
        normalized.setdefault("user_playbooks_enabled", optimize and adopt)
        for key in legacy_user_keys:
            normalized.pop(key, None)
    normalized = canonicalize_agent_optimizer_policy(normalized)
    if "use_merge" in data:
        enabled = _LEGACY_OPTIMIZER_BOOL.validate_python(data["use_merge"])
        limit = _LEGACY_MERGE_LIMIT.validate_python(
            data.get("max_merge_invocations", DEFAULT_MAX_MERGE_INVOCATIONS)
        )
        normalized.pop("use_merge")
        normalized["max_merge_invocations"] = limit if enabled else 0
    return strip_retired_optimizer_stop_score(normalized)


class ExtractionPreset(StrEnum):
    """Named extraction presets that bundle window_size and stride_size.

    Each preset targets a specific conversation pattern:
    - quick_chat: Short conversations (support bots, quick Q&A)
    - standard: General-purpose conversational agents (default)
    - long_form: Long conversations (coding assistants, research)
    - high_volume: High-traffic agents (1000+ daily interactions)
    """

    QUICK_CHAT = "quick_chat"
    STANDARD = "standard"
    LONG_FORM = "long_form"
    HIGH_VOLUME = "high_volume"


# Historical preset decoding only; these do not select current execution defaults.
_PRESET_VALUES: dict[ExtractionPreset, tuple[int, int]] = {
    ExtractionPreset.QUICK_CHAT: (5, 3),
    ExtractionPreset.STANDARD: (10, 8),
    ExtractionPreset.LONG_FORM: (25, 10),
    ExtractionPreset.HIGH_VOLUME: (15, 8),
}
_LEGACY_EXTRACTION_PRESET = TypeAdapter(ExtractionPreset | None)


# ---------------------------------------------------------------------------
# Field migration maps (old stored JSON name → new Python attr name)
# ---------------------------------------------------------------------------
_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "batch_size": "window_size",
    "batch_interval": "stride_size",
    "extraction_window_size": "window_size",
    "extraction_window_stride": "stride_size",
}

_AGGREGATOR_FIELD_MIGRATION: dict[str, str] = {
    "min_feedback_threshold": "min_cluster_size",
    "refresh_count": "reaggregation_trigger_count",
    "similarity_threshold": "clustering_similarity",
}

_EXTRACTOR_OVERRIDE_MIGRATION: dict[str, str] = {
    "batch_size_override": "window_size_override",
    "batch_interval_override": "stride_size_override",
    "extraction_window_size_override": "window_size_override",
    "extraction_window_stride_override": "stride_size_override",
}

_PROFILE_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "profile_content_definition_prompt": "extraction_definition_prompt",
}

_PLAYBOOK_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "feedback_definition_prompt": "extraction_definition_prompt",
    "playbook_definition_prompt": "extraction_definition_prompt",
    "feedback_aggregator_config": "aggregation_config",
    "playbook_aggregator_config": "aggregation_config",
    "playbook_name": "extractor_name",
    "feedback_name": "extractor_name",
}


def _migrate_dict(data: Any, mapping: dict[str, str]) -> Any:
    """Rename old field names to new ones in a raw dict before Pydantic validates.

    Creates a shallow copy to avoid mutating the caller's dict.
    """
    if isinstance(data, dict):
        data = dict(data)
        for old, new in mapping.items():
            if old not in data:
                continue
            # New name wins when both are present; always drop the old key so it
            # doesn't survive into validation and trip ``extra="forbid"``.
            if new in data:
                data.pop(old)
            else:
                data[new] = data.pop(old)
    return data


# Retired list-valued config fields and the singular field that replaced them.
# The first configured entry wins when an old list contains multiple items.
_LEGACY_SINGLE_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("profile_extractor_configs", "profile_extractor_config"),
    ("user_playbook_extractor_configs", "user_playbook_extractor_config"),
    ("playbook_configs", "user_playbook_extractor_config"),
    ("agent_feedback_configs", "user_playbook_extractor_config"),
    ("agent_success_configs", "agent_success_config"),
)


def _first_config_entry(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_legacy_config_shape(data: dict[str, Any]) -> dict[str, Any]:
    """Map retired list-valued config fields onto current singular fields.

    This is a stored-data upgrade path applied at storage load boundaries: any
    config persisted before the single-extractor refactor still carries list
    keys (e.g. ``agent_success_configs``) that ``Config`` would otherwise drop
    as unknown fields, silently losing the user's customization. Legacy keys are
    removed from the returned payload and the first configured entry wins.

    Returns a shallow copy; the caller's dict is not mutated.
    """
    normalized = dict(data)
    for legacy_field, current_field in _LEGACY_SINGLE_CONFIG_FIELDS:
        if legacy_field not in normalized:
            continue
        if current_field not in normalized:
            normalized[current_field] = _first_config_entry(normalized[legacy_field])
        del normalized[legacy_field]
    return normalized


def strip_retired_config_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize retired settings without losing custom policies or opaque metadata.

    Copy changed dictionaries so ordinary saves and raw tuner-reset upgrades
    can share this cleanup without mutating a caller's saved configuration.
    """
    # These policy modules import migration errors during their own initialization.
    from reflexio.housekeeping_policy import strip_legacy_housekeeping_fields
    from reflexio.pending_tool_policy import (
        strip_legacy_pending_tool_timing,
        strip_retired_pending_followup_warning,
    )

    payload = strip_legacy_housekeeping_fields(data)
    if "shadow_comparison_judge_prompt_version" in payload:
        # Bootstrap-only inference images do not package model schemas.
        from reflexio.models.api_schema.validators import NonEmptyStr

        try:
            TypeAdapter(NonEmptyStr).validate_python(
                payload["shadow_comparison_judge_prompt_version"]
            )
        except ValidationError:
            raise ConfigurationMigrationError(
                "Invalid obsolete shadow-comparison rubric label."
            ) from None
        # This label never selected a template. Current provenance comes from
        # the actual template; do not rewrite historical verdicts or metadata.
        payload = {
            key: value
            for key, value in payload.items()
            if key != "shadow_comparison_judge_prompt_version"
        }
    if "extraction_preset" in payload:
        # A preset was an input shortcut, never independent execution policy.
        # Resolve it before raw upgrades can preserve it as opaque metadata.
        payload = _migrate_dict(payload, _CONFIG_FIELD_MIGRATION)
        preset = _LEGACY_EXTRACTION_PRESET.validate_python(
            payload.pop("extraction_preset")
        )
        if preset is not None:
            for name, value in zip(
                ("window_size", "stride_size"), _PRESET_VALUES[preset], strict=True
            ):
                if payload.get(name) is None:
                    payload[name] = value
    api_keys = payload.get("api_key_config")
    openai = api_keys.get("openai") if isinstance(api_keys, dict) else None
    azure = openai.get("azure_config") if isinstance(openai, dict) else None
    if (
        isinstance(api_keys, dict)
        and isinstance(openai, dict)
        and isinstance(azure, dict)
        and "deployment_name" in azure
    ):
        # Azure deployments are selected by the model ID. This field never
        # affected dispatch; raw upgrades must not preserve it as opaque metadata.
        payload["api_key_config"] = {
            **api_keys,
            "openai": {
                **openai,
                "azure_config": {
                    key: value
                    for key, value in azure.items()
                    if key != "deployment_name"
                },
            },
        }
    if "pending_tool_call_config" in payload:
        payload["pending_tool_call_config"] = strip_retired_pending_followup_warning(
            strip_legacy_pending_tool_timing(payload["pending_tool_call_config"])
        )
    if "playbook_optimizer_config" in payload:
        payload["playbook_optimizer_config"] = _normalize_optimizer_policy(
            payload["playbook_optimizer_config"]
        )
    if "retrieval_floor" in payload:
        payload["retrieval_floor"] = canonicalize_retrieval_floor_policy(
            payload["retrieval_floor"]
        )
    if "governance_retention" in payload:
        payload["governance_retention"] = canonicalize_audit_retention_policy(
            payload["governance_retention"]
        )
    payload.pop("shadow_mode_enabled", None)
    names = {
        "profile_extractor_config": {"extractor_name"},
        "user_playbook_extractor_config": {
            "extractor_name",
            "playbook_name",
            "feedback_name",
        },
        "agent_success_config": {
            "evaluation_name",
            "window_size_override",
            "stride_size_override",
            *_EXTRACTOR_OVERRIDE_MIGRATION,
        },
    }
    for config_field, current in (
        *((name, name) for name in names),
        *_LEGACY_SINGLE_CONFIG_FIELDS,
    ):
        value = payload.get(config_field)
        retired = names[current]
        if isinstance(value, dict):
            payload[config_field] = {k: v for k, v in value.items() if k not in retired}
        elif isinstance(value, list):
            payload[config_field] = [
                {k: v for k, v in item.items() if k not in retired}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        if current == "user_playbook_extractor_config":
            cleaned = payload.get(config_field)
            for item in cleaned if isinstance(cleaned, list) else [cleaned]:
                if isinstance(item, dict) and isinstance(
                    policy := item.get("deduplication_config"), dict
                ):
                    item["deduplication_config"] = {
                        key: value
                        for key, value in policy.items()
                        if key != "max_unified_content_chars"
                    }
    return payload


def _normalize_storage_backend(data: Any) -> Any:
    """Disambiguate historical native references before union validation.

    Explicit tags and model instances retain authority. Old HTTP-bearing
    references select Supabase even if malformed, so required-DSN validation
    cannot turn a broken tenant reference into another backend.
    """
    if not isinstance(data, dict) or any(
        key in data for key in ("type", "storage_type", "managed_by")
    ):
        return data
    if any(key in data for key in ("url", "key", "read_url", "read_key")):
        return {**data, "type": "supabase"}
    if "db_url" in data:
        return {**data, "type": "postgres"}
    return data


def canonicalize_stored_config_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite legacy policy names without serializing away opaque metadata.

    This is a raw migration, not validation or default materialization. Callers
    must still use ``validate_stored_config`` before persisting or executing it.
    Storage identity/credentials retain their separate strict validation path.
    """
    payload = strip_retired_config_fields(normalize_legacy_config_shape(data))
    payload = _migrate_dict(payload, _CONFIG_FIELD_MIGRATION)
    for config_field, mapping in (
        ("profile_extractor_config", _PROFILE_CONFIG_FIELD_MIGRATION),
        ("user_playbook_extractor_config", _PLAYBOOK_CONFIG_FIELD_MIGRATION),
    ):
        value = payload.get(config_field)
        if not isinstance(value, dict):
            continue
        value = _migrate_dict(
            _migrate_dict(value, mapping), _EXTRACTOR_OVERRIDE_MIGRATION
        )
        if config_field == "user_playbook_extractor_config" and isinstance(
            aggregator := value.get("aggregation_config"), dict
        ):
            value["aggregation_config"] = _migrate_dict(
                aggregator, _AGGREGATOR_FIELD_MIGRATION
            )
        payload[config_field] = value
    return payload


def canonicalize_config_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert saved JSON names while preserving effective policy and extensions.

    This explicit migration interface does not connect to storage or select a
    new backend. Persist the validated backend tag before dropping historical
    HTTP selectors, whose presence otherwise influences untagged storage.
    """
    from reflexio.models.config_schema import validate_stored_config

    try:
        converted = canonicalize_stored_config_fields(payload)
        raw_storage = converted.get("storage_config")
        if isinstance(raw_storage, dict):
            storage = dict(raw_storage)
            for old, canonical in (("storage_type", "type"), ("schema_name", "schema")):
                if old in storage:
                    # The current storage models forbid both spellings together.
                    # Do not turn a previously invalid document into a valid one.
                    if canonical in storage:
                        raise StoredConfigurationError("Conflicting storage fields")
                    storage[canonical] = storage.pop(old)
            # Capture historical backend identity before removing HTTP selectors.
            # This raw migration is independent of runtime alias acceptance.
            storage = _normalize_storage_backend(storage)
            storage.setdefault(
                "type", "supabase" if "managed_by" in storage else "sqlite"
            )
            if storage["type"] == "supabase":
                for key in ("url", "key", "read_url", "read_key"):
                    storage.pop(key, None)
            converted["storage_config"] = storage
        # Runtime validation sees only the converted document. In particular,
        # explicit upgrades must keep working after runtime aliases are removed.
        validate_stored_config(converted)
        return converted
    except ConfigurationMigrationError:
        # These policy errors already contain only known field names and actionable
        # upgrade guidance; retain their fail-closed contract for explicit tools.
        raise
    except ValueError:
        # Legacy validators can include values in ValidationError text. This
        # migration interface must never print configuration contents to a CLI.
        raise StoredConfigurationError(
            "Configuration cannot be converted safely; correct invalid settings before retrying."
        ) from None
