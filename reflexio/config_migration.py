"""Stored-configuration errors and explicit canonical document conversion."""

from typing import Any


class StoredConfigurationError(ValueError):
    """The saved configuration must be corrected before it can load."""


class ConfigurationMigrationError(StoredConfigurationError):
    """An explicit migration is required before this saved policy can load."""


def canonicalize_config_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert saved JSON names while preserving effective policy and extensions.

    This explicit migration interface does not connect to storage or select a
    new backend. Persist the validated backend tag before dropping historical
    HTTP selectors, whose presence otherwise influences untagged storage.
    """
    from reflexio.models.config_schema import (
        canonicalize_stored_config_fields,
        validate_stored_config,
    )

    try:
        before = validate_stored_config(payload)
        converted = canonicalize_stored_config_fields(payload)
        raw_storage = converted.get("storage_config")
        if isinstance(raw_storage, dict) and before.storage_config is not None:
            storage = dict(raw_storage)
            storage["type"] = before.storage_config.storage_type
            storage.pop("storage_type", None)
            if "schema_name" in storage:
                storage["schema"] = storage.pop("schema_name")
            if storage["type"] == "supabase":
                for key in ("url", "key", "read_url", "read_key"):
                    storage.pop(key, None)
            converted["storage_config"] = storage
        after = validate_stored_config(converted)
        if before.model_dump_json() != after.model_dump_json():
            raise StoredConfigurationError("Conversion changed effective policy")
        return converted
    except ValueError:
        # Legacy validators can include values in ValidationError text. This
        # migration interface must never print configuration contents to a CLI.
        raise StoredConfigurationError(
            "Configuration cannot be converted safely; correct invalid settings before retrying."
        ) from None
