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
        _normalize_storage_backend,
        canonicalize_stored_config_fields,
        validate_stored_config,
    )

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
    except ValueError:
        # Legacy validators can include values in ValidationError text. This
        # migration interface must never print configuration contents to a CLI.
        raise StoredConfigurationError(
            "Configuration cannot be converted safely; correct invalid settings before retrying."
        ) from None
