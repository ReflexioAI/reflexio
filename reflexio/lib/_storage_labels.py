"""Shared helpers for describing + masking storage configurations.

Used by the publish diagnostics, the whoami endpoint, and the config
show/pull commands so every surface labels storage the same way and
masks credentials with the same rules.
"""

from __future__ import annotations

from reflexio.models.config_schema import (
    StorageConfig,
    StorageConfigDisk,
    StorageConfigSQLite,
)


def describe_storage(
    storage_config: StorageConfig | None,
) -> tuple[str | None, str | None]:
    """Map a ``StorageConfig`` to a ``(storage_type, masked_label)`` pair.

    The type is a short slug ("sqlite", "disk", "supabase", "local_dir")
    suitable for grouping; the label is a short, human-readable string
    with any secret material masked so it's safe to print anywhere.

    Enterprise-only storage types (Supabase, S3) are matched by class name
    so we don't take a hard import dependency on ``reflexio_ext`` from the
    open-source package.

    Args:
        storage_config: The storage configuration to describe.

    Returns:
        tuple[str | None, str | None]: ``(storage_type, storage_label)``,
        or ``(None, None)`` when no storage is configured.
    """
    if storage_config is None:
        return None, None

    cls_name = type(storage_config).__name__

    if isinstance(storage_config, StorageConfigSQLite):
        return "sqlite", storage_config.db_path or "<default sqlite>"
    if isinstance(storage_config, StorageConfigDisk):
        return "disk", storage_config.dir_path or "<default dir>"

    if cls_name == "StorageConfigSupabase":
        url = getattr(storage_config, "url", None)
        return "supabase", mask_url(url) if url else "<supabase>"
    if cls_name == "StorageConfigLocal":
        return "local_dir", getattr(storage_config, "dir_path", None) or "<local>"

    return cls_name.removeprefix("StorageConfig").lower() or None, cls_name


def mask_url(value: str) -> str:
    """Mask the middle of a URL/host while keeping scheme + domain tail.

    Examples:
        ``https://jpkjckbyxrdefzomiyse.supabase.co`` →
        ``https://jpkj...supabase.co``

        ``postgresql://user:pass@host.supabase.com:6543/postgres`` →
        ``postgresql://***@host.supabase.com:6543/postgres``
    """
    if not value:
        return ""

    # Postgres / user-info URLs: hide the credentials segment entirely.
    if "@" in value and "://" in value:
        scheme, rest = value.split("://", 1)
        _, host_part = rest.split("@", 1)
        return f"{scheme}://***@{host_part}"

    # HTTP(S) URLs: keep scheme, show first 4 chars of host, then tail domain.
    if "://" in value:
        scheme, rest = value.split("://", 1)
        host = rest.split("/", 1)[0]
        if "." not in host:
            return f"{scheme}://{_short_head(host)}..."
        head, tail = host.split(".", 1)
        return f"{scheme}://{_short_head(head)}...{tail}"

    # Plain strings (keys, paths): keep the first 4 chars.
    return _short_head(value) + "..."


def mask_secret(value: str) -> str:
    """Mask a secret string, keeping only the first 4 and last 2 characters.

    Returns ``"<empty>"`` for empty inputs so the output is always printable.
    """
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-2:]}"


def _short_head(value: str) -> str:
    """Return the first 4 characters of ``value`` (or the full string if shorter)."""
    return value[:4] if len(value) > 4 else value
