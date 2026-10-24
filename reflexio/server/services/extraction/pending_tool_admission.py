"""One org-owned admission decision for fresh and resumed pending-info tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.models.config_schema import PendingToolCallConfig
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.services.storage.storage_base import BaseStorage


def _reject_retired_rollout() -> None:
    """Verify an old rollout file is neutral; never resolve its cohort policy.

    Non-neutral files require explicit adoption into each org's existing opt-in.
    Missing, unreadable and malformed files must not be conflated: the manager
    returns None for both absence and read errors, so check presence first.
    """
    manager = SiteVarManager()
    source = Path(manager.source_dir)
    if not any(
        (source / name).exists() for name in ("feature_flags.json", "feature_flags.txt")
    ):
        return
    legacy = manager.get_site_var("feature_flags")
    if isinstance(legacy, dict):
        rollout = legacy.get("resumable_extraction_agent")
        if rollout is None or (
            isinstance(rollout, dict) and rollout.get("enabled") is True
        ):
            return
    raise ValueError(
        "Retired resumable_extraction_agent rollout policy requires explicit "
        "adoption into pending_tool_call_config.enabled. Preserve existing "
        "org opt-outs and disable previously denied orgs before removing the "
        "retired policy from feature_flags.json/txt. No values were copied."
    )


def _disabled_reason(
    pending: PendingToolCallConfig | None, storage: BaseStorage | None
) -> str | None:
    if pending is None:
        return "no pending_tool_call_config on the resolved config"
    if not pending.enabled:
        return "pending_tool_call_config.enabled is false"
    _reject_retired_rollout()
    if storage is None:
        return "the request context has no storage"
    return None


def pending_tool_calls_disabled_reason(context: RequestContext) -> str | None:
    """Explain org/storage refusal without changing the permanent extraction loop."""
    config = context.configurator.get_config()
    pending = getattr(config, "pending_tool_call_config", None) if config else None
    return _disabled_reason(pending, context.storage)


def pending_tool_calls_enabled(context: RequestContext) -> bool:
    return pending_tool_calls_disabled_reason(context) is None


def require_pending_tool_admission(
    pending: PendingToolCallConfig, storage: BaseStorage | None
) -> None:
    """Enforce the same decision before a direct tool can read/write or dispatch."""
    reason = _disabled_reason(pending, storage)
    if reason is not None:
        raise ValueError(reason)
