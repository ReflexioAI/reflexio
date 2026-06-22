"""
Feature flags module for gating features per organization.

Reads feature flag configuration from site_var and provides helpers
to check whether a given feature is enabled for an organization.
"""

import logging

from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)


def _get_feature_flags_config() -> dict:
    """
    Load the feature_flags site var configuration.

    Returns:
        dict: The full feature flags config, or empty dict if not found.
    """
    config = SiteVarManager().get_site_var("feature_flags")
    if config is None or not isinstance(config, dict):
        logger.warning(
            "feature_flags site var not found or invalid, defaulting to empty config"
        )
        return {}
    return config


def is_feature_enabled(org_id: str, feature_name: str) -> bool:
    """
    Check if a feature is enabled for a given organization.

    A feature is enabled if:
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    If the feature is not found in config, it defaults to enabled (fail-open).

    Args:
        org_id (str): The organization ID to check
        feature_name (str): The feature flag name (e.g. "deduplicator")

    Returns:
        bool: True if the feature is enabled for this org
    """
    config = _get_feature_flags_config()
    feature_config = config.get(feature_name)

    if feature_config is None:
        # Unknown feature — default to enabled (fail-open)
        return True

    if feature_config.get("enabled", False):
        return True

    enabled_org_ids = feature_config.get("enabled_org_ids", []) or []
    return org_id in enabled_org_ids


def get_all_feature_flags(org_id: str) -> dict[str, bool]:
    """
    Get the resolved enabled/disabled state of all feature flags for an organization.

    Args:
        org_id (str): The organization ID to check

    Returns:
        dict[str, bool]: Mapping of feature name to enabled status
    """
    config = _get_feature_flags_config()
    result: dict[str, bool] = {}
    for feature_name in config:
        result[feature_name] = is_feature_enabled(org_id, feature_name)
    return result


def is_invitation_only_enabled() -> bool:
    """
    Check if invitation-only registration mode is enabled globally.

    Returns:
        bool: True if invitation-only mode is enabled
    """
    config = _get_feature_flags_config()
    invitation_config = config.get("invitation_only")
    if invitation_config is None:
        return False
    return invitation_config.get("enabled", False)


def is_deduplicator_enabled(org_id: str) -> bool:
    """
    Convenience check for whether the deduplicator is enabled for an org.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True if deduplicator is enabled
    """
    return is_feature_enabled(org_id, "deduplicator")


def _is_fail_closed_flag_enabled(org_id: str, feature_key: str) -> bool:
    """
    Shared fail-closed helper: returns False if key absent or config malformed.

    Unlike is_feature_enabled (fail-open), this function returns False when the
    feature key is absent or when its value is not a dict. This is the safety
    invariant for soft-delete flags — unconfigured or malformed entries must
    never accidentally activate tombstone-producing paths.

    A feature is enabled if:
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    Args:
        org_id (str): The organization ID to check
        feature_key (str): The feature flag key in the config dict

    Returns:
        bool: True only if the feature is explicitly enabled for this org
    """
    config = _get_feature_flags_config()
    feature_config = config.get(feature_key)

    if feature_config is None:
        # Key absent — fail-CLOSED
        return False

    if not isinstance(feature_config, dict):
        logger.warning(
            "feature_flags[%s] is not a dict (got %s), defaulting to OFF",
            feature_key,
            type(feature_config).__name__,
        )
        return False

    # Strict bool identity — truthy strings like "false" must not enable (#195).
    enabled = feature_config.get("enabled", False)
    if enabled is True:
        return True

    # Reject non-list values — a string does substring `in` match, not membership (#195).
    org_ids = feature_config.get("enabled_org_ids", [])
    if not isinstance(org_ids, list):
        return False
    return org_id in org_ids


def _is_default_open_flag_enabled(org_id: str, feature_key: str) -> bool:
    """
    Shared default-open helper for soft-delete flags.

    Returns True when the feature key is absent from config (default ON), but
    preserves all explicit-disable and per-org override semantics:
    - Key absent or None → True (default ON — GC is also ON by default).
    - Key present but malformed (not a dict) → False with a warning (safe fallback).
    - Key present, enabled=True → True for all orgs.
    - Key present, enabled=False, org in enabled_org_ids → True.
    - Key present, enabled=False, org NOT in enabled_org_ids → False (explicit disable).

    Strict-bool and strict-list guards from _is_fail_closed_flag_enabled are
    preserved: truthy strings and non-bool ints do NOT enable, and a string
    enabled_org_ids does NOT match via substring (anti-#195).

    Args:
        org_id (str): The organization ID to check
        feature_key (str): The feature flag key in the config dict

    Returns:
        bool: True when the flag is on (including when absent/unconfigured)
    """
    config = _get_feature_flags_config()
    feature_config = config.get(feature_key)

    if feature_config is None:
        # Key absent — default OPEN (soft-delete is on by default; GC runs too).
        return True

    if not isinstance(feature_config, dict):
        logger.warning(
            "feature_flags[%s] is not a dict (got %s), defaulting to OFF",
            feature_key,
            type(feature_config).__name__,
        )
        return False

    # Strict bool identity — truthy strings like "false" must not enable (#195).
    enabled = feature_config.get("enabled", False)
    if enabled is True:
        return True

    # Reject non-list values — a string does substring `in` match, not membership (#195).
    org_ids = feature_config.get("enabled_org_ids", [])
    if not isinstance(org_ids, list):
        return False
    return org_id in org_ids


def is_dedup_soft_delete_enabled(org_id: str) -> bool:
    """
    Check if deduplication soft-delete is enabled for a given organization.

    Defaults to ENABLED when the key is absent from config (default-open).
    GC is also enabled by default (LineageGCConfig.enabled=True), so tombstones
    created by this path are reclaimed automatically.

    Explicit disable: set ``dedup_soft_delete: {enabled: false, enabled_org_ids: []}``
    in the feature_flags site var to disable globally, or omit an org from
    ``enabled_org_ids`` while setting ``enabled: false`` to disable per-org.

    A feature is enabled if:
    - The feature key is absent from config (default ON), OR
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True unless the feature is explicitly disabled for this org
    """
    return _is_default_open_flag_enabled(org_id, "dedup_soft_delete")


def is_aggregation_soft_delete_enabled(org_id: str) -> bool:
    """
    Check if aggregation soft-delete is enabled for a given organization.

    Defaults to ENABLED when the key is absent from config (default-open).
    GC is also enabled by default (LineageGCConfig.enabled=True), so SUPERSEDED
    tombstones created by this path are reclaimed automatically after the 90-day
    grace window.

    Explicit disable: set ``aggregation_soft_delete: {enabled: false, enabled_org_ids: []}``
    in the feature_flags site var to disable globally, or omit an org from
    ``enabled_org_ids`` while setting ``enabled: false`` to disable per-org.

    A feature is enabled if:
    - The feature key is absent from config (default ON), OR
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True unless the feature is explicitly disabled for this org
    """
    return _is_default_open_flag_enabled(org_id, "aggregation_soft_delete")


def is_resumable_extraction_agent_enabled(org_id: str) -> bool:
    """
    Convenience check for whether classic extraction should use the resumable agent.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True if the resumable extraction agent is enabled
    """
    return is_feature_enabled(org_id, "resumable_extraction_agent")
