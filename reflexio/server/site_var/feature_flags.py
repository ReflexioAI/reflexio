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

    if feature_config.get("enabled", False):
        return True

    enabled_org_ids = feature_config.get("enabled_org_ids", []) or []
    return org_id in enabled_org_ids


def is_dedup_soft_delete_enabled(org_id: str) -> bool:
    """
    Check if deduplication soft-delete is enabled for a given organization.

    This is a FAIL-CLOSED flag: if the key is absent from config or the value
    is not a dict, it returns False. This is the opposite of is_feature_enabled
    (which is fail-open). The difference is intentional — soft-delete must never
    activate for unconfigured orgs, as tombstone growth without a GC pass would
    be unbounded.

    A feature is enabled if:
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    If the feature key is absent from config, it defaults to disabled
    (fail-CLOSED). This function does NOT delegate to is_feature_enabled.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True only if the feature is explicitly enabled for this org
    """
    return _is_fail_closed_flag_enabled(org_id, "dedup_soft_delete")


def is_aggregation_soft_delete_enabled(org_id: str) -> bool:
    """
    Check if aggregation soft-delete is enabled for a given organization.

    This is a FAIL-CLOSED flag: if the key is absent from config or the value
    is not a dict, it returns False. This is the opposite of is_feature_enabled
    (which is fail-open). The difference is intentional — soft-delete must never
    activate for unconfigured orgs, as tombstone growth without a GC pass would
    be unbounded.

    The flag gates soft-supersede (durable replacement of hard-delete for
    playbook aggregation removal). It must only be turned ON for an org once
    Phase B2 GC is enabled for that org — B2 GC is the only reclaimer of the
    SUPERSEDED tombstones this will later create.

    A feature is enabled if:
    - The feature's "enabled" field is True (globally enabled), OR
    - The org_id is in the feature's "enabled_org_ids" list.

    If the feature key is absent from config, it defaults to disabled
    (fail-CLOSED). This function does NOT delegate to is_feature_enabled.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True only if the feature is explicitly enabled for this org
    """
    return _is_fail_closed_flag_enabled(org_id, "aggregation_soft_delete")


def is_resumable_extraction_agent_enabled(org_id: str) -> bool:
    """
    Convenience check for whether classic extraction should use the resumable agent.

    Args:
        org_id (str): The organization ID to check

    Returns:
        bool: True if the resumable extraction agent is enabled
    """
    return is_feature_enabled(org_id, "resumable_extraction_agent")
