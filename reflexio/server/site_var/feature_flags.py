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

    enabled_org_ids = feature_config.get("enabled_org_ids", [])
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
