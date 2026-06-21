import unittest
from unittest.mock import patch

from reflexio.server.site_var.feature_flags import (
    get_all_feature_flags,
    is_dedup_soft_delete_enabled,
    is_deduplicator_enabled,
    is_feature_enabled,
    is_resumable_extraction_agent_enabled,
)

MOCK_CONFIG = {
    "analytics_v2": {
        "enabled": True,
        "enabled_org_ids": [],
    },
    "beta_feature": {
        "enabled": False,
        "enabled_org_ids": [],
    },
    "pre_retrieval": {
        "enabled": True,
        "enabled_org_ids": [],
    },
    "deduplicator": {
        "enabled": False,
        "enabled_org_ids": ["org-dedup"],
    },
    "resumable_extraction_agent": {
        "enabled": False,
        "enabled_org_ids": ["org-resumable"],
    },
}


class TestFeatureFlags(unittest.TestCase):
    """Unit tests for the feature_flags module."""

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_globally_enabled_feature(self, _mock):
        """A feature with enabled=True should be enabled for any org."""
        self.assertTrue(is_feature_enabled("org-999", "analytics_v2"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_org_specific_enabled(self, _mock):
        """A feature disabled globally but with org in enabled_org_ids should be enabled for that org."""
        self.assertTrue(is_feature_enabled("org-dedup", "deduplicator"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_org_not_in_enabled_list(self, _mock):
        """A feature disabled globally with org NOT in enabled_org_ids should be disabled."""
        self.assertFalse(is_feature_enabled("org-999", "deduplicator"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_completely_disabled_feature(self, _mock):
        """A feature with enabled=False and empty enabled_org_ids should be disabled for everyone."""
        self.assertFalse(is_feature_enabled("org-123", "beta_feature"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_unknown_feature_defaults_enabled(self, _mock):
        """An unknown feature name should default to enabled (fail-open)."""
        self.assertTrue(is_feature_enabled("org-123", "nonexistent_feature"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_get_all_feature_flags(self, _mock):
        """get_all_feature_flags should return a dict of feature -> bool for the given org."""
        result = get_all_feature_flags("org-123")
        self.assertEqual(
            result,
            {
                "analytics_v2": True,
                "beta_feature": False,
                "pre_retrieval": True,
                "deduplicator": False,
                "resumable_extraction_agent": False,
            },
        )

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_get_all_feature_flags_no_match(self, _mock):
        """get_all_feature_flags for an org with no specific access should reflect global state."""
        result = get_all_feature_flags("org-999")
        self.assertEqual(
            result,
            {
                "analytics_v2": True,
                "beta_feature": False,
                "pre_retrieval": True,
                "deduplicator": False,
                "resumable_extraction_agent": False,
            },
        )

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_empty_config_defaults_enabled(self, _mock):
        """With empty config, all features should default to enabled."""
        self.assertTrue(is_feature_enabled("org-123", "some_feature"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_get_all_flags_empty_config(self, _mock):
        """get_all_feature_flags with empty config should return empty dict."""
        result = get_all_feature_flags("org-123")
        self.assertEqual(result, {})

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_is_deduplicator_enabled_for_enabled_org(self, _mock):
        """is_deduplicator_enabled should return True for orgs in enabled_org_ids."""
        self.assertTrue(is_deduplicator_enabled("org-dedup"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_is_deduplicator_disabled_for_other_org(self, _mock):
        """is_deduplicator_enabled should return False for orgs not in enabled_org_ids."""
        self.assertFalse(is_deduplicator_enabled("org-123"))
        self.assertFalse(is_deduplicator_enabled("org-999"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_is_deduplicator_enabled_unknown_defaults_enabled(self, _mock):
        """is_deduplicator_enabled with empty config should default to enabled."""
        self.assertTrue(is_deduplicator_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_is_resumable_extraction_agent_enabled_for_enabled_org(self, _mock):
        """resumable extraction agent can be enabled for selected orgs."""
        self.assertTrue(is_resumable_extraction_agent_enabled("org-resumable"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,
    )
    def test_is_resumable_extraction_agent_disabled_for_other_orgs(self, _mock):
        """resumable extraction agent defaults to classic extraction for other orgs."""
        self.assertFalse(is_resumable_extraction_agent_enabled("org-123"))


class TestDedupSoftDeleteFlag(unittest.TestCase):
    """Tests for is_dedup_soft_delete_enabled — a FAIL-CLOSED flag.

    Unlike is_feature_enabled (fail-open), a missing key must return False.
    This is the safety invariant: unconfigured orgs must never get soft-delete
    enabled accidentally (tombstone growth without B2 GC would be unbounded).
    """

    # F1 regression: the critical case — key absent → OFF
    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_unconfigured_org_returns_false(self, _mock):
        """FAIL-CLOSED: key absent from config → False (not True like is_feature_enabled)."""
        self.assertFalse(is_dedup_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,  # MOCK_CONFIG has no dedup_soft_delete key
    )
    def test_key_missing_from_populated_config_returns_false(self, _mock):
        """FAIL-CLOSED: key missing from a non-empty config still returns False."""
        self.assertFalse(is_dedup_soft_delete_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "dedup_soft_delete": {"enabled": False, "enabled_org_ids": []},
        },
    )
    def test_explicitly_disabled_returns_false(self, _mock):
        """Explicitly disabled (enabled=False, no org list) → False."""
        self.assertFalse(is_dedup_soft_delete_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "dedup_soft_delete": {"enabled": True, "enabled_org_ids": []},
        },
    )
    def test_globally_enabled_returns_true(self, _mock):
        """Globally enabled (enabled=True) → True for any org."""
        self.assertTrue(is_dedup_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "dedup_soft_delete": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_in_enabled_list_returns_true(self, _mock):
        """Org in enabled_org_ids → True even when global enabled=False."""
        self.assertTrue(is_dedup_soft_delete_enabled("org-pilot"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "dedup_soft_delete": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_not_in_enabled_list_returns_false(self, _mock):
        """Org NOT in enabled_org_ids with global disabled → False."""
        self.assertFalse(is_dedup_soft_delete_enabled("org-other"))

    # Contrast test: documents deliberate divergence from is_feature_enabled
    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_contrast_with_fail_open_helper(self, _mock):
        """Contrast: is_feature_enabled returns True for unknown key; is_dedup_soft_delete_enabled returns False.

        This documents the deliberate divergence — the two functions have
        opposite defaults for unconfigured keys.
        """
        key = "dedup_soft_delete"
        self.assertTrue(is_feature_enabled("org-any", key))
        self.assertFalse(is_dedup_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "dedup_soft_delete": {"enabled": False, "enabled_org_ids": None},
        },
    )
    def test_explicit_null_enabled_org_ids_returns_false(self, _mock):
        """Explicit null enabled_org_ids must not raise TypeError.

        When ``enabled_org_ids`` is explicitly null (None) in config, the
        membership check must treat it as an empty list and return False,
        not raise TypeError.
        """
        self.assertFalse(is_dedup_soft_delete_enabled("org-pilot"))


class TestIsFeatureFlagsNullOrgIds(unittest.TestCase):
    """Null-safety tests for is_feature_enabled with null enabled_org_ids."""

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "some_feature": {"enabled": False, "enabled_org_ids": None},
        },
    )
    def test_explicit_null_enabled_org_ids_returns_false(self, _mock):
        """Explicit null enabled_org_ids must not raise TypeError in is_feature_enabled.

        When ``enabled_org_ids`` is explicitly null (None) in config, the
        membership check must treat it as an empty list and return False,
        not raise TypeError.
        """
        self.assertFalse(is_feature_enabled("org-any", "some_feature"))


if __name__ == "__main__":
    unittest.main()
