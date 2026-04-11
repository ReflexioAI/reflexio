import unittest
from unittest.mock import patch

from reflexio.server.site_var.feature_flags import (
    get_all_feature_flags,
    is_deduplicator_enabled,
    is_feature_enabled,
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


if __name__ == "__main__":
    unittest.main()
