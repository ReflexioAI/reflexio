import unittest
from unittest.mock import patch

from reflexio.server.site_var.feature_flags import (
    get_all_feature_flags,
    is_aggregation_soft_delete_enabled,
    is_deduplicator_enabled,
    is_feature_enabled,
    is_lineage_dual_read_diff_enabled,
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


class TestAggregationSoftDeleteFlag(unittest.TestCase):
    """Tests for is_aggregation_soft_delete_enabled — a DEFAULT-OPEN flag.

    The key is absent from config → True (default ON, because GC is also ON by
    default so tombstones are reclaimed). Explicit disable still works via
    enabled=False with no org list. Malformed config (non-dict) → False with a
    warning (safe fallback).
    """

    # Default-open: the critical case — key absent → ON
    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_unconfigured_org_returns_true(self, _mock):
        """DEFAULT-OPEN: key absent from config → True (soft-delete on by default)."""
        self.assertTrue(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,  # MOCK_CONFIG has no aggregation_soft_delete key
    )
    def test_key_missing_from_populated_config_returns_true(self, _mock):
        """DEFAULT-OPEN: key missing from a non-empty config still returns True."""
        self.assertTrue(is_aggregation_soft_delete_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {"enabled": False, "enabled_org_ids": []},
        },
    )
    def test_explicitly_disabled_returns_false(self, _mock):
        """Explicitly disabled (enabled=False, no org list) → False."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {"enabled": True, "enabled_org_ids": []},
        },
    )
    def test_globally_enabled_returns_true(self, _mock):
        """Globally enabled (enabled=True) → True for any org."""
        self.assertTrue(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_in_enabled_list_returns_true(self, _mock):
        """Org in enabled_org_ids → True even when global enabled=False."""
        self.assertTrue(is_aggregation_soft_delete_enabled("org-pilot"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_not_in_enabled_list_returns_false(self, _mock):
        """Org NOT in enabled_org_ids with global disabled → False (explicit disable)."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-other"))

    # Both is_feature_enabled and is_aggregation_soft_delete_enabled are now default-open.
    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_both_helpers_agree_on_absent_key(self, _mock):
        """Both is_feature_enabled and is_aggregation_soft_delete_enabled return True for absent key.

        Previously they diverged (fail-open vs fail-closed). Now both are default-open
        for this flag. The contrast test is retained to document the current behavior.
        """
        key = "aggregation_soft_delete"
        self.assertTrue(is_feature_enabled("org-any", key))
        self.assertTrue(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {"enabled": False, "enabled_org_ids": None},
        },
    )
    def test_explicit_null_enabled_org_ids_returns_false(self, _mock):
        """Explicit null enabled_org_ids must not raise TypeError.

        When ``enabled_org_ids`` is explicitly null (None) in config, the
        membership check must treat it as an empty list and return False,
        not raise TypeError.
        """
        self.assertFalse(is_aggregation_soft_delete_enabled("org-pilot"))


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


class TestFailClosedFlagMalformedConfig(unittest.TestCase):
    """F2 regression: non-dict feature_config must return False (fail-closed), not AttributeError.

    If the site var value is a scalar or list instead of a dict, calling
    .get("enabled", ...) raises AttributeError. Both fail-closed flag functions
    must guard against this by returning False with a warning log.
    """

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={"aggregation_soft_delete": True},
    )
    def test_aggregation_scalar_value_returns_false(self, _mock):
        """aggregation_soft_delete with scalar config value → False, no AttributeError."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={"aggregation_soft_delete": 42},
    )
    def test_aggregation_integer_value_returns_false(self, _mock):
        """aggregation_soft_delete with integer config value → False, no AttributeError."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-any"))


class TestFailClosedFlagStrictBoolValidation(unittest.TestCase):
    """#195: strict enabled is True — truthy strings and non-bool ints must not enable.

    CodeRabbit finding: `if feature_config.get("enabled", False):` accepts any truthy
    value including the string "false". Fix: `if enabled is True:` (strict identity).
    """

    # --- is_aggregation_soft_delete_enabled ---

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {"enabled": "false", "enabled_org_ids": []}
        },
    )
    def test_aggregation_string_false_returns_false(self, _mock):
        """enabled='false' (string) must not enable aggregation soft-delete."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={"aggregation_soft_delete": {"enabled": 1, "enabled_org_ids": []}},
    )
    def test_aggregation_int_one_returns_false(self, _mock):
        """enabled=1 (int) must not enable aggregation soft-delete."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {
                "enabled": False,
                "enabled_org_ids": "org-pilot",
            }
        },
    )
    def test_aggregation_string_org_ids_returns_false(self, _mock):
        """enabled_org_ids='org-pilot' (string) must not match 'org-pilot' or 'pilot'."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-pilot"))
        self.assertFalse(is_aggregation_soft_delete_enabled("pilot"))


class TestSoftDeleteDefaultOn(unittest.TestCase):
    """Regression suite for the default-ON semantics of the soft-delete flags.

    aggregation_soft_delete must return True for any org when the site var has no
    entry for that key (absent → enabled). The GC is also enabled by default so
    tombstones are reclaimed automatically.
    """

    # --- aggregation_soft_delete ---

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_aggregation_absent_key_any_org_returns_true(self, _mock):
        """Any org with absent aggregation_soft_delete key → True (default ON)."""
        for org_id in ("org-a", "org-b", "org-c", "org-xyz-456"):
            with self.subTest(org_id=org_id):
                self.assertTrue(is_aggregation_soft_delete_enabled(org_id))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {"enabled": False, "enabled_org_ids": []}
        },
    )
    def test_aggregation_explicit_global_disable_overrides_default(self, _mock):
        """Explicit enabled=False + empty list → False for all orgs (override respected)."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "aggregation_soft_delete": {
                "enabled": False,
                "enabled_org_ids": ["org-exempt"],
            }
        },
    )
    def test_aggregation_explicit_per_org_disable_respected(self, _mock):
        """Explicit per-org exclusion via enabled=False + list without org → False."""
        self.assertFalse(is_aggregation_soft_delete_enabled("org-not-exempt"))
        self.assertTrue(is_aggregation_soft_delete_enabled("org-exempt"))


class TestLineageDualReadDiffFlag(unittest.TestCase):
    """Tests for is_lineage_dual_read_diff_enabled — a DEFAULT-CLOSED flag.

    The key is absent from config → False (fail-CLOSED). The flag must never
    activate on unconfigured deployments. Explicit enable via enabled=True or
    per-org enabled_org_ids still works. Malformed config (non-dict) → False
    with a warning (safe fallback). Strict-bool identity guards prevent truthy
    strings and non-bool ints from enabling the flag.
    """

    # Default-closed: the critical case — key absent → OFF
    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={},
    )
    def test_absent_key_returns_false(self, _mock):
        """DEFAULT-CLOSED: key absent from config → False (dual-read diff off by default)."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value=MOCK_CONFIG,  # MOCK_CONFIG has no lineage_dual_read_diff key
    )
    def test_key_missing_from_populated_config_returns_false(self, _mock):
        """DEFAULT-CLOSED: key missing from a non-empty config still returns False."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-123"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "lineage_dual_read_diff": {"enabled": True, "enabled_org_ids": []},
        },
    )
    def test_globally_enabled_returns_true(self, _mock):
        """Globally enabled (enabled=True) → True for any org."""
        self.assertTrue(is_lineage_dual_read_diff_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "lineage_dual_read_diff": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_in_enabled_list_returns_true(self, _mock):
        """Org in enabled_org_ids → True even when global enabled=False."""
        self.assertTrue(is_lineage_dual_read_diff_enabled("org-pilot"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "lineage_dual_read_diff": {
                "enabled": False,
                "enabled_org_ids": ["org-pilot"],
            },
        },
    )
    def test_org_not_in_enabled_list_returns_false(self, _mock):
        """Org NOT in enabled_org_ids with global disabled → False."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-other"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={"lineage_dual_read_diff": "yes"},
    )
    def test_malformed_scalar_value_returns_false(self, _mock):
        """Non-dict config value (scalar) → False, no AttributeError."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "lineage_dual_read_diff": {"enabled": "true", "enabled_org_ids": []}
        },
    )
    def test_string_true_enabled_returns_false(self, _mock):
        """enabled='true' (string) is not bool True — strict identity must reject it."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={"lineage_dual_read_diff": {"enabled": 1, "enabled_org_ids": []}},
    )
    def test_int_one_enabled_returns_false(self, _mock):
        """enabled=1 (int) is not bool True — strict identity must reject it."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-any"))

    @patch(
        "reflexio.server.site_var.feature_flags._get_feature_flags_config",
        return_value={
            "lineage_dual_read_diff": {
                "enabled": False,
                "enabled_org_ids": "org-pilot",
            }
        },
    )
    def test_string_org_ids_returns_false(self, _mock):
        """enabled_org_ids='org-pilot' (string) must not match via substring in (#195)."""
        self.assertFalse(is_lineage_dual_read_diff_enabled("org-pilot"))
        self.assertFalse(is_lineage_dual_read_diff_enabled("pilot"))


if __name__ == "__main__":
    unittest.main()
