"""
Tests for Reflexio instance cache in reflexio.server.cache.reflexio_cache.

Covers:
1. get_reflexio creates and caches a new instance
2. Cache hit returns the same instance
3. Different cache keys (org_id, storage_base_dir) produce separate entries
4. invalidate_reflexio_cache removes found entries and returns False for missing
5. clear_reflexio_cache empties all entries
6. get_cache_stats returns correct structure and values
7. Version-based auto-invalidation: stale entries are evicted on next get
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.cache.reflexio_cache import (
    clear_reflexio_cache,
    get_cache_stats,
    get_reflexio,
    invalidate_reflexio_cache,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear the module-level cache before and after every test."""
    clear_reflexio_cache()
    yield
    clear_reflexio_cache()


# =============================================================================
# get_reflexio Tests
# =============================================================================


class TestGetReflexio:
    """Tests for get_reflexio creation and caching behavior."""

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_creates_new_instance_on_cache_miss(self, mock_reflexio_cls: MagicMock):
        """First call with a given key creates a new Reflexio instance."""
        mock_instance = MagicMock()
        mock_reflexio_cls.return_value = mock_instance

        result = get_reflexio("org-1")
        assert result is mock_instance
        mock_reflexio_cls.assert_called_once_with(org_id="org-1", storage_base_dir=None)

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_cache_hit_returns_same_instance(self, mock_reflexio_cls: MagicMock):
        """Second call with the same key returns the cached instance without constructing again."""
        mock_instance = MagicMock()
        mock_reflexio_cls.return_value = mock_instance

        first = get_reflexio("org-1")
        second = get_reflexio("org-1")

        assert first is second
        mock_reflexio_cls.assert_called_once()

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_different_org_ids_produce_separate_entries(
        self, mock_reflexio_cls: MagicMock
    ):
        """Different org_id values result in separate cache entries."""
        mock_a = MagicMock()
        mock_b = MagicMock()
        mock_reflexio_cls.side_effect = [mock_a, mock_b]

        result_a = get_reflexio("org-a")
        result_b = get_reflexio("org-b")

        assert result_a is not result_b
        assert mock_reflexio_cls.call_count == 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_different_storage_base_dirs_produce_separate_entries(
        self, mock_reflexio_cls: MagicMock
    ):
        """Same org_id with different storage_base_dir creates separate entries."""
        mock_a = MagicMock()
        mock_b = MagicMock()
        mock_reflexio_cls.side_effect = [mock_a, mock_b]

        result_a = get_reflexio("org-1", storage_base_dir="/path/a")
        result_b = get_reflexio("org-1", storage_base_dir="/path/b")

        assert result_a is not result_b
        assert mock_reflexio_cls.call_count == 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_none_vs_string_storage_base_dir(self, mock_reflexio_cls: MagicMock):
        """storage_base_dir=None and a string value produce different cache entries."""
        mock_a = MagicMock()
        mock_b = MagicMock()
        mock_reflexio_cls.side_effect = [mock_a, mock_b]

        result_a = get_reflexio("org-1", storage_base_dir=None)
        result_b = get_reflexio("org-1", storage_base_dir="/data")

        assert result_a is not result_b
        assert mock_reflexio_cls.call_count == 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_passes_storage_base_dir_to_constructor(self, mock_reflexio_cls: MagicMock):
        """storage_base_dir is forwarded to the Reflexio constructor."""
        mock_reflexio_cls.return_value = MagicMock()

        get_reflexio("org-1", storage_base_dir="/custom/dir")
        mock_reflexio_cls.assert_called_once_with(
            org_id="org-1", storage_base_dir="/custom/dir"
        )


# =============================================================================
# invalidate_reflexio_cache Tests
# =============================================================================


class TestInvalidateReflexioCache:
    """Tests for invalidate_reflexio_cache."""

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_invalidate_existing_entry_returns_true(self, mock_reflexio_cls: MagicMock):
        """Invalidating an existing cache entry returns True."""
        mock_reflexio_cls.return_value = MagicMock()
        get_reflexio("org-1")

        assert invalidate_reflexio_cache("org-1") is True

    def test_invalidate_missing_entry_returns_false(self):
        """Invalidating a non-existent key returns False."""
        assert invalidate_reflexio_cache("no-such-org") is False

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_invalidated_entry_is_recreated_on_next_get(
        self, mock_reflexio_cls: MagicMock
    ):
        """After invalidation, the next get_reflexio creates a fresh instance."""
        mock_first = MagicMock()
        mock_second = MagicMock()
        mock_reflexio_cls.side_effect = [mock_first, mock_second]

        first = get_reflexio("org-1")
        invalidate_reflexio_cache("org-1")
        second = get_reflexio("org-1")

        assert first is not second
        assert mock_reflexio_cls.call_count == 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_invalidate_with_storage_base_dir(self, mock_reflexio_cls: MagicMock):
        """Invalidation uses the full cache key including storage_base_dir."""
        mock_reflexio_cls.return_value = MagicMock()
        get_reflexio("org-1", storage_base_dir="/data")

        # Wrong storage_base_dir should not find the entry
        assert invalidate_reflexio_cache("org-1", storage_base_dir=None) is False
        # Correct storage_base_dir should find and remove it
        assert invalidate_reflexio_cache("org-1", storage_base_dir="/data") is True


# =============================================================================
# clear_reflexio_cache Tests
# =============================================================================


class TestClearReflexioCache:
    """Tests for clear_reflexio_cache."""

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_clear_empties_all_entries(self, mock_reflexio_cls: MagicMock):
        """Clearing the cache removes all entries."""
        mock_reflexio_cls.return_value = MagicMock()
        get_reflexio("org-1")
        get_reflexio("org-2")
        get_reflexio("org-3")

        stats_before = get_cache_stats()
        assert stats_before["current_size"] == 3

        clear_reflexio_cache()

        stats_after = get_cache_stats()
        assert stats_after["current_size"] == 0

    def test_clear_on_empty_cache_is_safe(self):
        """Clearing an already empty cache does not raise."""
        clear_reflexio_cache()
        assert get_cache_stats()["current_size"] == 0


# =============================================================================
# get_cache_stats Tests
# =============================================================================


class TestGetCacheStats:
    """Tests for get_cache_stats."""

    def test_returns_expected_keys(self):
        """Stats dict contains current_size, max_size, and ttl_seconds."""
        stats = get_cache_stats()
        assert "current_size" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats

    def test_empty_cache_current_size_is_zero(self):
        """Empty cache reports current_size=0."""
        assert get_cache_stats()["current_size"] == 0

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_current_size_increments(self, mock_reflexio_cls: MagicMock):
        """current_size reflects the number of cached entries."""
        mock_reflexio_cls.return_value = MagicMock()

        get_reflexio("org-1")
        assert get_cache_stats()["current_size"] == 1

        get_reflexio("org-2")
        assert get_cache_stats()["current_size"] == 2

    def test_max_size_matches_constant(self):
        """max_size matches the module-level REFLEXIO_CACHE_MAX_SIZE constant."""
        from reflexio.server.cache.reflexio_cache import REFLEXIO_CACHE_MAX_SIZE

        assert get_cache_stats()["max_size"] == REFLEXIO_CACHE_MAX_SIZE

    def test_ttl_matches_constant(self):
        """ttl_seconds matches the module-level REFLEXIO_CACHE_TTL_SECONDS constant."""
        from reflexio.server.cache.reflexio_cache import REFLEXIO_CACHE_TTL_SECONDS

        assert get_cache_stats()["ttl_seconds"] == REFLEXIO_CACHE_TTL_SECONDS


# =============================================================================
# Version-based auto-invalidation tests
# =============================================================================


def _stub_reflexio(version: tuple[str, object] | None) -> MagicMock:
    """Build a Reflexio mock that returns *version* from current_config_version().

    Each call to ``current_config_version()`` returns the same tuple,
    matching the contract that the probe is cheap and idempotent
    between out-of-band writes.
    """
    instance = MagicMock()
    instance.current_config_version.return_value = version
    return instance


class TestVersionBasedInvalidation:
    """Tests for auto-eviction when the persisted config version changes."""

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_cache_evicts_when_config_file_modified(self, mock_reflexio_cls: MagicMock):
        """A cached entry whose probed version differs from the cached one is rebuilt."""
        # First instance is constructed at version mtime=100.0 — cached.
        # Then someone edits the file out-of-band, bumping mtime to 200.0.
        # The cached instance now reports the new version when probed.
        first = _stub_reflexio(("file", 100.0))
        second = _stub_reflexio(("file", 200.0))
        mock_reflexio_cls.side_effect = [first, second]

        a = get_reflexio("org-1")
        # Simulate an out-of-band file edit by changing what the cached
        # mock reports on its NEXT probe.
        first.current_config_version.return_value = ("file", 200.0)

        b = get_reflexio("org-1")
        assert a is not b
        assert b is second
        assert mock_reflexio_cls.call_count == 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_cache_keeps_when_config_file_unchanged(self, mock_reflexio_cls: MagicMock):
        """Identical probe value across hits leaves the cached instance in place."""
        first = _stub_reflexio(("file", 100.0))
        mock_reflexio_cls.return_value = first

        a = get_reflexio("org-1")
        b = get_reflexio("org-1")
        c = get_reflexio("org-1")

        assert a is b is c
        assert mock_reflexio_cls.call_count == 1
        # Probe was called on the second + third hits (not the first
        # construction, where the result is captured for stamping).
        assert first.current_config_version.call_count >= 2

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_cache_handles_none_version(self, mock_reflexio_cls: MagicMock):
        """When the backend can't probe (returns None), the entry is never auto-evicted."""
        first = _stub_reflexio(None)
        mock_reflexio_cls.return_value = first

        a = get_reflexio("org-1")
        b = get_reflexio("org-1")

        assert a is b
        # The cache short-circuits the probe call when the stamped
        # version is None — no point in re-probing a backend that
        # already declared "I can't tell you". One call happened at
        # construction to capture the stamp; the second hit must NOT
        # add another call.
        assert first.current_config_version.call_count == 1
        assert mock_reflexio_cls.call_count == 1

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_probe_failure_treated_as_fresh(self, mock_reflexio_cls: MagicMock):
        """A probe that raises must not break the cache hit; the cached instance is kept."""
        first = _stub_reflexio(("file", 100.0))
        mock_reflexio_cls.return_value = first

        a = get_reflexio("org-1")
        # Subsequent probes blow up — we want the cache to swallow it
        # and serve the cached instance, not crash the request.
        first.current_config_version.side_effect = OSError("disk gone")

        b = get_reflexio("org-1")
        assert a is b
        assert mock_reflexio_cls.call_count == 1

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_db_version_tuple_evicts_on_change(self, mock_reflexio_cls: MagicMock):
        """The eviction path is generic — works for ('db', int) tuples too, not just file mtimes."""
        first = _stub_reflexio(("db", 1))
        second = _stub_reflexio(("db", 2))
        mock_reflexio_cls.side_effect = [first, second]

        a = get_reflexio("org-1")
        first.current_config_version.return_value = ("db", 2)

        b = get_reflexio("org-1")
        assert a is not b
        assert b is second

    @patch("reflexio.server.cache.reflexio_cache.Reflexio")
    def test_construction_probe_failure_does_not_disable_future_eviction(
        self, mock_reflexio_cls: MagicMock
    ):
        """A transient probe failure at construction must not poison the entry.

        Earlier code stamped the entry with ``cached_version=None``
        whenever ``_probe_version_safe`` returned None — which conflated
        "backend can't probe" with "probe raised". That permanently
        disabled auto-eviction for the entry. The fix: don't cache an
        instance whose construction probe raised, so the next request
        re-builds and re-probes successfully, and version-based eviction
        works from then on.
        """
        # First instance's construction-time probe raises. Second
        # instance probes cleanly at version 100, then jumps to 200.
        # Third instance (post-eviction) probes at 200.
        first = MagicMock()
        first.org_id = "org-1"
        first.current_config_version.side_effect = OSError("disk gone")

        second = _stub_reflexio(("file", 100.0))
        third = _stub_reflexio(("file", 200.0))
        mock_reflexio_cls.side_effect = [first, second, third]

        # Request 1: construction-time probe raises → instance returned
        # but NOT cached.
        a = get_reflexio("org-1")
        assert a is first
        assert get_cache_stats()["current_size"] == 0

        # Request 2: backend recovers, second construction succeeds and
        # gets cached at version 100.
        b = get_reflexio("org-1")
        assert b is second
        assert get_cache_stats()["current_size"] == 1

        # Request 3: out-of-band edit bumps version to 200; eviction
        # runs because the entry has a real stamp now.
        second.current_config_version.return_value = ("file", 200.0)
        c = get_reflexio("org-1")
        assert c is third
        assert mock_reflexio_cls.call_count == 3
