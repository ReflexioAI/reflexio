"""Unit tests for storage label masking helpers.

Covers ``describe_storage`` (maps StorageConfig → (type, masked_label))
and the two masking helpers used by whoami, config show, and the
publish summary. Masking is a security-relevant primitive — any
regression here could leak secrets into logs or bug reports.
"""

from reflexio.lib._storage_labels import describe_storage, mask_secret, mask_url
from reflexio.models.config_schema import (
    StorageConfigDisk,
    StorageConfigSQLite,
)


class TestDescribeStorage:
    def test_none_returns_none_tuple(self):
        assert describe_storage(None) == (None, None)

    def test_sqlite_default(self):
        storage_type, label = describe_storage(StorageConfigSQLite())
        assert storage_type == "sqlite"
        assert label is not None
        assert "sqlite" in label.lower()

    def test_sqlite_with_custom_path(self):
        storage_type, label = describe_storage(StorageConfigSQLite(db_path="/tmp/x.db"))
        assert storage_type == "sqlite"
        assert label == "/tmp/x.db"

    def test_disk_with_dir_path(self):
        storage_type, label = describe_storage(StorageConfigDisk(dir_path="/tmp/data"))
        assert storage_type == "disk"
        assert label == "/tmp/data"

    def test_supabase_matched_by_class_name(self):
        """StorageConfigSupabase is enterprise-only — we match by name.

        This lets the OS helper describe a Supabase config without
        importing reflexio_ext (which may not be installed).
        """

        class StorageConfigSupabase:
            url = "https://jpkjckbyxrdefzomiyse.supabase.co"

        storage_type, label = describe_storage(StorageConfigSupabase())  # type: ignore[arg-type]
        assert storage_type == "supabase"
        # Label is masked — first 4 chars of host, then tail
        assert label is not None
        assert label.startswith("https://jpkj")
        assert "supabase.co" in label
        assert "xrdef" not in label

    def test_supabase_contract_ad_hoc_type(self):
        """Regression: describe_storage matches StorageConfigSupabase by exact
        class name without importing reflexio_ext.

        Uses ``type(...)`` to mint an ad-hoc class with the right name, which
        pins the string-match contract from the opposite direction: even a
        class that shares no lineage with the enterprise type must still be
        recognized as ``"supabase"`` as long as the name matches exactly.
        """
        fake_cls = type(
            "StorageConfigSupabase",
            (),
            {"url": "https://xyzwabcdefghij.supabase.co"},
        )
        storage_type, label = describe_storage(fake_cls())  # type: ignore[arg-type]
        assert storage_type == "supabase"
        assert label is not None
        assert label.startswith("https://xyzw")
        assert "supabase.co" in label

    def test_supabase_rename_falls_back_to_generic(self):
        """Regression: if enterprise renames the class (e.g. to V2), the
        shared helper must NOT silently keep returning ``"supabase"``.

        This pins the string-match contract — a rename should take the
        generic fallback branch. The test deliberately uses a class name
        that is *close* to the matched name to catch accidental partial
        matching (e.g. ``startswith``) in future refactors.
        """

        class StorageConfigSupabaseV2:
            url = "https://jpkjckbyxrdefzomiyse.supabase.co"

        storage_type, label = describe_storage(StorageConfigSupabaseV2())  # type: ignore[arg-type]
        # The generic fallback strips "StorageConfig" prefix and lowercases.
        assert storage_type == "supabasev2"
        # Explicitly NOT the supabase branch — the label is the raw class
        # name, not a masked URL.
        assert storage_type != "supabase"
        assert label == "StorageConfigSupabaseV2"

    def test_unknown_class_falls_back_to_generic(self):
        """A completely unrelated class name also falls back to the generic
        branch — exercises the last ``return`` in describe_storage.
        """

        class StorageConfigSB:
            url = "https://example.com"

        storage_type, label = describe_storage(StorageConfigSB())  # type: ignore[arg-type]
        assert storage_type == "sb"
        assert storage_type != "supabase"
        assert label == "StorageConfigSB"


class TestMaskUrl:
    def test_empty(self):
        assert mask_url("") == ""

    def test_https_host(self):
        masked = mask_url("https://jpkjckbyxrdefzomiyse.supabase.co")
        assert masked.startswith("https://jpkj")
        assert masked.endswith("supabase.co")
        assert "xrdef" not in masked

    def test_postgres_user_info(self):
        """User credentials in the URL must be stripped."""
        masked = mask_url(
            "postgresql://postgres.abc:verysecret@aws-1.pooler.supabase.com:6543/postgres"
        )
        assert "verysecret" not in masked
        assert "***@aws-1.pooler.supabase.com:6543/postgres" in masked

    def test_plain_string(self):
        masked = mask_url("sk_live_abcdefghij")
        assert masked.startswith("sk_l")
        assert "abcdefghij" not in masked

    def test_url_without_dot_in_host(self):
        """Host with no dot (e.g. ``http://host``) has no tail domain to
        preserve. The helper falls through to the no-dot branch and still
        masks the body rather than leaking it whole.

        Observed behavior: since ``"host"`` is exactly 4 chars, it's not
        longer than 4, so ``_short_head`` returns the full string, yielding
        ``"http://host..."``. This is safe (no secret to leak on a bare
        hostname) and documents the current behavior.
        """
        masked = mask_url("http://host")
        assert masked == "http://host..."

    def test_url_without_dot_in_long_host(self):
        """Longer host without a dot — the head is truncated to 4 chars."""
        masked = mask_url("http://longhostname")
        assert masked == "http://long..."
        assert "hostname" not in masked

    def test_very_short_plain_string(self):
        """Plain string shorter than 4 chars — entire string is kept
        because ``_short_head`` only truncates when len > 4. The trailing
        ``...`` still signals truncation to the reader.
        """
        masked = mask_url("a")
        assert masked == "a..."

    def test_empty_string_returns_empty(self):
        """Empty input short-circuits to empty (re-verifies the guard)."""
        assert mask_url("") == ""


class TestMaskSecret:
    def test_empty_returns_placeholder(self):
        assert mask_secret("") == "<empty>"

    def test_short_string_fully_masked(self):
        assert mask_secret("abcd") == "****"

    def test_long_string_keeps_head_and_tail(self):
        masked = mask_secret("sk_live_verysecretkeyhere")
        assert masked.startswith("sk_l")
        assert masked.endswith("re")
        assert "verysecret" not in masked
