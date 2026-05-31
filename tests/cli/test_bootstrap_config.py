"""Tests for CLI bootstrap config: priority chain, write-back, and container safety."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from reflexio.cli.bootstrap_config import (
    load_storage_from_config,
    resolve_storage,
    save_storage_to_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(base_dir: str, org_id: str, storage_type: str) -> None:
    """Write a minimal config file with the given storage type for testing."""
    save_storage_to_config(storage_type, org_id=org_id, base_dir=base_dir)


# ---------------------------------------------------------------------------
# resolve_storage — priority chain
# ---------------------------------------------------------------------------


class TestResolveStorage:
    """Tests for resolve_storage() priority: CLI flag > env var > config > default."""

    def test_cli_flag_wins_over_env_and_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_STORAGE", "postgres")
        _write_config(str(tmp_path), "self-host-org", "sqlite")
        # Patch home to use tmp_path for config lookup
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value="supabase",
        ):
            assert resolve_storage("sqlite") == "sqlite"

    def test_env_var_wins_over_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_STORAGE", "postgres")
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value="sqlite",
        ):
            assert resolve_storage(None) == "postgres"

    def test_config_wins_over_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REFLEXIO_STORAGE", raising=False)
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value="postgres",
        ):
            assert resolve_storage(None) == "postgres"

    def test_default_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REFLEXIO_STORAGE", raising=False)
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value=None,
        ):
            assert resolve_storage(None) == "sqlite"

    def test_invalid_storage_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid storage backend"):
            resolve_storage("redis")

    def test_case_insensitive_flag(self) -> None:
        assert resolve_storage("SQLite") == "sqlite"
        assert resolve_storage("SUPABASE") == "supabase"
        assert resolve_storage("POSTGRES") == "postgres"

    def test_env_var_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REFLEXIO_STORAGE", "SUPABASE")
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value=None,
        ):
            assert resolve_storage(None) == "supabase"

    def test_invalid_env_var_falls_to_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_STORAGE", "invalid_backend")
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value="postgres",
        ):
            assert resolve_storage(None) == "postgres"

    def test_empty_env_var_falls_to_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_STORAGE", "")
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value="postgres",
        ):
            assert resolve_storage(None) == "postgres"


# ---------------------------------------------------------------------------
# save_storage_to_config / load_storage_from_config — round-trip
# ---------------------------------------------------------------------------


class TestSaveAndLoadStorage:
    """Tests for config file round-trip persistence."""

    def test_round_trip_sqlite(self, tmp_path: Path) -> None:
        save_storage_to_config("sqlite", org_id="test-org", base_dir=str(tmp_path))
        result = load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
        assert result == "sqlite"

    def test_round_trip_supabase_with_creds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("DATA_SUPABASE_KEY", "test-key-123")
        monkeypatch.setenv("DATA_DB_URL", "postgresql://localhost/test")
        save_storage_to_config("supabase", org_id="test-org", base_dir=str(tmp_path))
        result = load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
        assert result == "supabase"

    def test_round_trip_supabase_with_legacy_db_url_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATA_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("DATA_SUPABASE_KEY", "test-key-123")
        monkeypatch.delenv("DATA_DB_URL", raising=False)
        monkeypatch.setenv("DATA_SUPABASE_DB_URL", "postgresql://localhost/test")
        save_storage_to_config("supabase", org_id="test-org", base_dir=str(tmp_path))
        result = load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
        assert result == "supabase"

    def test_supabase_without_creds_preserves_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        save_storage_to_config("sqlite", org_id="test-org", base_dir=str(tmp_path))
        assert (
            load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
            == "sqlite"
        )

        monkeypatch.delenv("DATA_SUPABASE_URL", raising=False)
        monkeypatch.delenv("DATA_SUPABASE_KEY", raising=False)
        monkeypatch.delenv("DATA_DB_URL", raising=False)
        monkeypatch.delenv("DATA_SUPABASE_DB_URL", raising=False)
        save_storage_to_config("supabase", org_id="test-org", base_dir=str(tmp_path))
        result = load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
        assert result == "sqlite"

    def test_preserves_existing_config_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Updating storage_config must not clobber extractors or other fields."""
        from reflexio.models.config_schema import (
            Config,
            ProfileExtractorConfig,
            StorageConfigSQLite,
        )
        from reflexio.server.services.configurator.local_file_config_storage import (
            LocalFileConfigStorage,
        )

        # Create a config with a custom extractor
        storage_obj = LocalFileConfigStorage("test-org", base_dir=str(tmp_path))
        config = Config(
            storage_config=StorageConfigSQLite(),
            profile_extractor_config=ProfileExtractorConfig(
                extractor_name="custom_extractor",
                extraction_definition_prompt="Custom prompt for testing",
            ),
            agent_context_prompt="test context",
        )
        storage_obj.save_config(config)

        # Now update storage to postgres (via canonical data DB URL)
        monkeypatch.setenv("DATA_DB_URL", "postgresql://localhost/test")
        save_storage_to_config("postgres", org_id="test-org", base_dir=str(tmp_path))

        # Verify extractor and context are preserved
        reloaded = storage_obj.load_config()
        assert reloaded.agent_context_prompt == "test context"
        assert reloaded.profile_extractor_config is not None
        assert reloaded.profile_extractor_config.extractor_name == "custom_extractor"
        assert (
            load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
            == "postgres"
        )

    def test_postgres_without_data_db_url_preserves_existing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        save_storage_to_config("sqlite", org_id="test-org", base_dir=str(tmp_path))
        monkeypatch.delenv("DATA_DB_URL", raising=False)

        save_storage_to_config("postgres", org_id="test-org", base_dir=str(tmp_path))

        assert (
            load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
            == "sqlite"
        )
        assert "DATA_DB_URL" in caplog.text

    def test_load_returns_none_when_no_file(self, tmp_path: Path) -> None:
        result = load_storage_from_config(org_id="nonexistent", base_dir=str(tmp_path))
        assert result is None

    def test_load_returns_none_for_nonexistent_dir(self, tmp_path: Path) -> None:
        result = load_storage_from_config(
            org_id="test", base_dir=str(tmp_path / "does_not_exist")
        )
        assert result is None


# ---------------------------------------------------------------------------
# .env write-back
# ---------------------------------------------------------------------------


class TestEnvFileWriteBack:
    """Tests for .env file write-back behavior."""

    def test_explicit_flag_updates_env_file(self, tmp_path: Path) -> None:
        """When --storage is explicitly passed, .env file should be updated."""
        from reflexio.cli.env_loader import set_env_var

        env_file = tmp_path / ".env"
        env_file.write_text('REFLEXIO_STORAGE="postgres"\n')

        set_env_var(env_file, "REFLEXIO_STORAGE", "sqlite")

        content = env_file.read_text()
        assert 'REFLEXIO_STORAGE="sqlite"' in content
        assert "postgres" not in content

    def test_set_env_var_preserves_other_vars(self, tmp_path: Path) -> None:
        """set_env_var should only modify the target variable."""
        from reflexio.cli.env_loader import set_env_var

        env_file = tmp_path / ".env"
        env_file.write_text(
            'OPENAI_API_KEY="sk-test"\n'
            'REFLEXIO_STORAGE="postgres"\n'
            'JWT_SECRET_KEY="secret"\n'
        )

        set_env_var(env_file, "REFLEXIO_STORAGE", "sqlite")

        content = env_file.read_text()
        assert 'OPENAI_API_KEY="sk-test"' in content
        assert 'REFLEXIO_STORAGE="sqlite"' in content
        assert 'JWT_SECRET_KEY="secret"' in content

    def test_set_env_var_creates_file_if_absent(self, tmp_path: Path) -> None:
        """If .env doesn't exist, set_env_var creates it and writes the variable."""
        from reflexio.cli.env_loader import set_env_var

        env_file = tmp_path / ".env"
        assert not env_file.exists()

        # set_env_var reads existing content or empty string if file doesn't exist
        set_env_var(env_file, "REFLEXIO_STORAGE", "sqlite")
        assert env_file.exists()
        assert 'REFLEXIO_STORAGE="sqlite"' in env_file.read_text()


# ---------------------------------------------------------------------------
# Layer consistency
# ---------------------------------------------------------------------------


class TestLayerConsistency:
    """Tests for consistency across process env, .env file, and config file."""

    def test_explicit_flag_aligns_all_layers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--storage sqlite should update process env, .env, and config file."""
        from reflexio.cli.env_loader import set_env_var

        env_file = tmp_path / ".env"
        env_file.write_text('REFLEXIO_STORAGE="postgres"\n')

        # Simulate the start() flow when storage="sqlite" (explicit flag)
        resolved = resolve_storage("sqlite")
        monkeypatch.setenv("REFLEXIO_STORAGE", resolved)
        save_storage_to_config(resolved, org_id="test-org", base_dir=str(tmp_path))
        set_env_var(env_file, "REFLEXIO_STORAGE", resolved)

        # Verify all three layers
        assert os.environ["REFLEXIO_STORAGE"] == "sqlite"
        assert 'REFLEXIO_STORAGE="sqlite"' in env_file.read_text()
        assert (
            load_storage_from_config(org_id="test-org", base_dir=str(tmp_path))
            == "sqlite"
        )

    def test_env_override_does_not_modify_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REFLEXIO_STORAGE=postgres (no flag) -> config updated, .env unchanged."""
        env_file = tmp_path / ".env"
        env_file.write_text('REFLEXIO_STORAGE="postgres"\n')
        original_content = env_file.read_text()

        monkeypatch.setenv("REFLEXIO_STORAGE", "postgres")

        # Simulate start() with storage=None (no flag)
        resolved = resolve_storage(None)
        assert resolved == "postgres"

        # Config updated, but .env NOT modified
        save_storage_to_config(
            resolved,
            org_id="test-org",
            base_dir=str(tmp_path),
        )
        assert env_file.read_text() == original_content  # .env unchanged

    def test_sequential_starts_preserve_choice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run 1: --storage sqlite. Run 2: no flag. Both should use sqlite."""
        from reflexio.cli.env_loader import set_env_var

        env_file = tmp_path / ".env"
        env_file.write_text('REFLEXIO_STORAGE="postgres"\n')

        # Run 1: explicit --storage sqlite
        resolved1 = resolve_storage("sqlite")
        assert resolved1 == "sqlite"
        monkeypatch.setenv("REFLEXIO_STORAGE", resolved1)
        save_storage_to_config(resolved1, org_id="test-org", base_dir=str(tmp_path))
        set_env_var(env_file, "REFLEXIO_STORAGE", resolved1)

        # Run 2: no flag — .env now says "sqlite" from run 1
        resolved2 = resolve_storage(None)
        assert resolved2 == "sqlite"  # consistent with run 1


# ---------------------------------------------------------------------------
# Docker / container safety
# ---------------------------------------------------------------------------


class TestContainerSafety:
    """Tests for graceful behavior in container environments."""

    def test_no_config_dir_returns_none(self, tmp_path: Path) -> None:
        """In Docker, ~/.reflexio/configs/ doesn't exist. Graceful fallback."""
        result = load_storage_from_config(
            org_id="test", base_dir=str(tmp_path / "nonexistent")
        )
        assert result is None

    def test_env_var_wins_when_no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Container with REFLEXIO_STORAGE=supabase, no config file -> supabase."""
        monkeypatch.setenv("REFLEXIO_STORAGE", "supabase")
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value=None,
        ):
            assert resolve_storage(None) == "supabase"

    def test_default_without_env_or_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env var, no config file -> falls to 'sqlite' default."""
        monkeypatch.delenv("REFLEXIO_STORAGE", raising=False)
        with patch(
            "reflexio.cli.bootstrap_config.load_storage_from_config",
            return_value=None,
        ):
            assert resolve_storage(None) == "sqlite"
