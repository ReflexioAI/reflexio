"""Contract tests for ConfigStorage implementations.

Verifies that every ConfigStorage backend satisfies the same behavioral
contract defined by the ABC.  Only LocalFileConfigStorage is locally
testable today; the parametrize structure makes it trivial to add more
backends later.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reflexio.models.config_schema import Config
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

if TYPE_CHECKING:
    from reflexio.server.services.configurator.config_storage import ConfigStorage

pytestmark = pytest.mark.integration


def _make_local_file_storage(tmp_dir: str) -> LocalFileConfigStorage:
    return LocalFileConfigStorage(org_id="test-org", base_dir=tmp_dir)


@pytest.fixture(
    params=[
        pytest.param("local_file", id="LocalFileConfigStorage"),
    ]
)
def config_storage(
    request: pytest.FixtureRequest,
) -> Generator[ConfigStorage, None, None]:
    """Yield a fresh ConfigStorage instance backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        match request.param:
            case "local_file":
                yield _make_local_file_storage(tmp_dir)
            case _:
                raise ValueError(f"Unknown storage backend: {request.param}")


class TestConfigStorageContract:
    """Behavioural contract that every ConfigStorage implementation must satisfy."""

    def test_get_default_config_returns_valid(
        self, config_storage: ConfigStorage
    ) -> None:
        """get_default_config() returns a well-formed Config instance."""
        default = config_storage.get_default_config()
        assert isinstance(default, Config)

    def test_save_and_load_round_trip(self, config_storage: ConfigStorage) -> None:
        """A saved config can be loaded back with key fields intact."""
        original = config_storage.get_default_config()
        original.window_size = 20
        original.stride_size = 10

        config_storage.save_config(original)
        loaded = config_storage.load_config()

        assert loaded.window_size == original.window_size
        assert loaded.stride_size == original.stride_size
        assert loaded.storage_config == original.storage_config

    def test_load_without_save_returns_default(
        self, config_storage: ConfigStorage
    ) -> None:
        """On a fresh storage, load_config() returns the default config."""
        loaded = config_storage.load_config()
        default = config_storage.get_default_config()

        assert isinstance(loaded, Config)
        assert loaded.storage_config == default.storage_config
        assert loaded.window_size == default.window_size

    def test_load_ignores_retired_fields_without_resetting_user_values(
        self, tmp_path: Path
    ) -> None:
        """Schema deletions drop only retired fields from stored config."""
        local_storage = LocalFileConfigStorage(
            org_id="legacy-config-org", base_dir=str(tmp_path)
        )
        config_path = Path(local_storage.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "storage_config": {"db_path": None},
                    "agent_context_prompt": "preserve this",
                    "window_size": 23,
                    "stride_size": 8,
                    "profile_extractor_config": {
                        "extraction_definition_prompt": "preserve nested value",
                        "retired_nested_field": True,
                    },
                    "retired_after_deployment": {"enabled": True},
                }
            ),
            encoding="utf-8",
        )

        loaded = local_storage.load_config()

        assert loaded.agent_context_prompt == "preserve this"
        assert loaded.window_size == 23
        assert loaded.profile_extractor_config is not None
        assert (
            loaded.profile_extractor_config.extraction_definition_prompt
            == "preserve nested value"
        )
        assert loaded.enable_document_expansion is False
        assert "retired_after_deployment" not in loaded.model_dump()
        assert (
            "retired_nested_field" not in loaded.profile_extractor_config.model_dump()
        )

    def test_get_version_returns_none_or_tuple(
        self, config_storage: ConfigStorage
    ) -> None:
        """get_version() returns a (kind, value) tuple or None — never raises."""
        version = config_storage.get_version()
        assert version is None or (isinstance(version, tuple) and len(version) == 2)

    def test_get_version_returns_none_when_file_missing(self, tmp_path) -> None:
        """LocalFileConfigStorage.get_version() returns None for an unsaved org.

        The file is created lazily by load_config() / save_config(); a
        fresh storage instance with no prior writes must report None
        rather than raising or stamping a phantom mtime — the cache
        treats None as "still fresh".
        """
        storage = LocalFileConfigStorage(org_id="brand-new-org", base_dir=str(tmp_path))
        # Don't trigger load_config (which creates the file). Probe directly.
        assert storage.get_version() is None

    def test_save_config_writes_atomically(self, tmp_path) -> None:
        """save_config writes via a per-write unique temp file and renames into place.

        This is the multi-worker safety property the F2 audit remediation
        enforces: ``LocalFileConfigStorage._save_config_to_local_dir``
        must write to a unique ``config_<org>.json.<pid>.<ns>.tmp`` then
        atomically rename over the final path. Three assertions:

        1. After a successful save, no ``.tmp`` sidecar is left behind
           in the directory (catches both legacy shared names and any
           new per-write names).
        2. The persisted config round-trips through ``load_config``.
        3. The tmp filename pattern is unique per-write (a glob for
           ``config_*.json.*.tmp`` returns nothing once the rename is done).

        We don't simulate a crash here — a true "leave the previous good
        file intact on crash" test would require process-kill plumbing.
        The leak-check is a cheap proxy that fails loudly if someone
        reverts to a non-atomic ``open("w")`` write or a shared tmp name.
        """
        storage = LocalFileConfigStorage(org_id="atomic-org", base_dir=str(tmp_path))
        cfg = storage.get_default_config()
        cfg.window_size = 7
        cfg.stride_size = 7
        storage.save_config(cfg)

        final = Path(storage.config_file)
        assert final.exists(), "config file should be written"
        leaked = list(final.parent.glob(f"{final.name}*.tmp"))
        assert not leaked, f"atomic write should not leak a .tmp sidecar: {leaked}"

        loaded = storage.load_config()
        assert loaded.window_size == 7
        assert loaded.stride_size == 7

    def test_save_config_propagates_write_failure(self, tmp_path, monkeypatch) -> None:
        """save_config must raise on OSError; callers must not see a false success.

        The pre-fix implementation caught Exception, printed a traceback,
        and returned normally — masking write failures from operators and
        leaving in-memory state diverged from disk. Verify the rewrite
        re-raises.
        """
        storage = LocalFileConfigStorage(org_id="raises-org", base_dir=str(tmp_path))
        cfg = storage.get_default_config()

        def boom(self: Path, *args, **kwargs) -> int:
            raise OSError("simulated disk failure")

        monkeypatch.setattr(Path, "write_text", boom)

        with pytest.raises(OSError, match="simulated disk failure"):
            storage.save_config(cfg)

        leaked = list(Path(tmp_path).rglob(f"{Path(storage.config_file).name}*.tmp"))
        assert not leaked, f"tmp file should be cleaned up on failure: {leaked}"

    def test_get_version_changes_after_save(
        self, config_storage: ConfigStorage
    ) -> None:
        """A backend that supports versioning must report a fresh value after save_config().

        Skipped for backends that legitimately can't probe (return None).
        """
        import time

        # Establish a baseline by saving once so the version stamp exists.
        cfg = config_storage.get_default_config()
        config_storage.save_config(cfg)
        first = config_storage.get_version()
        if first is None:
            pytest.skip("backend does not support cheap version probing")

        # Mutate + save again — the stamp must move forward.
        # File backends key on mtime, so we sleep enough to cross the
        # filesystem's 1-second resolution on common Linux distros.
        time.sleep(1.1)
        cfg.window_size = (cfg.window_size or 0) + 1
        config_storage.save_config(cfg)
        second = config_storage.get_version()

        assert second is not None
        assert second != first


def test_legacy_shadow_file_load_patch_save_roundtrip(tmp_path: Path) -> None:
    storage = LocalFileConfigStorage(org_id="legacy-org", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = {
        "storage_config": {"type": "sqlite"},
        "shadow_mode_enabled": True,
        "agent_context_prompt": "customer prompt",
        "window_size": 23,
        "stride_size": 7,
    }
    path.write_text(json.dumps(original))
    loaded = storage.load_config()
    assert loaded.agent_context_prompt == "customer prompt"
    assert loaded.window_size == 23
    assert loaded.stride_size == 7
    loaded.enable_document_expansion = True
    storage.save_config(loaded)
    saved = json.loads(path.read_text())
    assert "shadow_mode_enabled" not in saved
    assert saved["window_size"] == 23
    assert saved["stride_size"] == 7
    assert storage.load_config().enable_document_expansion is True


@pytest.mark.parametrize(
    "section,field",
    [
        ("pending_tool_call_config", "resume_poll_interval_seconds"),
        ("pending_tool_call_config", "resume_claim_ttl_seconds"),
        ("playbook_optimizer_config", "scheduler_jitter_seconds"),
        ("playbook_optimizer_config", "abort_cooldown_threshold"),
        ("playbook_optimizer_config", "cooldown_after_aborts_seconds"),
        ("lineage_gc", "poll_interval_seconds"),
        ("governance_retention", "audit_events_delete_batch_limit"),
    ],
)
def test_retired_custom_policy_propagates_without_resetting_file(
    tmp_path, section, field
):
    from reflexio.cli.bootstrap_config import (
        load_storage_from_config,
        save_storage_to_config,
    )
    from reflexio.config_migration import ConfigurationMigrationError

    storage = _make_local_file_storage(str(tmp_path))
    payload = {
        "storage_config": {"type": "sqlite"},
        "lineage_gc": {"enabled": False},
        "agent_context_prompt": "preserve me",
    }
    payload.setdefault(section, {})[field] = "private-custom-policy"
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    for load in (
        storage.load_config,
        lambda: load_storage_from_config("test-org", base_dir=str(tmp_path)),
        lambda: save_storage_to_config("sqlite", "test-org", base_dir=str(tmp_path)),
    ):
        with pytest.raises(ConfigurationMigrationError, match=field) as error:
            load()
        assert "private-custom-policy" not in str(error.value)
        assert path.read_bytes() == before
