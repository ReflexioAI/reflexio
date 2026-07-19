"""Contract tests for ConfigStorage implementations.

Verifies that every ConfigStorage backend satisfies the same behavioral
contract defined by the ABC.  Only LocalFileConfigStorage is locally
testable today; the parametrize structure makes it trivial to add more
backends later.
"""

from __future__ import annotations

import fcntl
import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reflexio.models.config_schema import Config
from reflexio.server.services.configurator.config_storage import ConfigWriteConflict
from reflexio.server.services.configurator.configurator import DefaultConfigurator
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

    def test_save_config_fails_immediately_when_lock_is_held(self, tmp_path) -> None:
        storage = LocalFileConfigStorage(org_id="locked-org", base_dir=str(tmp_path))
        lock_path = Path(storage.config_file).with_suffix(".json.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(
                ConfigWriteConflict,
                match="already in progress",
            ):
                storage.save_config(storage.get_default_config())

    def test_first_load_under_held_lock_returns_default_config(self, tmp_path) -> None:
        """A concurrent first-load must serve defaults, not raise a conflict."""
        storage = LocalFileConfigStorage(org_id="fresh-org", base_dir=str(tmp_path))
        lock_path = Path(storage.config_file).with_suffix(".json.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            loaded = storage.load_config()

        assert loaded == storage.get_default_config()

    def test_schema_cleanup_conflict_returns_valid_stored_config(
        self, tmp_path
    ) -> None:
        storage = LocalFileConfigStorage(org_id="read-locked", base_dir=str(tmp_path))
        payload = storage.get_default_config().model_dump(mode="json")
        payload["retired_field"] = "ignored"
        payload["agent_context_prompt"] = "user value survives"
        config_path = Path(storage.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        lock_path = config_path.with_suffix(".json.lock")

        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            loaded = storage.load_config()

        assert loaded.storage_config == storage.get_default_config().storage_config
        assert loaded.agent_context_prompt == "user value survives"
        assert json.loads(config_path.read_text())["retired_field"] == "ignored"

    def test_schema_cleanup_preserves_concurrent_newer_write(
        self, tmp_path, monkeypatch
    ) -> None:
        """The cleanup rewrite must normalize the latest payload, not the
        pre-lock snapshot, so a write completing between the read and the
        lock is cleaned rather than silently overwritten."""
        storage = LocalFileConfigStorage(org_id="cleanup-race", base_dir=str(tmp_path))
        stale = storage.get_default_config().model_dump(mode="json")
        stale["retired_field"] = "drop me"
        config_path = Path(storage.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(stale), encoding="utf-8")

        newer = dict(stale)
        newer["agent_context_prompt"] = "written after the pre-lock read"
        monkeypatch.setattr(
            storage,
            "_read_payload_unlocked",
            lambda _path: dict(newer),
        )

        loaded = storage.load_config()

        persisted = json.loads(config_path.read_text(encoding="utf-8"))
        assert persisted["agent_context_prompt"] == "written after the pre-lock read"
        assert "retired_field" not in persisted
        # The returned config still reflects the payload this reader saw.
        assert loaded.agent_context_prompt is None

    def test_sequential_partial_commits_reload_latest_payload(self, tmp_path) -> None:
        configurator = DefaultConfigurator(
            org_id="partial-org",
            base_dir=str(tmp_path),
        )
        first = configurator.prepare_config_write(
            {"agent_context_prompt": "first"},
            partial=True,
        )
        second = configurator.prepare_config_write(
            {"window_size": 23},
            partial=True,
        )

        assert configurator.commit_config_write(first) is True
        assert configurator.commit_config_write(second) is True

        reloaded = DefaultConfigurator(
            org_id="partial-org",
            base_dir=str(tmp_path),
        ).get_config()
        assert reloaded.agent_context_prompt == "first"
        assert reloaded.window_size == 23

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
