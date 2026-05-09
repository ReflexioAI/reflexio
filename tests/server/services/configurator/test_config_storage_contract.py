"""Contract tests for ConfigStorage implementations.

Verifies that every ConfigStorage backend satisfies the same behavioral
contract defined by the ABC.  Only LocalFileConfigStorage is locally
testable today; the parametrize structure makes it trivial to add more
backends later.
"""

from __future__ import annotations

import tempfile
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
def config_storage(request: pytest.FixtureRequest) -> ConfigStorage:
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
