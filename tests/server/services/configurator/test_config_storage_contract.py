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
        original.batch_size = 20
        original.batch_interval = 10

        config_storage.save_config(original)
        loaded = config_storage.load_config()

        assert loaded.batch_size == original.batch_size
        assert loaded.batch_interval == original.batch_interval
        assert loaded.storage_config == original.storage_config

    def test_load_without_save_returns_default(
        self, config_storage: ConfigStorage
    ) -> None:
        """On a fresh storage, load_config() returns the default config."""
        loaded = config_storage.load_config()
        default = config_storage.get_default_config()

        assert isinstance(loaded, Config)
        assert loaded.storage_config == default.storage_config
        assert loaded.batch_size == default.batch_size
