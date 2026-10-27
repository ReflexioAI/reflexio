"""Configuration reads retain originals; initialization cannot replace a winner."""

import errno
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app
from reflexio.config_document import decode_config_document
from reflexio.config_migration import StoredConfigurationError
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.fixture
def storage(tmp_path):
    return LocalFileConfigStorage("lifecycle", base_dir=str(tmp_path))


def write_source(storage, raw):
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o640)
    return path


@pytest.mark.parametrize(
    "geometry",
    ['"window_size":23,"stride_size":7', '"batch_size":23,"batch_interval":7'],
)
def test_existing_read_preserves_sparse_source_and_nested_metadata(
    storage, monkeypatch, geometry
):
    raw = (
        '{"storage_config":{"type":"sqlite","db_path":"/tmp/source.sqlite"},'
        + geometry
        + ',"profile_extractor_config":{"extraction_definition_prompt":"Custom prompt","extension":{"fraction":0.12345678901234567890123456789}},"opaque":{"scaled":1.00,"huge":1e1000}}'
    ).encode()
    path = write_source(storage, raw)
    before = path.stat()
    monkeypatch.setattr(
        storage,
        "_save_config_to_local_dir",
        Mock(side_effect=AssertionError("reading must not publish")),
    )
    if '"batch_size"' in geometry:
        with pytest.raises(StoredConfigurationError):
            storage.load_config()
        assert path.read_bytes() == raw
        return
    loaded = storage.load_config()
    assert (loaded.window_size, loaded.stride_size) == (23, 7)
    assert (
        loaded.profile_extractor_config.extraction_definition_prompt == "Custom prompt"
    )
    assert path.read_bytes() == raw
    after = path.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_mode) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_mode,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"private":"secret-sentinel",',
        b"",
        b"null",
        b"[]",
        b'{"opaque":NaN}',
        b'{"opaque":1e9999999999999999999999999999999}',
        b'{"window_size":0}',
        b'{"storage_config":{"type":"disk"}}',
        b"\xff",
    ],
    ids=[
        "malformed",
        "empty",
        "null",
        "array",
        "nonfinite",
        "exponent",
        "invalid-policy",
        "unsupported-backend",
        "encoding",
    ],
)
def test_unreadable_existing_file_never_becomes_defaults_or_writes(
    storage, monkeypatch, raw
):
    path = write_source(storage, raw)
    defaults = Mock(
        side_effect=AssertionError("existing configuration must not use defaults")
    )
    publish = Mock(side_effect=AssertionError("invalid configuration must not publish"))
    monkeypatch.setattr(storage, "get_default_config", defaults)
    monkeypatch.setattr(storage, "_save_config_to_local_dir", publish)
    with pytest.raises(StoredConfigurationError) as error:
        storage.load_config()
    assert "secret-sentinel" not in str(error.value)
    assert path.read_bytes() == raw
    defaults.assert_not_called()
    publish.assert_not_called()


def test_read_io_error_propagates_without_initializing(storage, monkeypatch):
    path = write_source(storage, b'{"storage_config":null}')
    failure = PermissionError("blocked-read")
    defaults = Mock(side_effect=AssertionError("not missing"))
    monkeypatch.setattr(storage, "get_default_config", defaults)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", Mock(side_effect=failure))
        with pytest.raises(PermissionError) as error:
            storage.load_config()
        assert error.value is failure
    assert path.read_bytes() == b'{"storage_config":null}'
    defaults.assert_not_called()


@pytest.mark.parametrize(
    "winner",
    [
        b'{"storage_config":null,"window_size":31,"stride_size":7,"agent_context_prompt":"concurrent"}',
        b'{"window_size":0}',
    ],
    ids=["valid-winner", "invalid-winner"],
)
def test_initialization_loads_concurrent_winner_without_replacing_it(
    storage, monkeypatch, winner
):
    original_link = os.link
    calls = []

    def competing_link(source, destination):
        calls.append(destination)
        Path(destination).write_bytes(winner)
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    if b'"window_size":0' in winner:
        with pytest.raises(StoredConfigurationError):
            storage.load_config()
    else:
        config = storage.load_config()
        assert (config.window_size, config.stride_size) == (31, 7)
        assert config.agent_context_prompt == "concurrent"
    assert len(calls) == 1
    assert Path(storage.config_file).read_bytes() == winner
    assert not list(Path(storage.base_dir).glob("*.tmp"))


def test_initialization_failure_preserves_primary_error_and_cleans_temp(
    storage, monkeypatch
):
    failure = OSError(errno.ENOTSUP, "exclusive publication unsupported")
    monkeypatch.setattr(os, "link", Mock(side_effect=failure))
    with pytest.raises(OSError) as error:
        storage.load_config()
    assert error.value is failure
    assert not Path(storage.config_file).exists()
    assert not list(Path(storage.base_dir).glob("*.tmp"))


def test_deleted_winner_does_not_restart_default_publication(storage, monkeypatch):
    original_link = os.link
    calls = []

    def deleted_winner(source, destination):
        calls.append(destination)
        Path(destination).write_text('{"storage_config":null}')
        try:
            return original_link(source, destination)
        finally:
            Path(destination).unlink()

    monkeypatch.setattr(os, "link", deleted_winner)
    with pytest.raises(FileNotFoundError):
        storage.load_config()
    assert len(calls) == 1
    assert not Path(storage.config_file).exists()


def test_real_offline_upgrade_retains_original_and_canonical_load_is_read_only(
    storage, tmp_path
):
    source = tmp_path / "retained-original.json"
    original = b'{"storage_config":{"type":"sqlite","db_path":"/tmp/original.sqlite"},"batch_size":23,"batch_interval":7,"opaque":{"fraction":0.12345678901234567890123456789,"scaled":1.00}}'
    source.write_bytes(original)
    source.chmod(0o640)
    target = Path(storage.config_file)
    target.parent.mkdir(parents=True)
    result = CliRunner().invoke(
        create_app(),
        ["--json", "config", "upgrade-file", str(source), "--output", str(target)],
    )
    assert result.exit_code == 0, result.output
    canonical = target.read_bytes()
    before = target.stat()
    config = storage.load_config()
    assert (config.window_size, config.stride_size) == (23, 7)
    assert source.read_bytes() == original and source.stat().st_mode & 0o777 == 0o640
    assert target.read_bytes() == canonical
    after = target.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_mode) == (
        after.st_ino,
        after.st_mtime_ns,
        after.st_mode,
    )
    assert (
        decode_config_document(canonical.decode())["opaque"]
        == decode_config_document(original.decode())["opaque"]
    )
