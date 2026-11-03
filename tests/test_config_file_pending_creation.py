"""Skipped-generation preparation preserves consent and exact recovery bytes."""

import hashlib
import json

import pytest

from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile
from reflexio.models.config_schema import validate_stored_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", range(14))
def test_offline_copy_preserves_generation_owned_creation_consent(tmp_path, version):
    from typer.testing import CliRunner

    from reflexio.cli.app import create_app

    source, destination = tmp_path / "source.json", tmp_path / "current.json"
    document = {
        "storage_config": None,
        "pending_tool_call_config": {"pending_ttl_seconds": 37},
    }
    if version:
        document["_reflexio_config_file"] = {"version": version}
    original = json.dumps(document).encode()
    source.write_bytes(original)
    result = CliRunner().invoke(
        create_app(),
        ["config", "upgrade-file", str(source), "--output", str(destination)],
    )
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original
    converted = LocalConfigFile(destination).read_document()
    assert converted["pending_tool_call_config"]["pending_ttl_seconds"] == (
        37 if version == 13 else 0
    )
    assert (
        json.loads(destination.read_bytes())["_reflexio_config_file"]["version"]
        == FILE_FORMAT_VERSION
    )
    assert LocalConfigFile(destination).prepare() is False


@pytest.mark.parametrize("version", range(13))
@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, 37)])
def test_historical_creation_prepares_once_and_preserves_original(
    tmp_path, version, enabled, expected
):
    path = tmp_path / "config.json"
    source = {
        "storage_config": None,
        "pending_tool_call_config": {
            "enabled": enabled,
            "pending_ttl_seconds": 37,
            "tool_overrides": {"ask_human": {"pending_ttl_seconds": 53}},
        },
        "opaque": {"keep": [17, False]},
    }
    if version:
        source["_reflexio_config_file"] = {"version": version}
    original = json.dumps(source).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    assert owner.prepare() is True
    document = owner.read_document()
    assert document["pending_tool_call_config"]["pending_ttl_seconds"] == expected
    assert "enabled" not in document["pending_tool_call_config"]
    assert document["opaque"] == source["opaque"]
    assert validate_stored_config(document).pending_tool_call_config.for_tool(
        "ask_human"
    ).pending_ttl_seconds == (53 if enabled else 0)
    digest = hashlib.sha256(original).hexdigest()
    assert (
        tmp_path / ".config-migrations/config.json" / f"{digest}.json"
    ).read_bytes() == original
    prepared = path.read_bytes()
    assert (
        json.loads(prepared)["_reflexio_config_file"]["version"] == FILE_FORMAT_VERSION
    )
    assert owner.prepare() is False and path.read_bytes() == prepared


@pytest.mark.parametrize(
    "policy",
    [
        {"enabled": False, "pending_ttl_seconds": 0},
        {"enabled": True, "pending_ttl_seconds": -1},
        {"pending_ttl_seconds": 0},
        {"enabled": "PRIVATE_INVALID_SWITCH", "pending_ttl_seconds": 37},
    ],
)
def test_invalid_old_creation_keeps_original_without_backup(tmp_path, policy):
    path = tmp_path / "config.json"
    original = json.dumps(
        {"storage_config": None, "pending_tool_call_config": policy}
    ).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError) as error:
        LocalConfigFile(path).prepare()
    assert "PRIVATE_INVALID_SWITCH" not in str(error.value)
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


def test_current_format_does_not_replay_old_implicit_opt_out(tmp_path):
    from reflexio.config_file import encode_local_config_document

    path = tmp_path / "config.json"
    original = encode_local_config_document(
        {
            "storage_config": None,
            "pending_tool_call_config": {"pending_ttl_seconds": 37},
        }
    ).encode()
    path.write_bytes(original)
    assert LocalConfigFile(path).prepare() is False
    assert path.read_bytes() == original
    assert (
        LocalConfigFile(path).read_document()["pending_tool_call_config"][
            "pending_ttl_seconds"
        ]
        == 37
    )


def test_actual_generation_12_reader_refuses_new_creation_format(tmp_path):
    import re
    import runpy
    from pathlib import Path

    from reflexio.config_document import decode_config_document

    old_decode = runpy.run_path(
        str(Path(__file__).parent / "fixtures/config_file_v12_reader.txt"),
        init_globals={
            "FILE_FORMAT_VERSION": 12,
            "_METADATA_KEY": "_reflexio_config_file",
            "decode_config_document": decode_config_document,
            "StoredConfigurationError": StoredConfigurationError,
            "re": re,
        },
    )["_decode"]
    path = tmp_path / "config.json"
    original = b'{"_reflexio_config_file":{"version":12},"storage_config":null,"pending_tool_call_config":{"enabled":true,"pending_ttl_seconds":37}}'
    assert old_decode(original)[1]["version"] == 12
    path.write_bytes(original)
    assert LocalConfigFile(path).prepare() is True
    with pytest.raises(
        StoredConfigurationError,
        match="Unrecognized local configuration file generation",
    ):
        old_decode(path.read_bytes())
