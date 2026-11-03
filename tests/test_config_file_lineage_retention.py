"""Saved lineage consent survives skipped-generation upgrades without aliases."""

import hashlib
import json

import pytest

from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile
from reflexio.models.config_schema import validate_stored_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", range(12))
@pytest.mark.parametrize("enabled,days,expected", [(False, 37, 0), (True, 37, 37)])
def test_historical_hold_or_custom_grace_survives_preparation(
    tmp_path, version, enabled, days, expected
):
    path = tmp_path / "config.json"
    payload = {
        "storage_config": None,
        "lineage_gc": {"enabled": enabled, "tombstone_grace_window_days": days},
        "opaque": {"keep": "metadata"},
    }
    if version:
        payload["_reflexio_config_file"] = {"version": version}
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    assert owner.prepare() is True
    document = owner.read_document()
    assert document["lineage_gc"] == {"tombstone_grace_window_days": expected}
    assert document["opaque"] == payload["opaque"]
    assert (
        validate_stored_config(document).lineage_gc.tombstone_grace_window_days
        == expected
    )
    digest = hashlib.sha256(original).hexdigest()
    assert (
        tmp_path / ".config-migrations/config.json" / f"{digest}.json"
    ).read_bytes() == original
    prepared = path.read_bytes()
    assert (
        json.loads(prepared)["_reflexio_config_file"]["version"] == FILE_FORMAT_VERSION
    )
    assert owner.prepare() is False
    assert path.read_bytes() == prepared


@pytest.mark.parametrize(
    "policy",
    [
        {"enabled": False, "tombstone_grace_window_days": 0},
        {"enabled": False, "tombstone_grace_window_days": -1},
        {"enabled": "private-invalid-value", "tombstone_grace_window_days": 37},
        {"tombstone_grace_window_days": 0},
        {"tombstone_grace_window_days": "0"},
        {"tombstone_grace_window_days": False},
    ],
)
def test_invalid_old_policy_preserves_original_and_creates_no_backup(tmp_path, policy):
    path = tmp_path / "config.json"
    original = json.dumps({"storage_config": None, "lineage_gc": policy}).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError) as error:
        LocalConfigFile(path).prepare()
    assert "private-invalid-value" not in str(error.value)
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


def test_current_generation_cannot_replay_legacy_enable_conversion(tmp_path):
    path = tmp_path / "config.json"
    original = json.dumps(
        {
            "storage_config": None,
            "lineage_gc": {"enabled": False},
            "_reflexio_config_file": {"version": FILE_FORMAT_VERSION},
        }
    ).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original


def test_generation_11_reader_refuses_new_retention_format(tmp_path):
    import re
    import runpy
    from pathlib import Path

    from reflexio.config_document import decode_config_document

    decoder = runpy.run_path(
        str(Path(__file__).parent / "fixtures/config_file_v11_reader.txt"),
        init_globals={
            "FILE_FORMAT_VERSION": 11,
            "_METADATA_KEY": "_reflexio_config_file",
            "decode_config_document": decode_config_document,
            "StoredConfigurationError": StoredConfigurationError,
            "re": re,
        },
    )["_decode"]
    path = tmp_path / "config.json"
    original = b'{"_reflexio_config_file":{"version":11},"storage_config":null,"lineage_gc":{"enabled":false,"tombstone_grace_window_days":37}}'
    assert decoder(original)[1]["version"] == 11
    path.write_bytes(original)
    assert LocalConfigFile(path).prepare()
    with pytest.raises(
        StoredConfigurationError,
        match="Unrecognized local configuration file generation",
    ):
        decoder(path.read_bytes())
