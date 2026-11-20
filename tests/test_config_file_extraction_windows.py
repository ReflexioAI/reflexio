"""Real file preparation, originals, current rejection and storage admission."""

import hashlib
from decimal import Decimal
from unittest.mock import Mock

import pytest

from reflexio.config_document import decode_config_document, encode_config_document
from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import (
    EXTRACTION_WINDOW_POLICY_VERSION,
    FILE_FORMAT_VERSION,
    LocalConfigFile,
)
from reflexio.extraction_window_migration import EXTRACTION_WINDOW_FIELDS
from reflexio.models.config_schema import validate_stored_config


def _historical_document(version):
    payload = {
        "storage_config": None,
        "window_size": 12,
        "stride_size": 4,
        "profile_extractor_config": {
            "extraction_definition_prompt": "Synthetic profile policy",
            "window_size_override": 16,
            "stride_size_override": 3,
        },
        "user_playbook_extractor_config": None,
        "opaque": {"exact": Decimal("1.234567890123456789"), "huge": Decimal("1e1000")},
    }
    if version:
        payload["_reflexio_config_file"] = {"version": version}
    return payload


@pytest.mark.parametrize("version", [0, 1, 10, 14])
def test_preparation_keeps_original_and_admits_storage_before_generation15(
    tmp_path, version
):
    path = tmp_path / "config.json"
    original = encode_config_document(_historical_document(version)).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    with pytest.raises(StoredConfigurationError):
        owner.read_document()
    assert path.read_bytes() == original
    admission = Mock()
    assert owner.prepare(admit=admission)
    admission.assert_called_once()
    logical = owner.read_document()
    assert {name: logical[name] for name in EXTRACTION_WINDOW_FIELDS} == {
        "profile_window_size": 16,
        "profile_stride_size": 3,
        "playbook_window_size": 12,
        "playbook_stride_size": 4,
    }
    assert logical["user_playbook_extractor_config"] is None
    assert logical["opaque"] == _historical_document(version)["opaque"]
    validate_stored_config(logical)
    digest = hashlib.sha256(original).hexdigest()
    assert (
        tmp_path / ".config-migrations/config.json" / f"{digest}.json"
    ).read_bytes() == original
    assert decode_config_document(path.read_text())["_reflexio_config_file"] == {
        "version": FILE_FORMAT_VERSION,
        "original_sha256": digest,
    }
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert owner.prepare(admit=admission) is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    admission.assert_called_once()
    assert EXTRACTION_WINDOW_POLICY_VERSION == 15


def test_admission_failure_preserves_the_original_file_without_backup(tmp_path):
    path = tmp_path / "config.json"
    original = encode_config_document(_historical_document(14)).encode()
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match="held"):
        LocalConfigFile(path).prepare(admit=Mock(side_effect=RuntimeError("held")))
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


@pytest.mark.parametrize("name", sorted(EXTRACTION_WINDOW_FIELDS))
@pytest.mark.parametrize("value", [None, 20])
def test_ignored_new_fields_refuse_before_any_generation_or_original_write(
    tmp_path, name, value
):
    path = tmp_path / "config.json"
    payload = _historical_document(14)
    payload[name] = value
    original = encode_config_document(payload).encode()
    path.write_bytes(original)
    admission = Mock()
    with pytest.raises(StoredConfigurationError, match="ignored"):
        LocalConfigFile(path).prepare(admit=admission)
    admission.assert_not_called()
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()
    with pytest.raises(StoredConfigurationError, match="ignored"):
        LocalConfigFile(path).read_document_for_conversion()


def test_falsely_current_old_selectors_cannot_be_reconverted_or_loaded(tmp_path):
    path = tmp_path / "config.json"
    original = encode_config_document(_historical_document(15)).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    for operation in (
        owner.prepare,
        owner.read_document,
        owner.read_document_for_conversion,
    ):
        with pytest.raises(StoredConfigurationError):
            operation()
        assert path.read_bytes() == original


def test_old_format_cannot_be_relabelled_by_a_full_publication(tmp_path):
    path = tmp_path / "config.json"
    document = {
        "storage_config": None,
        "profile_window_size": 12,
        "profile_stride_size": 4,
        "_reflexio_config_file": {"version": 14},
    }
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).publish_document(encode_config_document(document))
    assert not path.exists()
