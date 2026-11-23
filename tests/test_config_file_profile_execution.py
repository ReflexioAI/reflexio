"""The existing file owner preserves originals before profile-policy upgrade."""

import hashlib
from unittest.mock import Mock

import pytest

from reflexio.config_document import encode_config_document
from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile
from reflexio.config_migration import canonicalize_config_document

pytestmark = pytest.mark.unit


def _historical(version, manual):
    result = {
        "storage_config": None,
        "profile_extractor_config": {
            "extraction_definition_prompt": "Synthetic profile policy",
            "manual_trigger": manual,
        },
        "opaque": {"customer": ["untouched"]},
    }
    if version < 15:
        result.update(window_size=23, stride_size=11)
    else:
        result.update(profile_window_size=23, profile_stride_size=11)
    if version:
        result["_reflexio_config_file"] = {"version": version}
    return result


@pytest.mark.parametrize("version", [0, 1, 10, 14, 15, 16])
@pytest.mark.parametrize("manual", [False, True])
def test_preparation_preserves_exact_original_and_mode_across_skipped_generations(
    tmp_path, version, manual
):
    path = tmp_path / "config.json"
    original = encode_config_document(_historical(version, manual)).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    with pytest.raises(StoredConfigurationError):
        owner.read_document()
    assert path.read_bytes() == original
    admit = Mock()
    assert owner.prepare(admit=admit)
    admit.assert_called_once()
    converted = owner.read_document()
    assert converted["profile_stride_size"] == (0 if manual else 11)
    assert converted["profile_window_size"] == 23
    assert "manual_trigger" not in converted["profile_extractor_config"]
    assert converted["opaque"] == {"customer": ["untouched"]}
    original_hash = hashlib.sha256(original).hexdigest()
    assert (
        tmp_path / ".config-migrations" / "config.json" / f"{original_hash}.json"
    ).read_bytes() == original
    prepared = path.read_bytes()
    assert not owner.prepare(admit=admit)
    assert path.read_bytes() == prepared
    assert admit.call_count == 1


@pytest.mark.parametrize("version", [0, 14, 15, 16])
@pytest.mark.parametrize("manual", [False, True])
def test_explicit_conversion_resolves_old_window_before_manual_mode(
    tmp_path, version, manual
):
    path = tmp_path / "config.json"
    original = encode_config_document(_historical(version, manual)).encode()
    path.write_bytes(original)
    converted = canonicalize_config_document(
        LocalConfigFile(path).read_document_for_conversion()
    )
    assert converted["profile_window_size"] == 23
    assert converted["profile_stride_size"] == (0 if manual else 11)
    assert path.read_bytes() == original


def test_current_zero_is_preserved_and_old_writer_envelopes_refuse(tmp_path):
    path = tmp_path / "config.json"
    owner = LocalConfigFile(path)
    current = {"storage_config": None, "profile_stride_size": 0}
    assert owner.publish_document(encode_config_document(current))
    assert owner.read_document()["profile_stride_size"] == 0
    before = path.read_bytes()
    with pytest.raises(StoredConfigurationError, match="generation17"):
        owner.publish_document(
            encode_config_document(
                {
                    **current,
                    "_reflexio_config_file": {"version": FILE_FORMAT_VERSION - 1},
                }
            )
        )
    assert path.read_bytes() == before


def test_old_zero_and_invalid_dormant_cadence_refuse_before_original_or_write(tmp_path):
    path = tmp_path / "config.json"
    for stride in (None, 0, -1, "invalid"):
        original = _historical(16, True)
        original["profile_stride_size"] = stride
        raw = encode_config_document(original).encode()
        path.write_bytes(raw)
        owner = LocalConfigFile(path)
        with pytest.raises(StoredConfigurationError):
            owner.prepare()
        with pytest.raises(StoredConfigurationError):
            owner.read_document_for_conversion()
        assert path.read_bytes() == raw
        assert not (tmp_path / ".config-migrations").exists()
