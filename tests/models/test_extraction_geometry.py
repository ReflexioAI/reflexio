"""Window/stride are the sole persisted extraction geometry authority."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.config_migration import (
    StoredConfigurationError,
    canonicalize_config_document,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import Config, validate_stored_config
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.mark.parametrize(
    "preset,geometry",
    [
        ("quick_chat", (5, 3)),
        ("standard", (10, 8)),
        ("long_form", (25, 10)),
        ("high_volume", (15, 8)),
        (None, (10, 8)),
    ],
)
@pytest.mark.parametrize("loader", [Config.model_validate, validate_stored_config])
def test_legacy_shortcuts_resolve_once_without_persisted_authority(
    preset, geometry, loader
):
    payload = {"storage_config": None, "extraction_preset": preset}
    before = deepcopy(payload)
    with pytest.raises(ValueError):
        loader(payload)
    config = loader(canonicalize_config_document(payload))
    assert (config.window_size, config.stride_size) == geometry
    assert payload == before
    dumped = config.model_dump(mode="json")
    assert "extraction_preset" not in dumped
    assert "extraction_preset" not in Config.model_json_schema()["properties"]
    assert Config.model_validate(dumped) == config


@pytest.mark.parametrize(
    "fields,geometry",
    [
        ({"window_size": 42}, (42, 10)),
        ({"stride_size": 3}, (25, 3)),
        ({"window_size": 42, "stride_size": 3}, (42, 3)),
        ({"window_size": None, "stride_size": None}, (25, 10)),
        ({"extraction_window_size": 42, "batch_interval": 3}, (42, 3)),
    ],
)
def test_explicit_geometry_and_old_field_aliases_keep_precedence(fields, geometry):
    config = Config.model_validate(
        canonicalize_config_document(
            {"storage_config": None, "extraction_preset": "long_form", **fields}
        )
    )
    assert (config.window_size, config.stride_size) == geometry


@pytest.mark.parametrize("preset", ["unknown", 42, {}, False])
def test_invalid_legacy_shortcuts_are_not_silently_dropped(preset):
    # Runtime constructors reject ``extraction_preset`` outright (#747); the
    # explicit upgrade must reject an invalid preset rather than drop it.
    payload = {
        "storage_config": None,
        "extraction_preset": "standard",
        "window_size": 42,
        "stride_size": 3,
    }
    assert "extraction_preset" not in canonicalize_config_document(payload)
    with pytest.raises(StoredConfigurationError):
        canonicalize_config_document({**payload, "extraction_preset": preset})


def test_raw_cleanup_preserves_extensions_and_removes_preset_before_upgrade():
    payload = {
        "storage_config": None,
        "extraction_preset": "long_form",
        "extension": {"opaque": [1, 2]},
    }
    before = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    assert payload == before
    assert cleaned == {
        "storage_config": None,
        "window_size": 25,
        "stride_size": 10,
        "extension": {"opaque": [1, 2]},
    }
    assert strip_retired_config_fields(cleaned) == cleaned


def test_real_config_file_load_patch_save_and_reload(tmp_path: Path):
    store = LocalFileConfigStorage("geometry", base_dir=str(tmp_path))
    path = Path(store.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"storage_config": None, "extraction_preset": "long_form"})
    )
    from reflexio.cli.config_upgrade import convert_config_file

    original = path.read_bytes()
    with pytest.raises(StoredConfigurationError):
        store.load_config()
    legacy = tmp_path / "legacy.json"
    path.rename(legacy)
    convert_config_file(legacy, path)
    assert legacy.read_bytes() == original
    configurator = DefaultConfigurator("geometry", base_dir=str(tmp_path))
    assert configurator.get_config().window_size == 25
    # Shallow patches already contain explicit geometry: an old label alone
    # must not gain new authority over it.
    before = path.read_bytes()
    with pytest.raises(ValueError):
        configurator.prepare_config_patch({"extraction_preset": "quick_chat"})
    with pytest.raises((ValueError, KeyError)):
        configurator.set_config_by_name("extraction_preset", "standard")
    assert path.read_bytes() == before
    assert (
        configurator.get_config().window_size,
        configurator.get_config().stride_size,
    ) == (25, 10)
    assert "extraction_preset" not in json.loads(path.read_text())
    reloaded = store.load_config()
    assert reloaded is not None
    assert (reloaded.window_size, reloaded.stride_size) == (25, 10)
    configurator.set_config(
        configurator.prepare_config_patch({"window_size": 31, "stride_size": 7})
    )
    reloaded = store.load_config()
    assert reloaded is not None
    assert (reloaded.window_size, reloaded.stride_size) == (31, 7)
