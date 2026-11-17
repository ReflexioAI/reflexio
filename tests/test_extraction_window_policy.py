"""The current schema has four independent policies and no legacy selectors."""

from copy import deepcopy

import pytest

from reflexio.extraction_window_migration import (
    convert_canonical_legacy_extraction_windows,
)
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
    validate_stored_config,
)
from reflexio.models.extraction_window import get_extractor_window_params


def test_current_pairs_are_independent_and_keep_documented_defaults():
    config = Config(storage_config=None)
    assert get_extractor_window_params(config, "profile") == (
        10,
        8,
    )
    assert get_extractor_window_params(config, "playbook") == (
        10,
        8,
    )
    config = Config(
        storage_config=None,
        profile_window_size=16,
        profile_stride_size=4,
        playbook_window_size=24,
        playbook_stride_size=2,
    )
    assert get_extractor_window_params(config, "profile") == (
        16,
        4,
    )
    assert get_extractor_window_params(config, "playbook") == (
        24,
        2,
    )


@pytest.mark.parametrize("name", ["window_size", "stride_size"])
@pytest.mark.parametrize("value", [None, 16])
def test_old_global_names_refuse_in_current_and_stored_readers(name, value):
    for reader in (Config.model_validate, validate_stored_config):
        with pytest.raises(ValueError):
            reader({"storage_config": None, name: value})


@pytest.mark.parametrize(
    "field,model,prompt",
    [
        (
            "profile_extractor_config",
            ProfileExtractorConfig,
            "extraction_definition_prompt",
        ),
        (
            "user_playbook_extractor_config",
            UserPlaybookExtractorConfig,
            "extraction_definition_prompt",
        ),
    ],
)
@pytest.mark.parametrize("name", ["window_size_override", "stride_size_override"])
@pytest.mark.parametrize("value", [None, 4])
def test_old_per_kind_names_are_rejected_even_when_null(
    field, model, prompt, name, value
):
    with pytest.raises(ValueError):
        model.model_validate({prompt: "Synthetic extraction policy", name: value})
    with pytest.raises(ValueError):
        validate_stored_config(
            {
                "storage_config": None,
                field: {prompt: "Synthetic extraction policy", name: value},
            }
        )


def test_historical_projection_preserves_effective_current_policy_and_source():
    source = {
        "storage_config": None,
        "window_size": 12,
        "stride_size": 4,
        "profile_extractor_config": {
            "extraction_definition_prompt": "Synthetic profile policy",
            "window_size_override": 16,
            "stride_size_override": 3,
        },
        "user_playbook_extractor_config": None,
    }
    before = deepcopy(source)
    current = Config.model_validate(convert_canonical_legacy_extraction_windows(source))
    assert source == before
    assert get_extractor_window_params(current, "profile") == (
        16,
        3,
    )
    assert current.user_playbook_extractor_config is None
    assert (current.playbook_window_size, current.playbook_stride_size) == (12, 4)
    dumped = current.model_dump()
    assert "window_size" not in dumped and "stride_size" not in dumped
    assert "window_size_override" not in dumped["profile_extractor_config"]
    assert "stride_size_override" not in dumped["profile_extractor_config"]


@pytest.mark.parametrize("manual_trigger", [False, True, None])
def test_persisted_policy_does_not_enroll_manual_or_filtered_work_in_automatic_cadence(
    manual_trigger,
):
    extractor = (
        None
        if manual_trigger is None
        else ProfileExtractorConfig(
            extraction_definition_prompt="Synthetic profile policy",
            manual_trigger=manual_trigger,
        )
    )
    config = Config(
        storage_config=None,
        profile_extractor_config=extractor,
        profile_window_size=4,
        profile_stride_size=8,
    )
    window = get_extractor_window_params(config, "profile")
    assert window == (4, 8)
    with pytest.raises(ValueError):
        window.require_automatic()


def test_historical_small_override_remains_readable_and_persistable(tmp_path):
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    historical = {
        "storage_config": None,
        "window_size": 42,
        "stride_size": 3,
        "profile_extractor_config": {
            "extraction_definition_prompt": "Synthetic manual collection",
            "window_size_override": 2,
            "stride_size_override": 8,
            "manual_trigger": False,
        },
    }
    current = Config.model_validate(
        convert_canonical_legacy_extraction_windows(historical)
    )
    storage = LocalFileConfigStorage("small-window", base_dir=str(tmp_path))
    storage.save_config(current)
    loaded = storage.load_config()
    assert loaded is not None and loaded.profile_extractor_config is not None
    assert loaded.profile_extractor_config.manual_trigger is False
    assert get_extractor_window_params(loaded, "profile") == (2, 8)
    assert get_extractor_window_params(loaded, "playbook") == (42, 3)
    with pytest.raises(ValueError):
        get_extractor_window_params(loaded, "profile").require_automatic()
