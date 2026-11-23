"""Historical profile mode conversion cannot grant automatic extraction."""

from copy import deepcopy

import pytest

from reflexio.config_env import ConfigurationMigrationError
from reflexio.config_migration import retire_profile_manual_trigger

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("manual", [True, "true", 1])
@pytest.mark.parametrize("stride", [None, 3, "11"])
def test_manual_only_remains_off_without_replacing_unrelated_policy(manual, stride):
    original = {
        "profile_stride_size": stride,
        "profile_window_size": 23,
        "profile_extractor_config": {
            "manual_trigger": manual,
            "extraction_definition_prompt": "customer prompt",
            "opaque": {"value": ["preserve"]},
        },
        "playbook_stride_size": 7,
        "extension": {"value": ["preserve"]},
    }
    before = deepcopy(original)
    converted = retire_profile_manual_trigger(original)
    assert converted["profile_stride_size"] == 0
    assert "manual_trigger" not in converted["profile_extractor_config"]
    assert original == before
    assert converted == {
        **before,
        "profile_stride_size": 0,
        "profile_extractor_config": {
            key: value
            for key, value in before["profile_extractor_config"].items()
            if key != "manual_trigger"
        },
    }
    converted["extension"]["value"].append("changed")
    converted["profile_extractor_config"]["opaque"]["value"].append("changed")
    assert original == before


@pytest.mark.parametrize(
    "extractor", [None, {}, {"manual_trigger": False}, {"manual_trigger": "false"}]
)
def test_auto_or_absent_extractor_preserves_sparse_custom_cadence(extractor):
    original = {"profile_extractor_config": extractor, "profile_stride_size": 13}
    converted = retire_profile_manual_trigger(original)
    assert converted["profile_stride_size"] == 13
    if isinstance(extractor, dict):
        assert converted["profile_extractor_config"] == {}
    else:
        assert converted["profile_extractor_config"] is None
    sparse = retire_profile_manual_trigger({})
    assert sparse == {}


@pytest.mark.parametrize("manual", [True, False])
@pytest.mark.parametrize("stride", [0, -1, "invalid", [], 1.5])
def test_invalid_dormant_stride_is_not_laundered_by_conversion(manual, stride):
    original = {
        "profile_stride_size": stride,
        "profile_extractor_config": {"manual_trigger": manual},
    }
    before = deepcopy(original)
    with pytest.raises(ConfigurationMigrationError, match="profile_stride_size"):
        retire_profile_manual_trigger(original)
    assert original == before


@pytest.mark.parametrize("manual", [None, "invalid", [], {}])
def test_invalid_historical_switch_refuses_without_exposing_values(manual):
    original = {"profile_extractor_config": {"manual_trigger": manual}}
    before = deepcopy(original)
    with pytest.raises(ConfigurationMigrationError) as error:
        retire_profile_manual_trigger(original)
    assert (
        str(error.value) == "Invalid obsolete profile_extractor_config.manual_trigger"
    )
    assert original == before


@pytest.mark.parametrize("extractor", [[], "invalid", True])
def test_invalid_extractor_shape_is_not_repaired(extractor):
    with pytest.raises(ConfigurationMigrationError, match="profile_extractor_config"):
        retire_profile_manual_trigger({"profile_extractor_config": extractor})


def test_missing_stride_does_not_materialize_a_positive_default():
    assert retire_profile_manual_trigger(
        {"profile_extractor_config": {"manual_trigger": True}}
    ) == {
        "profile_extractor_config": {},
        "profile_stride_size": 0,
    }
    assert retire_profile_manual_trigger(
        {"profile_extractor_config": {"manual_trigger": False}}
    ) == {
        "profile_extractor_config": {},
    }
