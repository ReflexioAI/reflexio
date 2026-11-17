"""Historical policy projection, without granting startup/write authority."""

from copy import deepcopy
from decimal import Decimal
from itertools import product
from typing import Any

import pytest

from reflexio.config_env import ConfigurationMigrationError
from reflexio.config_migration import canonicalize_stored_config_fields
from reflexio.extraction_window_migration import (
    EXTRACTION_WINDOW_FIELDS,
    convert_canonical_legacy_extraction_windows,
    convert_canonical_legacy_project_windows,
)


def test_effective_policy_and_opaque_originals_are_preserved():
    original = {
        "window_size": 12,
        "stride_size": 4,
        "profile_extractor_config": {
            "window_size_override": 16,
            "stride_size_override": None,
            "extraction_definition_prompt": "private prompt",
        },
        "user_playbook_extractor_config": None,
        "opaque": {"exact": Decimal("1.234567890123456789"), "huge": Decimal("1e1000")},
    }
    before = deepcopy(original)
    converted = convert_canonical_legacy_extraction_windows(original)
    assert original == before
    assert {name: converted[name] for name in EXTRACTION_WINDOW_FIELDS} == {
        "profile_window_size": 16,
        "profile_stride_size": 4,
        "playbook_window_size": 12,
        "playbook_stride_size": 4,
    }
    assert "window_size" not in converted and "stride_size" not in converted
    assert converted["profile_extractor_config"] == {
        "extraction_definition_prompt": "private prompt"
    }
    assert converted["user_playbook_extractor_config"] is None
    assert converted["opaque"] == before["opaque"]
    converted["opaque"]["exact"] = 0
    assert original == before


@pytest.mark.parametrize("value", [None, "12", 12])
def test_global_null_and_scalar_coercion_follow_the_historical_schema(value):
    result = convert_canonical_legacy_extraction_windows(
        {"window_size": value, "stride_size": None}
    )
    width = 10 if value is None else 12
    assert result == {
        "profile_window_size": width,
        "profile_stride_size": 8,
        "playbook_window_size": width,
        "playbook_stride_size": 8,
    }


def test_sixteen_project_masks_and_four_overlay_shapes():
    for pw, ps, bw, bs, project_width, project_stride in product(
        [None, 16], [None, 3], [None, 20], [None, 2], [False, True], [False, True]
    ):
        parent = {
            "window_size": 12,
            "stride_size": 4,
            "profile_extractor_config": {
                "window_size_override": pw,
                "stride_size_override": ps,
            },
            "user_playbook_extractor_config": {
                "window_size_override": bw,
                "stride_size_override": bs,
            },
        }
        project: dict[str, Any] = {"opaque": {"exact": Decimal("1e1000")}}
        if project_width:
            project["window_size"] = 24
        if project_stride:
            project["stride_size"] = 2
        before = deepcopy((parent, project))
        masked = (
            project_width
            and (pw is not None or bw is not None)
            or project_stride
            and (ps is not None or bs is not None)
        )
        if masked:
            with pytest.raises(ConfigurationMigrationError, match="masked"):
                convert_canonical_legacy_project_windows(parent, project)
        else:
            output = convert_canonical_legacy_project_windows(parent, project)
            expected: dict[str, Any] = {"opaque": {"exact": Decimal("1e1000")}}
            if project_width:
                expected.update(profile_window_size=24, playbook_window_size=24)
            if project_stride:
                expected.update(profile_stride_size=2, playbook_stride_size=2)
            assert output == expected
        assert (parent, project) == before


def test_empty_equal_and_null_overlays_keep_their_distinct_intent():
    parent = {"window_size": 12, "stride_size": 4, "profile_extractor_config": None}
    assert convert_canonical_legacy_project_windows(parent, {}) == {}
    assert convert_canonical_legacy_project_windows(parent, {"window_size": 12}) == {
        "profile_window_size": 12,
        "playbook_window_size": 12,
    }
    assert convert_canonical_legacy_project_windows(parent, {"window_size": None}) == {
        "profile_window_size": 10,
        "playbook_window_size": 10,
    }


def test_ignored_project_metadata_is_not_validated_as_workspace_policy():
    project = {"profile_extractor_config": {"window_size_override": "private-canary"}}
    assert convert_canonical_legacy_project_windows({}, project) == project


@pytest.mark.parametrize("name", sorted(EXTRACTION_WINDOW_FIELDS))
def test_ignored_future_fields_cannot_acquire_policy_authority(name):
    with pytest.raises(ConfigurationMigrationError) as org_error:
        convert_canonical_legacy_extraction_windows({name: "private-canary"})
    with pytest.raises(ConfigurationMigrationError) as project_error:
        convert_canonical_legacy_project_windows({}, {name: "private-canary"})
    assert "private-canary" not in str(org_error.value) + str(project_error.value)


@pytest.mark.parametrize(
    "raw",
    [
        {"batch_size": 16},
        {"extraction_preset": "long_form"},
        {"profile_extractor_configs": [{"window_size_override": 16}]},
        {"profile_extractor_config": {"batch_size_override": 16}},
    ],
)
def test_skipping_raw_name_conversion_refuses_instead_of_selecting_defaults(raw):
    with pytest.raises(ConfigurationMigrationError):
        convert_canonical_legacy_extraction_windows(raw)
    converted = convert_canonical_legacy_extraction_windows(
        canonicalize_stored_config_fields(raw)
    )
    assert EXTRACTION_WINDOW_FIELDS.issubset(converted)


@pytest.mark.parametrize(
    "policy",
    [
        {"window_size": "private-canary"},
        {"window_size": 0},
        {"window_size": 4, "stride_size": 8},
        {"profile_extractor_config": {"stride_size_override": -1}},
    ],
)
def test_invalid_historical_geometry_refuses_without_values(policy):
    with pytest.raises(ConfigurationMigrationError) as error:
        convert_canonical_legacy_extraction_windows(policy)
    assert "private-canary" not in str(error.value)
