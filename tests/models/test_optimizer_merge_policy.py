"""Legacy merge switches normalize to a single budget without enabling work."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    PlaybookOptimizerConfig,
    strip_retired_config_fields,
)
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ({}, 5),
        ({"use_merge": True}, 5),
        ({"use_merge": False}, 0),
        ({"use_merge": False, "max_merge_invocations": 9}, 0),
        ({"use_merge": True, "max_merge_invocations": 9}, 9),
        ({"use_merge": True, "max_merge_invocations": 0}, 0),
        ({"use_merge": "off", "max_merge_invocations": 9}, 0),
        ({"max_merge_invocations": 0}, 0),
        ({"max_merge_invocations": 9}, 9),
    ],
)
def test_legacy_merge_policy_round_trip(legacy, expected):
    before = deepcopy(legacy)
    config = PlaybookOptimizerConfig.model_validate(legacy)
    assert config.max_merge_invocations == expected
    assert "use_merge" not in config.model_dump()
    assert "use_merge" not in config.model_json_schema()["properties"]
    assert (
        PlaybookOptimizerConfig.model_validate(
            config.model_dump()
        ).max_merge_invocations
        == expected
    )
    assert legacy == before


@pytest.mark.parametrize(
    "legacy",
    [
        {"use_merge": "invalid"},
        {"use_merge": None},
        {"use_merge": False, "max_merge_invocations": -1},
        {"use_merge": False, "max_merge_invocations": "invalid"},
        {"max_merge_invocations": -1},
    ],
)
def test_retirement_does_not_hide_invalid_policy(legacy):
    with pytest.raises(ValidationError):
        PlaybookOptimizerConfig.model_validate(legacy)


def test_raw_cleanup_preserves_inactive_policy_and_extensions():
    payload = {
        "playbook_optimizer_config": {
            "use_merge": False,
            "max_merge_invocations": 9,
            "opaque": {"use_merge": "unrelated"},
        },
        "extension": {"use_merge": True},
    }
    before = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    assert cleaned["playbook_optimizer_config"] == {
        "max_merge_invocations": 0,
        "opaque": {"use_merge": "unrelated"},
    }
    assert cleaned["extension"] == {"use_merge": True}
    assert payload == before


def test_local_file_canonicalization_never_enables_legacy_disabled_merges(tmp_path):
    storage = LocalFileConfigStorage("merge-policy", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "playbook_optimizer_config": {
                    "use_merge": False,
                    "max_merge_invocations": 9,
                },
            }
        )
    )
    for _ in range(2):
        assert (
            storage.load_config().playbook_optimizer_config.max_merge_invocations == 0
        )
    saved = json.loads(path.read_text())["playbook_optimizer_config"]
    assert saved["max_merge_invocations"] == 0
    assert "use_merge" not in saved


@pytest.mark.parametrize("enabled,limit", [(True, 0), (False, 9), (True, 9)])
def test_real_gepa_merge_admission_matches_legacy_policy(enabled, limit):
    from unittest.mock import Mock

    from gepa.core.data_loader import ListDataLoader
    from gepa.proposer.merge import MergeProposer

    config = PlaybookOptimizerConfig.model_validate(
        {"use_merge": enabled, "max_merge_invocations": limit}
    )
    evaluator = Mock(side_effect=AssertionError("admission must not invoke a model"))
    legacy = MergeProposer(
        logger=Mock(),
        valset=ListDataLoader([]),
        evaluator=evaluator,
        use_merge=enabled,
        max_merge_invocations=limit,
    )
    canonical = MergeProposer(
        logger=Mock(),
        valset=ListDataLoader([]),
        evaluator=evaluator,
        use_merge=config.max_merge_invocations > 0,
        max_merge_invocations=config.max_merge_invocations,
    )
    legacy.schedule_if_needed()
    canonical.schedule_if_needed()
    assert legacy.merges_due == canonical.merges_due
    evaluator.assert_not_called()
