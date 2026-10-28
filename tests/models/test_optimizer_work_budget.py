"""One work budget admits optimization without reviving old disabled policy."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from reflexio.config_file import LocalConfigFile
from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import PlaybookOptimizerConfig


@pytest.mark.parametrize("budget", [0, 1, 37])
@pytest.mark.parametrize(
    "agent,user", [(False, False), (True, False), (False, True), (True, True)]
)
def test_work_budget_and_target_consent_are_both_required(budget, agent, user):
    policy = PlaybookOptimizerConfig.model_validate(
        {
            "metric_call_budget": budget,
            "agent_playbooks_enabled": agent,
            "user_playbooks_enabled": user,
        }
    )
    assert policy.allows_target("agent_playbook") is (budget > 0 and agent)
    assert policy.allows_target("user_playbook") is (budget > 0 and user)
    assert policy.metric_call_budget == budget


def test_default_is_disabled_and_old_switches_are_not_runtime_inputs():
    assert PlaybookOptimizerConfig().metric_call_budget == 0
    schema = PlaybookOptimizerConfig.model_json_schema()["properties"]
    for name, value in [
        ("enabled", False),
        ("enabled", True),
        ("max_metric_calls", 37),
    ]:
        assert name not in schema
        with pytest.raises(ValueError):
            PlaybookOptimizerConfig.model_validate({name: value})


@pytest.mark.parametrize("version", [0, 1, 2, 3, 4])
def test_startup_never_activates_previously_ignored_new_budget(tmp_path, version):
    path = tmp_path / "config.json"
    payload = {
        "storage_config": None,
        "playbook_optimizer_config": {
            "metric_call_budget": 37,
            "user_playbooks_enabled": True,
        },
    }
    if version:
        payload["_reflexio_config_file"] = {"version": version}
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="metric_call_budget"):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == raw
    assert not (tmp_path / ".config-migrations").exists()


@pytest.mark.parametrize("enabled", [None, False, True, "off", "on"])
@pytest.mark.parametrize("limit", [None, 1, 37])
def test_historical_conversion_preserves_effective_budget(enabled, limit):
    old = {"user_playbooks_enabled": True, "extension": {"enabled": "opaque"}}
    if enabled is not None:
        old["enabled"] = enabled
    if limit is not None:
        old["max_metric_calls"] = limit
    source = {"storage_config": None, "playbook_optimizer_config": old}
    before = deepcopy(source)
    # Empty/sparse policy remains absent/disabled. Only an actual old field is
    # translated, so applying the converter again cannot disable a new budget.
    converted = canonicalize_config_document(source)
    policy = PlaybookOptimizerConfig.model_validate(
        converted["playbook_optimizer_config"]
    )
    expected = (limit or 20) if enabled in (True, "on") else 0
    assert policy.metric_call_budget == expected
    assert policy.allows_target("user_playbook") is (expected > 0)
    assert converted["playbook_optimizer_config"]["extension"] == old["extension"]
    assert canonicalize_config_document(converted) == converted
    assert source == before


@pytest.mark.parametrize(
    "old",
    [
        {"enabled": "private-invalid"},
        {"enabled": None},
        {"enabled": False, "max_metric_calls": 0},
        {"enabled": False, "max_metric_calls": -1},
        {"enabled": False, "max_metric_calls": "private-invalid"},
        {"enabled": False, "metric_call_budget": 37},
        {"max_metric_calls": 37, "metric_call_budget": 0},
    ],
)
def test_conversion_rejects_invalid_or_conflicting_policy_without_values(old):
    source = {"storage_config": None, "playbook_optimizer_config": old}
    before = deepcopy(source)
    with pytest.raises(ValueError) as error:
        canonicalize_config_document(source)
    assert "private-invalid" not in str(error.value)
    assert source == before


@pytest.mark.parametrize("enabled,limit,expected", [(False, 37, 0), (True, 37, 37)])
def test_local_startup_prepares_previous_generation_once_without_losing_original(
    tmp_path, enabled, limit, expected
):
    path = tmp_path / "config.json"
    payload = {
        "storage_config": {"type": "sqlite"},
        "playbook_optimizer_config": {
            "enabled": enabled,
            "max_metric_calls": limit,
            "user_playbooks_enabled": True,
        },
        "_reflexio_config_file": {"version": 4},
        "extension": {"private": "retained"},
    }
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    owner = LocalConfigFile(path)
    assert owner.prepare() is True
    current = json.loads(path.read_bytes())
    assert current["_reflexio_config_file"]["version"] == 6
    assert current["playbook_optimizer_config"] == {
        "metric_call_budget": expected,
        "user_playbooks_enabled": True,
    }
    assert current["extension"] == payload["extension"]
    saved = path.read_bytes()
    assert owner.prepare() is False
    assert path.read_bytes() == saved
    backup = (
        tmp_path / ".config-migrations" / path.name / f"{sha256(raw).hexdigest()}.json"
    )
    assert backup.read_bytes() == raw
