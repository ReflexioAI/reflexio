"""Canonical activation preserves legacy opt-in and frozen policy data."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.config_migration import (
    StoredConfigurationError,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.mark.parametrize("optimize", [None, False, True])
@pytest.mark.parametrize("adopt", [None, False, True])
def test_legacy_user_activation_truth_table_and_roundtrip(optimize, adopt):
    legacy = {"enabled": True}
    if optimize is not None:
        legacy["optimize_user_playbooks"] = optimize
    if adopt is not None:
        legacy["auto_update_user_playbooks"] = adopt
    before = deepcopy(legacy)
    config = _upgrade_policy(legacy)
    expected = bool(optimize and adopt)
    assert config.user_playbooks_enabled is expected
    assert config.allows_target("user_playbook") is expected
    saved = config.model_dump()
    for name in ("optimize_user_playbooks", "auto_update_user_playbooks"):
        assert name not in saved
        assert name not in config.model_json_schema()["properties"]
    assert _upgrade_policy(saved).allows_target("user_playbook") is expected
    assert legacy == before


@pytest.mark.parametrize("canonical", [False, True])
def test_explicit_canonical_field_wins_over_valid_legacy_values(canonical):
    config = _upgrade_policy(
        {
            "user_playbooks_enabled": canonical,
            "optimize_user_playbooks": not canonical,
            "auto_update_user_playbooks": not canonical,
        }
    )
    assert config.user_playbooks_enabled is canonical


@pytest.mark.parametrize(
    "key", ["optimize_user_playbooks", "auto_update_user_playbooks"]
)
@pytest.mark.parametrize("invalid", [None, "invalid"])
def test_inactive_or_superseded_legacy_switches_still_validate(key, invalid):
    legacy: dict[str, object] = {
        "optimize_user_playbooks": False,
        "auto_update_user_playbooks": False,
        "user_playbooks_enabled": False,
    }
    # Runtime constructors reject these names outright (#747), so the
    # boolean validation this guards now lives in the explicit upgrade.
    assert _upgrade_policy(legacy).user_playbooks_enabled is False
    legacy[key] = invalid
    with pytest.raises(StoredConfigurationError):
        _upgrade_policy(legacy)


@pytest.mark.parametrize(
    "optimize,adopt,expected",
    [("true", "on", True), ("true", "off", False), ("0", 1, False)],
)
def test_legacy_boolean_coercion_is_preserved(optimize, adopt, expected):
    config = _upgrade_policy(
        {"optimize_user_playbooks": optimize, "auto_update_user_playbooks": adopt}
    )
    assert config.user_playbooks_enabled is expected


def test_raw_normalization_does_not_rewrite_historical_authority():
    payload = {
        "playbook_optimizer_config": {
            "optimize_user_playbooks": True,
            "auto_update_user_playbooks": False,
        },
        "custom_extension": {"adoption_policy": {"auto_update_user_playbooks": True}},
    }
    before = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    assert cleaned["playbook_optimizer_config"] == {"user_playbooks_enabled": False}
    assert cleaned["custom_extension"] == before["custom_extension"]
    assert payload == before


@pytest.mark.parametrize("optimize,adopt", [(True, True), (True, False), (False, True)])
def test_real_local_file_upgrade_preserves_effective_activation(
    tmp_path, optimize, adopt
):
    storage = LocalFileConfigStorage("target-policy", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "playbook_optimizer_config": {
                    "enabled": True,
                    "optimize_user_playbooks": optimize,
                    "auto_update_user_playbooks": adopt,
                },
            }
        )
    )
    original = path.read_bytes()
    from reflexio.config_migration import StoredConfigurationError

    with pytest.raises(StoredConfigurationError):
        storage.load_config()
    assert path.read_bytes() == original
    from reflexio.cli.config_upgrade import convert_config_file

    canonical = tmp_path / "canonical.json"
    convert_config_file(path, canonical)
    storage.config_file = str(canonical)
    for _ in range(2):
        assert storage.load_config().playbook_optimizer_config.allows_target(
            "user_playbook"
        ) is (optimize and adopt)
    assert path.read_bytes() == original
    saved = json.loads(canonical.read_text())["playbook_optimizer_config"]
    assert saved["user_playbooks_enabled"] is (optimize and adopt)
    assert "optimize_user_playbooks" not in saved
    assert "auto_update_user_playbooks" not in saved


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("adopt", [False, True])
def test_agent_adoption_veto_remains_independent(enabled, optimize, adopt):
    config = _upgrade_policy(
        {
            "enabled": enabled,
            "optimize_agent_playbooks": optimize,
            "auto_update_pending_agent_playbooks": adopt,
            "user_playbooks_enabled": True,
        }
    )
    assert config.allows_target("agent_playbook") is (enabled and optimize and adopt)
    assert config.allows_target("user_playbook") is enabled


def _upgrade_policy(legacy):
    from reflexio.config_migration import canonicalize_config_document

    current = canonicalize_config_document(
        {"storage_config": None, "playbook_optimizer_config": legacy}
    )
    return PlaybookOptimizerConfig.model_validate(current["playbook_optimizer_config"])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("optimize", [None, False, True])
@pytest.mark.parametrize("adopt", [None, False, True])
def test_legacy_agent_upgrade_preserves_effective_activation(enabled, optimize, adopt):
    legacy = {"enabled": enabled, "user_playbooks_enabled": True}
    if optimize is not None:
        legacy["optimize_agent_playbooks"] = optimize
    if adopt is not None:
        legacy["auto_update_pending_agent_playbooks"] = adopt
    before = deepcopy(legacy)
    config = _upgrade_policy(legacy)
    expected = bool(optimize and (True if adopt is None else adopt))
    assert config.agent_playbooks_enabled is expected
    assert config.allows_target("agent_playbook") is (enabled and expected)
    assert config.allows_target("user_playbook") is enabled
    assert _upgrade_policy(config.model_dump()).model_dump() == config.model_dump()
    assert legacy == before
    for key in ("optimize_agent_playbooks", "auto_update_pending_agent_playbooks"):
        assert key not in config.model_dump()
        assert key not in config.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "key", ["optimize_agent_playbooks", "auto_update_pending_agent_playbooks"]
)
@pytest.mark.parametrize("value", [False, True, None, "private-invalid-value"])
def test_retired_agent_names_rejected_by_runtime_even_with_canonical_policy(key, value):
    with pytest.raises(ValueError, match=key) as error:
        PlaybookOptimizerConfig.model_validate(
            {"agent_playbooks_enabled": False, key: value}
        )
    assert "private-invalid-value" not in str(error.value)


@pytest.mark.parametrize(
    "key", ["optimize_agent_playbooks", "auto_update_pending_agent_playbooks"]
)
@pytest.mark.parametrize("value", [None, "private-invalid-value"])
def test_inactive_agent_upgrade_validates_legacy_policy_without_leaking_values(
    key, value
):
    with pytest.raises(StoredConfigurationError) as error:
        _upgrade_policy(
            {"enabled": False, "agent_playbooks_enabled": False, key: value}
        )
    assert "private-invalid-value" not in str(error.value)
