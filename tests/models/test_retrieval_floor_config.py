import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    Config,
    RetrievalFloorConfig,
    StorageConfigSQLite,
)


def test_retrieval_floor_defaults():
    cfg = RetrievalFloorConfig()
    assert cfg.candidate_pool_size == 0
    assert cfg.profile_floor is None
    assert cfg.user_playbook_floor is None
    assert cfg.agent_playbook_floor is None


def test_config_has_retrieval_floor_default():
    cfg = Config(storage_config=StorageConfigSQLite())
    assert isinstance(cfg.retrieval_floor, RetrievalFloorConfig)
    assert cfg.retrieval_floor.candidate_pool_size == 0


def test_candidate_budget_must_be_nonnegative():
    with pytest.raises(ValidationError):
        RetrievalFloorConfig(candidate_pool_size=-1)


def test_retrieval_floor_null_falls_back_to_default():
    # A stored config row missing the column (or carrying an explicit null) must
    # fall back to the default rather than failing validation — same as the other
    # defaulted sub-configs stripped by the before-validator.
    base = Config(storage_config=StorageConfigSQLite())
    payload = base.model_dump()
    payload["retrieval_floor"] = None
    cfg = Config.model_validate(payload)
    assert cfg.retrieval_floor == RetrievalFloorConfig()


@pytest.mark.parametrize(
    "old",
    [
        {},
        {"enabled": False},
        {"pool_size": 73},
        {"enabled": False, "pool_size": 73},
        {"enabled": True},
        {"enabled": True, "pool_size": 73},
    ],
)
def test_explicit_conversion_preserves_effective_retrieval_policy(old):
    from reflexio.config_migration import canonicalize_config_document

    source = {
        "storage_config": None,
        "retrieval_floor": {
            **old,
            "profile_floor": None,
            "user_playbook_floor": 0.0,
            "agent_playbook_floor": -4.25,
            "opaque": {"keep": True},
        },
    }
    converted = canonicalize_config_document(source)
    floor = converted["retrieval_floor"]
    config = Config.model_validate(converted).retrieval_floor
    assert config.candidate_pool_size == (
        old.get("pool_size", 30) if old.get("enabled", False) else 0
    )
    assert (
        config.profile_floor,
        config.user_playbook_floor,
        config.agent_playbook_floor,
    ) == (None, 0.0, -4.25)
    assert floor["opaque"] == {"keep": True}
    assert "enabled" not in floor and "pool_size" not in floor
    assert source["retrieval_floor"] == {
        **old,
        "profile_floor": None,
        "user_playbook_floor": 0.0,
        "agent_playbook_floor": -4.25,
        "opaque": {"keep": True},
    }


@pytest.mark.parametrize(
    "old",
    [
        {"enabled": None},
        {"enabled": "private-invalid"},
        {"enabled": False, "pool_size": 0},
        {"enabled": False, "pool_size": None},
        {"enabled": False, "pool_size": -1},
        {"enabled": False, "pool_size": 1.5},
    ],
)
def test_inactive_historical_capacity_is_validated_without_value_disclosure(old):
    from reflexio.config_env import ConfigurationMigrationError
    from reflexio.config_migration import canonicalize_retrieval_floor_policy

    with pytest.raises(ConfigurationMigrationError) as error:
        canonicalize_retrieval_floor_policy(old)
    assert "private-invalid" not in str(error.value)


@pytest.mark.parametrize(
    "old",
    [
        {"enabled": False},
        {"pool_size": 30},
        {"enabled": None},
        {"pool_size": None},
        {"candidate_pool_size": 17, "enabled": False},
        {"candidate_pool_size": 17, "pool_size": 30},
    ],
)
def test_current_runtime_rejects_retired_names_even_when_mixed(old):
    with pytest.raises(ValidationError, match="Obsolete configuration fields"):
        RetrievalFloorConfig.model_validate(old)


def test_deliberate_conversion_preserves_current_precedence():
    from reflexio.config_migration import canonicalize_retrieval_floor_policy

    assert canonicalize_retrieval_floor_policy(
        {"enabled": False, "pool_size": 65, "candidate_pool_size": 19}
    ) == {"candidate_pool_size": 19}
