import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    Config,
    RetrievalFloorConfig,
    StorageConfigSQLite,
)


def test_retrieval_floor_defaults():
    cfg = RetrievalFloorConfig()
    assert cfg.enabled is True
    assert cfg.pool_size == 30
    assert cfg.profile_floor == -3.0
    assert cfg.user_playbook_floor == -3.0
    assert cfg.agent_playbook_floor == -3.0


def test_config_has_retrieval_floor_default():
    cfg = Config(storage_config=StorageConfigSQLite())
    assert isinstance(cfg.retrieval_floor, RetrievalFloorConfig)
    assert cfg.retrieval_floor.enabled is True


def test_pool_size_must_be_positive():
    with pytest.raises(ValidationError):
        RetrievalFloorConfig(pool_size=0)


def test_retrieval_floor_null_falls_back_to_default():
    # A stored config row missing the column (or carrying an explicit null) must
    # fall back to the default rather than failing validation — same as the other
    # defaulted sub-configs stripped by the before-validator.
    base = Config(storage_config=StorageConfigSQLite())
    payload = base.model_dump()
    payload["retrieval_floor"] = None
    cfg = Config.model_validate(payload)
    assert cfg.retrieval_floor == RetrievalFloorConfig()
