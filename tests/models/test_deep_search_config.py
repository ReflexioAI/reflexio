"""Tests for DeepSearchConfig and its wiring into Config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    Config,
    DeepSearchConfig,
    StorageConfigSQLite,
)


def _config(**kwargs) -> Config:
    return Config(storage_config=StorageConfigSQLite(), **kwargs)


def test_defaults():
    cfg = DeepSearchConfig()
    assert cfg.enabled is True
    assert cfg.max_subqueries == 6
    assert cfg.planner_timeout_s == 15.0
    assert cfg.reflect_timeout_s == 45.0


def test_validation_rejects_nonpositive_values():
    with pytest.raises(ValidationError):
        DeepSearchConfig(max_subqueries=0)
    with pytest.raises(ValidationError):
        DeepSearchConfig(planner_timeout_s=0)
    with pytest.raises(ValidationError):
        DeepSearchConfig(reflect_timeout_s=-1)


def test_config_carries_default_deep_search_config():
    cfg = _config()
    assert cfg.deep_search_config.enabled is True


def test_config_round_trip_preserves_overrides():
    cfg = _config(deep_search_config=DeepSearchConfig(enabled=False, max_subqueries=2))
    reloaded = Config.model_validate(cfg.model_dump())
    assert reloaded.deep_search_config.enabled is False
    assert reloaded.deep_search_config.max_subqueries == 2


def test_none_from_stored_blob_falls_back_to_default():
    """Old stored configs may carry deep_search_config: null — must not fail."""
    data = _config().model_dump()
    data["deep_search_config"] = None
    reloaded = Config.model_validate(data)
    assert reloaded.deep_search_config.enabled is True


def test_llm_config_deep_search_model_overrides_round_trip():
    from reflexio.models.config_schema import LLMConfig

    llm = LLMConfig(
        deep_search_planner_model_name="gpt-5-nano",
        deep_search_reranker_model_name="claude-sonnet-4-6",
    )
    cfg = _config(llm_config=llm)
    reloaded = Config.model_validate(cfg.model_dump())
    assert reloaded.llm_config is not None
    assert reloaded.llm_config.deep_search_planner_model_name == "gpt-5-nano"
    assert reloaded.llm_config.deep_search_reranker_model_name == "claude-sonnet-4-6"
    # Defaults stay None (fall through to role resolution).
    assert LLMConfig().deep_search_planner_model_name is None
    assert LLMConfig().deep_search_reranker_model_name is None
