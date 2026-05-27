import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import ReflectionConfig


def test_reflection_config_default_post_horizon_size_is_3():
    cfg = ReflectionConfig()
    assert cfg.post_horizon_size == 3


def test_reflection_config_post_horizon_size_explicit():
    cfg = ReflectionConfig(post_horizon_size=5)
    assert cfg.post_horizon_size == 5


def test_reflection_config_post_horizon_size_zero_recovers_legacy():
    cfg = ReflectionConfig(post_horizon_size=0)
    assert cfg.post_horizon_size == 0


def test_reflection_config_post_horizon_size_rejects_negative():
    with pytest.raises(ValidationError):
        ReflectionConfig(post_horizon_size=-1)
