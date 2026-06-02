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


def test_reflection_config_default_max_revisions_per_pass_is_8():
    cfg = ReflectionConfig()
    assert cfg.max_revisions_per_pass == 8


def test_reflection_config_max_revisions_per_pass_explicit():
    cfg = ReflectionConfig(max_revisions_per_pass=3)
    assert cfg.max_revisions_per_pass == 3


def test_reflection_config_max_revisions_per_pass_rejects_zero():
    with pytest.raises(ValidationError):
        ReflectionConfig(max_revisions_per_pass=0)


def test_reflection_config_max_revisions_per_pass_rejects_negative():
    with pytest.raises(ValidationError):
        ReflectionConfig(max_revisions_per_pass=-1)
