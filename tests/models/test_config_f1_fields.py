"""Verify F1 config field on Config."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import Config, StorageConfigSQLite


def _minimal(**overrides) -> Config:
    return Config(storage_config=StorageConfigSQLite(), **overrides)


def test_shadow_comparison_judge_prompt_version_defaults_to_v1_0_0():
    c = _minimal()
    assert c.shadow_comparison_judge_prompt_version == "v1.0.0"


def test_shadow_comparison_judge_prompt_version_rejects_empty_string():
    with pytest.raises(ValidationError):
        _minimal(shadow_comparison_judge_prompt_version="")


def test_shadow_comparison_judge_prompt_version_accepts_arbitrary_semver():
    c = _minimal(shadow_comparison_judge_prompt_version="v2.1.3")
    assert c.shadow_comparison_judge_prompt_version == "v2.1.3"
    re = Config(**c.model_dump())
    assert re.shadow_comparison_judge_prompt_version == "v2.1.3"
