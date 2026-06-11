"""Verify Config exposes the F3 sampler + concurrency knobs with the
documented defaults."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    DEFAULT_AGENT_SUCCESS_DEFINITION_PROMPT,
    DEFAULT_AGENT_SUCCESS_SAMPLING_RATE,
    Config,
    StorageConfigSQLite,
)


def _minimal_config(**overrides) -> Config:
    return Config(storage_config=StorageConfigSQLite(), **overrides)


def test_eval_sample_n_per_stratum_defaults_to_200():
    c = _minimal_config()
    assert c.eval_sample_n_per_stratum == 200


def test_eval_concurrency_limit_defaults_to_10():
    c = _minimal_config()
    assert c.eval_concurrency_limit == 10


def test_agent_success_evaluation_defaults_on_at_five_percent():
    c = _minimal_config()

    assert c.agent_success_config is not None
    assert c.agent_success_config.sampling_rate == DEFAULT_AGENT_SUCCESS_SAMPLING_RATE
    assert c.agent_success_config.sampling_rate == 0.05
    assert (
        c.agent_success_config.success_definition_prompt
        == DEFAULT_AGENT_SUCCESS_DEFINITION_PROMPT
    )


def test_agent_success_evaluation_can_be_disabled_explicitly():
    c = _minimal_config(agent_success_config=None)

    assert c.agent_success_config is None


def test_eval_sample_n_per_stratum_must_be_positive():
    with pytest.raises(ValidationError):
        _minimal_config(eval_sample_n_per_stratum=0)
    with pytest.raises(ValidationError):
        _minimal_config(eval_sample_n_per_stratum=-1)


def test_eval_concurrency_limit_must_be_positive():
    with pytest.raises(ValidationError):
        _minimal_config(eval_concurrency_limit=0)
    with pytest.raises(ValidationError):
        _minimal_config(eval_concurrency_limit=-1)


def test_overrides_round_trip_through_model_dump():
    c = _minimal_config(eval_sample_n_per_stratum=50, eval_concurrency_limit=4)
    re_parsed = Config(**c.model_dump())
    assert re_parsed.eval_sample_n_per_stratum == 50
    assert re_parsed.eval_concurrency_limit == 4
