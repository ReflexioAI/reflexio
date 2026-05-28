"""Verify Config exposes the F3 sampler + concurrency knobs with the
documented defaults."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import Config, StorageConfigSQLite


def _minimal_config(**overrides) -> Config:
    return Config(storage_config=StorageConfigSQLite(), **overrides)


def test_eval_sample_n_per_stratum_defaults_to_200():
    c = _minimal_config()
    assert c.eval_sample_n_per_stratum == 200


def test_eval_concurrency_limit_defaults_to_10():
    c = _minimal_config()
    assert c.eval_concurrency_limit == 10


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
