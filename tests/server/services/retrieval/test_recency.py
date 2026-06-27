from types import SimpleNamespace

import pytest

from reflexio.server.services.retrieval.recency import (
    RecencyConfig,
    additive_penalty,
    decay,
    decay_for_item,
    multiplicative_factor,
)


def test_decay_half_life_is_half():
    assert decay(50, 100) > 0.70
    assert decay(100, 100) == pytest.approx(0.5)


def test_profile_decay_uses_ttl_lifespan():
    profile = SimpleNamespace(
        last_modified_timestamp=100,
        expiration_timestamp=300,
    )

    assert decay_for_item(profile, entity_type="profiles", now=200) == pytest.approx(
        0.5
    )


def test_missing_timestamps_are_noop():
    item = SimpleNamespace(created_at=None)

    assert decay_for_item(item, entity_type="user_playbooks", now=200) == 1.0


def test_factor_and_penalty_bounds():
    assert multiplicative_factor(0.0, 0.15) == 0.85
    assert multiplicative_factor(1.0, 0.15) == 1.0
    assert additive_penalty(0.0, 0.2) == 0.2
    assert additive_penalty(1.0, 0.2) == 0.0


def test_config_from_env_and_site_var_overrides():
    cfg = RecencyConfig.from_env(
        env={
            "REFLEXIO_SEARCH_RECENCY_ENABLED": "false",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_FRAC": "2",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_LOGIT": "-1",
            "REFLEXIO_SEARCH_RECENCY_POOL_SIZE": "0",
        }
    ).with_overrides(
        {
            "recency_enabled": True,
            "recency_max_penalty_frac": 0.25,
            "recency_max_penalty_logit": 0.4,
            "recency_pool_size": 40,
        }
    )

    assert cfg.enabled is True
    assert cfg.max_penalty_frac == 0.25
    assert cfg.max_penalty_logit == 0.4
    assert cfg.pool_size == 40
