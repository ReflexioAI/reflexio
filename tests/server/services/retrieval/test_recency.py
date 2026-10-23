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


def test_config_from_env_clamps_values():
    cfg = RecencyConfig.from_env(
        env={
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_FRAC": "2",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_LOGIT": "-1",
            "REFLEXIO_SEARCH_RECENCY_POOL_SIZE": "0",
        }
    )
    assert cfg.enabled is False
    assert cfg.max_penalty_frac == 1.0
    assert cfg.max_penalty_logit == 0.0
    assert cfg.pool_size == 0


def test_config_from_env_preserves_custom_policy():
    cfg = RecencyConfig.from_env(
        env={
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_FRAC": "0.25",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_LOGIT": "0.4",
            "REFLEXIO_SEARCH_RECENCY_POOL_SIZE": "40",
        }
    )
    assert cfg.enabled is True
    assert cfg.max_penalty_frac == 0.25
    assert cfg.max_penalty_logit == 0.4
    assert cfg.pool_size == 40


def test_zero_pool_disables_recency_without_changing_score_policy():
    cfg = RecencyConfig.from_env(
        env={
            "REFLEXIO_SEARCH_RECENCY_POOL_SIZE": "0",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_FRAC": "0.4",
            "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_LOGIT": "0.7",
        }
    )
    assert cfg.enabled is False
    assert cfg.pool_size == 0
    assert cfg.max_penalty_frac == 0.4
    assert cfg.max_penalty_logit == 0.7


@pytest.mark.parametrize("obsolete", ["", "false", "true", "private-test-value"])
def test_direct_reader_rejects_retired_switch_even_with_canonical_pool(obsolete):
    with pytest.raises(ValueError) as error:
        RecencyConfig.from_env(
            env={
                "REFLEXIO_SEARCH_RECENCY_ENABLED": obsolete,
                "REFLEXIO_SEARCH_RECENCY_POOL_SIZE": "0",
            }
        )
    message = str(error.value)
    assert "REFLEXIO_SEARCH_RECENCY_ENABLED" in message
    assert "REFLEXIO_SEARCH_RECENCY_POOL_SIZE" in message
    assert "private-test-value" not in message
