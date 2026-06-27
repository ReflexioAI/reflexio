"""Recency adjustment helpers for unified search ranking."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from reflexio.server.env_utils import env_str, env_truthy

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 24 * 60 * 60
PLAYBOOK_HALF_LIFE_SECONDS = 90 * SECONDS_PER_DAY
_PROFILE_TTL_HALF_LIFE_FRACTION = 0.5


@dataclass(frozen=True)
class ScoredItem[T]:
    item: T
    score: float | None


@dataclass(frozen=True)
class RecencyConfig:
    enabled: bool = True
    max_penalty_frac: float = 0.15
    max_penalty_logit: float = 0.2
    pool_size: int = 20

    @classmethod
    def from_env(cls, *, env: Mapping[str, str] | None = None) -> RecencyConfig:
        return cls(
            enabled=env_truthy(
                env_str("REFLEXIO_SEARCH_RECENCY_ENABLED", "true", env=env)
            ),
            max_penalty_frac=_float_env(
                "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_FRAC",
                cls.max_penalty_frac,
                env=env,
                minimum=0.0,
                maximum=1.0,
            ),
            max_penalty_logit=_float_env(
                "REFLEXIO_SEARCH_RECENCY_MAX_PENALTY_LOGIT",
                cls.max_penalty_logit,
                env=env,
                minimum=0.0,
            ),
            pool_size=_int_env(
                "REFLEXIO_SEARCH_RECENCY_POOL_SIZE",
                cls.pool_size,
                env=env,
                minimum=1,
            ),
        )

    def with_overrides(self, values: Mapping[str, object] | None) -> RecencyConfig:
        if not values:
            return self
        updates: dict[str, object] = {}
        if "recency_enabled" in values:
            updates["enabled"] = _bool_value(values["recency_enabled"], self.enabled)
        if "recency_max_penalty_frac" in values:
            updates["max_penalty_frac"] = _float_value(
                values["recency_max_penalty_frac"],
                self.max_penalty_frac,
                minimum=0.0,
                maximum=1.0,
            )
        if "recency_max_penalty_logit" in values:
            updates["max_penalty_logit"] = _float_value(
                values["recency_max_penalty_logit"],
                self.max_penalty_logit,
                minimum=0.0,
            )
        if "recency_pool_size" in values:
            updates["pool_size"] = _int_value(
                values["recency_pool_size"], self.pool_size, minimum=1
            )
        return replace(self, **updates)


def decay_for_item(
    item: object,
    *,
    entity_type: str,
    now: int | None = None,
) -> float:
    """Return a freshness decay in [0, 1], where 1 means no penalty."""
    now = now if now is not None else int(datetime.now(UTC).timestamp())
    if entity_type == "profiles":
        return _profile_decay(item, now=now)
    return _timestamp_decay(
        getattr(item, "created_at", None),
        half_life_seconds=PLAYBOOK_HALF_LIFE_SECONDS,
        now=now,
    )


def decay(age_seconds: float, half_life_seconds: float) -> float:
    if half_life_seconds <= 0:
        return 1.0
    age_seconds = max(0.0, age_seconds)
    return math.exp(-age_seconds / (half_life_seconds / math.log(2)))


def multiplicative_factor(decay_value: float, max_penalty_frac: float) -> float:
    return 1.0 - _clamp(max_penalty_frac, 0.0, 1.0) * (1.0 - _decay(decay_value))


def additive_penalty(decay_value: float, max_penalty_logit: float) -> float:
    return max(0.0, max_penalty_logit) * (1.0 - _decay(decay_value))


def _profile_decay(item: object, *, now: int) -> float:
    modified_at = _int_or_none(getattr(item, "last_modified_timestamp", None))
    expires_at = _int_or_none(getattr(item, "expiration_timestamp", None))
    if modified_at is None or expires_at is None or expires_at <= modified_at:
        return 1.0
    half_life = (expires_at - modified_at) * _PROFILE_TTL_HALF_LIFE_FRACTION
    return decay(now - modified_at, half_life)


def _timestamp_decay(value: object, *, half_life_seconds: int, now: int) -> float:
    timestamp = _int_or_none(value)
    if timestamp is None or timestamp <= 0:
        return 1.0
    return decay(now - timestamp, half_life_seconds)


def _float_env(
    name: str,
    default: float,
    *,
    env: Mapping[str, str] | None,
    minimum: float,
    maximum: float | None = None,
) -> float:
    return _float_value(
        env_str(name, str(default), env=env),
        default,
        minimum=minimum,
        maximum=maximum,
    )


def _int_env(
    name: str,
    default: int,
    *,
    env: Mapping[str, str] | None,
    minimum: int,
) -> int:
    return _int_value(env_str(name, str(default), env=env), default, minimum=minimum)


def _bool_value(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        return env_truthy(value)
    return default


def _float_value(
    value: object,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        if not isinstance(value, str | int | float):
            raise TypeError
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid recency float override %r; using %.3f", value, default)
        return default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _int_value(value: object, default: int, *, minimum: int) -> int:
    try:
        if not isinstance(value, str | int):
            raise TypeError
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid recency integer override %r; using %d", value, default)
        return default
    return max(minimum, parsed)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _decay(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
