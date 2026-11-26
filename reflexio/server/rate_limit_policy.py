"""Canonical process policy for SlowAPI, independent of its dotenv loader."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from limits import parse_many
from slowapi.extension import C

from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str


def _boolean(name: str, default: bool) -> bool:
    value = env_str(name, str(default)).lower()
    # Retain the dependency's valid true/false and 1/0 spellings. False must
    # mean false, including for the former false-default string parsing bug.
    choices = {"true": True, "1": True, "false": False, "0": False}
    if value not in choices:
        raise ValueError(f"{name} must be true, false, 1 or 0")
    return choices[value]


def _rules(name: str) -> str:
    value = env_str(name)
    if value:
        try:
            parse_many(value)
        except ValueError:
            raise ValueError(f"{name} must contain valid rate expressions") from None
    return value


@dataclass(frozen=True)
class RateLimitPolicy:
    """One policy owns backend selection, rule scopes and outage behavior."""

    enabled: bool
    storage_url: str
    strategy: str
    key_prefix: str
    application: str
    default: str
    swallow_errors: bool
    fallback: str

    @classmethod
    def from_environment(cls) -> "RateLimitPolicy":
        import os

        reject_removed_env(os.environ)
        fallback = env_str("RATELIMIT_IN_MEMORY_FALLBACK")
        if fallback != "inherit":
            fallback = _rules("RATELIMIT_IN_MEMORY_FALLBACK")
        return cls(
            enabled=_boolean("RATELIMIT_ENABLED", True),
            storage_url=env_str("RATELIMIT_STORAGE_URL", "memory://"),
            strategy=env_str("RATELIMIT_STRATEGY", "fixed-window"),
            key_prefix=env_str("RATELIMIT_KEY_PREFIX"),
            application=_rules("RATELIMIT_APPLICATION"),
            default=_rules("RATELIMIT_DEFAULT"),
            swallow_errors=_boolean("RATELIMIT_SWALLOW_ERRORS", False),
            fallback=fallback,
        )

    def dependency_values(self) -> Mapping[str, Any]:
        """Closed integration contract; response metadata belongs to the SDK."""
        return {
            C.ENABLED: self.enabled,
            C.STORAGE_URL: self.storage_url,
            C.STRATEGY: self.strategy,
            C.KEY_PREFIX: self.key_prefix,
            C.APPLICATION_LIMITS: self.application,
            C.DEFAULT_LIMITS: self.default,
            C.SWALLOW_ERRORS: self.swallow_errors,
            C.IN_MEMORY_FALLBACK_ENABLED: bool(self.fallback),
            C.IN_MEMORY_FALLBACK: "" if self.fallback == "inherit" else self.fallback,
            C.STORAGE_OPTIONS: {},
            # The SDK owns the rejection wire protocol. Vendor success-header
            # generation would perform backend I/O after response construction.
            C.HEADERS_ENABLED: False,
            C.HEADER_LIMIT: None,
            C.HEADER_REMAINING: None,
            C.HEADER_RESET: None,
            C.HEADER_RETRY_AFTER: None,
            C.HEADER_RETRY_AFTER_VALUE: None,
        }
