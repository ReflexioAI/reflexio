"""Platform MiniMax credential failover; never changes a model or endpoint."""

from collections.abc import Mapping
from typing import Any

import litellm

from reflexio.server.env_utils import env_str
from reflexio.server.llm.provider_circuit import ProviderCircuitOpenError


def backup_credentials(params: Mapping[str, Any]) -> tuple[str, str] | None:
    """Only the environment-owned primary may spend the environment backup."""
    if not str(params.get("model", "")).lower().startswith("minimax/"):
        return None
    # Custom endpoints may belong to a customer or another provider entirely.
    if (
        params.get("api_base")
        or params.get("base_url")
        or env_str("MINIMAX_API_BASE", "")
        or litellm.api_base
        or params.get("custom_llm_provider") not in (None, "minimax")
    ):
        return None
    primary = env_str("MINIMAX_API_KEY", "").strip()
    backup = env_str("MINIMAX_BACKUP_API_KEY", "").strip()
    resolved = params.get("api_key") or primary
    if not primary or not backup or backup == primary or resolved != primary:
        return None
    return primary, backup


def quota_exhausted(error: BaseException) -> bool:
    """Recognize MiniMax's explicit plan-exhaustion code, not arbitrary 429s.

    The isolated worker preserves the provider's error as text in
    LiteLLMClientError, so inspect the cause chain as well as typed errors.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderCircuitOpenError):
            return current.quota_exhausted
        message = str(current).lower()
        status = getattr(current, "upstream_status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status == 429 and "(2056)" in message and "usage limit reached" in message:
            return True
        current = current.__cause__ or current.__context__
    return False
