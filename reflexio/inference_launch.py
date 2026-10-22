"""Pure inference endpoint and launcher ownership; no model or server imports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from reflexio.config_env import ConfigurationMigrationError
from reflexio.env_utils import env_truthy

_SERVICE_URL = "REFLEXIO_EMBEDDING_SERVICE_URL"
_LEGACY_HOST = "REFLEXIO_EMBEDDING_DAEMON_HOST"
_LEGACY_DISABLE = "REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON"
_QUERY_URL = "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL"
_API_QUERY_URL = "http://127.0.0.1:8093"


@dataclass(frozen=True)
class InferenceLaunchPlan:
    """The owned process role and its startup health gate, if any."""

    mode: Literal["colocated", "attached", "query"]
    health_url: str | None


def inference_service_url(env: Mapping[str, str], *, default_port: int = 8072) -> str:
    """Resolve one endpoint; custom legacy hosts require explicit migration."""
    explicit = env.get(_SERVICE_URL, "").strip()
    if explicit:
        return explicit.rstrip("/")
    if env.get(_LEGACY_HOST, "").strip():
        raise ConfigurationMigrationError(
            f"{_LEGACY_HOST} is retired. Configure {_SERVICE_URL} with the full "
            "endpoint URL and choose REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS "
            "explicitly if retaining the previous local-service request budget."
        )
    port = env.get("EMBEDDING_PORT", "").strip() or str(default_port)
    return f"http://127.0.0.1:{port}"


def ordinary_inference_launch(
    env: Mapping[str, str], *, default_port: int = 8072
) -> InferenceLaunchPlan:
    """An explicit endpoint has its own owner; otherwise own the local child."""
    endpoint = inference_service_url(env, default_port=default_port)
    if env.get(_SERVICE_URL, "").strip():
        return InferenceLaunchPlan("attached", None)
    if env_truthy(env.get(_LEGACY_DISABLE, "")):
        raise ConfigurationMigrationError(
            f"{_LEGACY_DISABLE} is retired. To attach to an existing process, "
            f"configure {_SERVICE_URL} explicitly, including for localhost. "
            "Choose REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS explicitly if "
            "retaining the previous local-service request budget."
        )
    return InferenceLaunchPlan("colocated", endpoint + "/health")


def container_inference_launch(env: Mapping[str, str]) -> InferenceLaunchPlan:
    """Resolve Docker's independent query role before ordinary child ownership."""
    if env.get("REFLEXIO_API_QUERY_EMBEDDING_ENABLED", "false") == "true":
        if env.get(_QUERY_URL, "").strip().rstrip("/") != _API_QUERY_URL:
            raise ConfigurationMigrationError(
                "API-local query mode requires its reserved loopback URL."
            )
        # The ordinary disable switch never selected this independent child.
        inference_service_url(env, default_port=8089)
        return InferenceLaunchPlan("query", _API_QUERY_URL + "/health/model")
    return ordinary_inference_launch(env, default_port=8089)


def main() -> None:
    """Small shell adapter: fixed mode and health URL, never eval shell code."""
    try:
        plan = container_inference_launch(os.environ)
    except ConfigurationMigrationError as exc:
        raise SystemExit(str(exc)) from None
    print(plan.mode, plan.health_url or "-")


if __name__ == "__main__":
    main()
