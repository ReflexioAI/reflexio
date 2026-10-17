from __future__ import annotations

from collections.abc import Mapping

from reflexio.config_migration import ConfigurationMigrationError
from reflexio.server.env_utils import env_str, env_truthy
from reflexio.server.services.governance.subject_refs import (
    actor_ref,
    request_ref,
    subject_ref,
)

LOCAL_GOVERNANCE_REF_SECRET = "reflexio-local-governance-ref-secret"  # noqa: S105
_PRODUCTION_DEPLOYMENT_MODES = {"platform", "self_host"}
_PRODUCTION_ENVS = {"prod", "production", "staging"}


def _is_local_dev_or_test(env: Mapping[str, str] | None = None) -> bool:
    deployment_mode = env_str("DEPLOYMENT_MODE", env=env).lower()
    environment = env_str("ENVIRONMENT", "development", env=env).lower()
    # The escape hatch describes the running test process, never a target mapping.
    if env_truthy(env_str("REFLEXIO_TEST_MODE")):
        return True
    return (
        deployment_mode not in _PRODUCTION_DEPLOYMENT_MODES
        and environment not in _PRODUCTION_ENVS
    )


def get_governance_ref_secret(*, env: Mapping[str, str] | None = None) -> str:
    """Resolve the secret for runtime or startup using one deployment policy.

    An explicit mapping is authoritative for deployment settings; test mode is
    always read from the actual process. Custom retired settings require explicit
    migration even when a secret or test escape hatch is present.
    """
    legacy = env_str("REFLEXIO_ENV", env=env).lower()
    if legacy and legacy != "development":
        raise ConfigurationMigrationError(
            "REFLEXIO_ENV is retired; set ENVIRONMENT explicitly and remove REFLEXIO_ENV."
        )
    secret = env_str("REFLEXIO_GOVERNANCE_REF_SECRET", env=env)
    if secret:
        return secret
    if _is_local_dev_or_test(env):
        return LOCAL_GOVERNANCE_REF_SECRET
    raise RuntimeError("REFLEXIO_GOVERNANCE_REF_SECRET is required")


def governance_subject_ref(org_id: str, user_id: str, secret: str) -> str:
    return subject_ref(f"{org_id}:subject:{user_id}", secret)


def governance_request_ref(org_id: str, request_id: str, secret: str) -> str:
    return request_ref(f"{org_id}:request:{request_id}", secret)


def governance_actor_ref(
    org_id: str,
    credential_kind: str,
    principal_or_fingerprint: str,
    secret: str,
) -> str:
    return actor_ref(
        f"{org_id}:actor:{credential_kind}:{principal_or_fingerprint}",
        secret,
    )
