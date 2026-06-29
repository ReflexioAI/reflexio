from __future__ import annotations

from reflexio.server.env_utils import env_str, env_truthy
from reflexio.server.services.governance.subject_refs import (
    actor_ref,
    request_ref,
    subject_ref,
)

LOCAL_GOVERNANCE_REF_SECRET = "reflexio-local-governance-ref-secret"  # noqa: S105
_PRODUCTION_DEPLOYMENT_MODES = {"platform", "self_host"}
_PRODUCTION_ENVS = {"production", "staging"}


def _is_local_dev_or_test() -> bool:
    deployment_mode = env_str("DEPLOYMENT_MODE").lower()
    reflexio_env = env_str("REFLEXIO_ENV", "development").lower()
    if env_truthy(env_str("REFLEXIO_TEST_MODE")):
        return True
    return (
        deployment_mode not in _PRODUCTION_DEPLOYMENT_MODES
        and reflexio_env not in _PRODUCTION_ENVS
    )


def get_governance_ref_secret() -> str:
    secret = env_str("REFLEXIO_GOVERNANCE_REF_SECRET")
    if secret:
        return secret
    if _is_local_dev_or_test():
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
