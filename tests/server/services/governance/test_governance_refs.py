from __future__ import annotations

import pytest

from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_actor_ref,
    governance_request_ref,
    governance_subject_ref,
)


def test_refs_are_domain_separated_by_org_and_kind() -> None:
    secret = "test-secret"

    assert governance_subject_ref("org-a", "alice", secret).startswith("subref_v1_")
    assert governance_request_ref("org-a", "ticket-1", secret).startswith("reqref_v1_")
    assert governance_actor_ref("org-a", "jwt", "principal-1", secret).startswith(
        "actref_v1_"
    )
    assert governance_subject_ref("org-a", "alice", secret) != governance_subject_ref(
        "org-b", "alice", secret
    )
    assert governance_request_ref("org-a", "same", secret) != governance_subject_ref(
        "org-a", "same", secret
    )
    assert governance_actor_ref(
        "org-a", "jwt", "principal-1", secret
    ) != governance_actor_ref("org-a", "central_token", "principal-1", secret)


def test_secret_defaults_only_in_local_dev_or_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.setenv("REFLEXIO_ENV", "development")

    assert get_governance_ref_secret() == "reflexio-local-governance-ref-secret"


def test_secret_fails_closed_in_platform_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("REFLEXIO_ENV", "production")

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()


def test_blank_secret_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "   ")
    monkeypatch.setenv("DEPLOYMENT_MODE", "self_host")
    monkeypatch.setenv("REFLEXIO_ENV", "production")

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()
