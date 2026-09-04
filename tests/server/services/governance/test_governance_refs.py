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

    subject_ref_org_a = governance_subject_ref("org-a", "alice", secret)
    subject_ref_org_b = governance_subject_ref("org-b", "alice", secret)
    request_ref_org_a = governance_request_ref("org-a", "ticket-1", secret)
    request_ref_org_b = governance_request_ref("org-b", "ticket-1", secret)
    actor_ref_jwt_org_a = governance_actor_ref("org-a", "jwt", "principal-1", secret)
    actor_ref_jwt_org_b = governance_actor_ref("org-b", "jwt", "principal-1", secret)
    actor_ref_token_org_a = governance_actor_ref(
        "org-a", "central_token", "principal-1", secret
    )

    assert subject_ref_org_a.startswith("subref_v1_")
    assert request_ref_org_a.startswith("reqref_v1_")
    assert actor_ref_jwt_org_a.startswith("actref_v1_")
    assert subject_ref_org_a != subject_ref_org_b
    assert request_ref_org_a != request_ref_org_b
    assert actor_ref_jwt_org_a != actor_ref_jwt_org_b
    assert request_ref_org_a != subject_ref_org_a
    assert actor_ref_jwt_org_a != actor_ref_token_org_a


def test_secret_defaults_only_in_local_dev_or_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    monkeypatch.setenv("REFLEXIO_ENV", "development")

    assert get_governance_ref_secret() == "reflexio-local-governance-ref-secret"


def test_secret_defaults_in_test_mode_even_for_platform_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("REFLEXIO_ENV", "production")
    monkeypatch.setenv("REFLEXIO_TEST_MODE", "true")

    assert get_governance_ref_secret() == "reflexio-local-governance-ref-secret"


def test_secret_fails_closed_in_platform_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("REFLEXIO_ENV", "production")
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()


def test_blank_secret_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "   ")
    monkeypatch.setenv("DEPLOYMENT_MODE", "self_host")
    monkeypatch.setenv("REFLEXIO_ENV", "production")

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()
