from __future__ import annotations

import pytest

from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_actor_ref,
    governance_request_ref,
    governance_subject_ref,
)
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _validate_governance_target_ref,
)
from reflexio.server.services.storage.storage_base._governance import GovernanceMixin


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


def test_governance_mixin_tracks_new_barrier_methods_as_abstract() -> None:
    assert {
        "begin_subject_erasure_barrier",
        "assert_subject_writable",
        "complete_subject_erasure_barrier_after_empty_check",
        "fail_subject_erasure_barrier",
        "get_subject_write_barrier",
    } <= GovernanceMixin.__abstractmethods__


def test_agent_success_eval_result_is_delete_only_target() -> None:
    """``agent_success_evaluation_result`` is a canonical delete target, so its
    target_ref validation must enforce the delete-only contract (phase=='delete',
    target_ref=='all') — not fall through to the permissive generic path that
    would accept e.g. ``hide_for_rebuild``.
    """
    target = "agent_success_evaluation_result"
    assert target in _CANONICAL_DELETE_TARGET_NAMES

    # Valid canonical delete shape is accepted.
    assert (
        _validate_governance_target_ref(
            target_name=target, phase="delete", target_ref="all"
        )
        == "all"
    )

    # Previously accepted (fell through to generic path); must now be rejected.
    with pytest.raises(ValueError, match="must use delete phase"):
        _validate_governance_target_ref(
            target_name=target, phase="hide_for_rebuild", target_ref="all"
        )

    # delete phase but non-"all" ref must also be rejected.
    with pytest.raises(ValueError, match="must be all"):
        _validate_governance_target_ref(
            target_name=target, phase="delete", target_ref="123"
        )


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
