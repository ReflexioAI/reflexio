from __future__ import annotations

import pytest

from reflexio.config_migration import ConfigurationMigrationError
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


@pytest.mark.parametrize("environment", ["", "  ", "development", " Development "])
def test_secret_defaults_only_in_local_dev_or_test(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    monkeypatch.setenv("ENVIRONMENT", environment)

    assert get_governance_ref_secret() == "reflexio-local-governance-ref-secret"


def test_secret_defaults_in_test_mode_even_for_platform_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REFLEXIO_TEST_MODE", "true")

    assert get_governance_ref_secret() == "reflexio-local-governance-ref-secret"


def test_secret_fails_closed_in_platform_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()


def test_blank_secret_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "   ")
    monkeypatch.setenv("DEPLOYMENT_MODE", "self_host")
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()


@pytest.mark.parametrize(
    "environment", ["development", "production", "staging", "prodution", "qa"]
)
@pytest.mark.parametrize("mode", ["platform", "self_host"])
def test_hosted_modes_require_secret_independently_of_logging_label(
    monkeypatch, environment, mode
):
    monkeypatch.setenv("DEPLOYMENT_MODE", mode)
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    monkeypatch.delenv("REFLEXIO_GOVERNANCE_REF_SECRET", raising=False)
    monkeypatch.setenv("REFLEXIO_ENV", "development")
    monkeypatch.setenv("ENVIRONMENT", environment)
    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret()


@pytest.mark.parametrize(
    "legacy", ["production", "staging", "test", "unprinted-legacy-value"]
)
@pytest.mark.parametrize("test_mode", ["true", "false"])
def test_custom_legacy_environment_requires_explicit_migration(
    monkeypatch, legacy, test_mode
):
    monkeypatch.setenv("REFLEXIO_TEST_MODE", test_mode)
    monkeypatch.setenv("REFLEXIO_ENV", legacy)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "unprinted-governance-secret")
    with pytest.raises(
        ConfigurationMigrationError, match="REFLEXIO_ENV.*deployment-mode"
    ) as error:
        get_governance_ref_secret()
    assert legacy not in str(error.value)
    assert "unprinted-governance-secret" not in str(error.value)


@pytest.mark.parametrize("legacy", ["", "  ", "development", " Development "])
def test_former_default_does_not_select_an_environment(monkeypatch, legacy):
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    monkeypatch.setenv("REFLEXIO_ENV", legacy)
    assert (
        get_governance_ref_secret(env={"REFLEXIO_ENV": legacy})
        == "reflexio-local-governance-ref-secret"
    )
    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret(
            env={"REFLEXIO_ENV": legacy, "DEPLOYMENT_MODE": "self_host"}
        )


def test_resolved_mapping_isolated_from_process_settings_and_test_flag(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REFLEXIO_ENV", "unrelated-process-legacy")
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "process-secret")
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    assert get_governance_ref_secret(env={}) == "reflexio-local-governance-ref-secret"
    deployment = {"DEPLOYMENT_MODE": "self_host", "REFLEXIO_TEST_MODE": "true"}
    before = deployment.copy()
    with pytest.raises(RuntimeError, match="REFLEXIO_GOVERNANCE_REF_SECRET"):
        get_governance_ref_secret(env=deployment)
    assert deployment == before
    assert (
        get_governance_ref_secret(
            env={**deployment, "REFLEXIO_GOVERNANCE_REF_SECRET": "mapped-secret"}
        )
        == "mapped-secret"
    )


@pytest.mark.parametrize("environment", ["development", "production", "staging", "qa"])
@pytest.mark.parametrize("initial_mode", [None, "platfrom"])
def test_existing_local_sqlite_erasure_barrier_survives_environment_label(
    tmp_path, monkeypatch, environment, initial_mode
):
    import sqlite3
    from contextlib import closing

    from reflexio.server.services.storage.error import SubjectWriteBarrierError
    from reflexio.server.services.storage.sqlite_storage._subject_write_gate import (
        SubjectWriteGateMixin,
        init_subject_write_barrier_table,
    )

    for name in (
        "DEPLOYMENT_MODE",
        "REFLEXIO_TEST_MODE",
        "REFLEXIO_GOVERNANCE_REF_SECRET",
        "REFLEXIO_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", environment)
    if initial_mode:
        monkeypatch.setenv("DEPLOYMENT_MODE", initial_mode)
        with pytest.raises(ConfigurationMigrationError, match="DEPLOYMENT_MODE"):
            get_governance_ref_secret()
        # Explicit local adoption keeps the legacy epoch, rather than picking a key.
        monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    database = tmp_path / "existing.sqlite"
    # Pinned legacy HMAC, independent of the upgraded resolver/default constant.
    previous_ref = "subref_v1_59dcb8a7c57eb69a4e4213c0236418c5"
    with closing(sqlite3.connect(database)) as conn:
        init_subject_write_barrier_table(conn)
        conn.execute(
            "INSERT INTO subject_write_barriers (org_id,subject_ref,purge_id,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            ("org", previous_ref, "previous-purge", "erased", 1, 1),
        )
        conn.commit()
    with closing(sqlite3.connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        gate = SubjectWriteGateMixin()
        gate.conn = conn
        gate.org_id = "org"
        current_ref = gate._subject_ref_for_user_id("erased-user")
        assert current_ref == previous_ref
        with pytest.raises(SubjectWriteBarrierError, match="previous-purge"):
            gate._assert_subject_writable_locked(current_ref)


def test_explicit_adoption_preserves_existing_secret_and_hmac_epoch(monkeypatch):
    monkeypatch.delenv("REFLEXIO_TEST_MODE", raising=False)
    env = {
        "REFLEXIO_ENV": "production",
        "REFLEXIO_GOVERNANCE_REF_SECRET": "existing-private-secret",
    }
    previous_ref = governance_subject_ref(
        "org", "user", env["REFLEXIO_GOVERNANCE_REF_SECRET"]
    )
    with pytest.raises(ConfigurationMigrationError):
        get_governance_ref_secret(env=env)
    env.pop("REFLEXIO_ENV")
    env["DEPLOYMENT_MODE"] = "self_host"
    assert (
        governance_subject_ref("org", "user", get_governance_ref_secret(env=env))
        == previous_ref
    )


@pytest.mark.parametrize("mode", ["platfrom", "self-host", "opaque-mode-value"])
@pytest.mark.parametrize("secret", [None, "existing-private-secret"])
@pytest.mark.parametrize("test_mode", ["true", "false"])
def test_unknown_mode_requires_adoption_without_rotating_secret(
    monkeypatch, mode, secret, test_mode
):
    monkeypatch.setenv("REFLEXIO_TEST_MODE", test_mode)
    env = {"DEPLOYMENT_MODE": mode}
    if secret:
        env["REFLEXIO_GOVERNANCE_REF_SECRET"] = secret
    with pytest.raises(
        ConfigurationMigrationError, match="DEPLOYMENT_MODE.*local"
    ) as error:
        get_governance_ref_secret(env=env)
    assert mode not in str(error.value)
    if secret:
        assert secret not in str(error.value)
    env["DEPLOYMENT_MODE"] = "local"
    assert get_governance_ref_secret(env=env) == (
        secret or "reflexio-local-governance-ref-secret"
    )
