"""Stateless provenance cannot cross scopes or masquerade as a newer version."""

from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256

import pytest

from reflexio.models.api_schema.domain import RetrievedLearning, UserPlaybook
from reflexio.server.extensions import register_service
from reflexio.server.services.governance.config import governance_subject_ref
from reflexio.server.services.offline_tuner.exposure_receipts import (
    LOCAL_RECEIPT_PROJECT_ID,
    ExposureReceiptIssuer,
    ReceiptProjectUnboundError,
    ServedPlaybookReceipt,
    receipt_project_id,
    verify_exposure_receipt,
)
from reflexio.server.services.search_exposure import SearchExposureBatch
from reflexio.server.work_scope import WORK_SCOPE_PROVIDER, WorkScope

_SECRET = "receipt-test-secret"
_TIME = "reflexio.server.services.offline_tuner.exposure_receipts.time.time"


class _Provider:
    """A work-scope provider with a fixed current scope."""

    def __init__(self, scope: WorkScope | None) -> None:
        self._scope = scope

    def current(self) -> WorkScope | None:
        return self._scope

    def bind(self, scope: WorkScope) -> nullcontext[None]:  # noqa: ARG002
        return nullcontext()


def _playbook() -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=7,
        user_id="alice",
        request_id="creation",
        agent_version="v1",
        content="Served version one",
    )


def _batch(playbook: UserPlaybook) -> SearchExposureBatch:
    return SearchExposureBatch(
        org_id="org",
        user_id="alice",
        session_id="session",
        request_id=None,
        interaction_id=None,
        user_playbooks=(playbook,),
    )


def _issuer() -> ExposureReceiptIssuer:
    return ExposureReceiptIssuer(project_id=lambda: "project")


@pytest.fixture
def receipt(monkeypatch):
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", _SECRET)
    playbook = _playbook()
    batch = _batch(playbook)
    token = _issuer().issue(batch)[7]
    return token, batch, playbook


def verify(token, **changes):
    scope = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "session_id": "session",
        "user_playbook_id": 7,
        "owner_subject_ref": governance_subject_ref("org", "alice", _SECRET),
    }
    scope.update(changes)
    return verify_exposure_receipt(token, **scope)


def verified(token, **changes) -> ServedPlaybookReceipt:
    receipt = verify(token, **changes)
    assert receipt is not None
    return receipt


def test_receipt_retains_search_version_after_edit(receipt):
    token, batch, playbook = receipt
    original = verify(token)
    assert original is not None
    playbook.content = "Edited version two"
    edited = _issuer().issue(replace(batch, user_playbooks=(playbook,)))[7]
    assert (
        verified(edited).served_full_version_fingerprint
        != original.served_full_version_fingerprint
    )
    assert verify(token) == original


@pytest.mark.parametrize(
    "change",
    [
        {"org_id": "other"},
        {"project_id": "other"},
        {"user_id": "bob"},
        {"session_id": "other"},
        {"user_playbook_id": 8},
        {"owner_subject_ref": "wrong"},
    ],
)
def test_receipt_cannot_cross_authority_or_identity(receipt, change):
    assert verify(receipt[0], **change) is None


@pytest.mark.parametrize("token", [None, "", "v1.bad.signature", "x" * 8_193])
def test_missing_or_malformed_receipt_is_incomplete(receipt, token):
    assert verify(token) is None


def test_tampering_and_key_rotation_do_not_create_verified_evidence(
    receipt, monkeypatch
):
    token = receipt[0]
    assert verify(token[:-1] + ("0" if token[-1] != "0" else "1")) is None
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "new-receipt-key")
    assert verify(token) is None


def test_storage_strips_receipt_and_ignores_client_digest(receipt):
    token = receipt[0]
    ref = RetrievedLearning(
        kind="user_playbook",
        learning_id="7",
        exposure_receipt=token,
        exposure_receipt_digest="0" * 64,
    )
    stored = ref.persistence_payload()
    assert stored["exposure_receipt_digest"] == sha256(token.encode()).hexdigest()
    assert "exposure_receipt" not in stored
    assert ref.exposure_receipt == token
    forged = RetrievedLearning(
        kind="user_playbook", learning_id="7", exposure_receipt_digest="0" * 64
    )
    assert "exposure_receipt_digest" not in forged.persistence_payload()


def test_delayed_publish_uses_original_receipt_without_short_expiry(
    receipt, monkeypatch
):
    token = receipt[0]
    original = verified(token)
    monkeypatch.setattr(_TIME, lambda: original.exposed_at + 30 * 86400)
    assert verify(token) == original


@pytest.mark.parametrize(("offset", "valid"), [(300, True), (301, False)])
def test_future_receipt_boundary(receipt, monkeypatch, offset, valid):
    original = verified(receipt[0])
    monkeypatch.setattr(_TIME, lambda: original.exposed_at - offset)
    assert (verify(receipt[0]) is not None) is valid


# --- receipt_project_id: the resolver the issuer and a local verifier share --


def test_project_id_is_the_local_constant_without_a_provider():
    assert receipt_project_id() == LOCAL_RECEIPT_PROJECT_ID == "local"


def test_project_id_is_the_bound_project_under_a_provider():
    register_service(
        WORK_SCOPE_PROVIDER, _Provider(WorkScope("org", "prj_a")), override=True
    )
    assert receipt_project_id() == "prj_a"


@pytest.mark.parametrize(
    "scope", [None, WorkScope("org", None), WorkScope("org", "")], ids=repr
)
def test_project_id_fails_closed_when_a_provider_binds_no_project(scope):
    """A registered provider with no project never falls back to "local"."""
    register_service(WORK_SCOPE_PROVIDER, _Provider(scope), override=True)
    with pytest.raises(ReceiptProjectUnboundError):
        receipt_project_id()


def test_default_issuer_signs_local_receipts_without_a_provider(monkeypatch):
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", _SECRET)
    playbook = _playbook()
    token = ExposureReceiptIssuer().issue(_batch(playbook))[7]

    assert verify(token, project_id=LOCAL_RECEIPT_PROJECT_ID) is not None
    assert verify(token) is None


def test_default_issuer_refuses_when_a_provider_binds_no_project(monkeypatch):
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", _SECRET)
    register_service(WORK_SCOPE_PROVIDER, _Provider(None), override=True)
    playbook = _playbook()
    with pytest.raises(ReceiptProjectUnboundError):
        ExposureReceiptIssuer().issue(_batch(playbook))
