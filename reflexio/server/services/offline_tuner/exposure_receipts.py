"""Stateless search receipts, verified only when their interaction is published.

Search signs each returned user playbook with a receipt; publish verifies it
and records served-version provenance only from a verified receipt.

The receipt binds a project id. :func:`receipt_project_id` is the one resolver
the issuer and a local verifier share, so the two cannot disagree: with a
registered work-scope provider it is the bound project (and refuses when none
is bound); without one -- single-tenant local storage -- it is the constant
:data:`LOCAL_RECEIPT_PROJECT_ID`. A host with its own project binding injects
its resolver into :class:`ExposureReceiptIssuer` instead.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import time
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reflexio.models.api_schema.domain import Interaction
from reflexio.server.extensions import get_service
from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.playbook.publication import canonical_json_bytes
from reflexio.server.services.search_exposure import (
    SearchExposureBatch,
    UserPlaybookExposureEvent,
    build_user_playbook_exposure_event,
)
from reflexio.server.work_scope import WORK_SCOPE_PROVIDER, current_project_id

_DOMAIN = b"reflexio:published-playbook-receipt:v1\x00"
_Identity = Annotated[str, Field(min_length=1, max_length=1_000)]
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ServedPlaybookReceipt(BaseModel):
    """Signed search provenance; contains no playbook text or raw owner identity."""

    model_config = ConfigDict(extra="forbid", strict=True)

    nonce: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    org_id: _Identity
    project_id: _Identity
    user_id: _Identity | None
    session_id: _Identity | None
    user_playbook_id: Annotated[int, Field(gt=0)]
    served_semantic_digest: _Digest
    served_full_version_fingerprint: _Digest
    owner_subject_ref: Annotated[str, Field(pattern=r"^subref_v1_[0-9a-f]{32}$")]
    exposed_at: Annotated[int, Field(ge=0)]


def _mac(body: bytes) -> str:
    return hmac.new(
        get_governance_ref_secret().encode(), _DOMAIN + body, sha256
    ).hexdigest()


#: The project id a receipt binds when no work-scope provider is registered.
LOCAL_RECEIPT_PROJECT_ID = "local"


class ReceiptProjectUnboundError(RuntimeError):
    """A work-scope provider is registered but no project is bound."""


def receipt_project_id() -> str:
    """Resolve the project id a receipt binds, failing closed when scoped.

    Returns:
        str: The bound project when a work-scope provider is registered;
            :data:`LOCAL_RECEIPT_PROJECT_ID` when none is.

    Raises:
        ReceiptProjectUnboundError: A provider is registered but no project
            is bound. Falling back to the local constant there would sign a
            receipt under the wrong project.
    """
    if get_service(WORK_SCOPE_PROVIDER) is None:
        return LOCAL_RECEIPT_PROJECT_ID
    project_id = current_project_id()
    if project_id is None:
        raise ReceiptProjectUnboundError(
            "No project is bound for a search exposure receipt; a registered "
            "work-scope provider must bind one before search signs receipts."
        )
    return project_id


class ExposureReceiptIssuer:
    """Sign the already retrieved playbooks without database work."""

    def __init__(self, *, project_id: Callable[[], str] = receipt_project_id) -> None:
        """Build an issuer.

        Args:
            project_id (Callable[[], str]): Resolves the project a receipt
                binds, called once per search. Raises to refuse issuing.
        """
        self._project_id = project_id

    def issue(self, batch: SearchExposureBatch) -> dict[int, str]:
        project_id = self._project_id()
        secret = get_governance_ref_secret()
        now = int(time.time())
        receipts: dict[int, str] = {}
        for playbook in batch.user_playbooks:
            if not playbook.user_id:
                continue
            owner_ref = governance_subject_ref(batch.org_id, playbook.user_id, secret)
            event = build_user_playbook_exposure_event(
                batch,
                playbook,
                exposed_at=now,
                ingested_at=now,
                governance_subject_ref=None,
                playbook_owner_governance_subject_ref=owner_ref,
            )
            receipt = ServedPlaybookReceipt(
                nonce=batch.invocation_id,
                org_id=batch.org_id,
                project_id=project_id,
                user_id=batch.user_id,
                session_id=batch.session_id,
                user_playbook_id=playbook.user_playbook_id,
                served_semantic_digest=event.served_semantic_digest or "",
                served_full_version_fingerprint=event.served_full_version_fingerprint
                or "",
                owner_subject_ref=owner_ref,
                exposed_at=now,
            )
            body = canonical_json_bytes(receipt.model_dump(mode="json"))
            receipts[playbook.user_playbook_id] = (
                "v1."
                + base64.urlsafe_b64encode(body).decode().rstrip("=")
                + "."
                + _mac(body)
            )
        return receipts


def verify_exposure_receipt(
    token: str | None,
    *,
    org_id: str,
    project_id: str,
    user_id: str,
    session_id: str,
    user_playbook_id: int,
    owner_subject_ref: str,
) -> ServedPlaybookReceipt | None:
    """Return verified provenance, or None for honestly incomplete evidence.

    No expiry: offline publication can be delayed. Key rotation invalidates old
    receipts. Never use unverified contents to choose a tenant or owner.
    """
    if token is None or len(token) > 8_192:
        return None
    try:
        version, encoded, signature = token.split(".")
        if version != "v1" or len(signature) != 64 or not signature.isascii():
            return None
        body = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
        if not hmac.compare_digest(signature, _mac(body)):
            return None
        receipt = ServedPlaybookReceipt.model_validate_json(body)
    except (ValueError, binascii.Error, ValidationError):
        return None
    if (
        receipt.org_id != org_id
        or receipt.project_id != project_id
        or receipt.user_playbook_id != user_playbook_id
        or receipt.owner_subject_ref != owner_subject_ref
        or (receipt.user_id is not None and receipt.user_id != user_id)
        or (receipt.session_id is not None and receipt.session_id != session_id)
        or receipt.exposed_at > int(time.time()) + 300
    ):
        return None
    return receipt


#: Prefix of every exposure event recorded when an interaction is published.
PUBLISHED_EXPOSURE_PREFIX = "publish-v1:"


def published_playbook_id(value: str) -> int | None:
    """Parse a retrieved ``user_playbook`` reference id, bounded to a bigint.

    Args:
        value (str): ``RetrievedLearning.learning_id``.

    Returns:
        int | None: The playbook id, or ``None`` when the reference cannot name
            a stored playbook (not an integer, or outside ``(0, 2**63)``).
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 < number < 2**63 else None


def build_published_exposure_events(
    interaction: Interaction,
    *,
    org_id: str,
    project_id: str,
    session_id: str,
    owners: Mapping[int, str],
    subject_ref: Callable[[str], str],
    ingested_at: int,
) -> list[UserPlaybookExposureEvent]:
    """Build the exposure events one newly published interaction records.

    Pure: the caller reads the owners and writes the events inside the
    interaction's own transaction. One event per ``user_playbook`` reference
    whose playbook is in ``owners``, in reference order. Served digests and
    ``exposed_at`` come only from a receipt that verifies against this
    interaction and the playbook's CURRENT owner; otherwise they are ``None``
    and the event is honestly incomplete.

    The event id is ``publish-v1:`` + sha256 of the canonical
    ``{org_id, project_id, request_id, interaction_id, ordinal}``, where
    ``ordinal`` is the reference's position among ALL retrieved learnings, so
    a replay of the same interaction yields the same ids.

    Args:
        interaction (Interaction): The interaction, with its stored id.
        org_id (str): The storage org.
        project_id (str): The project receipts bind (see
            :func:`receipt_project_id`).
        session_id (str): The interaction's request session.
        owners (Mapping[int, str]): Playbook id -> owner user id, for the
            playbooks the store found. References to other ids are skipped.
        subject_ref (Callable[[str], str]): User id -> governance subject ref.
        ingested_at (int): Epoch seconds the events are recorded at.

    Returns:
        list[UserPlaybookExposureEvent]: The events, in reference order.
    """
    events: list[UserPlaybookExposureEvent] = []
    for ordinal, ref in enumerate(interaction.retrieved_learnings):
        if ref.kind != "user_playbook":
            continue
        playbook_id = published_playbook_id(ref.learning_id)
        if playbook_id is None or playbook_id not in owners:
            continue
        owner = owners[playbook_id]
        owner_ref = subject_ref(owner)
        receipt = verify_exposure_receipt(
            ref.exposure_receipt,
            org_id=org_id,
            project_id=project_id,
            user_id=interaction.user_id,
            session_id=session_id,
            user_playbook_id=playbook_id,
            owner_subject_ref=owner_ref,
        )
        identity = canonical_json_bytes(
            {
                "org_id": org_id,
                "project_id": project_id,
                "request_id": interaction.request_id,
                "interaction_id": interaction.interaction_id,
                "ordinal": ordinal,
            }
        )
        events.append(
            UserPlaybookExposureEvent(
                exposure_event_id=PUBLISHED_EXPOSURE_PREFIX
                + sha256(identity).hexdigest(),
                request_id=interaction.request_id,
                session_id=session_id,
                user_id=interaction.user_id,
                playbook_owner_user_id=owner,
                user_playbook_id=playbook_id,
                served_semantic_digest=receipt.served_semantic_digest
                if receipt
                else None,
                served_full_version_fingerprint=(
                    receipt.served_full_version_fingerprint if receipt else None
                ),
                exposed_at=receipt.exposed_at if receipt else None,
                ingested_at=ingested_at,
                governance_subject_ref=subject_ref(interaction.user_id),
                playbook_owner_governance_subject_ref=owner_ref,
            )
        )
    return events
