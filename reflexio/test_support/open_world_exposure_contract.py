# ruff: noqa: S101 -- Test-only assertions.
"""Shared contract for the published-exposure ledger, part of the port contract.

:class:`~reflexio.test_support.open_world_port_contract.OpenWorldPortContract`
inherits :class:`PublishedExposureContract`, so every backend binding of the
port contract runs these cases too. They drive storage through its public
publish, search, read and clear methods and read the ledger back through the
harness, so one definition holds SQLite's Python-computed integrity state and
Postgres's generated columns to the same answers.

All data is synthetic.

Invariants covered
------------------
I10 -- exposure atomicity: a published interaction's exposure rows commit with
it or not at all, both when the transaction fails after the write and when the
exposure write is refused; served digests and ``exposed_at`` come only from a
receipt that verifies, otherwise the row is ``incomplete``. The integrity
state and its ordered reasons agree on every backend.

I11 -- deterministic ``publish-v1:`` ids from ``{org_id, project_id,
request_id, interaction_id, ordinal}``; a replay of a stored interaction
records nothing new.

I12 -- for an unchanged row, search serves the playbook whose full-version
fingerprint ``get_user_playbook_by_id`` reads.

I18 (exposure part) -- clearing a user deletes every exposure naming them as
viewer or as playbook owner, and no other.

Backend difference, asserted per backend through
``refuses_unattributable_exposure``: a ``user_playbook`` reference the store
cannot attribute (no such playbook, or no stored request for the
interaction) fails the whole publish on Postgres; SQLite stores the
interaction and records no exposure for it.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any, NamedTuple, Protocol
from uuid import uuid4

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    Request,
    RetrievedLearning,
    UserPlaybook,
)
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.offline_tuner.exposure_receipts import (
    ExposureReceiptIssuer,
)
from reflexio.server.services.playbook.publication import (
    canonical_json_bytes,
    incumbent_user_playbook_semantic_digest,
)
from reflexio.server.services.search_exposure import (
    SearchExposureBatch,
    UserPlaybookExposureEvent,
    user_playbook_full_version_fingerprint,
)

_DIGEST_REASONS = (
    "missing_served_semantic_digest",
    "missing_served_full_version_fingerprint",
    "missing_exposed_at",
)
_DIGEST = "a" * 64
#: A prepared interaction embedding, as the publish path computes before its
#: transaction; storage that requires one refuses an empty vector.
_EMBEDDING = [1.0] + [0.0] * 511
_REF = "subref_v1_" + "b" * 32


class ExposureRow(NamedTuple):
    """One stored ``user_playbook_exposure_events`` row, as the harness reads it."""

    exposure_event_id: str
    request_id: str | None
    session_id: str | None
    user_id: str | None
    playbook_owner_user_id: str | None
    user_playbook_id: int | None
    served_semantic_digest: str | None
    served_full_version_fingerprint: str | None
    exposed_at: int | None
    ingested_at: int
    governance_subject_ref: str | None
    playbook_owner_governance_subject_ref: str | None
    integrity_state: str
    integrity_reasons: tuple[str, ...]


class ExposureHarness(Protocol):
    """The exposure half of the port harness."""

    @property
    def storage(self) -> Any: ...

    def now(self) -> int:
        """The store's own clock, in whole epoch seconds."""
        ...

    @property
    def receipt_project_id(self) -> str:
        """The project id this store's receipts bind and its verifier expects."""
        ...

    @property
    def refuses_unattributable_exposure(self) -> bool:
        """True when an unattributable reference fails the whole publish."""
        ...

    def exposure_rows(self, request_id: str) -> list[ExposureRow]:
        """The request's exposure rows, ordered by event id."""
        ...

    def insert_exposure(self, event: UserPlaybookExposureEvent) -> ExposureRow:
        """Store ``event`` directly, as the backend's writer would, and read it."""
        ...

    def block_subject(self, user_id: str) -> None:
        """Place an active erasure barrier on ``user_id``'s subject."""
        ...


def _uid(label: str) -> str:
    return f"{label}-{uuid4().hex[:10]}"


def _owned_playbook(harness: ExposureHarness, owner: str) -> UserPlaybook:
    """A current playbook saved through storage, findable by a unique word."""
    playbook = UserPlaybook(
        user_id=owner,
        playbook_name=_uid("pb"),
        agent_version="v1",
        request_id=_uid("seed"),
        content=f"synthetic guidance zq{uuid4().hex[:12]}",
        trigger="synthetic trigger",
    )
    harness.storage.save_user_playbooks([playbook])
    assert playbook.user_playbook_id
    return playbook


def _served(harness: ExposureHarness, playbook: UserPlaybook) -> UserPlaybook:
    """The playbook as keyword search serves it."""
    found = harness.storage.search_user_playbooks(
        SearchUserPlaybookRequest(
            query=playbook.content.rsplit(" ", 1)[-1],
            user_id=playbook.user_id,
            search_mode=SearchMode.FTS,
        )
    )
    (served,) = [
        item for item in found if item.user_playbook_id == playbook.user_playbook_id
    ]
    return served


def _receipt(
    harness: ExposureHarness,
    served: UserPlaybook,
    *,
    session_id: str,
    user_id: str | None,
) -> str:
    batch = SearchExposureBatch(
        org_id=harness.storage.org_id,
        request_id=None,
        session_id=session_id,
        interaction_id=None,
        user_id=user_id,
        user_playbooks=(served,),
    )
    issuer = ExposureReceiptIssuer(project_id=lambda: harness.receipt_project_id)
    return issuer.issue(batch)[served.user_playbook_id]


def _ref(playbook_id: int, receipt: str | None = None) -> RetrievedLearning:
    return RetrievedLearning(
        kind="user_playbook", learning_id=str(playbook_id), exposure_receipt=receipt
    )


def _interaction(
    user_id: str, request_id: str, refs: list[RetrievedLearning]
) -> Interaction:
    return Interaction(
        user_id=user_id,
        request_id=request_id,
        role="Assistant",
        content="synthetic answer",
        retrieved_learnings=refs,
        embedding=list(_EMBEDDING),
    )


def _publish(
    harness: ExposureHarness,
    *,
    user_id: str,
    session_id: str,
    interactions: list[Interaction],
    store_request: bool = True,
) -> None:
    """Publish one request and its interactions in one transaction."""
    storage = harness.storage
    with storage.commit_scope():
        if store_request:
            storage.add_request(
                Request(
                    request_id=interactions[0].request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
            )
        storage.add_user_interactions_bulk(
            user_id, interactions, embeddings_prepared=True
        )


def _event_id(harness: ExposureHarness, interaction: Interaction, ordinal: int) -> str:
    identity = {
        "org_id": harness.storage.org_id,
        "project_id": harness.receipt_project_id,
        "request_id": interaction.request_id,
        "interaction_id": interaction.interaction_id,
        "ordinal": ordinal,
    }
    return "publish-v1:" + sha256(canonical_json_bytes(identity)).hexdigest()


def _stored_interactions(harness: ExposureHarness, user_id: str) -> list[Interaction]:
    return list(harness.storage.get_user_interaction(user_id))


def _base_event(**changes: Any) -> UserPlaybookExposureEvent:
    event = UserPlaybookExposureEvent(
        exposure_event_id=_uid("publish-v1:integrity"),
        request_id="request",
        session_id="session",
        user_id="viewer",
        playbook_owner_user_id="owner",
        user_playbook_id=7,
        served_semantic_digest=_DIGEST,
        served_full_version_fingerprint=_DIGEST,
        exposed_at=1_000,
        ingested_at=1_000,
        governance_subject_ref=_REF,
        playbook_owner_governance_subject_ref=_REF,
    )
    return replace(event, **changes)


#: Each event shape and the integrity the ledger must store for it, on every
#: backend. Whitespace is trimmed of spaces only, as Postgres ``btrim`` does;
#: patterns match the whole value (a trailing newline is invalid).
_INTEGRITY_CASES: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "complete": ({}, ()),
    "session_only_is_correlated": ({"request_id": None}, ()),
    "tab_is_not_blank": ({"user_id": "\t"}, ()),
    "no_correlation": (
        {"request_id": "  ", "session_id": None},
        ("missing_correlation",),
    ),
    "blank_user": ({"user_id": "  "}, ("missing_user_id",)),
    "no_owner": ({"playbook_owner_user_id": None}, ("missing_playbook_owner_user_id",)),
    "no_playbook": ({"user_playbook_id": None}, ("missing_playbook_id",)),
    "zero_playbook": ({"user_playbook_id": 0}, ("invalid_playbook_id",)),
    "blank_semantic": (
        {"served_semantic_digest": " "},
        ("missing_served_semantic_digest",),
    ),
    "upper_semantic": (
        {"served_semantic_digest": "A" * 64},
        ("invalid_served_semantic_digest",),
    ),
    "newline_fingerprint": (
        {"served_full_version_fingerprint": _DIGEST + "\n"},
        ("invalid_served_full_version_fingerprint",),
    ),
    "negative_exposed_at": ({"exposed_at": -1}, ("invalid_exposed_at",)),
    "bad_subject": (
        {"governance_subject_ref": "subref_v1_xyz"},
        ("invalid_governance_identity",),
    ),
    "no_owner_subject": (
        {"playbook_owner_governance_subject_ref": None},
        ("missing_playbook_owner_governance_identity",),
    ),
    "every_reason_in_order": (
        {
            "request_id": None,
            "session_id": None,
            "user_id": None,
            "playbook_owner_user_id": "",
            "user_playbook_id": None,
            "served_semantic_digest": None,
            "served_full_version_fingerprint": "f" * 63,
            "exposed_at": None,
            "governance_subject_ref": "",
            "playbook_owner_governance_subject_ref": "subref_v2_" + "b" * 32,
        },
        (
            "missing_correlation",
            "missing_user_id",
            "missing_playbook_owner_user_id",
            "missing_playbook_id",
            "missing_served_semantic_digest",
            "invalid_served_full_version_fingerprint",
            "missing_exposed_at",
            "missing_governance_identity",
            "invalid_playbook_owner_governance_identity",
        ),
    ),
}


class PublishedExposureContract:
    """The published-exposure ledger, asserted identically on every backend."""

    # ----------------------------------------------------------------- I10
    # Integrity state: one Python definition, one SQL definition

    @pytest.mark.parametrize("case", sorted(_INTEGRITY_CASES))
    def test_integrity_state_and_reasons_agree(
        self, port_harness: ExposureHarness, case: str
    ) -> None:
        changes, reasons = _INTEGRITY_CASES[case]

        row = port_harness.insert_exposure(_base_event(**changes))

        assert row.integrity_reasons == reasons
        assert row.integrity_state == ("incomplete" if reasons else "complete")

    # ----------------------------------------------------------------- I10
    # Served provenance only from a verified receipt

    def test_a_verified_receipt_records_a_complete_exposure(
        self, port_harness: ExposureHarness
    ) -> None:
        owner, session, request_id = _uid("owner"), _uid("session"), _uid("request")
        playbook = _owned_playbook(port_harness, owner)
        served = _served(port_harness, playbook)
        before = port_harness.now()
        receipt = _receipt(port_harness, served, session_id=session, user_id=owner)
        interaction = _interaction(
            owner,
            request_id,
            [
                RetrievedLearning(kind="profile", learning_id="synthetic-profile"),
                _ref(served.user_playbook_id, receipt),
            ],
        )

        _publish(
            port_harness, user_id=owner, session_id=session, interactions=[interaction]
        )

        stored = port_harness.storage.get_user_playbook_by_id(served.user_playbook_id)
        (row,) = port_harness.exposure_rows(request_id)
        assert row.exposure_event_id == _event_id(port_harness, interaction, 1), (
            "the ordinal counts every retrieved learning, not only playbooks"
        )
        assert (row.request_id, row.session_id, row.user_id) == (
            request_id,
            session,
            owner,
        )
        assert (row.playbook_owner_user_id, row.user_playbook_id) == (
            owner,
            served.user_playbook_id,
        )
        assert row.served_full_version_fingerprint == (
            user_playbook_full_version_fingerprint(stored)
        )
        assert row.served_semantic_digest == incumbent_user_playbook_semantic_digest(
            content_digest=sha256(stored.content.encode()).hexdigest(),
            trigger=stored.trigger,
        )
        assert row.exposed_at is not None and row.exposed_at >= before - 1
        assert row.governance_subject_ref == row.playbook_owner_governance_subject_ref
        assert (row.integrity_state, row.integrity_reasons) == ("complete", ())

    @pytest.mark.parametrize(
        "receipt_kind", ["absent", "tampered", "other_session", "other_playbook"]
    )
    def test_without_a_verifiable_receipt_the_exposure_is_incomplete(
        self, port_harness: ExposureHarness, receipt_kind: str
    ) -> None:
        owner, session, request_id = _uid("owner"), _uid("session"), _uid("request")
        served = _served(port_harness, _owned_playbook(port_harness, owner))
        other = _served(port_harness, _owned_playbook(port_harness, owner))
        receipt: str | None = None
        if receipt_kind == "tampered":
            valid = _receipt(port_harness, served, session_id=session, user_id=owner)
            receipt = valid[:-1] + ("1" if valid[-1] == "0" else "0")
        elif receipt_kind == "other_session":
            receipt = _receipt(
                port_harness, served, session_id=_uid("session"), user_id=owner
            )
        elif receipt_kind == "other_playbook":
            receipt = _receipt(port_harness, other, session_id=session, user_id=owner)
        interaction = _interaction(
            owner, request_id, [_ref(served.user_playbook_id, receipt)]
        )

        _publish(
            port_harness, user_id=owner, session_id=session, interactions=[interaction]
        )

        (row,) = port_harness.exposure_rows(request_id)
        assert row.exposure_event_id == _event_id(port_harness, interaction, 0)
        assert (row.playbook_owner_user_id, row.user_playbook_id) == (
            owner,
            served.user_playbook_id,
        )
        assert row.served_semantic_digest is None
        assert row.served_full_version_fingerprint is None
        assert row.exposed_at is None
        assert (row.integrity_state, row.integrity_reasons) == (
            "incomplete",
            _DIGEST_REASONS,
        )

    # ----------------------------------------------------------------- I10
    # Atomicity

    def test_exposures_roll_back_with_their_interaction(
        self, port_harness: ExposureHarness
    ) -> None:
        owner, session = _uid("owner"), _uid("session")
        playbook_id = _owned_playbook(port_harness, owner).user_playbook_id
        aborted, committed = _uid("request"), _uid("request")

        with (
            pytest.raises(RuntimeError, match="abort the publish"),
            port_harness.storage.commit_scope(),
        ):
            _publish(
                port_harness,
                user_id=owner,
                session_id=session,
                interactions=[_interaction(owner, aborted, [_ref(playbook_id)])],
            )
            raise RuntimeError("abort the publish")

        assert port_harness.exposure_rows(aborted) == []
        assert _stored_interactions(port_harness, owner) == []
        _publish(
            port_harness,
            user_id=owner,
            session_id=session,
            interactions=[_interaction(owner, committed, [_ref(playbook_id)])],
        )
        assert len(port_harness.exposure_rows(committed)) == 1, (
            "the same publish, committed, does record"
        )

    def test_a_refused_exposure_rolls_back_its_interaction(
        self, port_harness: ExposureHarness
    ) -> None:
        owner, viewer = _uid("owner"), _uid("viewer")
        request_id = _uid("request")
        playbook = _owned_playbook(port_harness, owner)
        port_harness.block_subject(owner)
        interaction = _interaction(
            viewer, request_id, [_ref(playbook.user_playbook_id)]
        )

        with pytest.raises(Exception, match="erasure barrier"):
            _publish(
                port_harness,
                user_id=viewer,
                session_id=_uid("session"),
                interactions=[interaction],
            )

        assert port_harness.exposure_rows(request_id) == []
        assert _stored_interactions(port_harness, viewer) == []

    @pytest.mark.parametrize("cause", ["unknown_playbook", "no_stored_request"])
    def test_an_unattributable_reference(
        self, port_harness: ExposureHarness, cause: str
    ) -> None:
        owner, viewer = _uid("owner"), _uid("viewer")
        request_id = _uid("request")
        playbook_id = (
            2**62
            if cause == "unknown_playbook"
            else _owned_playbook(port_harness, owner).user_playbook_id
        )
        interaction = _interaction(viewer, request_id, [_ref(playbook_id)])

        def publish() -> None:
            _publish(
                port_harness,
                user_id=viewer,
                session_id=_uid("session"),
                interactions=[interaction],
                store_request=cause != "no_stored_request",
            )

        if port_harness.refuses_unattributable_exposure:
            with pytest.raises(Exception):  # noqa: B017 -- backend-specific type.
                publish()
            assert _stored_interactions(port_harness, viewer) == []
        else:
            publish()
            assert len(_stored_interactions(port_harness, viewer)) == 1
        assert port_harness.exposure_rows(request_id) == []

    # ----------------------------------------------------------------- I11

    def test_a_replay_records_nothing_new(self, port_harness: ExposureHarness) -> None:
        owner, session, request_id = _uid("owner"), _uid("session"), _uid("request")
        served = _served(port_harness, _owned_playbook(port_harness, owner))
        receipt = _receipt(port_harness, served, session_id=session, user_id=owner)
        interaction = _interaction(
            owner, request_id, [_ref(served.user_playbook_id, receipt)]
        )
        _publish(
            port_harness, user_id=owner, session_id=session, interactions=[interaction]
        )
        recorded = port_harness.exposure_rows(request_id)
        assert len(recorded) == 1

        assert interaction.interaction_id, "the store assigned the stored id"
        port_harness.storage.add_user_interactions_bulk(
            owner, [interaction], embeddings_prepared=True
        )

        assert port_harness.exposure_rows(request_id) == recorded
        assert len(_stored_interactions(port_harness, owner)) == 1

    def test_event_ids_are_one_per_interaction_and_ordinal(
        self, port_harness: ExposureHarness
    ) -> None:
        owner, session, request_id = _uid("owner"), _uid("session"), _uid("request")
        playbook_id = _owned_playbook(port_harness, owner).user_playbook_id
        first = _interaction(owner, request_id, [_ref(playbook_id), _ref(playbook_id)])
        second = _interaction(owner, request_id, [_ref(playbook_id)])

        _publish(
            port_harness,
            user_id=owner,
            session_id=session,
            interactions=[first, second],
        )

        assert first.interaction_id != second.interaction_id
        expected = sorted(
            [
                _event_id(port_harness, first, 0),
                _event_id(port_harness, first, 1),
                _event_id(port_harness, second, 0),
            ]
        )
        assert [
            row.exposure_event_id for row in port_harness.exposure_rows(request_id)
        ] == expected

    # ----------------------------------------------------------------- I12

    def test_search_serves_the_stored_full_version(
        self, port_harness: ExposureHarness
    ) -> None:
        playbook = _owned_playbook(port_harness, _uid("owner"))

        served = _served(port_harness, playbook)

        stored = port_harness.storage.get_user_playbook_by_id(playbook.user_playbook_id)
        assert stored is not None
        assert user_playbook_full_version_fingerprint(served) == (
            user_playbook_full_version_fingerprint(stored)
        )

    # ------------------------------------------------------- I18 (exposure)

    def test_clearing_a_user_deletes_their_exposures_as_viewer_and_owner(
        self, port_harness: ExposureHarness
    ) -> None:
        first_owner, second_owner = _uid("owner"), _uid("owner")
        viewer, bystander = _uid("viewer"), _uid("viewer")
        first = _owned_playbook(port_harness, first_owner).user_playbook_id
        second = _owned_playbook(port_harness, second_owner).user_playbook_id
        viewer_request, bystander_request = _uid("request"), _uid("request")
        _publish(
            port_harness,
            user_id=viewer,
            session_id=_uid("session"),
            interactions=[
                _interaction(viewer, viewer_request, [_ref(first), _ref(second)])
            ],
        )
        _publish(
            port_harness,
            user_id=bystander,
            session_id=_uid("session"),
            interactions=[_interaction(bystander, bystander_request, [_ref(second)])],
        )

        def remaining() -> set[tuple[str | None, int | None]]:
            return {
                (row.user_id, row.user_playbook_id)
                for request in (viewer_request, bystander_request)
                for row in port_harness.exposure_rows(request)
            }

        assert remaining() == {(viewer, first), (viewer, second), (bystander, second)}
        port_harness.storage.clear_user_data(first_owner)
        assert remaining() == {(viewer, second), (bystander, second)}, (
            "an owner's clear removes exposures of their playbooks only"
        )
        port_harness.storage.clear_user_data(viewer)
        assert remaining() == {(bystander, second)}, (
            "a viewer's clear removes their own exposures only"
        )
