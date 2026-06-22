"""Contract tests for the lineage legal-hold store (Task 1 of legal-hold feature).

Runs against every locally-testable backend via the parametrized ``storage``
fixture. Exercises the CRUD surface (place_hold / release_hold / get_holds) and
the ``_is_on_legal_hold`` resolver across all three hold scopes (org / user /
entity), including the agent_playbook special-case (no ``user_id`` column).

The held-row-survives-delete invariant skeleton at the bottom is populated by
Tasks 2-4.
"""

import sqlite3

import pytest

from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — minimal raw inserts so user-scope resolution can be tested.
# ---------------------------------------------------------------------------


def _insert_profile(storage: BaseStorage, profile_id: str, user_id: str) -> None:
    """Insert a minimal profiles row (profile_id TEXT, user_id TEXT NOT NULL)."""
    storage.conn.execute(  # type: ignore[attr-defined]
        "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp) "
        "VALUES (?, ?, ?)",
        (profile_id, user_id, 0),
    )
    storage.conn.commit()  # type: ignore[attr-defined]


def _insert_user_playbook(storage: BaseStorage, user_id: str | None) -> int:
    """Insert a minimal user_playbooks row; return its autoincrement id."""
    cur = storage.conn.execute(  # type: ignore[attr-defined]
        "INSERT INTO user_playbooks (user_id, created_at, request_id) VALUES (?, ?, ?)",
        (user_id, "1970-01-01T00:00:00", "req"),
    )
    storage.conn.commit()  # type: ignore[attr-defined]
    return int(cur.lastrowid)


def _insert_agent_playbook(storage: BaseStorage) -> int:
    """Insert a minimal agent_playbooks row (no user_id column); return its id."""
    cur = storage.conn.execute(  # type: ignore[attr-defined]
        "INSERT INTO agent_playbooks (created_at) VALUES (?)",
        ("1970-01-01T00:00:00",),
    )
    storage.conn.commit()  # type: ignore[attr-defined]
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# CRUD surface
# ---------------------------------------------------------------------------


def test_place_and_get_hold(storage: BaseStorage) -> None:
    pid = "p1"
    _insert_profile(storage, pid, "u1")
    hold_id = storage.place_hold(
        org_id="contract_test",
        scope="entity",
        entity_type="profile",
        entity_id=pid,
        matter_id="m1",
        legal_basis="litigation_hold",
        reason="lawsuit",
        placed_by="legal@x.com",
    )
    assert hold_id > 0

    holds = storage.get_holds("contract_test")
    assert len(holds) == 1
    h = holds[0]
    assert h.id == hold_id
    assert h.scope == "entity"
    assert h.entity_type == "profile"
    assert h.entity_id == pid
    assert h.matter_id == "m1"
    assert h.legal_basis == "litigation_hold"
    assert h.released_at is None

    assert storage._is_on_legal_hold("contract_test", "profile", pid) is True


def test_release_hold_by_id(storage: BaseStorage) -> None:
    pid = "p1"
    _insert_profile(storage, pid, "u1")
    hold_id = storage.place_hold(
        org_id="contract_test",
        scope="entity",
        entity_type="profile",
        entity_id=pid,
        matter_id="m1",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
    )

    released = storage.release_hold(
        org_id="contract_test", hold_id=hold_id, released_by="legal@x.com"
    )
    assert released == 1

    assert storage.get_holds("contract_test", active_only=True) == []
    assert storage._is_on_legal_hold("contract_test", "profile", pid) is False


def test_hold_row_not_deleted_after_release(storage: BaseStorage) -> None:
    hold_id = storage.place_hold(
        org_id="contract_test",
        scope="org",
        matter_id="m1",
        legal_basis="regulatory_order",
        reason="",
        placed_by="",
    )
    storage.release_hold(
        org_id="contract_test", hold_id=hold_id, released_by="legal@x.com"
    )

    active = storage.get_holds("contract_test", active_only=True)
    assert active == []

    all_holds = storage.get_holds("contract_test", active_only=False)
    assert len(all_holds) == 1
    h = all_holds[0]
    assert h.id == hold_id
    assert h.released_at is not None
    assert h.released_by == "legal@x.com"


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def test_scope_org_covers_all_entity_types(storage: BaseStorage) -> None:
    pid = "p1"
    _insert_profile(storage, pid, "u1")
    upid = _insert_user_playbook(storage, "u1")
    apid = _insert_agent_playbook(storage)

    storage.place_hold(
        org_id="contract_test",
        scope="org",
        matter_id="m1",
        legal_basis="legal_obligation",
        reason="",
        placed_by="",
    )

    assert storage._is_on_legal_hold("contract_test", "profile", pid) is True
    assert (
        storage._is_on_legal_hold("contract_test", "user_playbook", str(upid)) is True
    )
    assert (
        storage._is_on_legal_hold("contract_test", "agent_playbook", str(apid)) is True
    )


def test_scope_user_covers_profile_and_user_playbook_not_agent_playbook(
    storage: BaseStorage,
) -> None:
    _insert_profile(storage, "p_x", "user_x")
    _insert_profile(storage, "p_y", "user_y")
    upid_x = _insert_user_playbook(storage, "user_x")
    upid_y = _insert_user_playbook(storage, "user_y")
    apid = _insert_agent_playbook(storage)

    storage.place_hold(
        org_id="contract_test",
        scope="user",
        user_id="user_x",
        matter_id="m1",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
    )

    # user_x's entities are held.
    assert storage._is_on_legal_hold("contract_test", "profile", "p_x") is True
    assert (
        storage._is_on_legal_hold("contract_test", "user_playbook", str(upid_x)) is True
    )
    # user_y's profile and playbook are NOT held.
    assert storage._is_on_legal_hold("contract_test", "profile", "p_y") is False
    assert (
        storage._is_on_legal_hold("contract_test", "user_playbook", str(upid_y))
        is False
    )
    # agent_playbook has no user_id column — user-scope holds never cover it.
    assert (
        storage._is_on_legal_hold("contract_test", "agent_playbook", str(apid)) is False
    )


def test_scope_entity_covers_only_that_row(storage: BaseStorage) -> None:
    _insert_profile(storage, "p1", "u1")
    _insert_profile(storage, "p2", "u1")

    storage.place_hold(
        org_id="contract_test",
        scope="entity",
        entity_type="profile",
        entity_id="p1",
        matter_id="m1",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
    )

    assert storage._is_on_legal_hold("contract_test", "profile", "p1") is True
    # Same user, different profile: an entity-scope hold must not bleed.
    assert storage._is_on_legal_hold("contract_test", "profile", "p2") is False


def test_null_entity_id_org_hold_resolves(storage: BaseStorage) -> None:
    pid = "p1"
    _insert_profile(storage, pid, "u1")
    storage.place_hold(
        org_id="contract_test",
        scope="org",
        entity_type=None,
        entity_id=None,
        matter_id="m1",
        legal_basis="regulatory_order",
        reason="",
        placed_by="",
    )

    holds = storage.get_holds("contract_test")
    assert holds[0].entity_type is None
    assert holds[0].entity_id is None
    assert storage._is_on_legal_hold("contract_test", "profile", pid) is True


# ---------------------------------------------------------------------------
# Constraints & partial release
# ---------------------------------------------------------------------------


def test_check_released_at_constraint(storage: BaseStorage) -> None:
    hold_id = storage.place_hold(
        org_id="contract_test",
        scope="org",
        matter_id="m1",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
        placed_at=1000,
    )
    # A released_at strictly before placed_at violates the table CHECK.
    with pytest.raises(sqlite3.IntegrityError):
        storage.conn.execute(  # type: ignore[attr-defined]
            "UPDATE lineage_legal_hold SET released_at = ? WHERE id = ?",
            (500, hold_id),
        )
        storage.conn.commit()  # type: ignore[attr-defined]


def test_partial_release_by_matter(storage: BaseStorage) -> None:
    storage.place_hold(
        org_id="contract_test",
        scope="user",
        user_id="user_x",
        matter_id="shared",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
    )
    storage.place_hold(
        org_id="contract_test",
        scope="user",
        user_id="user_y",
        matter_id="shared",
        legal_basis="litigation_hold",
        reason="",
        placed_by="",
    )

    released = storage.release_hold(
        org_id="contract_test",
        matter_id="shared",
        scope="user",
        user_id="user_x",
        released_by="legal@x.com",
    )
    assert released == 1

    active = storage.get_holds("contract_test", active_only=True)
    assert len(active) == 1
    assert active[0].user_id == "user_y"


# ---------------------------------------------------------------------------
# Invariant skeleton (Tasks 2-4)
# ---------------------------------------------------------------------------


def test_held_row_survives_delete_paths_skeleton() -> None:
    """Invariant skeleton: a held row must survive every hard-delete path.

    TODO(T2): GC enforcement — add gc_expired_tombstones case.
    TODO(T3): Other delete paths (delete_all_*, retention trimmer).
    TODO(T4): Account deletion / org-purge.
    """
    pass  # Populated by Tasks 2-4
