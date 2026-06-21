"""Contract tests for generic row-retention storage methods."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    Interaction,
    ProfileTimeToLive,
    Request,
    UserActionType,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _make_request(request_id: str, created_at: int) -> Request:
    return Request(
        request_id=request_id,
        user_id="u1",
        session_id="test_session",
        created_at=created_at,
        source="test",
        agent_version="v1",
    )


def _make_interaction(
    interaction_id: int, request_id: str, created_at: int
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id="u1",
        request_id=request_id,
        content=f"interaction {interaction_id}",
        created_at=created_at,
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
    )


def _make_profile(profile_id: str) -> UserProfile:
    return UserProfile(
        user_id="u1",
        profile_id=profile_id,
        content=f"profile {profile_id}",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        source="test",
    )


def test_retention_deletes_oldest_interactions(storage: BaseStorage) -> None:
    now = int(datetime.now(UTC).timestamp())
    for i in range(1, 6):
        storage.add_user_interaction("u1", _make_interaction(i, f"req{i}", now + i))

    assert storage.count_retention_target_rows("interactions") == 5

    deleted = storage.delete_oldest_retention_target_rows("interactions", 2)

    assert deleted == 2
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {3, 4, 5}


def test_retention_deletes_oldest_profiles(storage: BaseStorage) -> None:
    storage.add_user_profile("u1", [_make_profile("p1")])
    storage.add_user_profile("u1", [_make_profile("p2")])
    storage.add_user_profile("u1", [_make_profile("p3")])
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("1", "p1"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("2", "p2"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("3", "p3"))
    conn.commit()

    deleted = storage.delete_oldest_retention_target_rows("profiles", 2)

    assert deleted == 2
    remaining = storage.get_all_profiles(limit=10, status_filter=[None])
    assert {profile.profile_id for profile in remaining} == {"p3"}


def test_retention_deletes_requests_before_orphaning_interactions(
    storage: BaseStorage,
) -> None:
    now = int(datetime.now(UTC).timestamp())
    for i in range(1, 4):
        request_id = f"req{i}"
        storage.add_request(_make_request(request_id, now + i))
        storage.add_user_interaction("u1", _make_interaction(i, request_id, now + i))

    deleted = storage.delete_oldest_retention_target_rows("requests", 2)

    assert deleted == 2
    assert storage.get_request("req1") is None
    assert storage.get_request("req2") is None
    assert storage.get_request("req3") is not None
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.request_id for interaction in remaining} == {"req3"}


def test_retention_interaction_delete_cleans_fts(storage: BaseStorage) -> None:
    """After retaining interactions, their fts rows must also be gone."""
    now = int(datetime.now(UTC).timestamp())
    for i in range(1, 4):
        storage.add_user_interaction("u1", _make_interaction(i, f"req{i}", now + i))

    conn = storage.conn  # type: ignore[attr-defined]

    # Verify fts rows exist before deletion.
    fts_before = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid IN (1, 2, 3)"
    ).fetchall()
    assert len(fts_before) == 3

    deleted = storage.delete_oldest_retention_target_rows("interactions", 2)

    assert deleted == 2
    fts_after = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid IN (1, 2)"
    ).fetchall()
    assert fts_after == [], "fts rows for deleted interactions must be gone"
    fts_kept = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid = 3"
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for retained interaction must remain"


def test_retention_profile_delete_cleans_fts(storage: BaseStorage) -> None:
    """After retaining profiles, their fts rows must also be gone."""
    storage.add_user_profile("u1", [_make_profile("p1")])
    storage.add_user_profile("u1", [_make_profile("p2")])
    storage.add_user_profile("u1", [_make_profile("p3")])
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("1", "p1"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("2", "p2"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("3", "p3"))
    conn.commit()

    # Verify fts rows exist before deletion.
    fts_before = conn.execute(
        "SELECT profile_id FROM profiles_fts WHERE profile_id IN ('p1', 'p2', 'p3')"
    ).fetchall()
    assert len(fts_before) == 3

    deleted = storage.delete_oldest_retention_target_rows("profiles", 2)

    assert deleted == 2
    fts_after = conn.execute(
        "SELECT profile_id FROM profiles_fts WHERE profile_id IN ('p1', 'p2')"
    ).fetchall()
    assert fts_after == [], "fts rows for deleted profiles must be gone"
    fts_kept = conn.execute(
        "SELECT profile_id FROM profiles_fts WHERE profile_id = 'p3'"
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for retained profile must remain"


def test_retention_request_cascade_cleans_interaction_fts(
    storage: BaseStorage,
) -> None:
    """Request-cascade delete must also remove interaction fts rows."""
    now = int(datetime.now(UTC).timestamp())
    for i in range(1, 4):
        request_id = f"req{i}"
        storage.add_request(_make_request(request_id, now + i))
        storage.add_user_interaction("u1", _make_interaction(i, request_id, now + i))

    conn = storage.conn  # type: ignore[attr-defined]
    fts_before = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid IN (1, 2, 3)"
    ).fetchall()
    assert len(fts_before) == 3

    deleted = storage.delete_oldest_retention_target_rows("requests", 2)

    assert deleted == 2
    # Interactions 1 and 2 were cascaded; their fts rows must be gone.
    fts_after = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid IN (1, 2)"
    ).fetchall()
    assert fts_after == [], "fts rows for cascaded interactions must be gone"
    fts_kept = conn.execute(
        "SELECT rowid FROM interactions_fts WHERE rowid = 3"
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for surviving interaction must remain"


# ---------------------------------------------------------------------------
# Playbook retention FTS + vec cleanup (B3h)
# ---------------------------------------------------------------------------


def _make_user_playbook(user_playbook_id: int) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id="u1",
        playbook_name="pb",
        agent_version="v1",
        request_id=f"req-{user_playbook_id}",
        content=f"content-{user_playbook_id}",
        created_at=user_playbook_id,
        source="test",
        source_interaction_ids=[],
    )


def _make_agent_playbook(agent_playbook_id: int) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=agent_playbook_id,
        playbook_name="pb",
        agent_version="v1",
        content=f"content-{agent_playbook_id}",
        created_at=agent_playbook_id,
    )


def test_retention_user_playbook_delete_cleans_fts(storage: BaseStorage) -> None:
    """After retention-deleting user_playbooks, their fts rows must be gone."""
    storage.save_user_playbooks(
        [_make_user_playbook(1), _make_user_playbook(2), _make_user_playbook(3)]
    )
    conn = storage.conn  # type: ignore[attr-defined]

    saved = conn.execute(
        "SELECT user_playbook_id FROM user_playbooks ORDER BY user_playbook_id"
    ).fetchall()
    assert len(saved) == 3
    saved_ids = [r["user_playbook_id"] for r in saved]

    # Assign distinct created_at values so deletion order is deterministic.
    for i, upid in enumerate(saved_ids, start=1):
        conn.execute(
            "UPDATE user_playbooks SET created_at = ? WHERE user_playbook_id = ?",
            (i, upid),
        )
    conn.commit()

    ph3 = ",".join("?" for _ in saved_ids)
    fts_before = conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph3})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(fts_before) == 3, "fts rows must exist before retention delete"

    deleted = storage.delete_oldest_retention_target_rows("user_playbooks", 2)

    assert deleted == 2

    # The two oldest entries' fts rows must be gone.
    oldest_ids = saved_ids[:2]
    ph2 = ",".join("?" for _ in oldest_ids)
    fts_after = conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph2})",  # noqa: S608
        oldest_ids,
    ).fetchall()
    assert fts_after == [], "fts rows for retention-deleted user_playbooks must be gone"

    # The surviving entry's fts row must remain.
    kept_id = saved_ids[2]
    fts_kept = conn.execute(
        "SELECT rowid FROM user_playbooks_fts WHERE rowid = ?", (kept_id,)
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for surviving user_playbook must remain"


def test_retention_agent_playbook_delete_cleans_fts(storage: BaseStorage) -> None:
    """After retention-deleting agent_playbooks, their fts rows must be gone."""
    storage.save_agent_playbooks(
        [_make_agent_playbook(1), _make_agent_playbook(2), _make_agent_playbook(3)]
    )
    conn = storage.conn  # type: ignore[attr-defined]

    saved = conn.execute(
        "SELECT agent_playbook_id FROM agent_playbooks ORDER BY agent_playbook_id"
    ).fetchall()
    assert len(saved) == 3
    saved_ids = [r["agent_playbook_id"] for r in saved]

    # Assign distinct created_at values so deletion order is deterministic.
    for i, apid in enumerate(saved_ids, start=1):
        conn.execute(
            "UPDATE agent_playbooks SET created_at = ? WHERE agent_playbook_id = ?",
            (i, apid),
        )
    conn.commit()

    ph3 = ",".join("?" for _ in saved_ids)
    fts_before = conn.execute(
        f"SELECT rowid FROM agent_playbooks_fts WHERE rowid IN ({ph3})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(fts_before) == 3, "fts rows must exist before retention delete"

    deleted = storage.delete_oldest_retention_target_rows("agent_playbooks", 2)

    assert deleted == 2

    # The two oldest entries' fts rows must be gone.
    oldest_ids = saved_ids[:2]
    ph2 = ",".join("?" for _ in oldest_ids)
    fts_after = conn.execute(
        f"SELECT rowid FROM agent_playbooks_fts WHERE rowid IN ({ph2})",  # noqa: S608
        oldest_ids,
    ).fetchall()
    assert fts_after == [], (
        "fts rows for retention-deleted agent_playbooks must be gone"
    )

    # The surviving entry's fts row must remain.
    kept_id = saved_ids[2]
    fts_kept = conn.execute(
        "SELECT rowid FROM agent_playbooks_fts WHERE rowid = ?", (kept_id,)
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for surviving agent_playbook must remain"
