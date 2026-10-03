"""Contract tests for generic row-retention storage methods."""

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    Interaction,
    ProfileTimeToLive,
    Request,
    Status,
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

    assert storage.count_retention_target_rows("interactions") == 5  # type: ignore[attr-defined]

    deleted = storage.delete_oldest_retention_target_rows("interactions", 2)  # type: ignore[attr-defined]
    assert deleted == 2
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {3, 4, 5}


def test_probe_counts_exactly_unless_the_estimate_settles_it(
    storage: BaseStorage,
) -> None:
    """The batched probe answers what the per-target estimate + count did."""
    now = int(datetime.now(UTC).timestamp())
    for i in range(1, 4):
        storage.add_user_interaction("u1", _make_interaction(i, f"req{i}", now + i))

    probes = storage.probe_retention_targets(  # type: ignore[attr-defined]
        {"interactions": 0.0, "profiles": 0.0}
    )

    assert set(probes) == {"interactions", "profiles"}
    assert (probes["interactions"].rows, probes["interactions"].exact) == (3, True)
    assert (probes["profiles"].rows, probes["profiles"].exact) == (0, True)
    assert all(probe.error is None for probe in probes.values())


def test_retention_deletes_oldest_profiles(storage: BaseStorage) -> None:
    storage.add_user_profile("u1", [_make_profile("p1")])
    storage.add_user_profile("u1", [_make_profile("p2")])
    storage.add_user_profile("u1", [_make_profile("p3")])
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("1", "p1"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("2", "p2"))
    conn.execute("UPDATE profiles SET created_at = ? WHERE profile_id = ?", ("3", "p3"))
    conn.commit()

    deleted = storage.delete_oldest_retention_target_rows("profiles", 2)  # type: ignore[attr-defined]
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

    deleted = storage.delete_oldest_retention_target_rows("requests", 2)  # type: ignore[attr-defined]
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

    deleted = storage.delete_oldest_retention_target_rows("interactions", 2)  # type: ignore[attr-defined]
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

    deleted = storage.delete_oldest_retention_target_rows("profiles", 2)  # type: ignore[attr-defined]
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

    deleted = storage.delete_oldest_retention_target_rows("requests", 2)  # type: ignore[attr-defined]
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


def test_retention_exposure_age_boundary_is_strict(storage: BaseStorage) -> None:
    """Only exposure evidence strictly older than 14 days is row-cap eligible."""
    from reflexio.server.services.storage.retention import (
        OPEN_WORLD_EVIDENCE_RETENTION_WINDOW_SECONDS,
    )

    now = 2_000_000_000
    cutoff = now - OPEN_WORLD_EVIDENCE_RETENTION_WINDOW_SECONDS
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute(
        """CREATE TABLE user_playbook_exposure_events (
            exposure_event_id TEXT PRIMARY KEY,
            ingested_at INTEGER NOT NULL
        )"""
    )
    conn.executemany(
        """INSERT INTO user_playbook_exposure_events
           (exposure_event_id, ingested_at)
           VALUES (?, ?)""",
        [
            ("older", cutoff - 1),
            ("exact", cutoff),
            ("newer", cutoff + 1),
        ],
    )
    conn.commit()

    retention_storage = cast(Any, storage)
    with patch(
        "reflexio.server.services.storage.retention_mixin.time.time",
        return_value=now,
    ):
        deleted = retention_storage.delete_oldest_retention_target_rows(
            "user_playbook_exposure_events", 3
        )

    assert deleted == 1
    remaining = conn.execute(
        "SELECT exposure_event_id FROM user_playbook_exposure_events "
        "ORDER BY exposure_event_id"
    ).fetchall()
    assert [row["exposure_event_id"] for row in remaining] == ["exact", "newer"]


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
        trigger=f"trigger-{user_playbook_id}",
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


def _seed_status_retention_rows(
    storage: BaseStorage, target: str, statuses: list[Status]
) -> tuple[str, str, list[str | int]]:
    """Seed one row per status and return their IDs in insertion order."""
    conn = storage.conn  # type: ignore[attr-defined]
    if target == "profiles":
        ids: list[str | int] = [f"retention-profile-{i}" for i in range(len(statuses))]
        storage.add_user_profile(
            "u1", [_make_profile(str(profile_id)) for profile_id in ids]
        )
        table_name, id_column = "profiles", "profile_id"
    elif target == "user_playbooks":
        storage.save_user_playbooks(
            [_make_user_playbook(i + 1) for i in range(len(statuses))]
        )
        table_name, id_column = "user_playbooks", "user_playbook_id"
        ids = [
            row[id_column]
            for row in conn.execute(
                f"SELECT {id_column} FROM {table_name} ORDER BY {id_column}"  # noqa: S608
            ).fetchall()
        ]
    else:
        storage.save_agent_playbooks(
            [_make_agent_playbook(i + 1) for i in range(len(statuses))]
        )
        table_name, id_column = "agent_playbooks", "agent_playbook_id"
        ids = [
            row[id_column]
            for row in conn.execute(
                f"SELECT {id_column} FROM {table_name} ORDER BY {id_column}"  # noqa: S608
            ).fetchall()
        ]

    for created_at, (row_id, status) in enumerate(zip(ids, statuses, strict=True), 1):
        conn.execute(
            f"UPDATE {table_name} SET created_at = ?, status = ? "  # noqa: S608
            f"WHERE {id_column} = ?",
            (created_at, status.value, row_id),
        )
    conn.commit()
    return table_name, id_column, ids


@pytest.mark.parametrize("target", ["profiles", "user_playbooks", "agent_playbooks"])
def test_retention_prioritizes_terminal_tombstones(
    storage: BaseStorage, target: str
) -> None:
    statuses = [
        Status.CURRENT,
        Status.PENDING,
        Status.ARCHIVE_IN_PROGRESS,
        Status.ARCHIVED,
        Status.MERGED,
        Status.SUPERSEDED,
        Status.EXPIRED,
    ]
    table_name, id_column, ids = _seed_status_retention_rows(storage, target, statuses)

    deleted = storage.delete_oldest_retention_target_rows(target, 4)  # type: ignore[attr-defined]

    assert deleted == 4
    remaining_ids = {
        row[id_column]
        for row in storage.conn.execute(  # type: ignore[attr-defined]
            f"SELECT {id_column} FROM {table_name}"  # noqa: S608
        ).fetchall()
    }
    assert remaining_ids == set(ids[:3])


@pytest.mark.parametrize("target", ["profiles", "user_playbooks", "agent_playbooks"])
def test_retention_falls_back_to_oldest_rows_after_tombstones(
    storage: BaseStorage, target: str
) -> None:
    statuses = [
        Status.CURRENT,
        Status.PENDING,
        Status.ARCHIVE_IN_PROGRESS,
        Status.ARCHIVED,
        Status.MERGED,
        Status.SUPERSEDED,
    ]
    table_name, id_column, ids = _seed_status_retention_rows(storage, target, statuses)

    deleted = storage.delete_oldest_retention_target_rows(target, 5)  # type: ignore[attr-defined]

    assert deleted == 5
    remaining_ids = {
        row[id_column]
        for row in storage.conn.execute(  # type: ignore[attr-defined]
            f"SELECT {id_column} FROM {table_name}"  # noqa: S608
        ).fetchall()
    }
    assert remaining_ids == {ids[2]}


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

    deleted = storage.delete_oldest_retention_target_rows("user_playbooks", 2)  # type: ignore[attr-defined]
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

    deleted = storage.delete_oldest_retention_target_rows("agent_playbooks", 2)  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# delete_all_user_playbooks_by_status — atomicity + FTS/vec cleanup (B3h Fix 1)
# ---------------------------------------------------------------------------


def test_delete_all_user_playbooks_by_status_cleans_search_rows(
    storage: BaseStorage,
) -> None:
    """delete_all_user_playbooks_by_status must remove both the playbook rows and
    their FTS (and vec when sqlite-vec is available) entries atomically.
    A non-matching playbook's row and FTS entry must survive.
    """
    from reflexio.models.api_schema.service_schemas import Status

    # Two ARCHIVED playbooks (to be deleted) + one active (no status) that must survive.
    archived_pb = _make_user_playbook(1)
    archived_pb2 = _make_user_playbook(2)
    surviving_pb = _make_user_playbook(3)

    storage.save_user_playbooks([archived_pb, archived_pb2, surviving_pb])

    conn = storage.conn  # type: ignore[attr-defined]

    # Retrieve auto-assigned IDs.
    all_rows = conn.execute(
        "SELECT user_playbook_id FROM user_playbooks ORDER BY user_playbook_id"
    ).fetchall()
    assert len(all_rows) == 3
    saved_ids = [r["user_playbook_id"] for r in all_rows]
    archived_id1, archived_id2, surviving_id = saved_ids

    # Mark first two as ARCHIVED; leave the third with no status (active).
    conn.execute(
        "UPDATE user_playbooks SET status = ? WHERE user_playbook_id IN (?, ?)",
        (Status.ARCHIVED.value, archived_id1, archived_id2),
    )
    conn.commit()

    # Confirm FTS rows exist for all three before deletion.
    ph3 = ",".join("?" for _ in saved_ids)
    fts_before = conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph3})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(fts_before) == 3, "all three fts rows must exist before delete"

    deleted = storage.delete_all_user_playbooks_by_status(Status.ARCHIVED)

    assert deleted == 2

    # Playbook rows for deleted IDs must be gone.
    rows_after = conn.execute("SELECT user_playbook_id FROM user_playbooks").fetchall()
    remaining_ids = {r["user_playbook_id"] for r in rows_after}
    assert archived_id1 not in remaining_ids
    assert archived_id2 not in remaining_ids
    assert surviving_id in remaining_ids

    # FTS rows for deleted playbooks must be gone.
    fts_deleted = conn.execute(
        "SELECT rowid FROM user_playbooks_fts WHERE rowid IN (?, ?)",
        (archived_id1, archived_id2),
    ).fetchall()
    assert fts_deleted == [], "fts rows for deleted playbooks must be gone"

    # FTS row for the surviving playbook must remain.
    fts_kept = conn.execute(
        "SELECT rowid FROM user_playbooks_fts WHERE rowid = ?", (surviving_id,)
    ).fetchall()
    assert len(fts_kept) == 1, "fts row for surviving playbook must remain"

    # Vec rows for deleted playbooks must be gone (when sqlite-vec is available).
    has_vec = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_playbooks_vec'"
    ).fetchone()
    if has_vec:
        vec_deleted = conn.execute(
            "SELECT rowid FROM user_playbooks_vec WHERE rowid IN (?, ?)",
            (archived_id1, archived_id2),
        ).fetchall()
        assert vec_deleted == [], "vec rows for deleted playbooks must be gone"
        vec_kept = conn.execute(
            "SELECT rowid FROM user_playbooks_vec WHERE rowid = ?", (surviving_id,)
        ).fetchall()
        assert len(vec_kept) == 1, "vec row for surviving playbook must remain"


def _seed_rle_row(
    storage: BaseStorage, *, session_id: str, learning_id: str, created_at: int
) -> None:
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute(
        """INSERT INTO retrieved_learning_evaluation
           (user_id, session_id, agent_version, kind, learning_id, is_relevant,
            relevance_reason, impact, impact_reason, created_at)
           VALUES ('u1', ?, 'v1', 'profile', ?, 1, 'r', 'positive', 'i', ?)""",
        (session_id, learning_id, created_at),
    )
    conn.commit()


def test_retention_removes_whole_retrieved_learning_sessions(
    storage: BaseStorage,
) -> None:
    """The grouped session target never leaves a partial session snapshot.

    Keys are (user_id, session_id): even when the requested delete count cuts
    inside the oldest session's rows, every row of each selected session is
    removed together.
    """
    for n in range(3):
        _seed_rle_row(
            storage, session_id="old-sess", learning_id=f"old-{n}", created_at=100
        )
    for n in range(2):
        _seed_rle_row(
            storage, session_id="new-sess", learning_id=f"new-{n}", created_at=200
        )
    assert (
        storage.count_retention_target_rows("retrieved_learning_evaluation") == 5  # type: ignore[attr-defined]
    )

    # Requesting 2 selects two key tuples, both from the oldest session —
    # deleting by (user_id, session_id) removes all 3 of its rows.
    storage.delete_oldest_retention_target_rows("retrieved_learning_evaluation", 2)  # type: ignore[attr-defined]
    remaining = storage.get_retrieved_learning_evaluation_results(limit=10)
    assert {row.session_id for row in remaining} == {"new-sess"}
    assert len(remaining) == 2


# -- Age-based expiry ---------------------------------------------------------
#
# Core invariant: the age pass deletes a row ONLY IF it is older than the
# cutoff AND the archiver accepted it -- including rows a cascade would take.

_DAY = 86_400


class _RecordingArchiver:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.batches: list[tuple[str, list[dict[str, Any]]]] = []

    def __call__(self, table_name: str, rows: Any) -> bool:
        self.batches.append((table_name, list(rows)))
        return self.accept

    def ids(self, table: str, column: str) -> set[Any]:
        return {
            row[column] for name, rows in self.batches if name == table for row in rows
        }


def _expire(
    storage: BaseStorage,
    target: str,
    cutoff: int,
    archiver: Any,
    *,
    budget: int = 1_000,
    batch_size: int = 1_000,
) -> Any:
    return storage.expire_retention_target_rows(  # type: ignore[attr-defined]
        target,
        older_than_epoch=cutoff,
        budget=budget,
        batch_size=batch_size,
        archiver=archiver,
        deadline=float("inf"),
    )


def _seed_aged_interactions(storage: BaseStorage, now: int) -> None:
    """Interactions 1-3 are 40 days old; 4-5 are 1 day old. One request each."""
    for i in range(1, 6):
        created = now - (40 if i <= 3 else 1) * _DAY + i
        storage.add_request(_make_request(f"req{i}", created))
        storage.add_user_interaction("u1", _make_interaction(i, f"req{i}", created))


def test_age_expiry_archives_then_deletes_only_rows_older_than_cutoff(
    storage: BaseStorage,
) -> None:
    now = int(datetime.now(UTC).timestamp())
    _seed_aged_interactions(storage, now)
    archiver = _RecordingArchiver()

    result = _expire(storage, "interactions", now - 30 * _DAY, archiver)

    assert (result.eligible, result.deleted, result.blocked) == (3, 3, None)
    assert archiver.ids("interactions", "interaction_id") == {1, 2, 3}
    assert all("embedding" not in row for _, rows in archiver.batches for row in rows)
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {4, 5}


def test_age_expiry_cutoff_is_strict_on_iso_timestamps(storage: BaseStorage) -> None:
    """A row exactly AT the cutoff second survives; one second older does not.

    Guards the cutoff's type: an integer compared against SQLite's ISO TEXT
    column matches nothing, which would make this test delete zero rows.
    """
    cutoff = int(datetime.now(UTC).timestamp()) - 30 * _DAY
    storage.add_user_interaction("u1", _make_interaction(1, "req1", cutoff - 1))
    storage.add_user_interaction("u1", _make_interaction(2, "req2", cutoff))

    result = _expire(storage, "interactions", cutoff, _RecordingArchiver())

    assert result.deleted == 1
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {2}


def test_age_expiry_deletes_nothing_when_the_archiver_declines(
    storage: BaseStorage,
) -> None:
    now = int(datetime.now(UTC).timestamp())
    _seed_aged_interactions(storage, now)

    result = _expire(
        storage, "interactions", now - 30 * _DAY, _RecordingArchiver(accept=False)
    )

    assert (result.deleted, result.blocked) == (0, "archive_failed")
    assert len(storage.get_all_interactions(limit=10)) == 5


def test_age_expiry_dry_run_counts_and_deletes_nothing(storage: BaseStorage) -> None:
    now = int(datetime.now(UTC).timestamp())
    _seed_aged_interactions(storage, now)

    result = _expire(storage, "interactions", now - 30 * _DAY, None)

    assert (result.eligible, result.deleted) == (3, 0)
    assert len(storage.get_all_interactions(limit=10)) == 5


def test_age_expiry_keeps_a_request_until_its_interactions_are_gone(
    storage: BaseStorage,
) -> None:
    """The requests -> interactions cascade must never take an unarchived row.

    req1 is old but still has a NEW interaction, so it is held back; req2 is old
    and childless once its own old interaction was aged out first.
    """
    now = int(datetime.now(UTC).timestamp())
    old, new = now - 40 * _DAY, now - _DAY
    storage.add_request(_make_request("req1", old))
    storage.add_user_interaction("u1", _make_interaction(1, "req1", new))
    storage.add_request(_make_request("req2", old + 1))
    storage.add_user_interaction("u1", _make_interaction(2, "req2", old + 1))
    archiver = _RecordingArchiver()
    cutoff = now - 30 * _DAY

    interactions = _expire(storage, "interactions", cutoff, archiver)
    requests = _expire(storage, "requests", cutoff, archiver)

    assert (interactions.deleted, requests.deleted) == (1, 1)
    assert storage.get_request("req1") is not None
    assert storage.get_request("req2") is None
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {1}
    assert archiver.ids("interactions", "interaction_id") == {2}
    assert archiver.ids("requests", "request_id") == {"req2"}


def test_age_expiry_steps_past_protected_rows(storage: BaseStorage) -> None:
    """A protected row at the head of the table must not stall the pass."""
    now = int(datetime.now(UTC).timestamp())
    _seed_aged_interactions(storage, now)

    def protect_first(target: str, keys: list[tuple[Any, ...]]) -> list[Any]:
        return (
            [key for key in keys if key[0] != 1] if target == "interactions" else keys
        )

    with patch.object(
        type(storage), "filter_extraction_retention", side_effect=protect_first
    ):
        result = _expire(
            storage,
            "interactions",
            now - 30 * _DAY,
            _RecordingArchiver(),
            batch_size=1,
        )

    assert (result.eligible, result.deleted, result.backlog) == (2, 2, False)
    remaining = storage.get_all_interactions(limit=10)
    assert {interaction.interaction_id for interaction in remaining} == {1, 4, 5}


def test_age_expiry_stops_at_its_budget(storage: BaseStorage) -> None:
    now = int(datetime.now(UTC).timestamp())
    _seed_aged_interactions(storage, now)

    result = _expire(
        storage, "interactions", now - 30 * _DAY, _RecordingArchiver(), budget=2
    )

    assert (result.deleted, result.backlog) == (2, True)
