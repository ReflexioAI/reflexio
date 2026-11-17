"""Real SQLite persistence, isolation and erasure of accepted-access evidence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.services.profile.attention import (
    candidate_attention,
    storage_snapshot,
)
from reflexio.server.services.profile.lifecycle import KEY, attention_epoch
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        store = SQLiteStorage(
            org_id="attention-test", db_path=str(tmp_path / "memory.db")
        )
        yield store
        store.conn.close()


def add_profile(storage, identity="old", owner="u", content="likes tea"):
    profile = UserProfile(
        profile_id=identity,
        user_id=owner,
        content=content,
        last_modified_timestamp=100,
        generated_from_request_id="r",
        custom_features={
            KEY: {
                "created_at": 100,
                "ancestor_ids": [],
                "explicit_storage_deadline": None,
            }
        },
    )
    storage.add_user_profile(owner, [profile])
    return profile


def test_access_is_monotonic_separate_and_owner_scoped(storage):
    add_profile(storage)
    add_profile(storage, "other", "another-owner")
    before = storage.conn.execute(
        "SELECT * FROM profiles WHERE profile_id='old'"
    ).fetchone()
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old", "other", "missing"], accessed_at=200
    )
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old"], accessed_at=150
    )
    assert storage.get_profile_access_epochs(
        user_id="u", profile_ids=["old", "other"]
    ) == {"old": 200}
    assert (
        storage.get_profile_access_epochs(
            user_id="another-owner", profile_ids=["old", "other"]
        )
        == {}
    )
    after = storage.conn.execute(
        "SELECT * FROM profiles WHERE profile_id='old'"
    ).fetchone()
    assert tuple(before) == tuple(after)


def test_retired_ancestor_retains_evidence_and_late_access(storage):
    add_profile(storage)
    storage.conn.execute(
        "UPDATE profiles SET status='merged',merged_into='new' WHERE profile_id='old'"
    )
    storage.conn.commit()
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old"], accessed_at=300
    )
    assert storage.get_profile_access_epochs(user_id="u", profile_ids=["old"]) == {
        "old": 300
    }


@pytest.mark.parametrize("physical", [True, False])
def test_partial_forget_keeps_ancestor_attention_until_owner_erasure(storage, physical):
    old = add_profile(storage, content="Home address 10 Example Street; likes tea")
    replacement = old.model_copy(
        update={
            "profile_id": "replacement",
            "content": "likes tea",
            "custom_features": {
                KEY: {
                    "created_at": 100,
                    "ancestor_ids": ["old"],
                    "explicit_storage_deadline": None,
                }
            },
        }
    )
    storage.add_user_profile("u", [replacement])
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old"], accessed_at=200
    )
    statement = (
        "DELETE FROM profiles WHERE profile_id='old'"
        if physical
        else "UPDATE profiles SET content='' WHERE profile_id='old'"
    )
    storage.conn.execute(statement)
    storage.conn.commit()
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old"], accessed_at=300
    )
    assert storage.get_profile_access_epochs(user_id="u", profile_ids=["old"]) == {
        "old": 200
    }
    with storage_snapshot(storage, "u"), candidate_attention([replacement]):
        assert attention_epoch(replacement, now=300) == 200
    storage.clear_user_data("u")
    storage.record_served_profile_access(
        user_id="u", profile_ids=["old"], accessed_at=400
    )
    assert (
        storage.get_profile_access_epochs(
            user_id="u", profile_ids=["old", "replacement"]
        )
        == {}
    )


def test_subject_barrier_prevents_late_access(storage):
    add_profile(storage)
    subject = storage._subject_ref_for_user_id("u")
    storage.conn.execute(
        "INSERT INTO subject_write_barriers(org_id,subject_ref,purge_id,status,created_at,updated_at) VALUES (?,?,?,'erased',1,1)",
        (storage.org_id, subject, "erase"),
    )
    storage.conn.commit()
    with pytest.raises(SubjectWriteBarrierError):
        storage.record_served_profile_access(
            user_id="u", profile_ids=["old"], accessed_at=300
        )
    assert storage.get_profile_access_epochs(user_id="u", profile_ids=["old"]) == {}


def test_startup_upgrade_retains_access(tmp_path):
    path = str(tmp_path / "upgrade.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        first = SQLiteStorage(org_id="upgrade", db_path=path)
        add_profile(first)
        first.record_served_profile_access(
            user_id="u", profile_ids=["old"], accessed_at=200
        )
        first.conn.close()
        second = SQLiteStorage(org_id="upgrade", db_path=path)
        try:
            assert second.get_profile_access_epochs(
                user_id="u", profile_ids=["old"]
            ) == {"old": 200}
        finally:
            second.conn.close()


def test_owner_clear_catches_access_inserted_before_last_content_purge(
    storage, monkeypatch
):
    add_profile(storage)
    add_profile(storage, "other", "other-owner")
    storage.conn.execute(
        "UPDATE profiles SET status='merged',merged_into='other' WHERE profile_id='old'"
    )
    storage.conn.commit()
    path = storage.conn.execute("PRAGMA database_list").fetchone()[2]
    other = SQLiteStorage(org_id=storage.org_id, db_path=path)
    original_purge = storage.purge_content
    observed = []

    def concurrent_access_then_purge(*, entity_type, entity_id, **kwargs):
        if entity_type == "profile" and entity_id == "old":
            other.record_served_profile_access(
                user_id="u", profile_ids=["old"], accessed_at=300
            )
            observed.append(
                other.get_profile_access_epochs(user_id="u", profile_ids=["old"])
            )
        return original_purge(entity_type=entity_type, entity_id=entity_id, **kwargs)

    monkeypatch.setattr(storage, "purge_content", concurrent_access_then_purge)
    try:
        result = storage.clear_user_data("u")
        assert result["purged_profiles"] == 1
        assert observed == [{"old": 300}]
        assert other.get_profile_access_epochs(user_id="u", profile_ids=["old"]) == {}
        other.record_served_profile_access(
            user_id="u", profile_ids=["old"], accessed_at=400
        )
        assert other.get_profile_access_epochs(user_id="u", profile_ids=["old"]) == {}
        assert other.get_user_profile("other-owner")[0].content == "likes tea"
    finally:
        other.conn.close()
