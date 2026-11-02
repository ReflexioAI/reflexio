"""Private progress survives owner turns without coupling users or work classes."""

import pytest

from reflexio.server.services.storage.storage_base._user_work_position import (
    ResumeCursor,
    UserWorkPosition,
    WorkClass,
)


def test_serialized_opportunities_rotate_without_resetting_class_sweeps():
    first, other = UserWorkPosition(), UserWorkPosition()
    seen = []
    classes: tuple[WorkClass, ...] = ("finalization", "continuation", "learning")
    for _ in range(4):
        for kind in classes:
            assert first.next_work_class == kind
            seen.append(kind)
            cursor = None if kind == "learning" else ResumeCursor("run_2", "run_9")
            # The next owner reads persisted state, rather than process history.
            first = UserWorkPosition.from_json(
                first.selected(kind, cursor=cursor).to_json()
            )
    assert seen == ["finalization", "continuation", "learning"] * 4
    assert first.finalization == first.continuation == ResumeCursor("run_2", "run_9")
    assert other == UserWorkPosition.from_json(None)
    assert other.next_work_class == "finalization"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "{}",
        "[]",
        {"last_work_class": [], "finalization": {}, "continuation": {}},
        '{"last_work_class":"learning"}',
        {"last_work_class": "learning", "finalization": {}, "continuation": {}},
        {
            "last_work_class": "learning",
            "finalization": {"after_id": "z", "through_id": None},
            "continuation": {"after_id": None, "through_id": None},
        },
    ],
)
def test_invalid_progress_is_not_silently_reset(value):
    with pytest.raises(ValueError):
        UserWorkPosition.from_json(value)


def test_selection_preserves_other_cursor_and_refuses_order_skips():
    position = UserWorkPosition().selected(
        "finalization", cursor=ResumeCursor("a", "c")
    )
    with pytest.raises(ValueError, match="out of order"):
        position.selected("learning")
    with pytest.raises(ValueError, match="preserve"):
        position.selected("continuation")
    position = position.selected("continuation", cursor=ResumeCursor(None, "q"))
    assert position.finalization == ResumeCursor("a", "c")
    with pytest.raises(ValueError, match="no resumable"):
        position.selected("learning", cursor=ResumeCursor())


def test_progress_survives_release_and_replacement_cannot_write(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
    from reflexio.server.services.storage.storage_base import LeaseLostError

    storage = SQLiteStorage(org_id="owner_position", db_path=str(tmp_path / "owner.db"))
    try:
        token = storage.claim_user_extraction("one", "first", 300)
        other = storage.claim_user_extraction("two", "other", 300)
        assert token is not None and other is not None
        with storage.commit_scope(), storage._stream_sql() as db:
            position = storage._owned_user_work_position(db, "one", token)
            storage._store_owned_user_work_position(
                db,
                "one",
                token,
                position.selected("finalization", cursor=ResumeCursor("a", "z")),
            )
        storage.release_user_extraction("one", token)
        new = storage.claim_user_extraction("one", "replacement", 300)
        assert new is not None and new != token
        with storage.commit_scope(), storage._stream_sql() as db:
            current = storage._owned_user_work_position(db, "one", new)
            assert current.next_work_class == "continuation"
            assert current.finalization == ResumeCursor("a", "z")
            assert (
                storage._owned_user_work_position(db, "two", other)
                == UserWorkPosition()
            )
        with (
            pytest.raises(LeaseLostError),
            storage.commit_scope(),
            storage._stream_sql() as db,
        ):
            storage._store_owned_user_work_position(
                db, "one", token, UserWorkPosition()
            )
        with (
            pytest.raises(LeaseLostError),
            storage.commit_scope(),
            storage._stream_sql() as db,
        ):
            storage._owned_user_work_position(db, "two", new)
        with storage.commit_scope(), storage._stream_sql() as db:
            assert storage._owned_user_work_position(db, "one", new) == current
        storage.fence_user_extraction("one", new)
    finally:
        storage.conn.close()


def test_progress_joins_failed_adoption_transaction(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(
        org_id="position_rollback", db_path=str(tmp_path / "owner.db")
    )
    try:
        token = storage.claim_user_extraction("one", "first", 300)
        assert token is not None
        with (
            pytest.raises(RuntimeError, match="adoption failed"),
            storage.commit_scope(),
            storage._stream_sql() as db,
        ):
            storage._store_owned_user_work_position(
                db,
                "one",
                token,
                UserWorkPosition().selected(
                    "finalization", cursor=ResumeCursor("a", "z")
                ),
            )
            raise RuntimeError("adoption failed")
        with storage.commit_scope(), storage._stream_sql() as db:
            assert (
                storage._owned_user_work_position(db, "one", token)
                == UserWorkPosition()
            )
    finally:
        storage.conn.close()
