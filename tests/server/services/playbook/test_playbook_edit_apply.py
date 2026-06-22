"""Tests for apply_playbook_edit() — the shared archive+insert primitive."""

import tempfile
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._lineage import (
    _EMPTY_REQUEST_ID_MSG,
)


def _storage(tmp: str) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        return SQLiteStorage(org_id="org_apply_1", db_path=f"{tmp}/t.db")


def _playbook(user_id: str = "u1", content: str = "old") -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version="v1",
        request_id="req_test",
        playbook_name="refund",
        content=content,
        trigger="refund",
    )


def test_apply_inserts_new_and_archives_incumbent():
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id
            assert old_id > 0

            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="offline_optimizer",
                request_id="run-abc",
            )
        assert new_id > 0

        # Only new_id should be CURRENT (status=None); old_id should be archived
        all_pbs = s.get_user_playbooks(user_id="u1")
        current_ids = {p.user_playbook_id for p in all_pbs if p.status is None}
        assert new_id in current_ids
        assert old_id not in current_ids


def test_apply_skips_archive_when_incumbent_not_current():
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id
            assert old_id > 0

            # Someone else archived it first
            s.archive_user_playbook_by_id(user_id="u1", user_playbook_id=old_id)

            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="offline_optimizer",
                request_id="run-abc",
            )
        # Optimistic-concurrency: incumbent was already archived → skip and return -1
        assert new_id == -1


def test_apply_expect_current_false_archives():
    """With expect_current=False, new playbook is inserted and incumbent is archived."""
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id
            assert old_id > 0

            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="offline_optimizer",
                request_id="run-abc",
            )
        assert new_id > 0

        # Only new_id should be CURRENT (status=None); old_id should be archived
        all_pbs = s.get_user_playbooks(user_id="u1")
        current_ids = {p.user_playbook_id for p in all_pbs if p.status is None}
        assert new_id in current_ids
        assert old_id not in current_ids


def test_apply_expect_current_false_returns_minus1_and_no_orphan():
    """When incumbent is already archived, supersede_record returns False.

    The new code deletes the just-inserted successor so no orphan CURRENT row
    remains — the -1 return value indicates the lost race, not an orphan.
    """
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id
            assert old_id > 0

            # Archive first so supersede_record will return False
            s.archive_user_playbook_by_id(user_id="u1", user_playbook_id=old_id)

            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="offline_optimizer",
                request_id="run-abc",
            )
        # supersede_record returned False → -1, successor cleaned up (no orphan)
        assert new_id == -1

        # No orphan: the inserted successor was deleted
        all_pbs = s.get_user_playbooks(user_id="u1")
        current_ids = {p.user_playbook_id for p in all_pbs if p.status is None}
        assert len(current_ids) == 0


def test_apply_raises_on_empty_request_id_before_write():
    """apply_playbook_edit raises ValueError on empty request_id before any storage write.

    The I2 (orphan) guard: an empty request_id is rejected immediately so no
    successor row is ever inserted when the caller forgets to supply a run id.
    """
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id

        # Patch save_user_playbooks to confirm it is never reached on empty request_id.
        with patch.object(s, "save_user_playbooks") as mock_save:
            with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
                apply_playbook_edit(
                    s,
                    incumbent_id=old_id,
                    new_playbook=_playbook(content="new"),
                    source="offline_optimizer",
                    request_id="",
                )
            mock_save.assert_not_called()

        # No orphan: no successor row was inserted (incumbent still CURRENT, count==1).
        count = s.conn.execute(
            "SELECT COUNT(*) FROM user_playbooks WHERE status IS NULL"
        ).fetchone()[0]
        assert count == 1, (
            "no orphan successor row should be inserted on empty request_id"
        )


@pytest.mark.parametrize("bad_request_id", ["", None])
def test_apply_raises_on_empty_or_none_request_id(bad_request_id):
    """apply_playbook_edit raises ValueError for both empty string and None request_id."""
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id

        with pytest.raises((ValueError, TypeError)):
            apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=_playbook(content="new"),
                source="offline_optimizer",
                request_id=bad_request_id,  # type: ignore[arg-type]
            )


def test_apply_lineage_event_carries_operation_run_id():
    """apply_playbook_edit records the operation-run request_id on the revise event.

    The lineage event must carry the operation request_id (the reflection run id),
    NOT the incumbent's birth request_id.  This enables correct run-correlation.
    """
    from reflexio.server.services.playbook.playbook_edit_apply import (
        apply_playbook_edit,
    )

    with tempfile.TemporaryDirectory() as tmp:
        s = _storage(tmp)
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            old = _playbook(content="old")
            s.save_user_playbooks([old])
            old_id = old.user_playbook_id
            assert old_id > 0

            operation_run_id = "reflection_run_xyz"
            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="reflection",
                request_id=operation_run_id,
            )
        assert new_id > 0

        events = s.get_lineage_events(
            entity_type="user_playbook", entity_id=str(new_id)
        )
        assert len(events) == 1
        assert events[0].op == "revise"
        assert events[0].request_id == operation_run_id, (
            f"lineage event must carry the operation run id {operation_run_id!r}, "
            f"not the incumbent's birth request_id {old.request_id!r}"
        )
