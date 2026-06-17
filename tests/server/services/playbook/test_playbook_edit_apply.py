"""Tests for apply_playbook_edit() — the shared archive+insert primitive."""

import tempfile
from unittest.mock import patch

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


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
                s, incumbent_id=old_id, new_playbook=new, source="offline_optimizer"
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
                s, incumbent_id=old_id, new_playbook=new, source="offline_optimizer"
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
                expect_current=False,
            )
        assert new_id > 0

        # Only new_id should be CURRENT (status=None); old_id should be archived
        all_pbs = s.get_user_playbooks(user_id="u1")
        current_ids = {p.user_playbook_id for p in all_pbs if p.status is None}
        assert new_id in current_ids
        assert old_id not in current_ids


def test_apply_expect_current_false_returns_minus1_when_archive_fails():
    """With expect_current=False, if archive fails the new playbook remains CURRENT (orphan)."""
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

            # Archive first, so archive_user_playbook_by_id will fail on second call
            s.archive_user_playbook_by_id(user_id="u1", user_playbook_id=old_id)

            new = _playbook(content="new")
            new_id = apply_playbook_edit(
                s,
                incumbent_id=old_id,
                new_playbook=new,
                source="offline_optimizer",
                expect_current=False,
            )
        # Archive fails, new playbook inserted but remains CURRENT (orphan)
        assert new_id == -1

        # Verify new playbook was inserted and is CURRENT (orphan)
        all_pbs = s.get_user_playbooks(user_id="u1")
        current_ids = {p.user_playbook_id for p in all_pbs if p.status is None}
        # The new playbook ID would be old_id + 1 (or next available)
        # Just verify there's at least one CURRENT playbook (the orphan)
        assert len(current_ids) > 0
