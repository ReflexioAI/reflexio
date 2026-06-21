"""Integration test: _retention_perform_delete rolls back on failure.

Verifies that if any step inside the retention critical section raises, the
whole transaction is rolled back and no partial writes are committed.
"""

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.retention import RETENTION_TARGETS_BY_NAME
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_USER_PLAYBOOKS_TARGET = RETENTION_TARGETS_BY_NAME["user_playbooks"]


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="org-rollback", db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


def _make_user_playbook(uid: int) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=uid,
        user_id="u1",
        playbook_name="pb",
        agent_version="v1",
        request_id=f"req-{uid}",
        content=f"content-{uid}",
        created_at=uid,
        source="test",
        source_interaction_ids=[],
    )


def test_retention_perform_delete_rolls_back_on_target_row_failure(
    tmp_path, monkeypatch
) -> None:
    """If _retention_delete_target_rows raises, no dependency deletes are committed."""
    s = _store(tmp_path)
    s.save_user_playbooks([_make_user_playbook(1), _make_user_playbook(2)])

    conn = s.conn
    saved = conn.execute(
        "SELECT user_playbook_id FROM user_playbooks ORDER BY user_playbook_id"
    ).fetchall()
    assert len(saved) == 2
    saved_ids = [r["user_playbook_id"] for r in saved]

    # Assign ordered created_at so deletion targets are deterministic.
    for i, upid in enumerate(saved_ids, start=1):
        conn.execute(
            "UPDATE user_playbooks SET created_at = ? WHERE user_playbook_id = ?",
            (i, upid),
        )
    conn.commit()

    # Verify FTS rows exist before the attempted retention delete.
    ph = ",".join("?" for _ in saved_ids)
    fts_before = conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(fts_before) == 2

    # Force the target-row delete to fail.
    def _fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(s, "_retention_delete_target_rows", _fail)

    keys = s._retention_select_oldest_keys(_USER_PLAYBOOKS_TARGET, 1)  # type: ignore[attr-defined]

    with pytest.raises(StorageError, match="injected failure"):
        s._retention_perform_delete(_USER_PLAYBOOKS_TARGET, keys)  # type: ignore[attr-defined]

    # After the rollback, the user_playbooks rows must be untouched.
    rows_after = conn.execute(
        f"SELECT user_playbook_id FROM user_playbooks WHERE user_playbook_id IN ({ph})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(rows_after) == 2, "rollback must leave all rows intact"

    # FTS rows must also be intact (dependency deletes were rolled back).
    fts_after = conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph})",  # noqa: S608
        saved_ids,
    ).fetchall()
    assert len(fts_after) == 2, "rollback must leave all FTS rows intact"


def test_retention_perform_delete_rolls_back_on_dependency_failure(
    tmp_path, monkeypatch
) -> None:
    """If _retention_delete_dependencies raises, the target row delete is not committed."""
    s = _store(tmp_path)
    s.save_user_playbooks([_make_user_playbook(1)])

    conn = s.conn
    saved = conn.execute("SELECT user_playbook_id FROM user_playbooks").fetchall()
    assert len(saved) == 1
    upid = saved[0]["user_playbook_id"]

    # Force the dependency delete to fail.
    def _fail(*args, **kwargs):
        raise RuntimeError("injected dep failure")

    monkeypatch.setattr(s, "_retention_delete_dependencies", _fail)

    keys = s._retention_select_oldest_keys(_USER_PLAYBOOKS_TARGET, 1)  # type: ignore[attr-defined]

    with pytest.raises(StorageError, match="injected dep failure"):
        s._retention_perform_delete(_USER_PLAYBOOKS_TARGET, keys)  # type: ignore[attr-defined]

    # The user_playbooks row must still be present after rollback.
    row_after = conn.execute(
        "SELECT user_playbook_id FROM user_playbooks WHERE user_playbook_id = ?",
        (upid,),
    ).fetchone()
    assert row_after is not None, "rollback must leave the target row intact"
