"""Characterization test: ``_delete_playbook_search_rows``'s ``commit=`` contract.

Pins the exact behavior a later verbatim move (peeling ``SQLiteDeletionMixin``
out of ``sqlite_storage/_base.py``) must preserve: ``commit=False`` participates
in the caller's outer transaction (so a rollback discards the deletes), while
the default ``commit=True`` durably persists them.

Falsifiability:
- If the move drops the ``commit=False`` path (i.e. always self-commits), the
  ``commit=False`` test's rollback would no longer restore the FTS rows and the
  "still present" assertion fails.
- If the move drops the ``commit=True`` self-commit, the default test's rows
  would reappear after the rollback and the "deleted" assertion fails.
"""

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path, name: str) -> SQLiteStorage:
    return SQLiteStorage(org_id=f"org-{name}", db_path=str(tmp_path / f"{name}.db"))


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


def _seed_playbooks_with_fts(storage: SQLiteStorage) -> list[int]:
    """Save two user playbooks via the real write path (which indexes FTS)."""
    storage.save_user_playbooks([_make_user_playbook(1), _make_user_playbook(2)])
    rows = storage.conn.execute(
        "SELECT user_playbook_id FROM user_playbooks ORDER BY user_playbook_id"
    ).fetchall()
    ids = [row["user_playbook_id"] for row in rows]
    assert len(ids) == 2
    return ids


def _fts_rowids_present(storage: SQLiteStorage, ids: list[int]) -> int:
    ph = ",".join("?" for _ in ids)
    rows = storage.conn.execute(
        f"SELECT rowid FROM user_playbooks_fts WHERE rowid IN ({ph})",  # noqa: S608
        ids,
    ).fetchall()
    return len(rows)


def test_delete_playbook_search_rows_commit_false_participates_in_txn(
    tmp_path,
) -> None:
    storage = _store(tmp_path, "commit-false")
    ids = _seed_playbooks_with_fts(storage)
    assert _fts_rowids_present(storage, ids) == 2

    # commit=False: the deletes join the caller's open transaction and must NOT
    # self-commit. Rolling back the outer transaction therefore restores them.
    storage._delete_playbook_search_rows("user", ids, commit=False)  # type: ignore[attr-defined]
    assert _fts_rowids_present(storage, ids) == 0  # pending within the txn

    storage.conn.rollback()

    assert _fts_rowids_present(storage, ids) == 2, (
        "commit=False must not self-commit; the outer rollback must restore the rows"
    )


def test_delete_playbook_search_rows_default_commit_persists_across_rollback(
    tmp_path,
) -> None:
    storage = _store(tmp_path, "commit-true")
    ids = _seed_playbooks_with_fts(storage)
    assert _fts_rowids_present(storage, ids) == 2

    # Default commit=True self-commits, so a subsequent rollback is a no-op.
    storage._delete_playbook_search_rows("user", ids)  # type: ignore[attr-defined]
    assert _fts_rowids_present(storage, ids) == 0

    storage.conn.rollback()

    assert _fts_rowids_present(storage, ids) == 0, (
        "default commit=True must durably persist the deletes past a rollback"
    )
