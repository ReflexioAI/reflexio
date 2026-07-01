"""Characterization tests: SQLite playbook atomicity invariants (Tier-1 Task 1).

Day-one invariant guards established BEFORE the PlaybookMixin decomposition. They
pin the CURRENT commit / lineage-event / no-op behavior of the top-risk,
atomicity-sensitive methods so a "tidying" reorder during the mixin split is caught
by a failing test. Modeled on the gold standard
``test_save_agent_playbook_with_aggregate_event_integration.py``.

Methods characterized here (SQLite side):
  - ``supersede_user_playbooks_by_ids`` — soft-delete to SUPERSEDED, per-row
    ``rowcount`` guard, single commit, one ``status_change`` event per updated row
    under a SHARED ``request_id`` (actor=``consolidator``). Uncovered before this file.
  - Crash-window / rollback for the hard-delete family
    (``delete_agent_playbook``, ``delete_all_agent_playbooks``,
    ``delete_all_agent_playbooks_by_playbook_name``,
    ``delete_archived_agent_playbooks_by_playbook_name``): a failing lineage emit
    leaves NEITHER the mutation NOR a phantom ``hard_delete`` event durable.
    (Happy-path + no-op coverage for these already lives in
    ``test_lineage_b1_harddelete_integration.py``.)

The invariant each test pins: a lineage event is recorded ONLY for a mutation that
actually committed. A refactor that reorders ``mutation -> rowcount-guarded emit ->
commit`` (e.g. emitting before/independently of the commit, or dropping the
``rowcount`` guard) must fail one of these tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import reflexio.server.services.storage.sqlite_storage.playbook._agent as _agent_playbook_mod
import reflexio.server.services.storage.sqlite_storage.playbook._user as _user_playbook_mod
from reflexio.models.api_schema.domain.entities import AgentPlaybook, UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "org-char") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_user_playbook(
    *, request_id: str = "r", content: str = "c", user_id: str = "u"
) -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version="v1",
        request_id=request_id,
        content=content,
    )


def _make_agent_playbook(
    *, playbook_name: str = "pb", agent_version: str = "v1", content: str = "c"
) -> AgentPlaybook:
    return AgentPlaybook(
        playbook_name=playbook_name, agent_version=agent_version, content=content
    )


def _supersede_events(s: SQLiteStorage, entity_id: int) -> list:
    return [
        e
        for e in s.get_lineage_events(
            entity_id=str(entity_id), entity_type="user_playbook"
        )
        if e.op == "status_change" and e.to_status == "superseded"
    ]


def _hard_delete_events(s: SQLiteStorage, entity_id: int) -> list:
    return [
        e
        for e in s.get_lineage_events(entity_id=str(entity_id))
        if e.op == "hard_delete"
    ]


# ---------------------------------------------------------------------------
# supersede_user_playbooks_by_ids — effect + lineage + no-op/phantom guard
# ---------------------------------------------------------------------------


class TestSupersedeUserPlaybooksByIds:
    def test_happy_path_supersedes_rows_and_emits_one_event_each(self, tmp_path):
        """CURRENT rows -> SUPERSEDED, content retained, one shared-request event each."""
        s = _store(tmp_path)
        pb1 = _make_user_playbook(request_id="r1", content="first")
        pb2 = _make_user_playbook(request_id="r2", content="second")
        s.save_user_playbooks([pb1, pb2])
        ids = [pb1.user_playbook_id, pb2.user_playbook_id]

        updated = s.supersede_user_playbooks_by_ids(ids, "shared-run")

        assert updated == 2
        # Excluded from default reads; readable as tombstone with content intact.
        for pb, content in ((pb1, "first"), (pb2, "second")):
            assert s.get_user_playbook_by_id(pb.user_playbook_id) is None
            tomb = s.get_user_playbook_by_id(
                pb.user_playbook_id, include_tombstones=True
            )
            assert tomb is not None
            assert tomb.status == Status.SUPERSEDED
            assert tomb.content == content

        # Exactly one status_change->superseded event per row, shared request_id.
        for pb in (pb1, pb2):
            evs = _supersede_events(s, pb.user_playbook_id)
            assert len(evs) == 1
            ev = evs[0]
            assert ev.request_id == "shared-run"
            assert ev.actor == "consolidator"
            assert ev.from_status is None  # CURRENT (NULL) prior status
            assert ev.to_status == "superseded"
            assert ev.reason == "None->superseded"
            assert ev.prov_relation == "wasInvalidatedBy"

    def test_already_superseded_row_is_noop_and_emits_no_new_event(self, tmp_path):
        """Tombstone-guarded WHERE: re-superseding emits NO second event (phantom guard)."""
        s = _store(tmp_path)
        pb = _make_user_playbook()
        s.save_user_playbooks([pb])

        first = s.supersede_user_playbooks_by_ids([pb.user_playbook_id], "run-1")
        assert first == 1
        second = s.supersede_user_playbooks_by_ids([pb.user_playbook_id], "run-2")
        assert second == 0

        # Only the first call's event exists — the no-op must not emit a phantom.
        evs = _supersede_events(s, pb.user_playbook_id)
        assert len(evs) == 1
        assert evs[0].request_id == "run-1"

    def test_nonexistent_id_returns_zero_and_emits_no_event(self, tmp_path):
        """A miss (id not present) updates nothing and emits no event."""
        s = _store(tmp_path)
        updated = s.supersede_user_playbooks_by_ids([99999], "run-miss")
        assert updated == 0
        assert _supersede_events(s, 99999) == []

    def test_partial_only_supersedes_and_emits_for_existing(self, tmp_path):
        """Mixed real+ghost ids: only the real row is updated and gets an event."""
        s = _store(tmp_path)
        pb = _make_user_playbook()
        s.save_user_playbooks([pb])

        updated = s.supersede_user_playbooks_by_ids(
            [pb.user_playbook_id, 88888], "run-partial"
        )
        assert updated == 1
        assert len(_supersede_events(s, pb.user_playbook_id)) == 1
        assert _supersede_events(s, 88888) == []

    def test_empty_list_returns_zero(self, tmp_path):
        s = _store(tmp_path)
        assert s.supersede_user_playbooks_by_ids([], "run-empty") == 0

    def test_empty_request_id_raises_before_write(self, tmp_path):
        """Empty request_id raises (no unreconstructable event); the row is untouched."""
        s = _store(tmp_path)
        pb = _make_user_playbook()
        s.save_user_playbooks([pb])

        with pytest.raises((ValueError, StorageError), match="non-empty"):
            s.supersede_user_playbooks_by_ids([pb.user_playbook_id], "")

        # Row remains CURRENT, no event.
        assert s.get_user_playbook_by_id(pb.user_playbook_id) is not None
        assert _supersede_events(s, pb.user_playbook_id) == []

    def test_crash_window_event_failure_leaves_no_partial_state(self, tmp_path):
        """If the supersede event emit raises, NEITHER the UPDATE nor an event is durable.

        Mirrors the gold-standard rollback test: the mutation and its lineage event
        commit together. A failing emit (before the single commit) must leave the
        row CURRENT and write no phantom status_change event.
        """
        s = _store(tmp_path)
        pb = _make_user_playbook()
        s.save_user_playbooks([pb])

        with (
            patch.object(
                _user_playbook_mod,
                "_emit_supersede_user_playbook",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.supersede_user_playbooks_by_ids([pb.user_playbook_id], "run-crash")

        # The method never committed; discard the would-be UPDATE and assert
        # durability: row still CURRENT, no superseded event recorded.
        s.conn.rollback()
        row = s.get_user_playbook_by_id(pb.user_playbook_id)
        assert row is not None
        assert row.status is None or row.status != Status.SUPERSEDED
        assert _supersede_events(s, pb.user_playbook_id) == []


# ---------------------------------------------------------------------------
# Hard-delete family — crash-window / rollback (no phantom hard_delete event)
# ---------------------------------------------------------------------------


class TestHardDeleteCrashWindow:
    """A failing hard_delete emit must leave NEITHER the row deleted NOR a phantom event.

    This is the exact emit-vs-commit ordering class flagged in
    ``.claude/rules/reflexio-patterns.md`` ("SQLite storage: lineage events &
    atomicity"): the event must be durable only in the same commit as the DELETE it
    attests to. If a refactor self-committed the event before the row delete, a
    phantom ``hard_delete`` would survive the rollback below and fail the assertion.
    Happy-path + no-op coverage lives in ``test_lineage_b1_harddelete_integration.py``.
    """

    def test_delete_agent_playbook_crash_window(self, tmp_path):
        s = _store(tmp_path)
        saved = s.save_agent_playbooks([_make_agent_playbook()])
        apid = saved[0].agent_playbook_id

        with (
            patch.object(
                _agent_playbook_mod,
                "_emit_hard_delete_playbook",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.delete_agent_playbook(apid)

        s.conn.rollback()
        assert s.get_agent_playbook_by_id(apid, include_tombstones=True) is not None
        assert _hard_delete_events(s, apid) == []

    def test_delete_all_agent_playbooks_crash_window(self, tmp_path):
        s = _store(tmp_path)
        saved = s.save_agent_playbooks([_make_agent_playbook(playbook_name="a")])
        apid = saved[0].agent_playbook_id

        with (
            patch.object(
                _agent_playbook_mod,
                "_emit_hard_delete_playbook",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.delete_all_agent_playbooks()

        s.conn.rollback()
        assert s.get_agent_playbook_by_id(apid, include_tombstones=True) is not None
        assert _hard_delete_events(s, apid) == []

    def test_delete_all_agent_playbooks_by_playbook_name_crash_window(self, tmp_path):
        s = _store(tmp_path)
        saved = s.save_agent_playbooks([_make_agent_playbook(playbook_name="named")])
        apid = saved[0].agent_playbook_id

        with (
            patch.object(
                _agent_playbook_mod,
                "_emit_hard_delete_playbook",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.delete_all_agent_playbooks_by_playbook_name("named")

        s.conn.rollback()
        assert s.get_agent_playbook_by_id(apid, include_tombstones=True) is not None
        assert _hard_delete_events(s, apid) == []

    def test_delete_archived_agent_playbooks_by_playbook_name_crash_window(
        self, tmp_path
    ):
        s = _store(tmp_path)
        ap = AgentPlaybook(
            playbook_name="archbook",
            agent_version="v1",
            content="c",
            status=Status.ARCHIVED,
        )
        saved = s.save_agent_playbooks([ap])
        apid = saved[0].agent_playbook_id

        with (
            patch.object(
                _agent_playbook_mod,
                "_emit_hard_delete_playbook",
                side_effect=RuntimeError("simulated emit failure"),
            ),
            pytest.raises(StorageError, match="simulated emit failure"),
        ):
            s.delete_archived_agent_playbooks_by_playbook_name("archbook")

        s.conn.rollback()
        assert s.get_agent_playbook_by_id(apid, include_tombstones=True) is not None
        assert _hard_delete_events(s, apid) == []
