"""Integration tests: reconstruct_profile_change_log — time-travel-stable model.

Phase B3 / Task 2: Rebuild the ProfileChangeLog view on demand using stable signals:
  - added(R)   = profiles whose immutable generated_from_request_id == R
                 (includes tombstones — a profile added in R1 and later
                 tombstoned in R2 is STILL added in R1).
  - removed(R) = entity_ids of status_change events with to_status=="superseded"
                 and request_id==R  (the dedup soft-delete signature; distinct
                 from reflection which emits op="revise").

Tested scenarios:
  - Dedup run (adds via generated_from_request_id + removes via supersede_profiles_by_ids)
    produces correct added/removed sets field-by-field — parity with legacy shape.
  - TIME-TRAVEL regression (F3): profile added in R1, tombstoned in R2 → still
    classified as added in R1, removed in R2.
  - Empty request_id group is skipped (never merged with unrelated runs).
  - Purged tombstone → blank removed content, no crash.
  - Reflection revise event does NOT get counted as a dedup removal.
  - limit, ordering, cross-org isolation, empty storage.
"""

from datetime import UTC, datetime

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "hello world",
    request_id: str = "",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=request_id,
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


# --------------------------------------------------------------------------
# Helper: seed a dedup run exactly as the dedup path would
# --------------------------------------------------------------------------


def _seed_dedup_run(
    s: SQLiteStorage,
    *,
    user_id: str,
    new_profiles: list[UserProfile],
    old_ids: list[str],
    request_id: str,
) -> None:
    """Seed a dedup run: add new profiles then supersede old ones.

    This mirrors the real dedup path:
      - new profiles carry generated_from_request_id == request_id (set at creation)
      - old profiles are soft-deleted via supersede_profiles_by_ids which emits
        status_change(to_status="superseded") events under request_id
    """
    s.add_user_profile(user_id, new_profiles)
    if old_ids:
        s.supersede_profiles_by_ids(user_id, old_ids, request_id)


# --------------------------------------------------------------------------
# Core parity test: dedup run -> reconstruct matches legacy shape
# --------------------------------------------------------------------------


def test_dedup_run_produces_one_changelog_row(tmp_path):
    """A single dedup run produces exactly one reconstructed change-log entry."""
    s = _store(tmp_path)
    old = _make_profile(
        user_id="u1", profile_id="p-old", content="old", request_id="req-seed"
    )
    s.add_user_profile("u1", [old])

    new = _make_profile(
        user_id="u1", profile_id="p-new", content="new", request_id="req-run1"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old"], request_id="req-run1"
    )

    result = reconstruct_profile_change_log(s)
    assert result.success
    # "req-run1" has both an added profile and a removed event
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    assert "req-run1" in rows_by_req


def test_dedup_run_added_profiles_by_generated_from_request_id(tmp_path):
    """added_profiles = profiles with generated_from_request_id == run's request_id."""
    s = _store(tmp_path)
    old = _make_profile(
        user_id="u1", profile_id="p-old", content="old facts", request_id="r0"
    )
    s.add_user_profile("u1", [old])

    new = _make_profile(
        user_id="u1", profile_id="p-new", content="new facts", request_id="r1"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old"], request_id="r1"
    )

    result = reconstruct_profile_change_log(s)
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    row = rows_by_req["r1"]
    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-new"
    assert row.added_profiles[0].content == "new facts"


def test_dedup_run_removed_profiles_from_status_change_events(tmp_path):
    """removed_profiles = profiles superseded in the run (via status_change events)."""
    s = _store(tmp_path)
    old = _make_profile(
        user_id="u1", profile_id="p-old", content="old facts", request_id="r0"
    )
    s.add_user_profile("u1", [old])

    new = _make_profile(
        user_id="u1", profile_id="p-new", content="new facts", request_id="r1"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old"], request_id="r1"
    )

    result = reconstruct_profile_change_log(s)
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    row = rows_by_req["r1"]
    assert len(row.removed_profiles) == 1
    assert row.removed_profiles[0].profile_id == "p-old"
    assert row.removed_profiles[0].content == "old facts"


def test_parity_with_legacy_shape(tmp_path):
    """Reconstructed row matches the legacy change-log shape field-by-field."""
    s = _store(tmp_path)
    old_content = "old known facts"
    new_content = "new known facts"
    old = _make_profile(
        user_id="u1", profile_id="p-old", content=old_content, request_id="r-seed"
    )
    s.add_user_profile("u1", [old])
    new = _make_profile(
        user_id="u1", profile_id="p-new", content=new_content, request_id="r-parity"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old"], request_id="r-parity"
    )

    result = reconstruct_profile_change_log(s)
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    row = rows_by_req["r-parity"]

    # added: content and profile_id match
    a = row.added_profiles[0]
    assert a.profile_id == "p-new"
    assert a.user_id == "u1"
    assert a.content == new_content

    # removed: content and profile_id match
    r = row.removed_profiles[0]
    assert r.profile_id == "p-old"
    assert r.user_id == "u1"
    assert r.content == old_content


# --------------------------------------------------------------------------
# TIME-TRAVEL regression (F3): the core stability guarantee
# --------------------------------------------------------------------------


def test_time_travel_added_in_r1_tombstoned_in_r2(tmp_path):
    """TIME-TRAVEL: profile added in R1, tombstoned in R2 -> still 'added in R1'.

    This is the F3 regression test. The old current-status model had a bug:
    if a profile added in run R1 was later superseded in run R2, it would be
    classified as 'removed in R1' (wrong). The stable column model is correct:
    generated_from_request_id never changes, so the profile is always 'added in R1'.
    """
    s = _store(tmp_path)

    # R0: seed an old profile to be removed in R1
    old_p = _make_profile(
        user_id="u1", profile_id="p-old", content="old content", request_id="r0"
    )
    s.add_user_profile("u1", [old_p])

    # R1: dedup run adds p-r1 (generated_from_request_id="r1"), removes p-old
    p_r1 = _make_profile(
        user_id="u1", profile_id="p-r1", content="r1 content", request_id="r1"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[p_r1], old_ids=["p-old"], request_id="r1"
    )

    # R2: later dedup run supersedes p-r1 (the profile added in R1)
    p_r2 = _make_profile(
        user_id="u1", profile_id="p-r2", content="r2 content", request_id="r2"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[p_r2], old_ids=["p-r1"], request_id="r2"
    )

    result = reconstruct_profile_change_log(s)
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}

    # R1: p-r1 must still appear as ADDED in R1 — even though it's now a tombstone
    assert "r1" in rows_by_req, "R1 must produce a changelog row"
    r1_row = rows_by_req["r1"]
    r1_added_ids = {p.profile_id for p in r1_row.added_profiles}
    r1_removed_ids = {p.profile_id for p in r1_row.removed_profiles}
    assert "p-r1" in r1_added_ids, "p-r1 was added in R1 — must appear as added in R1"
    assert "p-r1" not in r1_removed_ids, "p-r1 must NOT appear as removed in R1"

    # R2: p-r1 must appear as REMOVED in R2 (it was superseded there)
    assert "r2" in rows_by_req, "R2 must produce a changelog row"
    r2_row = rows_by_req["r2"]
    r2_removed_ids = {p.profile_id for p in r2_row.removed_profiles}
    r2_added_ids = {p.profile_id for p in r2_row.added_profiles}
    assert "p-r1" in r2_removed_ids, (
        "p-r1 was superseded in R2 — must appear as removed in R2"
    )
    assert "p-r2" in r2_added_ids, "p-r2 was added in R2"


# --------------------------------------------------------------------------
# Empty request_id group: must be skipped
# --------------------------------------------------------------------------


def test_empty_request_id_group_is_skipped(tmp_path):
    """A profile with generated_from_request_id='' must not create a changelog row.

    The empty-string request_id is a runtime guard — never merge unrelated runs
    under "".
    """
    s = _store(tmp_path)
    # Profile with empty generated_from_request_id
    p = _make_profile(
        user_id="u1", profile_id="p-empty", content="content", request_id=""
    )
    s.add_user_profile("u1", [p])

    result = reconstruct_profile_change_log(s)
    req_ids = {row.request_id for row in result.profile_change_logs}
    assert "" not in req_ids, "empty request_id must never appear in reconstruction"
    assert result.profile_change_logs == [], (
        "no dedup activity should produce empty log"
    )


# --------------------------------------------------------------------------
# Purged tombstone: graceful handling
# --------------------------------------------------------------------------


def test_purged_tombstone_no_crash(tmp_path):
    """When the removed profile's tombstone has been GC'd, no crash occurs.

    removed_profiles is empty for the purged entry; added_profiles still present.
    """
    s = _store(tmp_path)
    old = _make_profile(
        user_id="u1", profile_id="p-old-gc", content="to be purged", request_id="r-seed"
    )
    s.add_user_profile("u1", [old])
    new = _make_profile(
        user_id="u1", profile_id="p-new-gc", content="survivor", request_id="r-gc"
    )
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old-gc"], request_id="r-gc"
    )

    # Simulate GC: hard-delete the tombstone row
    s.conn.execute("DELETE FROM profiles WHERE profile_id = ?", ("p-old-gc",))
    s.conn.commit()

    result = reconstruct_profile_change_log(s)
    assert result.success
    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    assert "r-gc" in rows_by_req
    row = rows_by_req["r-gc"]
    # removed is empty — purged tombstone silently omitted
    assert row.removed_profiles == []
    # added still has the survivor
    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-new-gc"


# --------------------------------------------------------------------------
# Reflection revise event: must NOT be counted as a dedup removal
# --------------------------------------------------------------------------


def test_reflection_revise_event_not_counted_as_removal(tmp_path):
    """A reflection op="revise" event must NOT appear as a removed profile.

    The dedup removal signal is specifically op="status_change" with
    to_status="superseded". A revise event from reflection is a different op
    and must be ignored by reconstruct_profile_change_log.
    """
    s = _store(tmp_path)
    # Add two profiles; supersede one via supersede_record (reflection path,
    # emits op="revise") — NOT the dedup path.
    old = _make_profile(
        user_id="u1", profile_id="p-revise-old", content="old", request_id="r-reflect"
    )
    new = _make_profile(
        user_id="u1", profile_id="p-revise-new", content="new", request_id="r-reflect"
    )
    s.add_user_profile("u1", [old, new])
    ctx = LineageContext(
        op_kind="revise", actor="reflection", request_id="r-reflect-req"
    )
    s.supersede_record(
        entity_type="profile",
        incumbent_id="p-revise-old",
        successor_id="p-revise-new",
        context=ctx,
    )

    result = reconstruct_profile_change_log(s)
    # The "r-reflect-req" has events, but no status_change+superseded events,
    # so removed should be empty. The "r-reflect" request_id has profiles with
    # generated_from_request_id="r-reflect", so it may appear as adds-only.
    # The key invariant: p-revise-old must NOT appear as a removed profile via
    # the revise event's request_id.
    all_removed_ids = {
        p.profile_id for row in result.profile_change_logs for p in row.removed_profiles
    }
    assert "p-revise-old" not in all_removed_ids, (
        "reflection revise event must NOT count as dedup removal"
    )


# --------------------------------------------------------------------------
# Adds-only run: no removals
# --------------------------------------------------------------------------


def test_adds_only_run_produces_row(tmp_path):
    """An add-only dedup run (no removals) now produces a change-log row.

    B3-pre T6a closes the reconstruction-completeness gap: generated_from_request_id
    values are unioned into the candidate pool, so a run that only added profiles
    (no supersede_profiles_by_ids call, hence no lineage event) is still discovered
    and yields a row with the N added profiles and removed_profiles=[].
    """
    s = _store(tmp_path)
    # Adds-only: two new profiles with the same generated_from_request_id.
    p1 = _make_profile(
        user_id="u1", profile_id="p-a1", content="fact A", request_id="r-adds"
    )
    p2 = _make_profile(
        user_id="u1", profile_id="p-a2", content="fact B", request_id="r-adds"
    )
    s.add_user_profile("u1", [p1, p2])
    # No supersede_profiles_by_ids call — adds only.

    result = reconstruct_profile_change_log(s)
    assert result.success

    rows_by_req = {row.request_id: row for row in result.profile_change_logs}
    assert "r-adds" in rows_by_req, (
        "add-only run must now appear in reconstruction (B3-pre T6a gap closure)"
    )
    row = rows_by_req["r-adds"]

    added_ids = {p.profile_id for p in row.added_profiles}
    assert added_ids == {"p-a1", "p-a2"}, (
        f"expected both added profiles; got {added_ids}"
    )
    assert row.removed_profiles == [], "add-only run has no removed profiles"


# --------------------------------------------------------------------------
# status_change with non-superseded to_status does not produce removal
# --------------------------------------------------------------------------


def test_status_change_only_no_supersede_produces_no_removal(tmp_path):
    """A status_change to 'archived' does NOT count as a dedup removal.

    Only status_change with to_status=='superseded' is the dedup signature.
    """
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="psc-only", request_id="r-arc")
    s.add_user_profile("u1", [profile])
    s.archive_profile_by_id("u1", "psc-only")

    result = reconstruct_profile_change_log(s)
    # archive emits status_change to_status="archived", not "superseded"
    # → no removal signal, and no lineage event row for "r-arc" in the output
    for row in result.profile_change_logs:
        assert all(p.profile_id != "psc-only" for p in row.removed_profiles)


# --------------------------------------------------------------------------
# limit: bounds the number of reconstructed entries
# --------------------------------------------------------------------------


def test_limit_bounds_reconstructed_entries(tmp_path):
    """limit=N returns at most N reconstructed change-log entries."""
    s = _store(tmp_path)
    for i in range(5):
        old = _make_profile(
            user_id="u1", profile_id=f"p-old-{i}", request_id=f"r-seed-{i}"
        )
        s.add_user_profile("u1", [old])
        new = _make_profile(
            user_id="u1", profile_id=f"p-new-{i}", request_id=f"r-run-{i}"
        )
        _seed_dedup_run(
            s,
            user_id="u1",
            new_profiles=[new],
            old_ids=[f"p-old-{i}"],
            request_id=f"r-run-{i}",
        )
    result = reconstruct_profile_change_log(s, limit=3)
    assert result.success
    assert len(result.profile_change_logs) <= 3


def test_limit_zero_returns_empty(tmp_path):
    """limit=0 returns an empty list."""
    s = _store(tmp_path)
    old = _make_profile(user_id="u1", profile_id="p-old", request_id="r0")
    s.add_user_profile("u1", [old])
    new = _make_profile(user_id="u1", profile_id="p-new", request_id="r1")
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new], old_ids=["p-old"], request_id="r1"
    )
    result = reconstruct_profile_change_log(s, limit=0)
    assert result.success
    assert result.profile_change_logs == []


# --------------------------------------------------------------------------
# Most-recent-first ordering
# --------------------------------------------------------------------------


def test_most_recent_first_ordering(tmp_path):
    """Reconstructed rows are ordered most-recent first."""
    s = _store(tmp_path)

    # Run 1
    old0 = _make_profile(user_id="u1", profile_id="p-old-0", request_id="r-seed-0")
    s.add_user_profile("u1", [old0])
    new0 = _make_profile(user_id="u1", profile_id="p-new-0", request_id="r-first")
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new0], old_ids=["p-old-0"], request_id="r-first"
    )

    # Run 2 (later)
    old1 = _make_profile(user_id="u1", profile_id="p-old-1", request_id="r-seed-1")
    s.add_user_profile("u1", [old1])
    new1 = _make_profile(user_id="u1", profile_id="p-new-1", request_id="r-second")
    _seed_dedup_run(
        s, user_id="u1", new_profiles=[new1], old_ids=["p-old-1"], request_id="r-second"
    )

    result = reconstruct_profile_change_log(s)
    req_ids = [row.request_id for row in result.profile_change_logs]
    # r-second was written after r-first, so it should come first
    assert req_ids.index("r-second") < req_ids.index("r-first")


# --------------------------------------------------------------------------
# Cross-org isolation
# --------------------------------------------------------------------------


def test_cross_org_isolation(tmp_path):
    """reconstruct_profile_change_log for org A must not return org B's runs.

    SQLite is a per-org DB in production (each org has its own db_path), so
    cross-org isolation is enforced by the file path — not by an org_id column
    on the profiles table. This test uses separate DB files to mirror that
    production topology.
    """
    s_a = SQLiteStorage(org_id="org-a", db_path=str(tmp_path / "a.db"))
    s_a.migrate()
    s_b = SQLiteStorage(org_id="org-b", db_path=str(tmp_path / "b.db"))
    s_b.migrate()

    old_a = _make_profile(user_id="ua", profile_id="pa-old", request_id="r-a-seed")
    s_a.add_user_profile("ua", [old_a])
    new_a = _make_profile(user_id="ua", profile_id="pa-new", request_id="r-a-run")
    _seed_dedup_run(
        s_a,
        user_id="ua",
        new_profiles=[new_a],
        old_ids=["pa-old"],
        request_id="r-a-run",
    )

    old_b = _make_profile(user_id="ub", profile_id="pb-old", request_id="r-b-seed")
    s_b.add_user_profile("ub", [old_b])
    new_b = _make_profile(user_id="ub", profile_id="pb-new", request_id="r-b-run")
    _seed_dedup_run(
        s_b,
        user_id="ub",
        new_profiles=[new_b],
        old_ids=["pb-old"],
        request_id="r-b-run",
    )

    result_a = reconstruct_profile_change_log(s_a)
    assert result_a.success
    req_ids_a = {row.request_id for row in result_a.profile_change_logs}
    assert "r-a-run" in req_ids_a
    assert "r-b-run" not in req_ids_a

    result_b = reconstruct_profile_change_log(s_b)
    assert result_b.success
    req_ids_b = {row.request_id for row in result_b.profile_change_logs}
    assert "r-b-run" in req_ids_b
    assert "r-a-run" not in req_ids_b


# --------------------------------------------------------------------------
# Empty storage
# --------------------------------------------------------------------------


def test_no_events_returns_empty(tmp_path):
    """With no lineage events AND no profiles, returns an empty list."""
    s = _store(tmp_path)
    result = reconstruct_profile_change_log(s)
    assert result.success
    assert result.profile_change_logs == []


# --------------------------------------------------------------------------
# get_distinct_generated_from_request_ids storage query
# --------------------------------------------------------------------------


def test_get_distinct_generated_from_request_ids_returns_correct_set(tmp_path):
    """The storage query returns the right distinct set, including tombstoned profiles."""
    s = _store(tmp_path)

    # Two profiles from run "r-live" (both live), one from run "r-tomb" (tombstoned).
    p_live1 = _make_profile(user_id="u1", profile_id="p-live-1", request_id="r-live")
    p_live2 = _make_profile(user_id="u1", profile_id="p-live-2", request_id="r-live")
    p_tomb = _make_profile(user_id="u1", profile_id="p-tomb", request_id="r-tomb")
    s.add_user_profile("u1", [p_live1, p_live2, p_tomb])

    # Tombstone p_tomb by superseding it under a different run.
    p_newer = _make_profile(user_id="u1", profile_id="p-newer", request_id="r-newer")
    s.add_user_profile("u1", [p_newer])
    s.supersede_profiles_by_ids("u1", ["p-tomb"], "r-newer")

    result = set(s.get_distinct_generated_from_request_ids())

    # r-live and r-tomb must appear; r-newer must also appear (p_newer is live).
    assert "r-live" in result
    assert "r-tomb" in result, "tombstoned profiles must still contribute their run id"
    assert "r-newer" in result

    # No duplicates: r-live had two profiles but should appear once.
    raw = s.get_distinct_generated_from_request_ids()
    assert raw.count("r-live") == 1, "DISTINCT must deduplicate"


def test_get_distinct_generated_from_request_ids_excludes_empty(tmp_path):
    """Empty-string generated_from_request_id values are excluded from the result."""
    s = _store(tmp_path)
    # One profile with empty generated_from_request_id.
    p_empty = UserProfile(
        user_id="u1",
        profile_id="p-empty",
        content="c",
        last_modified_timestamp=1,
        generated_from_request_id="",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )
    s.add_user_profile("u1", [p_empty])

    result = s.get_distinct_generated_from_request_ids()
    assert "" not in result, "empty-string request_id must be excluded"
    assert result == [], "no non-empty ids → empty result"


def test_reconstruct_resolves_adds_in_one_bulk_read_not_per_candidate(tmp_path):
    """F1 perf-regression guard: the "added" side is one bulk read, not N.

    Regardless of how many runs exist, reconstruct must call
    get_all_generated_profiles exactly once and never the per-id
    get_profiles_by_generated_from_request_id — so the now-live endpoint does
    not fan out over the org's whole dedup history before the limit slice.
    """
    s = _store(tmp_path)
    for i in range(5):
        s.add_user_profile(
            "u1", [_make_profile("u1", f"p{i}", f"c{i}", request_id=f"run-{i}")]
        )

    calls = {"bulk": 0, "per_id": 0}
    orig_bulk = s.get_all_generated_profiles
    orig_per = s.get_profiles_by_generated_from_request_id

    def _bulk():
        calls["bulk"] += 1
        return orig_bulk()

    def _per(request_id):
        calls["per_id"] += 1
        return orig_per(request_id)

    s.get_all_generated_profiles = _bulk  # type: ignore[method-assign]
    s.get_profiles_by_generated_from_request_id = _per  # type: ignore[method-assign]

    resp = reconstruct_profile_change_log(s)

    assert calls["bulk"] == 1, "added side must be a single bulk read"
    assert calls["per_id"] == 0, "no per-candidate fan-out"
    assert {r.request_id for r in resp.profile_change_logs} == {
        f"run-{i}" for i in range(5)
    }
