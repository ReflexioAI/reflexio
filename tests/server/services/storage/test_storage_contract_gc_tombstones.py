"""Storage contract tests for gc_expired_tombstones (Lineage Phase B2, Task 5).

Parametrized over locally-testable backends via the shared ``storage`` fixture
in conftest.py (currently SQLite only).  Enterprise backends add their params
in Task 6; they will skip here without DATA_* env vars.

Seeding note
------------
GC now ages on ``retired_at`` (INTEGER epoch, set at every tombstone write-path).
After calling ``merge_records`` (which sets ``retired_at = now``), tests that need
an "aged" tombstone must back-date ``retired_at`` via a raw SQL helper.  The
SQLite-backed ``storage`` fixture exposes ``storage.conn`` for this purpose.

Profile tables are used for contract cases because ``merge_records`` works on all
entity types and the ``retired_at`` column exists on all tombstone-bearing tables.
"""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.service_schemas import UserProfile

pytestmark = pytest.mark.integration

# A fixed epoch well in the past, used to seed "aged" retired_at values.
_EPOCH_2020 = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
# A cutoff after the old epoch — any tombstone retired_at < this is eligible.
_CUTOFF_2021 = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
# A future epoch for "fresh" retired_at values — after any historical cutoff.
_EPOCH_FUTURE = int(datetime(2035, 1, 1, tzinfo=UTC).timestamp())


def _make_profile(profile_id: str, ts: int | None = None) -> UserProfile:
    from datetime import UTC, datetime

    ts = ts or int(datetime.now(UTC).timestamp())
    return UserProfile(
        user_id="u1",
        profile_id=profile_id,
        content=f"content for {profile_id}",
        last_modified_timestamp=ts,
        generated_from_request_id=f"req-{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _merge_into_survivor(
    storage, source_profile_id: str, survivor_profile_id: str
) -> None:
    """Tombstone the source into the survivor via the public merge_records API."""
    storage.merge_records(
        entity_type="profile",
        survivor_id=survivor_profile_id,
        source_ids=[source_profile_id],
        context=LineageContext(
            op_kind="merge",
            actor="test",
            request_id=f"req-merge-{source_profile_id}",
        ),
    )


def _backdate_retired_at(storage, profile_id: str, retired_at: int) -> None:
    """Back-date retired_at on a profile row for test seeding.

    merge_records sets retired_at = now; tests that need an "aged" tombstone must
    call this after the merge.
    """
    storage.conn.execute(
        "UPDATE profiles SET retired_at = ? WHERE profile_id = ?",
        (retired_at, profile_id),
    )
    storage.conn.commit()


# ---------------------------------------------------------------------------
# Case 1: Aged tombstone (MERGED, old retired_at) is hard-deleted
# and a hard_delete lineage event is emitted.
# ---------------------------------------------------------------------------


def test_gc_deletes_aged_merged_profile_and_emits_hard_delete(storage) -> None:
    """Aged MERGED profile (retired_at past cutoff) is deleted; a hard_delete event is recorded."""
    old_profile = _make_profile("gc-aged-src")
    survivor_profile = _make_profile("gc-aged-survivor")
    storage.add_user_profile("u1", [old_profile])
    storage.add_user_profile("u1", [survivor_profile])

    # Tombstone old_profile (MERGED) via public API; retired_at is set to now.
    _merge_into_survivor(storage, "gc-aged-src", "gc-aged-survivor")
    # Back-date retired_at so it falls before the cutoff.
    _backdate_retired_at(storage, "gc-aged-src", _EPOCH_2020)

    deleted = storage.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=_CUTOFF_2021
    )

    assert deleted == 1

    # The tombstoned row must be physically gone (even with include_tombstones=True).
    assert storage.get_profile_by_id("gc-aged-src", include_tombstones=True) is None

    # A hard_delete lineage event must have been emitted for the deleted row.
    events = storage.get_lineage_events(entity_type="profile", entity_id="gc-aged-src")
    hd_events = [e for e in events if e.op == "hard_delete"]
    assert len(hd_events) == 1


# ---------------------------------------------------------------------------
# Case 2: CURRENT row (status None) is NOT deleted, even with a future cutoff.
# ---------------------------------------------------------------------------


def test_gc_does_not_delete_current_row(storage) -> None:
    """A CURRENT profile (status None, retired_at NULL) must never be deleted by gc_expired_tombstones."""
    current = _make_profile("gc-current")
    storage.add_user_profile("u1", [current])

    # Cutoff is far in the future — would delete any tombstone — but status is NULL.
    deleted = storage.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=_EPOCH_FUTURE
    )

    assert deleted == 0
    # CURRENT profile must still exist.
    assert storage.get_profile_by_id("gc-current", include_tombstones=True) is not None


# ---------------------------------------------------------------------------
# Case 3: A fresh tombstone (recent retired_at) is NOT deleted by a historical cutoff.
# ---------------------------------------------------------------------------


def test_gc_does_not_delete_fresh_tombstone(storage) -> None:
    """A tombstone whose retired_at is after the cutoff must survive GC."""
    fresh_src = _make_profile("gc-fresh-src")
    fresh_survivor = _make_profile("gc-fresh-survivor")
    storage.add_user_profile("u1", [fresh_src])
    storage.add_user_profile("u1", [fresh_survivor])

    # Tombstone the source — retired_at is set to now by merge_records.
    _merge_into_survivor(storage, "gc-fresh-src", "gc-fresh-survivor")
    # Leave retired_at as-is (current timestamp, well after _CUTOFF_2021).

    # Cutoff is 2021 — the tombstone's retired_at (now) is after the cutoff; must NOT be GC'd.
    deleted = storage.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=_CUTOFF_2021
    )

    assert deleted == 0
    assert (
        storage.get_profile_by_id("gc-fresh-src", include_tombstones=True) is not None
    ), "Fresh tombstone must survive when retired_at is after cutoff"

    # The fresh tombstone must still be retrievable.
    events = storage.get_lineage_events(entity_type="profile", entity_id="gc-fresh-src")
    hd_events = [e for e in events if e.op == "hard_delete"]
    assert len(hd_events) == 0


# ---------------------------------------------------------------------------
# Case 4: Idempotent — second gc call returns 0 and emits no new events.
# ---------------------------------------------------------------------------


def test_gc_idempotent_returns_zero_on_second_call(storage) -> None:
    """After GC deletes a tombstone, a second identical call returns 0 with no new events."""
    old_profile = _make_profile("gc-idem-src")
    survivor = _make_profile("gc-idem-survivor")
    storage.add_user_profile("u1", [old_profile])
    storage.add_user_profile("u1", [survivor])
    _merge_into_survivor(storage, "gc-idem-src", "gc-idem-survivor")
    _backdate_retired_at(storage, "gc-idem-src", _EPOCH_2020)

    first = storage.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=_CUTOFF_2021
    )
    assert first == 1

    events_after_first = storage.get_lineage_events(
        entity_type="profile", entity_id="gc-idem-src"
    )

    second = storage.gc_expired_tombstones(
        entity_type="profile", older_than_epoch=_CUTOFF_2021
    )
    assert second == 0

    events_after_second = storage.get_lineage_events(
        entity_type="profile", entity_id="gc-idem-src"
    )
    assert len(events_after_second) == len(events_after_first)
