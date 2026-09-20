"""Dashboard profile metrics describe current profiles by their creation time."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage, _extras
from reflexio.server.services.storage.sqlite_storage.profiles import _profile_store


@pytest.mark.parametrize("age_days", [1, 40, 70])
def test_upsert_preserves_profile_creation_cohort(tmp_path, monkeypatch, age_days):
    """Retries and content edits must not move a profile into today's cohort."""
    now = 1_800_000_000
    day = 86400
    created = now - age_days * day
    created_at = datetime.fromtimestamp(created, UTC).isoformat()
    monkeypatch.setattr(_extras, "_epoch_now", lambda: now)
    monkeypatch.setattr(_profile_store, "_iso_now", lambda: created_at)
    storage = SQLiteStorage(db_path=str(tmp_path / "profiles.db"), org_id="dashboard")
    profile = UserProfile(
        profile_id="existing",
        user_id="u",
        content="original",
        generated_from_request_id="r",
        last_modified_timestamp=created,
    )
    storage.add_user_profile("u", [profile], skip_embedding=True)
    before = storage.get_dashboard_stats(days_back=30)
    assert before["current_period"]["total_profiles"] == int(age_days <= 30)
    assert before["previous_period"]["total_profiles"] == int(30 < age_days <= 60)

    monkeypatch.setattr(
        _profile_store, "_iso_now", lambda: datetime.fromtimestamp(now, UTC).isoformat()
    )
    for replacement in [
        profile,
        profile.model_copy(
            update={"content": "edited", "last_modified_timestamp": now}
        ),
    ]:
        storage.add_user_profile("u", [replacement], skip_embedding=True)
        row = storage.conn.execute(
            "SELECT created_at, content, last_modified_timestamp FROM profiles WHERE profile_id=?",
            (profile.profile_id,),
        ).fetchone()
        assert row["created_at"] == created_at
        assert row["content"] == replacement.content
        assert row["last_modified_timestamp"] == replacement.last_modified_timestamp
        after = storage.get_dashboard_stats(days_back=30)
        for key in ("current_period", "previous_period", "profiles_time_series"):
            assert after[key] == before[key]
        assert [
            p.profile_id
            for p in storage.get_all_profiles(
                start_time=created, end_time=created, date_field="created_at"
            )
        ] == [profile.profile_id]
        assert storage.get_all_profiles(start_time=now, date_field="created_at") == []


def test_current_creation_cohorts_and_boundaries(tmp_path, monkeypatch):
    now = 1_800_000_000
    day = 86400
    monkeypatch.setattr(_extras, "_epoch_now", lambda: now)
    storage = SQLiteStorage(db_path=str(tmp_path / "profiles.db"), org_id="dashboard")
    rows = [
        ("new", None, now - day),
        ("start", None, now - 30 * day),
        ("end", None, now),
        ("previous", None, now - 30 * day - 1),
        ("previous_start", None, now - 60 * day),
        ("old_edited_today", None, now - 70 * day),
        ("future", None, now + 1),
    ] + [
        (s, s, now - day)
        for s in ["superseded", "merged", "archived", "pending", "expired"]
    ]
    for pid, status, created in rows:
        storage.add_user_profile(
            "u",
            [
                UserProfile(
                    profile_id=pid,
                    user_id="u",
                    content=pid,
                    generated_from_request_id="r",
                    last_modified_timestamp=now,
                )
            ],
        )
        storage._execute(
            "UPDATE profiles SET created_at=?,status=? WHERE profile_id=?",
            (
                datetime.fromtimestamp(created, UTC).isoformat(),
                status,
                pid,
            ),
        )
    stats = storage.get_dashboard_stats(days_back=30)
    assert stats["current_period"]["total_profiles"] == 3
    assert stats["previous_period"]["total_profiles"] == 2
    assert sum(p["value"] for p in stats["profiles_time_series"]) == 3
    cohort = storage.get_all_profiles(
        start_time=now - 30 * day, end_time=now, date_field="created_at"
    )
    assert {p.profile_id for p in cohort} == {"new", "start", "end"}
    assert len(cohort) == stats["current_period"]["total_profiles"]
    assert "old_edited_today" in {
        p.profile_id
        for p in storage.get_all_profiles(start_time=now - 30 * day, end_time=now)
    }
    assert [
        p.profile_id
        for p in storage.get_all_profiles(
            limit=1,
            profile_id="new",
            user_id="u",
            start_time=now - day,
            end_time=now - day,
            date_field="created_at",
        )
    ] == ["new"]
    # Retiring a profile removes it from both the tile and its creation-day bucket.
    storage._execute("UPDATE profiles SET status='superseded' WHERE profile_id='new'")
    stats = storage.get_dashboard_stats(days_back=30)
    assert stats["current_period"]["total_profiles"] == 2
    assert sum(p["value"] for p in stats["profiles_time_series"]) == 2
