"""Dashboard profile metrics describe current profiles by their creation time."""

from datetime import UTC, datetime

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage, _extras


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
