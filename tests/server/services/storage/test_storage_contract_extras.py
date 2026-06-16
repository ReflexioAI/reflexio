"""Contract tests for ExtrasMixin — run against every local storage backend."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.service_schemas import (
    Citation,
    Interaction,
    ProfileChangeLog,
    UserProfile,
)
from reflexio.server.services.storage.storage_base._extras import ExtrasMixin

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile_change_log(user_id: str, change_description: str) -> ProfileChangeLog:
    return ProfileChangeLog(
        id=0,
        user_id=user_id,
        request_id=f"req-{user_id}",
        created_at=1_700_000_000,
        added_profiles=[
            UserProfile(
                user_id=user_id,
                profile_id=f"prof-{user_id}",
                content=change_description,
                last_modified_timestamp=1_700_000_000,
                generated_from_request_id=f"req-{user_id}",
            )
        ],
        removed_profiles=[],
        mentioned_profiles=[],
    )


# ---------------------------------------------------------------------------
# TestProfileChangeLogs
# ---------------------------------------------------------------------------


class TestProfileChangeLogs:
    def test_add_and_get_profile_change_logs(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "added greeting"))
        storage.add_profile_change_log(
            _make_profile_change_log("u2", "added preference")
        )

        logs = storage.get_profile_change_logs()
        assert len(logs) == 2

    def test_delete_profile_change_log_for_user(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "log for u1"))
        storage.add_profile_change_log(_make_profile_change_log("u2", "log for u2"))

        storage.delete_profile_change_log_for_user("u1")

        logs = storage.get_profile_change_logs()
        assert len(logs) == 1
        assert logs[0].user_id == "u2"

    def test_delete_all_profile_change_logs(self, storage):
        storage.add_profile_change_log(_make_profile_change_log("u1", "log 1"))
        storage.add_profile_change_log(_make_profile_change_log("u2", "log 2"))

        storage.delete_all_profile_change_logs()
        assert storage.get_profile_change_logs() == []


# ---------------------------------------------------------------------------
# TestPlaybookApplicationStats
# ---------------------------------------------------------------------------


def _make_interaction(
    request_id: str,
    created_at: int,
    citations: list[Citation],
) -> Interaction:
    return Interaction(
        interaction_id=0,
        user_id="u1",
        request_id=request_id,
        created_at=created_at,
        role="Assistant",
        content="answer",
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
        shadow_content="",
        expert_content="",
        tools_used=[],
        citations=citations,
    )


class TestPlaybookApplicationStats:
    def test_empty_when_no_citations(self, storage):
        # Backends that have no implementation return [] from the default; SQLite
        # returns [] when no interactions carry citations. Either way: empty.
        assert storage.get_playbook_application_stats(days_back=30) == []

    def test_aggregates_by_kind_and_real_id(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        # Two interactions cite playbook 42; one also cites profile p-99.
        storage._insert_interaction(
            _make_interaction(
                "r1",
                now - 100,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                    Citation(
                        kind="profile", real_id="p-99", tag="p1-99", title="terse"
                    ),
                ],
            )
        )
        storage._insert_interaction(
            _make_interaction(
                "r2",
                now,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    )
                ],
            )
        )

        stats = storage.get_playbook_application_stats(days_back=30)
        assert len(stats) == 2

        # Most-applied row sorts first.
        top = stats[0]
        assert top.kind == "playbook"
        assert top.real_id == "42"
        assert top.applied_count == 2
        assert top.title == "timeline"
        # last_applied_at should be the LATER of the two interactions.
        assert top.last_applied_at == now

        profile_row = stats[1]
        assert profile_row.kind == "profile"
        assert profile_row.real_id == "p-99"
        assert profile_row.applied_count == 1

    def test_respects_days_back_window(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        old = now - 60 * 24 * 60 * 60  # 60 days ago
        storage._insert_interaction(
            _make_interaction(
                "r_old",
                old,
                [Citation(kind="playbook", real_id="42", tag="s1-2a", title="old")],
            )
        )
        # 30-day window excludes the 60-day-old citation.
        assert storage.get_playbook_application_stats(days_back=30) == []
        # 90-day window includes it.
        stats = storage.get_playbook_application_stats(days_back=90)
        assert len(stats) == 1 and stats[0].applied_count == 1

    def test_counts_duplicate_citations_once_per_interaction(self, storage):
        if not _backend_supports_application_stats(storage):
            pytest.skip("Backend does not implement get_playbook_application_stats")
        now = int(datetime.now(tz=UTC).timestamp())
        storage._insert_interaction(
            _make_interaction(
                "r_duplicate",
                now,
                [
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                    Citation(
                        kind="playbook", real_id="42", tag="s1-2a", title="timeline"
                    ),
                ],
            )
        )

        stats = storage.get_playbook_application_stats(days_back=30)

        assert len(stats) == 1
        assert stats[0].applied_count == 1


def _backend_supports_application_stats(storage) -> bool:
    """True when the storage backend has a real (non-default) implementation.

    The default in ``ExtrasMixin`` returns ``[]`` for any input — backends
    that haven't been wired up yet (supabase, postgres) hit that path
    and have nothing to test.
    """
    return (
        storage.__class__.get_playbook_application_stats
        is not ExtrasMixin.get_playbook_application_stats
    )


# ---------------------------------------------------------------------------
# TestUsageEvents (per-entity observability)
# ---------------------------------------------------------------------------


def _backend_supports_usage_events(storage) -> bool:
    """True when the storage backend implements ``record_usage_event`` and
    ``get_injection_stats`` (not the default no-op / ``[]`` path)."""
    return (
        storage.__class__.record_usage_event
        is not ExtrasMixin.record_usage_event
    ) and (
        storage.__class__.get_injection_stats
        is not ExtrasMixin.get_injection_stats
    )


def _backend_supports_memory_review(storage) -> bool:
    """True when the storage backend implements ``get_memory_review_candidates``."""
    return (
        storage.__class__.get_memory_review_candidates
        is not ExtrasMixin.get_memory_review_candidates
    )


class TestUsageEvents:
    def test_record_usage_event_persists_a_row(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement record_usage_event")
        storage.record_usage_event(
            org_id=storage.org_id,
            event_name="learning_injection",
            event_category="application",
            entity_type="playbook",
            entity_id="42",
            caller_type="production_agent",
            session_id="s1",
            request_id="r1",
            prompt_tokens=10,
        )
        # The row is queryable via get_injection_stats (it filters on
        # event_name='learning_injection' and entity_id IS NOT NULL).
        stats = storage.get_injection_stats(days_back=30)
        assert len(stats) == 1
        assert stats[0].entity_type == "playbook"
        assert stats[0].entity_id == "42"
        assert stats[0].surfaced_count == 1
        assert stats[0].total_prompt_tokens == 10

    def test_record_usage_event_rejects_org_mismatch(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement record_usage_event")
        from reflexio.server.services.storage.error import StorageError
        with pytest.raises(StorageError, match="org_id mismatch"):
            storage.record_usage_event(
                org_id="wrong-org",
                event_name="learning_injection",
                event_category="application",
            )

    def test_get_injection_stats_empty_when_no_events(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement get_injection_stats")
        assert storage.get_injection_stats(days_back=30) == []

    def test_get_injection_stats_empty_for_zero_days_back(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement get_injection_stats")
        # Defensive guard: zero/negative days_back returns [] (mirrors
        # get_playbook_application_stats). Pydantic at the API layer
        # already enforces gt=0, but storage should be robust on its own.
        assert storage.get_injection_stats(days_back=0) == []
        assert storage.get_injection_stats(days_back=-1) == []

    def test_get_injection_stats_aggregates_by_entity(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement get_injection_stats")
        # Two events for playbook 42 (one in s1, one in s2), one for profile p-99.
        for sid, rid in (("s1", "r1"), ("s2", "r2")):
            storage.record_usage_event(
                org_id=storage.org_id,
                event_name="learning_injection",
                event_category="application",
                entity_type="playbook",
                entity_id="42",
                caller_type="production_agent",
                session_id=sid,
                request_id=rid,
                prompt_tokens=5,
            )
        storage.record_usage_event(
            org_id=storage.org_id,
            event_name="learning_injection",
            event_category="application",
            entity_type="profile",
            entity_id="p-99",
            caller_type="production_agent",
            session_id="s1",
            request_id="r1",
            prompt_tokens=3,
        )
        stats = storage.get_injection_stats(days_back=30)
        assert len(stats) == 2
        # Most-injected sorts first.
        top = stats[0]
        assert top.entity_type == "playbook"
        assert top.entity_id == "42"
        assert top.surfaced_count == 2
        assert top.total_prompt_tokens == 10
        assert top.distinct_session_count is None or top.distinct_session_count >= 1
        # Profile row second.
        bottom = stats[1]
        assert bottom.entity_type == "profile"
        assert bottom.entity_id == "p-99"
        assert bottom.surfaced_count == 1
        assert bottom.total_prompt_tokens == 3

    def test_get_injection_stats_ignores_other_event_names(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement get_injection_stats")
        # The aggregation only counts ``learning_injection``. A row of
        # ``extraction_tokens`` for the same (entity_type, entity_id)
        # must NOT inflate the surfaced_count.
        storage.record_usage_event(
            org_id=storage.org_id,
            event_name="extraction_tokens",
            event_category="learning",
            entity_type="playbook",
            entity_id="42",
            caller_type="internal",
            prompt_tokens=999,
        )
        stats = storage.get_injection_stats(days_back=30)
        assert stats == []

    def test_get_injection_stats_respects_days_back_window(self, storage):
        if not _backend_supports_usage_events(storage):
            pytest.skip("Backend does not implement get_injection_stats")
        # Insert one event 60 days ago; the 30-day window excludes it.
        # We use a custom created_at to bypass the auto-now default.
        # The storage layer's row insert accepts created_at via the
        # `record_usage_event` signature? No — the helper does not
        # accept created_at. We rely on the default ``now()`` and a
        # direct SQL insert for the backdated row.
        now_iso = "2020-01-01T00:00:00.000Z"
        with storage._lock:
            storage.conn.execute(
                """INSERT INTO usage_events
                     (org_id, event_name, event_category, entity_type,
                      entity_id, count_value, created_at)
                   VALUES (?, 'learning_injection', 'application',
                           'playbook', '99', 1, ?)""",
                (storage.org_id, now_iso),
            )
            storage.conn.commit()
        # 30-day window excludes the 2020 row.
        assert storage.get_injection_stats(days_back=30) == []
        # 3650-day window includes it.
        stats = storage.get_injection_stats(days_back=3650)
        assert len(stats) == 1
        assert stats[0].entity_id == "99"


# ---------------------------------------------------------------------------
# TestMemoryReviewCandidates
# ---------------------------------------------------------------------------


class TestMemoryReviewCandidates:
    def test_empty_when_no_playbooks(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        assert storage.get_memory_review_candidates(days_back=60) == []

    def test_empty_for_zero_days_back(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        # Defensive guard: zero/negative days_back returns [].
        assert storage.get_memory_review_candidates(days_back=0) == []
        assert storage.get_memory_review_candidates(days_back=-1) == []

    def test_stale_signal_for_unused_playbook(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        # Insert a playbook with no injection events in the window and
        # a creation timestamp older than the window. Use a direct SQL
        # insert to control the timestamp.
        old_iso = "2020-01-01T00:00:00.000Z"
        with storage._lock:
            storage.conn.execute(
                """INSERT INTO user_playbooks
                     (user_id, request_id, agent_version, content,
                      playbook_name, created_at, status)
                   VALUES ('u1', 'r1', 'v1', 'old rule', 'stale-rule', ?, NULL)""",
                (old_iso,),
            )
            storage.conn.commit()
        candidates = storage.get_memory_review_candidates(days_back=60)
        # The 2020 row is outside the 60-day window; the staleness
        # signal is computed from current_time - created_at >= days_back.
        stale = [c for c in candidates if "stale" in c.signals]
        assert len(stale) == 1
        assert stale[0].entity_type == "playbook"
        assert stale[0].title == "stale-rule"
        assert stale[0].injection_count == 0
        assert stale[0].citation_count == 0

    def test_no_stale_signal_for_fresh_playbook(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        # A playbook created NOW (within the look-back window) is NOT
        # stale, even with zero injection events.
        with storage._lock:
            storage.conn.execute(
                """INSERT INTO user_playbooks
                     (user_id, request_id, agent_version, content,
                      playbook_name, created_at, status)
                   VALUES ('u1', 'r1', 'v1', 'fresh rule', 'fresh-rule',
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), NULL)"""
            )
            storage.conn.commit()
        candidates = storage.get_memory_review_candidates(days_back=60)
        assert candidates == []

    def test_high_cost_low_cite_signal(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        # Set up: playbook 42 was injected 5 times (low cite rate) and
        # cited 1 time. Should be flagged.
        with storage._lock:
            # First create a playbook
            cur = storage.conn.execute(
                """INSERT INTO user_playbooks
                     (user_id, request_id, agent_version, content,
                      playbook_name, created_at, status)
                   VALUES ('u1', 'r1', 'v1', 'noisy rule', 'noisy-rule',
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), NULL)"""
            )
            pb_id = cur.lastrowid
            # Then inject it 5 times in the recent window
            for _ in range(5):
                storage.conn.execute(
                    """INSERT INTO usage_events
                         (org_id, event_name, event_category, entity_type,
                          entity_id, count_value, prompt_tokens, created_at)
                       VALUES (?, 'learning_injection', 'application',
                               'playbook', ?, 1, 10,
                               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                    (storage.org_id, str(pb_id)),
                )
            # And cite it once on an interaction
            import json as _json
            storage.conn.execute(
                """INSERT INTO interactions
                     (user_id, request_id, created_at, role, citations)
                   VALUES ('u1', 'r-cite-1', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                           'Assistant', ?)""",
                (_json.dumps([
                    {"kind": "playbook", "real_id": str(pb_id), "tag": "s1-42", "title": "noisy"}
                ]),),
            )
            storage.conn.commit()
        candidates = storage.get_memory_review_candidates(days_back=60)
        high_cost = [c for c in candidates if "high_cost_low_cite" in c.signals]
        assert len(high_cost) == 1
        assert high_cost[0].entity_id == str(pb_id)
        assert high_cost[0].injection_count == 5
        assert high_cost[0].citation_count == 1

    def test_no_high_cost_signal_when_well_cited(self, storage):
        if not _backend_supports_memory_review(storage):
            pytest.skip("Backend does not implement get_memory_review_candidates")
        # Same injection volume but cited as often as injected — should
        # NOT be flagged as high_cost_low_cite.
        import json as _json
        with storage._lock:
            cur = storage.conn.execute(
                """INSERT INTO user_playbooks
                     (user_id, request_id, agent_version, content,
                      playbook_name, created_at, status)
                   VALUES ('u1', 'r1', 'v1', 'useful rule', 'useful-rule',
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), NULL)"""
            )
            pb_id = cur.lastrowid
            for _ in range(5):
                storage.conn.execute(
                    """INSERT INTO usage_events
                         (org_id, event_name, event_category, entity_type,
                          entity_id, count_value, created_at)
                       VALUES (?, 'learning_injection', 'application',
                               'playbook', ?, 1,
                               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                    (storage.org_id, str(pb_id)),
                )
            # 5 citations — equal to injection count
            for i in range(5):
                storage.conn.execute(
                    """INSERT INTO interactions
                         (user_id, request_id, created_at, role, citations)
                       VALUES ('u1', ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                               'Assistant', ?)""",
                    (
                        f"r-cite-{i}",
                        _json.dumps([
                            {"kind": "playbook", "real_id": str(pb_id), "tag": "s1-42", "title": "useful"}
                        ]),
                    ),
                )
            storage.conn.commit()
        candidates = storage.get_memory_review_candidates(days_back=60)
        high_cost = [c for c in candidates if "high_cost_low_cite" in c.signals]
        assert high_cost == []
