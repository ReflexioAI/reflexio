"""Unit tests for the per-session seen-cache used by unified search dedup."""

import unittest

from reflexio.server.services.retrieval.session_dedup import SessionSeenCache


class TestSessionSeenCache(unittest.TestCase):
    def test_seen_record_round_trip(self):
        cache = SessionSeenCache()
        self.assertEqual(cache.seen("org", "s1"), frozenset())

        cache.record("org", "s1", [("profile", "p1"), ("user_playbook", "7")])
        self.assertEqual(
            cache.seen("org", "s1"),
            frozenset({("profile", "p1"), ("user_playbook", "7")}),
        )

    def test_sessions_are_isolated(self):
        cache = SessionSeenCache()
        cache.record("org", "s1", [("profile", "p1")])

        self.assertEqual(cache.seen("org", "s2"), frozenset())
        self.assertEqual(cache.seen("org", "s1"), frozenset({("profile", "p1")}))

    def test_orgs_are_isolated(self):
        cache = SessionSeenCache()
        cache.record("org-a", "s1", [("profile", "p1")])

        self.assertEqual(cache.seen("org-b", "s1"), frozenset())

    def test_per_session_entry_cap_evicts_oldest_first(self):
        cache = SessionSeenCache(max_entries_per_session=3)
        cache.record("org", "s1", [("profile", f"p{i}") for i in range(5)])

        self.assertEqual(
            cache.seen("org", "s1"),
            frozenset({("profile", "p2"), ("profile", "p3"), ("profile", "p4")}),
        )

    def test_session_cap_evicts_least_recently_used(self):
        cache = SessionSeenCache(max_sessions=2)
        cache.record("org", "s1", [("profile", "p1")])
        cache.record("org", "s2", [("profile", "p2")])
        # Touch s1 so s2 becomes the least recently used.
        cache.seen("org", "s1")
        cache.record("org", "s3", [("profile", "p3")])

        self.assertEqual(cache.seen("org", "s2"), frozenset())
        self.assertEqual(cache.seen("org", "s1"), frozenset({("profile", "p1")}))
        self.assertEqual(cache.seen("org", "s3"), frozenset({("profile", "p3")}))

    def test_idle_sessions_evicted(self):
        cache = SessionSeenCache(idle_eviction_seconds=0.0)
        cache.record("org", "s1", [("profile", "p1")])

        # Any later access observes the zero-idle session as expired.
        self.assertEqual(cache.seen("org", "s1"), frozenset())

    def test_clear_drops_everything(self):
        cache = SessionSeenCache()
        cache.record("org", "s1", [("profile", "p1")])
        cache.clear()

        self.assertEqual(cache.seen("org", "s1"), frozenset())

    def test_empty_record_does_not_allocate_a_session_slot(self):
        cache = SessionSeenCache(max_sessions=2)
        cache.record("org", "s1", [("profile", "p1")])
        cache.record("org", "s2", [("profile", "p2")])
        # Empty recordings for new sessions must not evict live sessions.
        cache.record("org", "empty-1", [])
        cache.record("org", "empty-2", [])

        self.assertEqual(cache.seen("org", "s1"), frozenset({("profile", "p1")}))
        self.assertEqual(cache.seen("org", "s2"), frozenset({("profile", "p2")}))

    def test_empty_record_still_refreshes_an_existing_session(self):
        cache = SessionSeenCache(max_sessions=2)
        cache.record("org", "s1", [("profile", "p1")])
        cache.record("org", "s2", [("profile", "p2")])
        # Empty record on s1 marks it most recently used...
        cache.record("org", "s1", [])
        # ...so adding a third session evicts s2, not s1.
        cache.record("org", "s3", [("profile", "p3")])

        self.assertEqual(cache.seen("org", "s1"), frozenset({("profile", "p1")}))
        self.assertEqual(cache.seen("org", "s2"), frozenset())

    def test_re_recording_same_key_is_idempotent(self):
        cache = SessionSeenCache(max_entries_per_session=2)
        cache.record("org", "s1", [("profile", "p1")])
        cache.record("org", "s1", [("profile", "p1"), ("profile", "p2")])

        self.assertEqual(
            cache.seen("org", "s1"),
            frozenset({("profile", "p1"), ("profile", "p2")}),
        )


if __name__ == "__main__":
    unittest.main()
