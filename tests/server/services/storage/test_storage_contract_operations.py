"""Contract tests for OperationMixin — run against every local storage backend."""

import time

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# TestOperationStateCRUD
# ---------------------------------------------------------------------------


class TestOperationStateCRUD:
    def test_create_and_get_operation_state(self, storage):
        state = {"status": "running", "progress": 50}
        storage.create_operation_state("svc_a", state)

        result = storage.get_operation_state("svc_a")
        assert result is not None
        assert result["operation_state"]["status"] == "running"
        assert result["operation_state"]["progress"] == 50

    def test_get_nonexistent_returns_none(self, storage):
        assert storage.get_operation_state("missing") is None

    def test_upsert_creates_new(self, storage):
        state = {"status": "idle"}
        storage.upsert_operation_state("svc_new", state)

        result = storage.get_operation_state("svc_new")
        assert result is not None
        assert result["operation_state"]["status"] == "idle"

    def test_upsert_updates_existing(self, storage):
        storage.create_operation_state("svc_up", {"status": "running", "progress": 10})
        storage.upsert_operation_state("svc_up", {"status": "done", "progress": 100})

        result = storage.get_operation_state("svc_up")
        assert result is not None
        assert result["operation_state"]["status"] == "done"
        assert result["operation_state"]["progress"] == 100

    def test_update_operation_state(self, storage):
        storage.create_operation_state("svc_upd", {"status": "running"})
        storage.update_operation_state(
            "svc_upd", {"status": "completed", "result": "ok"}
        )

        result = storage.get_operation_state("svc_upd")
        assert result is not None
        assert result["operation_state"]["status"] == "completed"
        assert result["operation_state"]["result"] == "ok"

    def test_delete_operation_state(self, storage):
        storage.create_operation_state("svc_del", {"status": "running"})
        storage.delete_operation_state("svc_del")

        assert storage.get_operation_state("svc_del") is None

    def test_delete_all_operation_states(self, storage):
        storage.create_operation_state("svc_1", {"status": "a"})
        storage.create_operation_state("svc_2", {"status": "b"})

        storage.delete_all_operation_states()

        assert storage.get_operation_state("svc_1") is None
        assert storage.get_operation_state("svc_2") is None

    def test_get_all_operation_states(self, storage):
        storage.create_operation_state("svc_x", {"status": "x"})
        storage.create_operation_state("svc_y", {"status": "y"})

        all_states = storage.get_all_operation_states()
        assert len(all_states) == 2


# ---------------------------------------------------------------------------
# TestPendingRequestQueue — R2
# ---------------------------------------------------------------------------


class TestPendingRequestQueue:
    """Contract: when the lock is held, blocked requests queue FIFO with payloads."""

    def _state(self, storage, key):
        record = storage.get_operation_state(key)
        if record is None:
            return None
        return record.get("operation_state", record)

    def test_first_acquire_creates_empty_queue(self, storage):
        result = storage.try_acquire_in_progress_lock("svc_lock_1", "req_1")
        assert result["acquired"] is True

        state = self._state(storage, "svc_lock_1")
        assert state is not None
        assert state.get("pending_request_queue", []) == []

    def test_blocked_request_appends_to_queue(self, storage):
        storage.try_acquire_in_progress_lock("svc_lock_2", "req_1")
        result = storage.try_acquire_in_progress_lock(
            "svc_lock_2", "req_2", payload={"user_id": "user_b"}
        )
        assert result["acquired"] is False

        state = self._state(storage, "svc_lock_2")
        assert state is not None
        queue = state.get("pending_request_queue", [])
        assert len(queue) == 1
        assert queue[0]["request_id"] == "req_2"
        assert queue[0]["payload"] == {"user_id": "user_b"}

    def test_multiple_blocked_requests_queue_fifo(self, storage):
        storage.try_acquire_in_progress_lock("svc_lock_3", "req_1")
        storage.try_acquire_in_progress_lock(
            "svc_lock_3", "req_2", payload={"user_id": "user_b"}
        )
        storage.try_acquire_in_progress_lock(
            "svc_lock_3", "req_3", payload={"user_id": "user_c"}
        )

        state = self._state(storage, "svc_lock_3")
        assert state is not None
        queue = state.get("pending_request_queue", [])
        assert [q["request_id"] for q in queue] == ["req_2", "req_3"]
        assert queue[0]["payload"] == {"user_id": "user_b"}
        assert queue[1]["payload"] == {"user_id": "user_c"}

    def test_queue_ignores_duplicate_request_id(self, storage):
        """If the same request_id retries while blocked (e.g. publish retry)
        we should not enqueue it twice. Pre-fix, the single slot just got
        overwritten; queue must dedupe by request_id."""
        storage.try_acquire_in_progress_lock("svc_lock_4", "req_1")
        storage.try_acquire_in_progress_lock(
            "svc_lock_4", "req_2", payload={"user_id": "user_b"}
        )
        storage.try_acquire_in_progress_lock(
            "svc_lock_4", "req_2", payload={"user_id": "user_b_v2"}
        )

        state = self._state(storage, "svc_lock_4")
        assert state is not None
        queue = state.get("pending_request_queue", [])
        # Only one entry for req_2 — second attempt is a noop.
        assert [q["request_id"] for q in queue] == ["req_2"]

    def test_holder_retry_does_not_self_enqueue(self, storage):
        """Holder calling try_acquire again for its own request_id is a noop."""
        storage.try_acquire_in_progress_lock("svc_lock_5", "req_1")
        result = storage.try_acquire_in_progress_lock("svc_lock_5", "req_1")
        # Same request — treated as already-acquired.
        assert result["acquired"] is True

        state = self._state(storage, "svc_lock_5")
        assert state is not None
        assert state.get("pending_request_queue", []) == []

    def test_lock_stays_held_across_a_handover(self, storage):
        """A handover must not look like a release to the next acquirer.

        ``release_lock_pop_queue`` hands the lock to the next queued holder by
        upserting a state blob with ``in_progress: True``. ``upsert_operation_state``
        REPLACES the blob rather than merging it, so any acquirer that keys off a
        different flag than the handover writes will read a live lock as free —
        and ``try_acquire`` also RESETS ``pending_request_queue`` when it acquires,
        so the queued work is silently dropped along with the lock.

        Three requests are the minimum to reach it: two to build a queue, and a
        third to arrive after the handover.
        """
        assert storage.try_acquire_in_progress_lock("svc_handover", "req_1")["acquired"]
        assert not storage.try_acquire_in_progress_lock("svc_handover", "req_2")[
            "acquired"
        ]

        # Exactly what release_lock_pop_queue writes when it promotes req_2 —
        # including a CURRENT started_at. A fixed past timestamp here would make
        # the new holder instantly stale and the assertion below meaningless.
        storage.upsert_operation_state(
            "svc_handover",
            {
                "in_progress": True,
                "started_at": int(time.time()),
                "current_request_id": "req_2",
                "pending_request_id": None,
                "pending_request_queue": [],
            },
        )

        result = storage.try_acquire_in_progress_lock("svc_handover", "req_3")
        assert result["acquired"] is False, (
            "req_3 took a lock still held by req_2 — the acquirer and the "
            "handover disagree about which key means 'held'"
        )
        state = self._state(storage, "svc_handover")
        assert state is not None
        assert state.get("current_request_id") == "req_2"

    def test_a_legacy_status_keyed_lock_is_still_honoured(self, storage):
        """Rows written before the canonical flag must not read as free.

        An older build wrote the held-flag as ``status: "in_progress"``. On first
        run after upgrade those rows are still live locks; without the shim every
        one of them would be read as free and stolen. Nothing writes this shape
        any more, so this test is what keeps the shim honest until it is removed.
        """
        storage.upsert_operation_state(
            "svc_legacy",
            {
                "status": "in_progress",
                "current_request_id": "req_old",
                "pending_request_queue": [],
            },
        )

        result = storage.try_acquire_in_progress_lock("svc_legacy", "req_new")
        assert result["acquired"] is False
        legacy_state = self._state(storage, "svc_legacy")
        assert legacy_state is not None
        assert legacy_state.get("current_request_id") == "req_old"

    def test_a_rejected_acquire_does_not_extend_the_holders_lock(self, storage):
        """Contention must not refresh the staleness clock.

        The queue-append branch rewrites the row on every rejected acquire. When
        staleness was measured from the row's ``updated_at``, each new contender
        pushed the deadline out, so a crashed holder's lock could never expire
        under sustained load. Measuring from the holder's ``started_at`` fixes it.
        """
        storage.try_acquire_in_progress_lock("svc_stale", "req_holder")

        # Backdate the holder well past any plausible staleness window, leaving
        # the row's own updated_at fresh — exactly the post-contention shape.
        state = self._state(storage, "svc_stale")
        assert state is not None
        state["started_at"] = int(time.time()) - 10_000
        storage.upsert_operation_state("svc_stale", state)

        result = storage.try_acquire_in_progress_lock(
            "svc_stale", "req_new", stale_lock_seconds=300
        )
        assert result["acquired"] is True, (
            "a lock whose holder started 10000s ago was not reclaimed — staleness "
            "is being measured from the row clock, which contention refreshes"
        )

    def test_acquire_records_started_at_for_the_shared_reader(self, storage):
        """``acquire_simple_lock`` reads ``started_at``, defaulting to 0.

        A held lock written without it computes ``now - 0`` and is therefore
        always stale, so the shared reader would steal a lock this backend
        considers held. The two must agree on the payload, not just the flag.
        """
        storage.try_acquire_in_progress_lock("svc_started_at", "req_1")
        state = self._state(storage, "svc_started_at")
        assert state is not None
        assert state.get("in_progress") is True
        assert isinstance(state.get("started_at"), int)
        assert state["started_at"] > 0
