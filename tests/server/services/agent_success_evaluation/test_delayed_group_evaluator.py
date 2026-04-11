"""Tests for the GroupEvaluationScheduler singleton and scheduling behavior."""

import time
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    GroupEvaluationScheduler,
    GroupKey,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test to avoid cross-test pollution."""
    GroupEvaluationScheduler._instance = None
    yield
    GroupEvaluationScheduler._instance = None


# ===============================
# Tests for get_instance() singleton
# ===============================


class TestGetInstance:
    """Tests for the singleton get_instance() class method."""

    def test_returns_same_instance(self):
        """get_instance() returns the same object on repeated calls."""
        instance1 = GroupEvaluationScheduler.get_instance()
        instance2 = GroupEvaluationScheduler.get_instance()
        assert instance1 is instance2

    def test_creates_instance_when_none(self):
        """get_instance() creates a new instance when none exists."""
        assert GroupEvaluationScheduler._instance is None
        instance = GroupEvaluationScheduler.get_instance()
        assert instance is not None
        assert isinstance(instance, GroupEvaluationScheduler)


# ===============================
# Tests for scheduler daemon thread
# ===============================


class TestSchedulerThread:
    """Tests for the scheduler's internal daemon thread."""

    def test_scheduler_thread_is_daemon(self):
        """The scheduler thread must be a daemon so it does not block process exit."""
        scheduler = GroupEvaluationScheduler.get_instance()
        assert scheduler._thread.daemon is True

    def test_scheduler_thread_is_alive(self):
        """The scheduler thread starts running on construction."""
        scheduler = GroupEvaluationScheduler.get_instance()
        assert scheduler._thread.is_alive()

    def test_scheduler_thread_name(self):
        """The scheduler thread has a descriptive name."""
        scheduler = GroupEvaluationScheduler.get_instance()
        assert scheduler._thread.name == "group-eval-scheduler"


# ===============================
# Tests for schedule()
# ===============================


class TestSchedule:
    """Tests for the schedule() method."""

    def test_callback_stored_with_correct_fire_time(self):
        """schedule() stores the callback with a fire time in the future."""
        scheduler = GroupEvaluationScheduler.get_instance()
        callback = MagicMock()
        key: GroupKey = ("org_1", "user_1", "session_1")

        before = time.monotonic()
        scheduler.schedule(key, callback)
        after = time.monotonic()

        # Verify the key is in the scheduled map
        assert key in scheduler._scheduled
        fire_time, stored_callback = scheduler._scheduled[key]
        assert stored_callback is callback
        # Fire time should be roughly now + delay
        assert fire_time >= before
        assert fire_time <= after + 700  # generous upper bound

    def test_reschedule_updates_fire_time(self):
        """Rescheduling the same key updates the fire time forward."""
        scheduler = GroupEvaluationScheduler.get_instance()
        callback1 = MagicMock()
        callback2 = MagicMock()
        key: GroupKey = ("org_1", "user_1", "session_1")

        scheduler.schedule(key, callback1)
        first_fire_time, _ = scheduler._scheduled[key]

        # Small delay to ensure monotonic time advances
        time.sleep(0.01)

        scheduler.schedule(key, callback2)
        second_fire_time, stored_callback = scheduler._scheduled[key]

        # Fire time should have been pushed forward
        assert second_fire_time > first_fire_time
        # Callback should be updated to the latest one
        assert stored_callback is callback2

    def test_schedule_multiple_keys(self):
        """Multiple different keys can be scheduled independently."""
        scheduler = GroupEvaluationScheduler.get_instance()
        key1: GroupKey = ("org_1", "user_1", "session_1")
        key2: GroupKey = ("org_1", "user_1", "session_2")
        cb1 = MagicMock()
        cb2 = MagicMock()

        scheduler.schedule(key1, cb1)
        scheduler.schedule(key2, cb2)

        assert key1 in scheduler._scheduled
        assert key2 in scheduler._scheduled
        assert scheduler._scheduled[key1][1] is cb1
        assert scheduler._scheduled[key2][1] is cb2

    def test_schedule_pushes_to_heap(self):
        """schedule() adds an entry to the min-heap."""
        scheduler = GroupEvaluationScheduler.get_instance()
        key: GroupKey = ("org_1", "user_1", "session_1")
        callback = MagicMock()

        initial_heap_len = len(scheduler._heap)
        scheduler.schedule(key, callback)

        assert len(scheduler._heap) > initial_heap_len


# ===============================
# Tests for _run_callback()
# ===============================


class TestRunCallback:
    """Tests for the static _run_callback() method."""

    def test_success_path(self):
        """_run_callback invokes the callback on success."""
        callback = MagicMock()
        key: GroupKey = ("org_1", "user_1", "session_1")

        GroupEvaluationScheduler._run_callback(key, callback)

        callback.assert_called_once()

    def test_exception_path_does_not_raise(self):
        """_run_callback catches exceptions without propagating."""
        callback = MagicMock(side_effect=RuntimeError("evaluation failed"))
        key: GroupKey = ("org_1", "user_1", "session_1")

        # Should not raise
        GroupEvaluationScheduler._run_callback(key, callback)

        callback.assert_called_once()

    def test_exception_is_logged(self):
        """_run_callback logs the exception when callback fails."""
        callback = MagicMock(side_effect=ValueError("bad value"))
        key: GroupKey = ("org_1", "user_1", "session_1")

        with patch(
            "reflexio.server.services.agent_success_evaluation.delayed_group_evaluator.logger"
        ) as mock_logger:
            GroupEvaluationScheduler._run_callback(key, callback)

            mock_logger.exception.assert_called_once()
            assert "failed" in mock_logger.exception.call_args[0][0].lower()
