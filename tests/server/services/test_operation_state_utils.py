"""
Unit tests for OperationStateManager.

Tests all 5 use cases: progress tracking, concurrency lock,
extractor bookmark, aggregator bookmark, and simple lock.
"""

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import Interaction, OperationStatus
from reflexio.server.services.operation_state_utils import (
    GENERATION_STALE_LOCK_SECONDS,
    OperationStateManager,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_storage():
    """Create a mock storage with all operation state methods."""
    storage = MagicMock()
    storage.get_operation_state.return_value = None
    storage.upsert_operation_state.return_value = None
    storage.update_operation_state.return_value = None
    storage.try_acquire_in_progress_lock.return_value = {"acquired": True}
    storage.get_operation_state_with_new_request_interaction.return_value = ({}, [])
    return storage


@pytest.fixture
def manager(mock_storage):
    """Create an OperationStateManager instance."""
    return OperationStateManager(
        storage=mock_storage, org_id="org_123", service_name="test_service"
    )


# ===============================
# Key Builder Tests
# ===============================


class TestKeyBuilders:
    """Tests for private key builder methods."""

    def test_progress_key(self, manager):
        assert manager._progress_key() == "test_service::org_123::progress"

    def test_lock_key_without_scope(self, manager):
        assert manager._lock_key() == "test_service::org_123::lock"

    def test_lock_key_with_scope(self, manager):
        assert manager._lock_key("user_1") == "test_service::org_123::user_1::lock"

    def test_bookmark_key_name_only(self, manager):
        assert (
            manager._bookmark_key("extractor_a") == "test_service::org_123::extractor_a"
        )

    def test_bookmark_key_with_scope(self, manager):
        assert (
            manager._bookmark_key("extractor_a", scope_id="user_1")
            == "test_service::org_123::user_1::extractor_a"
        )

    def test_bookmark_key_with_version(self, manager):
        assert (
            manager._bookmark_key("aggregator_b", version="v2")
            == "test_service::org_123::aggregator_b::v2"
        )

    def test_cancellation_key(self, manager):
        assert manager._cancellation_key() == "test_service::org_123::cancellation"

    def test_bookmark_key_with_scope_and_version(self, manager):
        assert (
            manager._bookmark_key("name", scope_id="scope", version="v1")
            == "test_service::org_123::scope::name::v1"
        )


# ===============================
# Use Case 1: Progress Tracking
# ===============================


class TestCheckInProgress:
    """Tests for check_in_progress method."""

    def test_no_existing_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        assert manager.check_in_progress() is None

    def test_existing_completed_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"status": OperationStatus.COMPLETED.value}
        }
        assert manager.check_in_progress() is None

    def test_existing_in_progress_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
                "started_at": int(datetime.now(UTC).timestamp()),
            }
        }
        result = manager.check_in_progress()
        assert result is not None
        assert "already in progress" in result

    def test_existing_failed_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"status": OperationStatus.FAILED.value}
        }
        assert manager.check_in_progress() is None

    def test_flat_state_structure(self, manager, mock_storage):
        """State without nested operation_state wrapper."""
        mock_storage.get_operation_state.return_value = {
            "status": OperationStatus.IN_PROGRESS.value,
            "started_at": int(datetime.now(UTC).timestamp()),
        }
        result = manager.check_in_progress()
        assert result is not None

    def test_stale_in_progress_auto_recovers(self, manager, mock_storage):
        """Stale IN_PROGRESS operation is auto-marked as FAILED."""
        stale_started_at = int(datetime.now(UTC).timestamp()) - 700
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
                "started_at": stale_started_at,
            }
        }

        result = manager.check_in_progress()
        assert result is None  # Allows new operations to proceed

        # Should have updated the state to FAILED
        mock_storage.update_operation_state.assert_called_once()
        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["status"] == OperationStatus.FAILED.value
        assert state["completed_at"] is not None
        assert "Auto-recovered" in state["error_message"]

    def test_in_progress_without_started_at_legacy(self, manager, mock_storage):
        """Legacy state without started_at is treated as legitimately in-progress."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
            }
        }

        result = manager.check_in_progress()
        assert result is not None
        assert "already in progress" in result

    def test_uses_progress_key(self, manager, mock_storage):
        manager.check_in_progress()
        mock_storage.get_operation_state.assert_called_once_with(
            "test_service::org_123::progress"
        )


class TestInitializeProgress:
    """Tests for initialize_progress method."""

    def test_basic_initialization(self, manager, mock_storage):
        manager.initialize_progress(total_users=5, request_params={"mode": "full"})

        # Called twice: once for progress state, once to clear cancellation flag
        assert mock_storage.upsert_operation_state.call_count == 2

        # First call: progress initialization
        progress_key, state = mock_storage.upsert_operation_state.call_args_list[0][0]
        assert progress_key == "test_service::org_123::progress"
        assert state["service_name"] == "test_service"
        assert state["status"] == OperationStatus.IN_PROGRESS.value
        assert state["total_users"] == 5
        assert state["processed_users"] == 0
        assert state["failed_users"] == 0
        assert state["request_params"] == {"mode": "full"}
        assert state["progress_percentage"] == 0.0
        assert state["stats"]["total_interactions_processed"] == 0
        assert state["stats"]["total_generated"] == 0

        # Second call: clear cancellation flag
        cancel_key, cancel_state = mock_storage.upsert_operation_state.call_args_list[
            1
        ][0]
        assert cancel_key == "test_service::org_123::cancellation"
        assert cancel_state["cancellation_requested"] is False

    def test_with_extra_stats(self, manager, mock_storage):
        manager.initialize_progress(
            total_users=3,
            request_params={},
            extra_stats={"custom_metric": 0},
        )

        # First call is the progress state
        _, state = mock_storage.upsert_operation_state.call_args_list[0][0]
        assert state["stats"]["custom_metric"] == 0
        assert state["stats"]["total_interactions_processed"] == 0


class TestSetCurrentItem:
    """Tests for set_current_item method."""

    def test_set_current_item(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"current_user_id": None, "status": "in_progress"}
        }
        manager.set_current_item("item_42")

        mock_storage.update_operation_state.assert_called_once()
        key, state = mock_storage.update_operation_state.call_args[0]
        assert key == "test_service::org_123::progress"
        assert state["current_user_id"] == "item_42"

    def test_set_current_item_no_existing_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        manager.set_current_item("item_1")

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["current_user_id"] == "item_1"


class TestUpdateProgress:
    """Tests for update_progress method."""

    def _make_in_progress_state(self):
        return {
            "operation_state": {
                "processed_users": 1,
                "processed_user_ids": ["prev"],
                "failed_users": 0,
                "failed_user_ids": [],
                "current_user_id": "current",
                "progress_percentage": 0.0,
                "stats": {"total_interactions_processed": 5},
            }
        }

    def test_successful_update(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = self._make_in_progress_state()

        manager.update_progress(item_id="user_2", count=3, success=True, total_users=4)

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["processed_users"] == 2
        assert "user_2" in state["processed_user_ids"]
        assert state["stats"]["total_interactions_processed"] == 8
        assert state["current_user_id"] is None
        assert state["progress_percentage"] == 50.0

    def test_failed_update(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = self._make_in_progress_state()

        manager.update_progress(
            item_id="user_3",
            count=0,
            success=False,
            total_users=4,
            error="timeout",
        )

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["failed_users"] == 1
        assert state["failed_user_ids"][0] == {"user_id": "user_3", "error": "timeout"}
        assert state["current_user_id"] is None

    def test_progress_percentage_calculation(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "processed_users": 2,
                "processed_user_ids": ["a", "b"],
                "failed_users": 0,
                "failed_user_ids": [],
                "current_user_id": "c",
                "progress_percentage": 0.0,
                "stats": {"total_interactions_processed": 10},
            }
        }

        manager.update_progress(item_id="c", count=5, success=True, total_users=4)

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["progress_percentage"] == 75.0


class TestFinalizeProgress:
    """Tests for finalize_progress method."""

    def test_finalize(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
                "stats": {"total_interactions_processed": 0, "total_generated": 0},
            }
        }

        manager.finalize_progress(total_processed=20, total_generated=5)

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["status"] == OperationStatus.COMPLETED.value
        assert state["completed_at"] is not None
        assert state["progress_percentage"] == 100.0
        assert state["stats"]["total_interactions_processed"] == 20
        assert state["stats"]["total_generated"] == 5

    def test_finalize_defaults_generated_to_zero(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
                "stats": {"total_interactions_processed": 0, "total_generated": 0},
            }
        }

        manager.finalize_progress(total_processed=10)

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["stats"]["total_generated"] == 0


class TestMarkProgressFailed:
    """Tests for mark_progress_failed method."""

    def test_mark_failed(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
            }
        }

        manager.mark_progress_failed("Something broke")

        _, state = mock_storage.update_operation_state.call_args[0]
        assert state["status"] == OperationStatus.FAILED.value
        assert state["error_message"] == "Something broke"
        assert state["completed_at"] is not None

    def test_mark_failed_no_existing_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        # Should not raise
        manager.mark_progress_failed("error")
        mock_storage.update_operation_state.assert_not_called()

    def test_mark_failed_swallows_exceptions(self, manager, mock_storage):
        mock_storage.get_operation_state.side_effect = RuntimeError("db down")
        # Should not raise
        manager.mark_progress_failed("error")


class TestGetProgress:
    """Tests for get_progress method."""

    def test_returns_state(self, manager, mock_storage):
        expected = {"status": "completed", "progress_percentage": 100.0}
        mock_storage.get_operation_state.return_value = {"operation_state": expected}
        assert manager.get_progress() == expected

    def test_returns_none_when_no_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        assert manager.get_progress() is None

    def test_flat_state_fallback(self, manager, mock_storage):
        state = {"status": "in_progress"}
        mock_storage.get_operation_state.return_value = state
        assert manager.get_progress() == state


# ===============================
# Use Case 2: Concurrency Lock
# ===============================


class TestAcquireLock:
    """Tests for acquire_lock method."""

    def test_acquire_lock_success(self, manager, mock_storage):
        mock_storage.try_acquire_in_progress_lock.return_value = {"acquired": True}

        result = manager.acquire_lock("req_1")
        assert result is True

        mock_storage.try_acquire_in_progress_lock.assert_called_once_with(
            "test_service::org_123::lock",
            "req_1",
            GENERATION_STALE_LOCK_SECONDS,
        )

    def test_acquire_lock_already_held(self, manager, mock_storage):
        mock_storage.try_acquire_in_progress_lock.return_value = {"acquired": False}

        result = manager.acquire_lock("req_2")
        assert result is False

    def test_acquire_lock_with_scope(self, manager, mock_storage):
        mock_storage.try_acquire_in_progress_lock.return_value = {"acquired": True}

        manager.acquire_lock("req_1", scope_id="user_5")

        mock_storage.try_acquire_in_progress_lock.assert_called_once_with(
            "test_service::org_123::user_5::lock",
            "req_1",
            GENERATION_STALE_LOCK_SECONDS,
        )

    def test_acquire_lock_custom_stale_seconds(self, manager, mock_storage):
        mock_storage.try_acquire_in_progress_lock.return_value = {"acquired": True}

        manager.acquire_lock("req_1", stale_seconds=60)

        mock_storage.try_acquire_in_progress_lock.assert_called_once_with(
            "test_service::org_123::lock", "req_1", 60
        )


class TestReleaseLock:
    """Tests for release_lock method."""

    def test_release_no_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        result = manager.release_lock("req_1")
        assert result is None

    def test_release_no_pending_request(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "current_request_id": "req_1",
                "pending_request_id": None,
            }
        }

        result = manager.release_lock("req_1")
        assert result is None

        # Lock should be cleared
        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert state["in_progress"] is False
        assert state["current_request_id"] is None
        assert state["pending_request_id"] is None

    def test_release_with_pending_request(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "current_request_id": "req_1",
                "pending_request_id": "req_2",
            }
        }

        result = manager.release_lock("req_1")
        assert result == "req_2"

        # Lock should be transferred to pending request
        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert state["in_progress"] is True
        assert state["current_request_id"] == "req_2"
        assert state["pending_request_id"] is None

    def test_release_not_owner(self, manager, mock_storage):
        """If we don't own the lock, do nothing."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "current_request_id": "someone_else",
                "pending_request_id": "req_3",
            }
        }

        result = manager.release_lock("req_1")
        assert result is None
        mock_storage.upsert_operation_state.assert_not_called()

    def test_release_pending_same_as_current(self, manager, mock_storage):
        """If pending == current, treat as no pending."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "current_request_id": "req_1",
                "pending_request_id": "req_1",
            }
        }

        result = manager.release_lock("req_1")
        assert result is None

        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert state["in_progress"] is False
        assert state["current_request_id"] is None
        assert state["pending_request_id"] is None

    def test_release_with_scope_id(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "current_request_id": "req_1",
                "pending_request_id": None,
            }
        }

        manager.release_lock("req_1", scope_id="user_5")

        mock_storage.get_operation_state.assert_called_once_with(
            "test_service::org_123::user_5::lock"
        )

    def test_release_flat_state(self, manager, mock_storage):
        """State record without nested operation_state."""
        mock_storage.get_operation_state.return_value = {
            "current_request_id": "req_1",
            "pending_request_id": None,
        }

        result = manager.release_lock("req_1")
        assert result is None
        mock_storage.upsert_operation_state.assert_called_once()

    def test_release_legacy_request_id_state(self, manager, mock_storage):
        """Legacy SQLite lock rows used request_id instead of current_request_id."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "request_id": "req_1",
                "pending_request_id": "req_2",
            }
        }

        result = manager.release_lock("req_1")

        assert result == "req_2"
        _, state = mock_storage.upsert_operation_state.call_args[0]
        assert state["in_progress"] is True
        assert state["current_request_id"] == "req_2"
        assert state["pending_request_id"] is None


class TestClearLock:
    """Tests for clear_lock method."""

    def test_clear_lock(self, manager, mock_storage):
        manager.clear_lock()

        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::lock"
        assert state["in_progress"] is False
        assert state["current_request_id"] is None
        assert state["pending_request_id"] is None

    def test_clear_lock_with_scope(self, manager, mock_storage):
        manager.clear_lock(scope_id="user_5")

        key, _ = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::user_5::lock"


# ===============================
# Use Case 3: Extractor Bookmark
# ===============================


class TestExtractorBookmark:
    """Tests for extractor bookmark methods."""

    def test_get_extractor_state_with_new_interactions(self, manager, mock_storage):
        mock_storage.get_operation_state_with_new_request_interaction.return_value = (
            {"last_processed_timestamp": 100},
            [],
        )

        state, interactions = manager.get_extractor_state_with_new_interactions(
            "my_extractor", user_id="user_1", sources=["web"]
        )

        mock_storage.get_operation_state_with_new_request_interaction.assert_called_once_with(
            "test_service::org_123::user_1::my_extractor", "user_1", ["web"]
        )
        assert state == {"last_processed_timestamp": 100}

    def test_get_extractor_state_no_user(self, manager, mock_storage):
        manager.get_extractor_state_with_new_interactions("extractor_a")

        mock_storage.get_operation_state_with_new_request_interaction.assert_called_once_with(
            "test_service::org_123::extractor_a", None, None
        )

    def test_update_extractor_bookmark(self, manager, mock_storage):
        interactions = [
            Interaction(
                interaction_id=10,
                user_id="u1",
                request_id="r1",
                created_at=1000,
            ),
            Interaction(
                interaction_id=20,
                user_id="u1",
                request_id="r2",
                created_at=2000,
            ),
        ]

        manager.update_extractor_bookmark("my_extractor", interactions, user_id="u1")

        key, payload = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::u1::my_extractor"
        assert payload["last_processed_interaction_ids"] == [10, 20]
        assert payload["last_processed_timestamp"] == 2000

    def test_update_extractor_bookmark_empty_list(self, manager, mock_storage):
        manager.update_extractor_bookmark("ext", [])
        mock_storage.upsert_operation_state.assert_not_called()

    def test_update_extractor_bookmark_no_user(self, manager, mock_storage):
        interactions = [
            Interaction(
                interaction_id=5,
                user_id="u1",
                request_id="r1",
                created_at=500,
            ),
        ]

        manager.update_extractor_bookmark("ext", interactions)

        key, _ = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::ext"

    def test_update_extractor_bookmark_none_timestamps(self, manager, mock_storage):
        """When all interactions have created_at=None, no timestamp is stored."""
        mock_interaction = MagicMock()
        mock_interaction.interaction_id = 1
        mock_interaction.created_at = None

        manager.update_extractor_bookmark("ext", [mock_interaction])

        _, payload = mock_storage.upsert_operation_state.call_args[0]
        assert "last_processed_timestamp" not in payload


# ===============================
# Use Case 4: Aggregator Bookmark
# ===============================


class TestAggregatorBookmark:
    """Tests for aggregator bookmark methods."""

    def test_get_aggregator_bookmark(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"last_processed_user_playbook_id": 42}
        }

        result = manager.get_aggregator_bookmark("playbook_a", "v1")
        assert result == 42

        mock_storage.get_operation_state.assert_called_once_with(
            "test_service::org_123::playbook_a::v1"
        )

    def test_get_aggregator_bookmark_no_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        assert manager.get_aggregator_bookmark("fb", "v1") is None

    def test_update_aggregator_bookmark(self, manager, mock_storage):
        manager.update_aggregator_bookmark("playbook_a", "v2", 99)

        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::playbook_a::v2"
        assert state == {"last_processed_user_playbook_id": 99}

    def test_get_aggregator_bookmark_ignores_top_level_keys(
        self, manager, mock_storage
    ):
        """Regression: bookmark must read from nested operation_state, not top-level dict."""
        mock_storage.get_operation_state.return_value = {
            "last_processed_user_playbook_id": 999,
            "operation_state": {},
        }
        result = manager.get_aggregator_bookmark("fb", "v1")
        assert result is None

    def test_get_cluster_fingerprints_ignores_top_level_keys(
        self, manager, mock_storage
    ):
        """Regression: fingerprints must read from nested operation_state, not top-level dict."""
        mock_storage.get_operation_state.return_value = {
            "cluster_fingerprints": {
                "fp1": {"agent_playbook_id": 1, "user_playbook_ids": [1, 2]}
            },
            "operation_state": {},
        }
        result = manager.get_cluster_fingerprints("fb", "v1")
        assert result == {}


# ===============================
# Use Case 5: Simple Lock
# ===============================


class TestSimpleLock:
    """Tests for simple lock methods."""

    def test_acquire_simple_lock_no_existing(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None

        result = manager.acquire_simple_lock()
        assert result is True

        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::lock"
        assert state["in_progress"] is True
        assert "started_at" in state

    def test_acquire_simple_lock_already_held(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "in_progress": True,
            "started_at": int(time.time()),
        }

        result = manager.acquire_simple_lock()
        assert result is False
        mock_storage.upsert_operation_state.assert_not_called()

    def test_acquire_simple_lock_stale(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "in_progress": True,
            "started_at": int(time.time()) - 600,
        }

        result = manager.acquire_simple_lock(stale_seconds=300)
        assert result is True
        mock_storage.upsert_operation_state.assert_called_once()

    def test_acquire_simple_lock_not_in_progress(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "in_progress": False,
            "completed_at": int(time.time()) - 10,
        }

        result = manager.acquire_simple_lock()
        assert result is True

    def test_release_simple_lock(self, manager, mock_storage):
        manager.release_simple_lock()

        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::lock"
        assert state["in_progress"] is False
        assert "completed_at" in state


# ===============================
# Use Case 6: Cancellation
# ===============================


class TestRequestCancellation:
    """Tests for request_cancellation method."""

    def test_request_cancellation_in_progress(self, manager, mock_storage):
        """Cancellation flag is set in separate row when operation is IN_PROGRESS."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"status": OperationStatus.IN_PROGRESS.value}
        }

        result = manager.request_cancellation()
        assert result is True

        # Should upsert to separate cancellation key, not update progress key
        mock_storage.upsert_operation_state.assert_called_once()
        key, state = mock_storage.upsert_operation_state.call_args[0]
        assert key == "test_service::org_123::cancellation"
        assert state["cancellation_requested"] is True

    def test_request_cancellation_not_in_progress(self, manager, mock_storage):
        """Cancellation returns False when operation is COMPLETED."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"status": OperationStatus.COMPLETED.value}
        }

        result = manager.request_cancellation()
        assert result is False
        mock_storage.upsert_operation_state.assert_not_called()

    def test_request_cancellation_no_state(self, manager, mock_storage):
        """Cancellation returns False when no state exists."""
        mock_storage.get_operation_state.return_value = None

        result = manager.request_cancellation()
        assert result is False

    def test_request_cancellation_failed_state(self, manager, mock_storage):
        """Cancellation returns False when operation is FAILED."""
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"status": OperationStatus.FAILED.value}
        }

        result = manager.request_cancellation()
        assert result is False


class TestIsCancellationRequested:
    """Tests for is_cancellation_requested method - reads from separate cancellation key."""

    def test_cancellation_requested_true(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"cancellation_requested": True}
        }
        assert manager.is_cancellation_requested() is True
        mock_storage.get_operation_state.assert_called_with(
            "test_service::org_123::cancellation"
        )

    def test_cancellation_requested_false(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {"cancellation_requested": False}
        }
        assert manager.is_cancellation_requested() is False

    def test_cancellation_not_set(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {"operation_state": {}}
        assert manager.is_cancellation_requested() is False

    def test_cancellation_no_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        assert manager.is_cancellation_requested() is False


class TestMarkCancelled:
    """Tests for mark_cancelled method."""

    def test_mark_cancelled(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = {
            "operation_state": {
                "status": OperationStatus.IN_PROGRESS.value,
            }
        }

        manager.mark_cancelled()

        # Should update progress key with CANCELLED status
        mock_storage.update_operation_state.assert_called_once()
        key, state = mock_storage.update_operation_state.call_args[0]
        assert key == "test_service::org_123::progress"
        assert state["status"] == OperationStatus.CANCELLED.value
        assert "completed_at" in state

        # Should clear separate cancellation flag
        mock_storage.upsert_operation_state.assert_called_once()
        cancel_key, cancel_state = mock_storage.upsert_operation_state.call_args[0]
        assert cancel_key == "test_service::org_123::cancellation"
        assert cancel_state["cancellation_requested"] is False

    def test_mark_cancelled_no_state(self, manager, mock_storage):
        mock_storage.get_operation_state.return_value = None
        manager.mark_cancelled()
        mock_storage.update_operation_state.assert_not_called()
