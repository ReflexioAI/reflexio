"""Centralized manager for all _operation_state table interactions.

Consolidates 5 use cases:
1. Progress tracking (rerun + manual batch operations)
2. Concurrency lock (atomic lock with request queuing)
3. Extractor bookmark (track last-processed interactions per extractor)
4. Aggregator bookmark (track last-processed user_playbook_id per aggregator)
5. Simple lock (non-queuing lock for cleanup operations)
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, OperationStatus
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

# Stale lock timeout - if generation started > 5 min ago and still "in_progress", assume it crashed
GENERATION_STALE_LOCK_SECONDS = 300

# Stale batch progress timeout - if batch operation started > 10 min ago and still IN_PROGRESS, auto-recover
BATCH_STALE_PROGRESS_SECONDS = 600


class OperationStateManager:
    """Centralized manager for all _operation_state table interactions.

    Provides methods for progress tracking, concurrency locks, extractor bookmarks,
    aggregator bookmarks, and simple locks.

    Args:
        storage: Storage instance with operation state methods
        org_id: Organization identifier
        service_name: Name of the service (e.g., "profile_generation", "playbook_extractor")
    """

    def __init__(self, storage: BaseStorage, org_id: str, service_name: str):
        self.storage = storage
        self.org_id = org_id
        self.service_name = service_name

    # ── Key Builders (private) ──

    def _progress_key(self) -> str:
        """Build progress tracking key.

        Returns:
            str: Key in format '{service_name}::{org_id}::progress'
        """
        return f"{self.service_name}::{self.org_id}::progress"

    def _cancellation_key(self) -> str:
        """Build cancellation flag key.

        Uses a separate row from progress to avoid lost-update race conditions
        where progress updates overwrite the cancellation flag.

        Returns:
            str: Key in format '{service_name}::{org_id}::cancellation'
        """
        return f"{self.service_name}::{self.org_id}::cancellation"

    def _lock_key(self, scope_id: str | None = None) -> str:
        """Build concurrency lock key.

        Args:
            scope_id: Optional scope identifier (e.g., user_id for profile generation)

        Returns:
            str: Key in format '{service_name}::{org_id}[::scope_id]::lock'
        """
        if scope_id:
            return f"{self.service_name}::{self.org_id}::{scope_id}::lock"
        return f"{self.service_name}::{self.org_id}::lock"

    def _bookmark_key(
        self,
        name: str,
        scope_id: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build bookmark key for extractor/aggregator state.

        Args:
            name: Extractor or aggregator name
            scope_id: Optional scope identifier (e.g., user_id)
            version: Optional version identifier (for aggregator)

        Returns:
            str: Key in format '{service_name}::{org_id}[::scope_id]::{name}[::version]'
        """
        parts = [self.service_name, self.org_id]
        if scope_id:
            parts.append(scope_id)
        parts.append(name)
        if version:
            parts.append(version)
        return "::".join(parts)

    # ── Use Case 1: Progress Tracking ──
    # (Batch operations: rerun + manual)

    def check_in_progress(self) -> str | None:
        """Check if there's an existing in-progress operation.

        If the operation has been in progress for longer than BATCH_STALE_PROGRESS_SECONDS,
        auto-marks it as FAILED and returns None to allow new operations to proceed.

        Returns:
            Error message if operation is in progress, None otherwise
        """
        key = self._progress_key()
        existing_state_entry = self.storage.get_operation_state(key)
        if existing_state_entry:
            existing_state = existing_state_entry.get(
                "operation_state", existing_state_entry
            )
            if existing_state.get("status") == OperationStatus.IN_PROGRESS.value:
                # Check if the operation is stale
                started_at = existing_state.get("started_at")
                if started_at is None:
                    # Legacy state without started_at — treat as legitimately in-progress
                    return (
                        f"A {self.service_name} operation is already in progress. "
                        "Please wait for it to complete."
                    )
                current_time = int(datetime.now(UTC).timestamp())
                elapsed = current_time - started_at

                if elapsed > BATCH_STALE_PROGRESS_SECONDS:
                    logger.warning(
                        "Stale %s batch operation detected (started %d seconds ago), "
                        "auto-marking as FAILED to allow recovery",
                        self.service_name,
                        elapsed,
                    )
                    existing_state["status"] = OperationStatus.FAILED.value
                    existing_state["completed_at"] = current_time
                    existing_state["error_message"] = (
                        f"Auto-recovered: operation was stuck IN_PROGRESS for {elapsed}s "
                        f"(threshold: {BATCH_STALE_PROGRESS_SECONDS}s)"
                    )
                    self.storage.update_operation_state(key, existing_state)
                    return None

                return (
                    f"A {self.service_name} operation is already in progress. "
                    "Please wait for it to complete."
                )
        return None

    def initialize_progress(
        self,
        total_users: int,
        request_params: dict,
        extra_stats: dict | None = None,
    ) -> None:
        """Initialize operation state with IN_PROGRESS status.

        Args:
            total_users: Total number of users to process
            request_params: Original request parameters for reference
            extra_stats: Optional additional stats fields to include
        """
        stats = {
            "total_interactions_processed": 0,
            "total_generated": 0,
        }
        if extra_stats:
            stats.update(extra_stats)

        key = self._progress_key()
        initial_state = {
            "service_name": self.service_name,
            "status": OperationStatus.IN_PROGRESS.value,
            "started_at": int(datetime.now(UTC).timestamp()),
            "completed_at": None,
            "total_users": total_users,
            "processed_users": 0,
            "failed_users": 0,
            "current_user_id": None,
            "processed_user_ids": [],
            "failed_user_ids": [],
            "request_params": request_params,
            "stats": stats,
            "error_message": None,
            "progress_percentage": 0.0,
        }
        self.storage.upsert_operation_state(key, initial_state)

        # Clear any stale cancellation flag from a previous operation
        cancel_key = self._cancellation_key()
        self.storage.upsert_operation_state(
            cancel_key, {"cancellation_requested": False}
        )

    def set_current_item(self, item_id: str) -> None:
        """Set the current item being processed.

        Args:
            item_id: Item ID currently being processed
        """
        key = self._progress_key()
        state_entry = self.storage.get_operation_state(key)
        current_state = (
            state_entry.get("operation_state", state_entry) if state_entry else {}
        )
        current_state["current_user_id"] = item_id
        self.storage.update_operation_state(key, current_state)

    def update_progress(
        self,
        item_id: str,
        count: int,
        success: bool,
        total_users: int,
        error: str | None = None,
    ) -> None:
        """Update operation state after processing a user.

        Args:
            item_id: User ID that was processed
            count: Number of interactions processed for this user
            success: Whether processing succeeded
            total_users: Total users being processed (for percentage calculation)
            error: Error message if processing failed
        """
        key = self._progress_key()
        state_entry = self.storage.get_operation_state(key)
        current_state = (
            state_entry.get("operation_state", state_entry) if state_entry else {}
        )

        if success:
            current_state["processed_users"] += 1
            current_state["processed_user_ids"].append(item_id)
            current_state["stats"]["total_interactions_processed"] += count
        else:
            current_state["failed_users"] += 1
            current_state["failed_user_ids"].append(
                {"user_id": item_id, "error": error}
            )

        current_state["current_user_id"] = None
        current_state["progress_percentage"] = (
            current_state["processed_users"] / total_users
        ) * 100

        self.storage.update_operation_state(key, current_state)

    def finalize_progress(self, total_processed: int, total_generated: int = 0) -> None:
        """Mark operation as COMPLETED and finalize state.

        Args:
            total_processed: Total number of items processed
            total_generated: Total number of profiles or playbooks generated
        """
        key = self._progress_key()
        state_entry = self.storage.get_operation_state(key)
        final_state = (
            state_entry.get("operation_state", state_entry) if state_entry else {}
        )
        final_state["status"] = OperationStatus.COMPLETED.value
        final_state["completed_at"] = int(datetime.now(UTC).timestamp())
        final_state["progress_percentage"] = 100.0
        final_state["stats"]["total_interactions_processed"] = total_processed
        final_state["stats"]["total_generated"] = total_generated
        self.storage.update_operation_state(key, final_state)

    def mark_progress_failed(self, error_message: str) -> None:
        """Mark operation as FAILED with error message.

        Args:
            error_message: Error description
        """
        try:
            key = self._progress_key()
            state_entry = self.storage.get_operation_state(key)
            if state_entry:
                failed_state = state_entry.get("operation_state", state_entry)
                failed_state["status"] = OperationStatus.FAILED.value
                failed_state["completed_at"] = int(datetime.now(UTC).timestamp())
                failed_state["error_message"] = error_message
                self.storage.update_operation_state(key, failed_state)
        except Exception:  # noqa: S110
            pass  # Ignore errors updating state during exception handling

    def get_progress(self) -> dict | None:
        """Get the current progress state.

        Returns:
            The progress state dict, or None if no state exists
        """
        key = self._progress_key()
        state_entry = self.storage.get_operation_state(key)
        if state_entry:
            return state_entry.get("operation_state", state_entry)
        return None

    def request_cancellation(self) -> bool:
        """Request cancellation of an in-progress operation.

        Writes the cancellation flag to a separate DB row from the progress state
        to avoid lost-update race conditions where concurrent progress updates
        overwrite the flag.

        Returns:
            bool: True if cancellation was requested (operation was in progress), False otherwise
        """
        # Verify the operation is actually in progress
        progress_key = self._progress_key()
        state_entry = self.storage.get_operation_state(progress_key)
        if not state_entry:
            return False

        state = state_entry.get("operation_state", state_entry)
        if state.get("status") != OperationStatus.IN_PROGRESS.value:
            return False

        # Write cancellation flag to a separate row to avoid race conditions
        cancel_key = self._cancellation_key()
        self.storage.upsert_operation_state(
            cancel_key, {"cancellation_requested": True}
        )
        logger.info(
            "Cancellation requested for %s (org=%s)", self.service_name, self.org_id
        )
        return True

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested for the current operation.

        Reads from a separate cancellation row to avoid race conditions with
        progress updates.

        Returns:
            bool: True if cancellation was requested
        """
        cancel_key = self._cancellation_key()
        cancel_entry = self.storage.get_operation_state(cancel_key)
        if not cancel_entry:
            return False
        cancel_state = cancel_entry.get("operation_state", cancel_entry)
        return cancel_state.get("cancellation_requested", False)

    def mark_cancelled(self) -> None:
        """Mark the current operation as CANCELLED.

        Sets status to CANCELLED in the progress row and clears the separate
        cancellation flag row.
        """
        # Update progress state to CANCELLED
        key = self._progress_key()
        state_entry = self.storage.get_operation_state(key)
        if not state_entry:
            return

        state = state_entry.get("operation_state", state_entry)
        state["status"] = OperationStatus.CANCELLED.value
        state["completed_at"] = int(datetime.now(UTC).timestamp())
        self.storage.update_operation_state(key, state)

        # Clear the separate cancellation flag
        cancel_key = self._cancellation_key()
        self.storage.upsert_operation_state(
            cancel_key, {"cancellation_requested": False}
        )
        logger.info(
            "Operation marked as cancelled for %s (org=%s)",
            self.service_name,
            self.org_id,
        )

    # ── Use Case 2: Concurrency Lock ──
    # (Atomic lock with request queuing for generation services)

    def acquire_lock(
        self,
        request_id: str,
        scope_id: str | None = None,
        stale_seconds: int = GENERATION_STALE_LOCK_SECONDS,
    ) -> bool:
        """Atomically check and acquire in-progress lock.

        Uses a single atomic database operation to prevent race conditions where
        multiple requests could both acquire the lock simultaneously.

        If a valid in-progress operation exists, updates pending_request_id so the
        running operation knows to re-run when it finishes. If no valid lock exists
        or the lock is stale, acquires the lock.

        Args:
            request_id: Current request ID
            scope_id: Optional scope identifier (e.g., user_id)
            stale_seconds: Seconds after which a lock is considered stale

        Returns:
            bool: True if lock acquired (proceed with generation), False if skipped
        """
        state_key = self._lock_key(scope_id)
        result = self.storage.try_acquire_in_progress_lock(
            state_key, request_id, stale_seconds
        )

        acquired = result.get("acquired", False)

        if acquired:
            logger.info(
                "Acquired in-progress lock for %s: state_key=%s, request_id=%s",
                self.service_name,
                state_key,
                request_id,
            )
        else:
            logger.info(
                "Skipping %s - another operation is in progress (state_key=%s). "
                "Updated pending_request_id to %s",
                self.service_name,
                state_key,
                request_id,
            )
        return acquired

    def release_lock(
        self,
        request_id: str,
        scope_id: str | None = None,
    ) -> str | None:
        """Release the in-progress lock and check if a new request came in.

        If a pending request exists (different from current), returns its ID so
        the caller can re-run. Otherwise clears the lock.

        Args:
            request_id: The request ID of the current operation
            scope_id: Optional scope identifier (e.g., user_id)

        Returns:
            Optional[str]: pending_request_id if a new request needs processing, None otherwise
        """
        state_key = self._lock_key(scope_id)
        state_record = self.storage.get_operation_state(state_key)

        if not state_record:
            return None

        # Extract operation_state from the record (storage returns nested structure)
        state = (
            state_record.get("operation_state", {})
            if isinstance(state_record.get("operation_state"), dict)
            else state_record
        )

        pending_request_id = state.get("pending_request_id")
        # Backward compatibility for locks written before current_request_id was introduced.
        current_request_id = state.get("current_request_id") or state.get("request_id")

        # Only process if we still own the lock
        if current_request_id == request_id:
            if pending_request_id and pending_request_id != request_id:
                # Another request came in, transfer ownership and signal re-run
                self.storage.upsert_operation_state(
                    state_key,
                    {
                        "in_progress": True,
                        "started_at": int(time.time()),
                        "current_request_id": pending_request_id,
                        "pending_request_id": None,
                    },
                )
                logger.info(
                    "New request %s came in during %s, will re-run (state_key=%s)",
                    pending_request_id,
                    self.service_name,
                    state_key,
                )
                return pending_request_id
            # No pending request, clear the lock
            self.storage.upsert_operation_state(
                state_key,
                {
                    "in_progress": False,
                    "current_request_id": None,
                    "pending_request_id": None,
                },
            )
            logger.info(
                "Released in-progress lock for %s: state_key=%s, request_id=%s",
                self.service_name,
                state_key,
                request_id,
            )

        return None

    def clear_lock(self, scope_id: str | None = None) -> None:
        """Clear the in-progress state (used for error cleanup).

        Args:
            scope_id: Optional scope identifier (e.g., user_id)
        """
        state_key = self._lock_key(scope_id)
        self.storage.upsert_operation_state(
            state_key,
            {
                "in_progress": False,
                "current_request_id": None,
                "pending_request_id": None,
            },
        )
        logger.debug(
            "Cleared in-progress lock for %s: state_key=%s",
            self.service_name,
            state_key,
        )

    # ── Use Case 3: Extractor Bookmark ──
    # (Track last-processed interactions per extractor)

    def get_extractor_state_with_new_interactions(
        self,
        extractor_name: str,
        user_id: str | None = None,
        sources: list[str] | None = None,
    ) -> tuple[dict, list[RequestInteractionDataModel]]:
        """Get extractor operation state and new interactions since last run.

        Args:
            extractor_name: Name of the extractor
            user_id: Optional user ID for user-level extractors
            sources: Optional source filter list

        Returns:
            Tuple of (state_dict, new_interactions_list)
        """
        state_key = self._bookmark_key(extractor_name, scope_id=user_id)
        return self.storage.get_operation_state_with_new_request_interaction(
            state_key, user_id, sources
        )

    def update_extractor_bookmark(
        self,
        extractor_name: str,
        processed_interactions: list[Interaction],
        user_id: str | None = None,
    ) -> None:
        """Update operation state for an extractor after processing.

        Args:
            extractor_name: Name of the extractor
            processed_interactions: Interactions that were processed
            user_id: Optional user ID for user-level extractors
        """
        if not processed_interactions:
            return

        state_key = self._bookmark_key(extractor_name, scope_id=user_id)

        last_processed_ids = [
            interaction.interaction_id for interaction in processed_interactions
        ]
        last_processed_timestamp = max(
            (
                interaction.created_at
                for interaction in processed_interactions
                if interaction.created_at is not None
            ),
            default=None,
        )

        state_payload: dict[str, Any] = {
            "last_processed_interaction_ids": last_processed_ids,
        }
        if last_processed_timestamp is not None:
            state_payload["last_processed_timestamp"] = last_processed_timestamp

        self.storage.upsert_operation_state(state_key, state_payload)

    # ── Use Case 4: Aggregator Bookmark ──
    # (Track last-processed user_playbook_id per aggregator)

    def get_aggregator_bookmark(self, name: str, version: str) -> int | None:
        """Get the last processed user_playbook_id for an aggregator.

        Args:
            name: Aggregator/playbook name
            version: Agent version

        Returns:
            Last processed user_playbook_id, or None if no state exists
        """
        state_key = self._bookmark_key(name, version=version)
        record = self.storage.get_operation_state(state_key)
        if record:
            state = record.get("operation_state", {})
            if isinstance(state, dict):
                return state.get("last_processed_user_playbook_id")
        return None

    def update_aggregator_bookmark(
        self, name: str, version: str, last_processed_id: int
    ) -> None:
        """Update the aggregator bookmark with the highest user_playbook_id processed.

        Args:
            name: Aggregator/playbook name
            version: Agent version
            last_processed_id: The highest user_playbook_id that was processed
        """
        state_key = self._bookmark_key(name, version=version)
        state = {"last_processed_user_playbook_id": last_processed_id}
        self.storage.upsert_operation_state(state_key, state)
        logger.info(
            "Updated aggregator bookmark for '%s' v%s with last_processed_user_playbook_id: %d",
            name,
            version,
            last_processed_id,
        )

    # ── Use Case 4b: Aggregator Cluster Fingerprints ──
    # (Track cluster fingerprints for change detection)

    def get_cluster_fingerprints(self, name: str, version: str) -> dict:
        """
        Get stored cluster fingerprints for an aggregator.

        Args:
            name: Aggregator/playbook name
            version: Agent version

        Returns:
            dict: Mapping of fingerprint_hash to {"agent_playbook_id": int, "user_playbook_ids": list[int]}.
                  Returns empty dict if no state exists.
        """
        state_key = self._bookmark_key(name, version=version) + "::clusters"
        record = self.storage.get_operation_state(state_key)
        if record:
            state = record.get("operation_state", {})
            if isinstance(state, dict):
                return state.get("cluster_fingerprints", {})
        return {}

    def update_cluster_fingerprints(
        self, name: str, version: str, fingerprints: dict
    ) -> None:
        """
        Store cluster fingerprint mapping for an aggregator.

        Args:
            name: Aggregator/playbook name
            version: Agent version
            fingerprints: Mapping of fingerprint_hash to {"agent_playbook_id": int, "user_playbook_ids": list[int]}
        """
        state_key = self._bookmark_key(name, version=version) + "::clusters"
        state = {"cluster_fingerprints": fingerprints}
        self.storage.upsert_operation_state(state_key, state)
        logger.info(
            "Updated cluster fingerprints for '%s' v%s with %d clusters",
            name,
            version,
            len(fingerprints),
        )

    # ── Use Case 5: Simple Lock ──
    # (Non-queuing lock for cleanup operations)

    def acquire_simple_lock(self, stale_seconds: int = 300) -> bool:
        """Acquire a simple non-queuing lock.

        Checks if a lock is already held. If the lock is stale (older than
        stale_seconds), it will be overridden.

        Args:
            stale_seconds: Seconds after which a lock is considered stale

        Returns:
            bool: True if lock acquired, False if another operation holds it
        """
        state_key = self._lock_key()
        state = self.storage.get_operation_state(state_key)
        current_time = int(time.time())

        if state and state.get("in_progress", False):
            started_at = state.get("started_at", 0)
            if current_time - started_at < stale_seconds:
                logger.info(
                    "Skipping %s - another operation is in progress", self.service_name
                )
                return False
            logger.warning(
                "Stale %s lock detected (started %d seconds ago), proceeding",
                self.service_name,
                current_time - started_at,
            )

        # Acquire lock
        self.storage.upsert_operation_state(
            state_key, {"in_progress": True, "started_at": current_time}
        )
        return True

    def release_simple_lock(self) -> None:
        """Release the simple lock."""
        state_key = self._lock_key()
        self.storage.upsert_operation_state(
            state_key,
            {"in_progress": False, "completed_at": int(time.time())},
        )
