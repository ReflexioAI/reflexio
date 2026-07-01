"""Upgrade/downgrade status-change orchestration for ``BaseGenerationService``.

``StatusChangeMixin`` holds the public ``run_upgrade``/``run_downgrade`` template
methods plus the six status-change hooks subclasses override. This bucket touches
NONE of the per-run ``self`` fields and no billing — the cleanest boundary of the
decomposition. Method bodies are moved verbatim from the former monolithic
``base_generation_service.py``; ``run_upgrade``/``run_downgrade`` are PUBLIC (called
from ``lib/_profiles.py`` and ``lib/_user_playbook.py``) so their signatures are
byte-identical.
"""

from enum import StrEnum
from typing import Any, Generic, TypeVar

from reflexio.models.api_schema.service_schemas import Status

TRequest = TypeVar("TRequest")


class StatusChangeOperation(StrEnum):
    """Operation type for upgrade/downgrade responses."""

    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


class StatusChangeMixin(Generic[TRequest]):  # noqa: UP046
    """Upgrade/downgrade template methods + overridable status-change hooks.

    Mixed into ``BaseGenerationService`` ahead of ``ABC``; every ``self.`` call in
    these methods resolves to a sibling hook on this same mixin, so the mixin needs
    no cross-mixin attribute stubs.
    """

    # ===============================
    # Upgrade/Downgrade methods (optional - override to enable)
    # ===============================

    def _has_items_with_status(self, status: Status | None, request: TRequest) -> bool:
        """Check if items exist with given status and filters from request.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            status: The status to check for (None for CURRENT)
            request: The upgrade/downgrade request object with filters

        Returns:
            bool: True if any matching items exist
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _delete_items_by_status(self, status: Status, request: TRequest) -> int:
        """Delete items with given status matching request filters.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            status: The status of items to delete
            request: The upgrade/downgrade request object with filters

        Returns:
            int: Number of items deleted
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _update_items_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        request: TRequest,
        user_ids: list[str] | None = None,
    ) -> int:
        """Update items from old_status to new_status with request filters.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            request: The upgrade/downgrade request object with filters
            user_ids: Optional pre-computed list of user IDs to filter by

        Returns:
            int: Number of items updated
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _get_affected_user_ids_for_upgrade(self, request: TRequest) -> list[str] | None:  # noqa: ARG002
        """Get user IDs to filter by for upgrade operations.

        Override this method to support the only_affected_users flag.
        By default returns None (no filtering).

        Args:
            request: The upgrade request object

        Returns:
            Optional[list[str]]: List of user IDs to filter by, or None for no filtering
        """
        return None

    def _get_affected_user_ids_for_downgrade(
        self,
        request: TRequest,  # noqa: ARG002
    ) -> list[str] | None:
        """Get user IDs to filter by for downgrade operations.

        Override this method to support the only_affected_users flag.
        By default returns None (no filtering).

        Args:
            request: The downgrade request object

        Returns:
            Optional[list[str]]: List of user IDs to filter by, or None for no filtering
        """
        return None

    def _create_status_change_response(
        self,
        operation: StatusChangeOperation,
        success: bool,
        counts: dict,
        msg: str,
    ) -> Any:
        """Create upgrade or downgrade response object based on operation type.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            operation: The operation type (UPGRADE or DOWNGRADE)
            success: Whether the operation succeeded
            counts: Dictionary of counts (upgrade: deleted/archived/promoted, downgrade: demoted/restored)
            msg: Status message

        Returns:
            A response object (e.g., UpgradeProfilesResponse, DowngradeUserPlaybooksResponse)
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def run_upgrade(self, request: TRequest) -> Any:
        """Run the upgrade workflow for the service.

        This template method orchestrates the upgrade process:
        1. Validate that pending items exist
        2. Delete old archived items
        3. Archive current items (None → ARCHIVED)
        4. Promote pending items (PENDING → None/CURRENT)

        Child classes must implement the hook methods to enable upgrade functionality:
        - _has_items_with_status()
        - _delete_items_by_status()
        - _update_items_status()
        - _create_status_change_response()

        Args:
            request: The upgrade request object with optional filters

        Returns:
            A response object with success status, counts, and message
        """
        try:
            # 1. Validate pending items exist
            if not self._has_items_with_status(Status.PENDING, request):
                return self._create_status_change_response(
                    StatusChangeOperation.UPGRADE,
                    False,
                    {"deleted": 0, "archived": 0, "promoted": 0},
                    "No pending items found to upgrade",
                )

            # Get affected user IDs once (child class determines the logic)
            affected_user_ids = self._get_affected_user_ids_for_upgrade(request)

            # 2. Delete old archived items (skip if archive_current=False)
            deleted = 0
            archived = 0
            if getattr(request, "archive_current", True):
                deleted = self._delete_items_by_status(Status.ARCHIVED, request)

                # 3. Archive current items (None → ARCHIVED)
                archived = self._update_items_status(
                    None, Status.ARCHIVED, request, user_ids=affected_user_ids
                )

            # 4. Promote pending items (PENDING → None)
            promoted = self._update_items_status(
                Status.PENDING, None, request, user_ids=affected_user_ids
            )

            msg = f"Upgraded: {promoted} promoted, {archived} archived, {deleted} old archived deleted"
            return self._create_status_change_response(
                StatusChangeOperation.UPGRADE,
                True,
                {"deleted": deleted, "archived": archived, "promoted": promoted},
                msg,
            )

        except Exception as e:
            return self._create_status_change_response(
                StatusChangeOperation.UPGRADE,
                False,
                {"deleted": 0, "archived": 0, "promoted": 0},
                f"Failed to upgrade: {str(e)}",
            )

    def run_downgrade(self, request: TRequest) -> Any:
        """Run the downgrade workflow for the service.

        This template method orchestrates the downgrade process:
        1. Validate that archived items exist
        2. Demote current items (None → ARCHIVE_IN_PROGRESS)
        3. Restore archived items (ARCHIVED → None/CURRENT)
        4. Complete archiving (ARCHIVE_IN_PROGRESS → ARCHIVED)

        Child classes must implement the hook methods to enable downgrade functionality:
        - _has_items_with_status()
        - _update_items_status()
        - _create_status_change_response()

        Args:
            request: The downgrade request object with optional filters

        Returns:
            A response object with success status, counts, and message
        """
        try:
            # 1. Validate archived items exist
            if not self._has_items_with_status(Status.ARCHIVED, request):
                return self._create_status_change_response(
                    StatusChangeOperation.DOWNGRADE,
                    False,
                    {"demoted": 0, "restored": 0},
                    "No archived items found to restore",
                )

            # Get affected user IDs once (child class determines the logic)
            affected_user_ids = self._get_affected_user_ids_for_downgrade(request)

            # 2. Demote current (None → ARCHIVE_IN_PROGRESS)
            demoted = self._update_items_status(
                None, Status.ARCHIVE_IN_PROGRESS, request, user_ids=affected_user_ids
            )

            # 3. Restore archived (ARCHIVED → None)
            restored = self._update_items_status(
                Status.ARCHIVED, None, request, user_ids=affected_user_ids
            )

            # 4. Complete archiving (ARCHIVE_IN_PROGRESS → ARCHIVED)
            self._update_items_status(
                Status.ARCHIVE_IN_PROGRESS,
                Status.ARCHIVED,
                request,
                user_ids=affected_user_ids,
            )

            msg = f"Downgraded: {demoted} archived, {restored} restored"
            return self._create_status_change_response(
                StatusChangeOperation.DOWNGRADE,
                True,
                {"demoted": demoted, "restored": restored},
                msg,
            )

        except Exception as e:
            return self._create_status_change_response(
                StatusChangeOperation.DOWNGRADE,
                False,
                {"demoted": 0, "restored": 0},
                f"Failed to downgrade: {str(e)}",
            )
