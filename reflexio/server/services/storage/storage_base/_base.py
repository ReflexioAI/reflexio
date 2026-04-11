from abc import ABC
from collections.abc import Sequence

from reflexio.models.api_schema.domain import Status


def matches_status_filter(
    item_status: Status | None,
    status_filter: Sequence[Status | None],
) -> bool:
    """Check whether an item's status matches a status filter list (Python-side filtering).

    Args:
        item_status (Status | None): The item's current status
        status_filter (Sequence[Status | None]): Allowed status values

    Returns:
        bool: True if the item passes the filter
    """
    has_none = None in status_filter
    status_strings = [
        s.value for s in status_filter if s is not None and hasattr(s, "value")
    ]
    if has_none and item_status is None:
        return True
    item_val = (
        item_status.value
        if item_status is not None and hasattr(item_status, "value")
        else item_status
    )
    if item_val in status_strings:
        return True
    return has_none and not status_strings and item_status is None


class BaseStorageCore(ABC):  # noqa: B024
    """Base class for storage. Abstract methods are defined in mixin classes."""

    def __init__(self, org_id: str, base_dir: str | None = None) -> None:
        self.org_id = org_id
        if base_dir is None:
            from reflexio.server import LOCAL_STORAGE_PATH

            base_dir = LOCAL_STORAGE_PATH
        self.base_dir = base_dir

    # Migrate
    def migrate(self) -> bool:
        """Handle migration to transform the storage to the latest format.

        Returns:
            A boolean indicating whether migration is successful.
        """
        return True

    def check_migration_needed(self) -> bool:
        """Check if storage needs migration. Returns False by default (no migration needed).

        Returns:
            bool: True if migration is needed, False otherwise
        """
        return False
