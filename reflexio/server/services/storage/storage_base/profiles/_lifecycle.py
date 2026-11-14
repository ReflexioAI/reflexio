"""Storage-owned profile permission and accepted-access contracts."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any


class ProfileLifecycleStoreMixin:
    """Keep profile lifetime effects inside each backend's transaction owner."""

    @abstractmethod
    def profile_lifecycle_due_users(
        self, *, now: int, after_user_id: str = ""
    ) -> list[str]:
        """Read a bounded project-scoped page of subjects with due permissions."""
        raise NotImplementedError

    @abstractmethod
    def profile_lifecycle_rows(
        self,
        user_id: str,
        project_id: str,
        *,
        after_profile_id: str = "",
        profile_id: str | None = None,
        lock: bool = False,
    ) -> list[dict[str, Any]]:
        """Read an owner-scoped bounded page, including retired/expired rows."""
        raise NotImplementedError

    @abstractmethod
    def delete_profile_lifecycle_payload(
        self,
        user_id: str,
        project_id: str,
        profile_id: str,
        *,
        reason: str,
    ) -> int:
        """Delete exactly one scoped payload through the existing write owner."""
        raise NotImplementedError

    @abstractmethod
    def get_profile_access_epochs(
        self, *, user_id: str, profile_ids: Sequence[str]
    ) -> dict[str, int]:
        """Read only requested scoped profile/ancestor access epochs."""
        raise NotImplementedError

    @abstractmethod
    def record_served_profile_access(
        self, *, user_id: str, profile_ids: Sequence[str], accessed_at: int
    ) -> None:
        """Monotonically record final accepted results without rewriting facts."""
        raise NotImplementedError
