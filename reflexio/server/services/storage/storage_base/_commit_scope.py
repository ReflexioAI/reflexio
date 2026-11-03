from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


class CommitScopeMixin(ABC):
    @property
    def in_commit_scope(self) -> bool:
        """Ownership introspection is required on use, not for read-only adapters."""
        raise NotImplementedError(
            "Storage adapter does not expose transaction ownership"
        )

    @abstractmethod
    def commit_scope(self) -> AbstractContextManager[None]:
        """Atomic multi-write transaction; see Workstream A plan Task 1."""
        raise NotImplementedError

    def _coordination_commit_scope(self) -> AbstractContextManager[None]:
        """Atomic private user-work progress, before any project is selected.

        Native adapters acquire their existing shared coordination writer here.
        This does not authorize tenant-table access or actual run adoption.
        """
        return self.commit_scope()
