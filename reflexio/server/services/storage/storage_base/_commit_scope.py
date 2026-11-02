from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


class CommitScopeMixin(ABC):
    @property
    @abstractmethod
    def in_commit_scope(self) -> bool:
        """Whether this thread already owns a multi-write transaction."""
        raise NotImplementedError

    @abstractmethod
    def commit_scope(self) -> AbstractContextManager[None]:
        """Atomic multi-write transaction; see Workstream A plan Task 1."""
        raise NotImplementedError
