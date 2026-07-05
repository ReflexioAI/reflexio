from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


class CommitScopeMixin(ABC):
    @abstractmethod
    def commit_scope(self) -> AbstractContextManager[None]:
        """Atomic multi-write transaction; see Workstream A plan Task 1."""
        raise NotImplementedError
