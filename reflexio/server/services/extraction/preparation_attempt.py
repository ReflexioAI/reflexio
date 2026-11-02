"""One ordinary secondary operation, admitted at its actual external dispatch."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from reflexio.server.llm.compute_admission import external_compute_admission
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.publication_preparation import (
    preparation_claim_digest,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    BaseStorage,
    LeaseLostError,
)
from reflexio.server.work_scope import current_project_id


class _PreparationAttempt:
    def __init__(self, storage: BaseStorage, run: AgentRunRecord):
        self.storage = storage
        self.run = run
        self.project = current_project_id()
        self.active = True
        self.identity: str | None = None
        self.lock = Lock()

    def enter(self) -> None:
        # Lifecycle mutex before W/R. Scope closes outside every DB writer;
        # neither provider I/O nor worker joins run while this mutex is held.
        with self.lock:
            if self.storage.in_commit_scope:
                raise InputBudgetConfigurationError(
                    "Secondary compute requires its own admission writer"
                )
            if not self.active or current_project_id() != self.project:
                raise InputBudgetConfigurationError("Preparation admission scope ended")
            if self.identity is None:
                entry = self.storage.begin_agent_preparation_attempt(self.run)
                self.identity = entry.identity
            else:
                entry = self.storage.pending_agent_preparation_attempt(self.run)
                if (
                    entry is None
                    or entry.identity != self.identity
                    or entry.claim_digest != preparation_claim_digest(self.run)
                ):
                    raise LeaseLostError("Preparation dispatch lost its admitted entry")


@contextmanager
def preparation_attempt_scope(
    storage: BaseStorage, run: AgentRunRecord
) -> Iterator[_PreparationAttempt]:
    if external_compute_admission.get() is not None:
        raise InputBudgetConfigurationError("Preparation admission already active")
    attempt = _PreparationAttempt(storage, run)
    guard_token = external_compute_admission.set(attempt.enter)
    try:
        yield attempt
    finally:
        with attempt.lock:
            attempt.active = False
        external_compute_admission.reset(guard_token)
