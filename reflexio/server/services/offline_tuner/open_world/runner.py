"""Shared job lease and stage-storage contracts for the offline-tuner worker.

The retired contrast/held-out state machine has been removed. The single active
method is RuleEvidenceAttemptRunner in rule_evidence_runtime.py.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    OptimizationJobStage,
    OptimizationTerminalOutcome,
    PlaybookOptimizationJob,
)


class OpenWorldStageStore(Protocol):
    """The job-lifecycle surface the runner advances an attempt through.

    Public because ``open_world_composition.OpenWorldStoragePort`` is built from
    it: a method added here is a method the composition check requires.
    """

    def advance_playbook_optimization_stage(
        self,
        job_id: int,
        fence: int,
        stage: OptimizationJobStage,
        *,
        terminal_outcome: OptimizationTerminalOutcome | None = None,
        now: int | None = None,
    ) -> bool: ...

    def get_playbook_optimization_job(
        self, job_id: int
    ) -> PlaybookOptimizationJob | None: ...

    def update_playbook_optimization_job(
        self, job_id: int, *, metadata_json: str | None = None
    ) -> None: ...

    def merge_playbook_optimization_job_metadata(
        self, job_id: int, patch: dict[str, Any]
    ) -> None: ...

    def renew_playbook_optimization_job_lease(
        self,
        job_id: int,
        owner: str,
        fence: int,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim: ...

    def reclaim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int = 60,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim: ...


@dataclass
class AttemptLease:
    """The claim an attempt is CURRENTLY running under.

    Mutable on purpose, and that is the whole point of it. Renewal leaves the
    FENCE alone -- ``renew_playbook_optimization_job_lease`` extends
    ``lease_expires_at`` and nothing else -- so every ``claim.fence`` consumer
    is untouched by a renewal. Exactly one reader compares the other half:
    ``attempt_store``'s publication binding check asserts
    ``job.lease_expires_at == claim.expires_at``, and it runs in
    ``complete_open_world_attempt``, OUTSIDE the runner. Handing it a claim
    object the runner has since renewed past would refuse every publication
    that renewed -- turning a fix for one durability bug into a new one on the
    only path that spends money successfully.

    So the composition root reads the ending claim from here rather than
    deriving it a second time from the job row, which would be two expressions
    of "the lease this attempt holds" free to disagree.
    """

    claim: OptimizationJobClaim
