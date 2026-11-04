"""Shared job lease for the offline-tuner worker.

The stage-storage contract (``OpenWorldStageStore``) lives in
``offline_tuner.open_world_port`` with the rest of the storage port. The single
active method is RuleEvidenceAttemptRunner in rule_evidence_runtime.py.
"""

from dataclasses import dataclass

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
)


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
