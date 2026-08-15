from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PurgeExecutionClaim:
    purge_id: str
    owner: str
    fence: int
    expires_at: int


def validate_purge_execution_claim(
    purge_id: str,
    execution_claim: PurgeExecutionClaim | None,
) -> PurgeExecutionClaim:
    if execution_claim is None:
        raise ValueError("purge execution claim is required")
    if type(execution_claim) is not PurgeExecutionClaim:
        raise ValueError("purge execution claim must be typed")
    if execution_claim.purge_id != purge_id:
        raise ValueError("purge execution claim purge_id mismatch")
    if not execution_claim.owner.strip():
        raise ValueError("purge execution claim owner is required")
    if execution_claim.fence <= 0:
        raise ValueError("purge execution claim fence is invalid")
    if execution_claim.expires_at <= 0:
        raise ValueError("purge execution claim expiry is invalid")
    return execution_claim
