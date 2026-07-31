"""Canonical identities for immutable session outcomes."""

from collections.abc import Collection
from hashlib import sha256

from reflexio.server.services.playbook.publication import canonical_json_bytes

__all__ = [
    "canonical_json_bytes",
    "outcome_contract_digest",
    "trajectory_digest",
]


def outcome_contract_digest(
    *,
    source: str,
    schema_version: int | str,
    allowed_values: Collection[str],
    finalization_rule: str,
) -> str:
    """Hash a server-owned structured outcome contract."""
    payload = {
        "allowed_values": sorted(set(allowed_values)),
        "finalization_rule": finalization_rule,
        "schema_version": schema_version,
        "source": source,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def trajectory_digest(trajectory: object) -> str:
    """Hash the canonical finalized session trajectory."""
    return sha256(canonical_json_bytes(trajectory)).hexdigest()
