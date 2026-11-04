"""The regeneration fence's pure digest derivations.

These three functions live in ``evidence/`` rather than beside the fence
protocol in ``open_world/fence.py`` so a storage backend can derive the fence
digests without importing the open-world package, which only the host's
composition root may import.

``open_world/fence.py`` re-exports all three, so every existing caller is
unchanged and there is still exactly ONE expression of each digest. That single
expression is the whole point: two encoders of "the same evidence" that drifted
apart would make two callers disagree about whether identical evidence is
identical, which is precisely the failure the fence exists to prevent -- and
they would disagree silently, because both copies would be correct on the day
they were written.

The digests themselves:

* **Evidence-prefix digest** -- incumbent fingerprint + discovery manifest +
  held-out manifest. All three are known *before* any analyst call, which is
  the entire point: keying the check on the full digest would prevent a
  redundant publication but not the spend that produced it.
* **Full digest** -- those three plus hypothesis and candidate. The auditable
  contract of record, stored on the terminal lifecycle row. It only exists
  *after* generation, because two of its five components are LLM outputs.

The prefix is a sound key for the full digest only because the design pins the
analyst to deterministic sampling: identical evidence yields an identical
hypothesis and candidate, so identical prefix implies identical full digest.
"""

from __future__ import annotations

from .manifests import canonical_json_bytes, sha256_hex

_EVIDENCE_FENCE_CONTRACT = "open_world_regeneration_evidence_fence"
_FULL_FENCE_CONTRACT = "open_world_regeneration_full_fence"
_ROLE_MANIFEST_CONTRACT = "open_world_regeneration_role_manifest"
_HEX = frozenset("0123456789abcdef")


def role_manifest_payload_digest(payload: object) -> str:
    """Digest one role manifest's CANONICAL PAYLOAD under the shared encoder.

    Takes the payload rather than the dataclass so a caller holding the stored
    JSON -- the storage adapter reading a frozen ``open_world_evidence_bundle``
    artifact -- derives the identical digest to a caller holding a live
    ``OpenWorldRoleManifest``, without rebuilding the object graph from JSON.

    It is framed with a contract type so this digest can never collide with a
    bare canonicalization of the same sessions produced for some other purpose.

    Args:
        payload (object): ``OpenWorldRoleManifest.canonical_payload()``, either
            freshly produced or read back verbatim from the frozen bundle
            artifact.

    Returns:
        str: Lowercase SHA-256 hex digest of the framed manifest.
    """
    return sha256_hex(
        canonical_json_bytes(
            {
                "contract_type": _ROLE_MANIFEST_CONTRACT,
                "schema_version": "v1",
                "payload": payload,
            }
        )
    )


def evidence_fence_digest(
    *,
    incumbent_fingerprint: str,
    discovery_manifest_digest: str,
    held_out_manifest_digest: str,
) -> str:
    """Digest the three components known before any analyst is invoked.

    Args:
        incumbent_fingerprint (str): The target's full-version fingerprint.
        discovery_manifest_digest (str): :func:`role_manifest_payload_digest`
            of the bundle's discovery manifest.
        held_out_manifest_digest (str): :func:`role_manifest_payload_digest`
            of the bundle's held-out manifest.

    Returns:
        str: Lowercase SHA-256 hex digest keying the attempt-start fence.
    """
    return sha256_hex(
        canonical_json_bytes(
            {
                "contract_type": _EVIDENCE_FENCE_CONTRACT,
                "schema_version": "v1",
                "payload": {
                    "incumbent_fingerprint": _require_digest(
                        "incumbent_fingerprint", incumbent_fingerprint
                    ),
                    "discovery_manifest_digest": _require_digest(
                        "discovery_manifest_digest", discovery_manifest_digest
                    ),
                    "held_out_manifest_digest": _require_digest(
                        "held_out_manifest_digest", held_out_manifest_digest
                    ),
                },
            }
        )
    )


def full_fence_digest(
    *,
    incumbent_fingerprint: str,
    discovery_manifest_digest: str,
    held_out_manifest_digest: str,
    hypothesis_digest: str,
    candidate_digest: str,
) -> str:
    """Digest the five-component contract of record stored on the terminal row.

    Args:
        incumbent_fingerprint (str): The target's full-version fingerprint.
        discovery_manifest_digest (str): Discovery manifest digest.
        held_out_manifest_digest (str): Held-out manifest digest.
        hypothesis_digest (str): Digest of the durable discovery memo.
        candidate_digest (str): Digest of the durable candidate artifact.

    Returns:
        str: Lowercase SHA-256 hex digest of the full regeneration fence.
    """
    return sha256_hex(
        canonical_json_bytes(
            {
                "contract_type": _FULL_FENCE_CONTRACT,
                "schema_version": "v1",
                "payload": {
                    "incumbent_fingerprint": _require_digest(
                        "incumbent_fingerprint", incumbent_fingerprint
                    ),
                    "discovery_manifest_digest": _require_digest(
                        "discovery_manifest_digest", discovery_manifest_digest
                    ),
                    "held_out_manifest_digest": _require_digest(
                        "held_out_manifest_digest", held_out_manifest_digest
                    ),
                    "hypothesis_digest": _require_digest(
                        "hypothesis_digest", hypothesis_digest
                    ),
                    "candidate_digest": _require_digest(
                        "candidate_digest", candidate_digest
                    ),
                },
            }
        )
    )


def _require_digest(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


__all__ = [
    "evidence_fence_digest",
    "full_fence_digest",
    "role_manifest_payload_digest",
]
