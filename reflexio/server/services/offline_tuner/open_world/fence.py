"""The regeneration fence: refuse a second attempt on identical evidence.

DORMANT, AND DELIBERATELY KEPT. The fence was built for a restore: when a
provisional successor reverted at its observation deadline, nothing stopped the
next tick from regenerating the same candidate from the same evidence and
republishing it -- the tuner would loop, paying for the same three analyst calls
every cycle. Under keep-unless-revised a publication is terminal, no lifecycle
reverts, and the restore RPC that was this fence's ONLY writer is never called.
So ``restored_open_world_regeneration_fence_exists`` answers ``False`` for every
attempt, and the check below is inert rather than wrong.

The loop it guarded cannot recur under this design for a different and stronger
reason: a republished successor changes the incumbent fingerprint, which is the
first component of both digests, so the next attempt keys on different evidence
whether or not this fence answers. And a successor that stays put is judged only
by sessions served AFTER its own ``published_at`` (INV-2), so the same evidence
is not re-read either.

The derivations stay because they are the durable naming of "this exact
evidence", which the lifecycle row still stores; only the read has lost its
producer. Delete the fence when the digest columns go, not before.

The fence is two digests:

* **Full digest** -- incumbent fingerprint + discovery manifest + held-out
  manifest + hypothesis + candidate. The auditable contract of record, stored
  on the terminal lifecycle row. It only exists *after* generation, because two
  of its five components are LLM outputs.
* **Evidence-prefix digest** -- incumbent fingerprint + discovery manifest +
  held-out manifest only. All three are known *before* any analyst call, which
  is the entire point: keying the check on the full digest would prevent a
  redundant publication but not the spend that produced it.

The prefix is a sound key for the full digest only because the design pins the
analyst to deterministic sampling: identical evidence yields an identical
hypothesis and candidate, so identical prefix implies identical full digest.

No lifecycle state fences today. ``restored`` was the only one that ever did
and is no longer reachable. ``confirmed`` never fenced -- a kept successor is an
ordinary incumbent and a later failure starts a fresh cycle -- and ``displaced``
and ``erased`` need no fence: a manual edit moves the incumbent fingerprint and
erasure destroys the evidence, so both already produce a different digest.

The three pure derivations themselves now live in
``evidence/regeneration_fence.py`` and are re-exported here, because storage
may not import this package. See that module's header.

There is exactly ONE canonicalization. Both digests go through
``evidence.manifests.canonical_json_bytes`` -- the RFC 8785 encoder that
``identity.py`` and ``open_world_models.py`` already use to derive every other
digest in this phase. A second encoder that drifted from the first would make
two callers disagree about whether the same evidence is the same evidence,
which is precisely the failure the fence exists to prevent.
"""

from __future__ import annotations

from typing import Protocol

from ..evidence.open_world_models import (
    OpenWorldEvidenceBundle,
    OpenWorldRoleManifest,
)
from ..evidence.regeneration_fence import (
    evidence_fence_digest,
    full_fence_digest,
    role_manifest_payload_digest,
)


class RegenerationFenceStore(Protocol):
    """The one durable read the attempt-start fence needs.

    Deliberately narrower than the attempt store that satisfies it:
    the fence asks a single yes/no question and must not be able to mutate
    anything on the path it short-circuits.
    """

    def restored_open_world_regeneration_fence_exists(
        self, *, evidence_digest: str
    ) -> bool: ...


def role_manifest_digest(manifest: OpenWorldRoleManifest) -> str:
    """Digest one evidence role manifest under the shared canonical encoder.

    ``OpenWorldRoleManifest`` carries no ``digest`` property of its own and the
    bundle's canonical payload does not record one, so the fence derives it
    here rather than reading a stored value. It is framed with a contract type
    so this digest can never collide with a bare canonicalization of the same
    sessions produced for some other purpose.

    Args:
        manifest (OpenWorldRoleManifest): Discovery, held-out, or safety-only
            manifest from a frozen evidence bundle.

    Returns:
        str: Lowercase SHA-256 hex digest of the manifest.
    """
    if type(manifest) is not OpenWorldRoleManifest:
        raise TypeError("manifest must be an OpenWorldRoleManifest")
    # Delegated, NOT reimplemented. The storage adapter derives this same digest
    # from the stored bundle artifact and cannot import this module without
    # breaking composition-boundary invariant I4, so the expression lives in
    # `evidence/regeneration_fence.py` and both callers share it. Inlining it back here
    # would recreate the two-encoders-disagreeing failure the fence exists to
    # prevent -- and would do so silently, since both copies would be correct on
    # the day they were written.
    return role_manifest_payload_digest(manifest.canonical_payload())


def bundle_evidence_fence_digest(bundle: OpenWorldEvidenceBundle) -> str:
    """Derive the attempt-start fence key from one frozen evidence bundle.

    Args:
        bundle (OpenWorldEvidenceBundle): The frozen bundle this attempt runs
            against.

    Returns:
        str: The evidence-prefix fence digest for that bundle.
    """
    if type(bundle) is not OpenWorldEvidenceBundle:
        raise TypeError("bundle must be an OpenWorldEvidenceBundle")
    return evidence_fence_digest(
        incumbent_fingerprint=bundle.target.incumbent_full_version_fingerprint,
        discovery_manifest_digest=role_manifest_digest(bundle.discovery),
        held_out_manifest_digest=role_manifest_digest(bundle.held_out),
    )


def regeneration_is_fenced(
    store: RegenerationFenceStore, bundle: OpenWorldEvidenceBundle
) -> bool:
    """Whether a restored lifecycle already burned this exact evidence.

    Always ``False`` in this design: nothing writes a restored fence any more.
    See the module header before reading a green answer as "the fence checked".

    Args:
        store (RegenerationFenceStore): The durable fence read.
        bundle (OpenWorldEvidenceBundle): The frozen bundle for this attempt.

    Returns:
        bool: True when the attempt must be refused before any analyst call.
    """
    return store.restored_open_world_regeneration_fence_exists(
        evidence_digest=bundle_evidence_fence_digest(bundle)
    )


__all__ = [
    "RegenerationFenceStore",
    "bundle_evidence_fence_digest",
    "evidence_fence_digest",
    "full_fence_digest",
    "regeneration_is_fenced",
    "role_manifest_digest",
]
