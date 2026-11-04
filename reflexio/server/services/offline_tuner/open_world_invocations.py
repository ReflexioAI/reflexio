"""The open-world refusal set and the durable invocation records.

Two things live here, and both are deliberately free of any ``storage``
import, so storage can name them. It sits at the ``offline_tuner`` package
level rather than in ``evidence/``: ``storage.open_world_evidence`` imports
it, and importing any ``evidence`` module runs that package's eager
``__init__``, which imports ``storage.open_world_evidence`` back.

**The refusal set.** Every way the durable open-world path refuses an attempt
is one member of :class:`OpenWorldRefusal`. Each refusal has exactly one
exception class, carrying it as its ``refusal`` class attribute under the one
base :class:`OpenWorldRefusalError`, and exactly one terminal reason in
:data:`TERMINAL_REASON_BY_REFUSAL`. That table is the whole policy: the runner
maps any :class:`OpenWorldRefusalError` through :func:`terminal_reason_for`
instead of keeping one ``except`` arm per class, where a subclass arm placed
after its parent silently never fired.

**The invocation records.** The three durable invocation calls used to take
sixteen-odd loose keyword arguments, rebuilt at four call sites by one
private helper. :class:`OpenWorldInvocationBinding` is the identity every call
rechecks, :class:`ExpectedProvenance` what a prepare promises, and
:class:`CompletionRecord` what a completion observed. Field names are the
tenant RPC payload keys, so storage serializes a record without renaming
anything.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar


class OpenWorldRefusal(StrEnum):
    """Every distinct way the durable open-world path refuses an attempt."""

    #: The job lease or fence is no longer this worker's.
    LEASE_LOST = "lease_lost"
    #: A session is not a member of the frozen evidence manifest -- in
    #: practice, a second tick collided with this playbook's frozen evidence.
    MANIFEST_CONFLICT = "manifest_conflict"
    #: This analyst-input identity already belongs to another job.
    IDENTITY_BOUND = "identity_bound"
    #: The durable slot the question resolves to is owned by another job.
    SLOT_PINNED = "slot_pinned"
    #: A durable identity is bound to different bytes or a different binding.
    INVOCATION_CONFLICT = "invocation_conflict"
    #: The qualification authority moved under the attempt.
    AUTHORITY_STALE = "authority_stale"
    #: Durable bytes, digests or a refusal the classifier does not know.
    INTEGRITY = "integrity"
    #: A governance subject's epoch moved, or it lost writability.
    GOVERNANCE_CHANGED = "governance_changed"
    #: The frozen bundle's governance vector no longer authorizes work.
    GOVERNANCE_INVALID = "governance_invalid"
    #: The target playbook is no longer the version the attempt froze.
    TARGET_STALE = "target_stale"
    #: The frozen bundle was written by an image with another manifest schema.
    BUNDLE_SCHEMA_UNSUPPORTED = "bundle_schema_unsupported"
    #: Frozen evidence bytes, digests or metadata no longer agree.
    EVIDENCE_INTEGRITY = "evidence_integrity"
    #: The frozen evidence could not be rehydrated against current rows.
    REHYDRATION_FAILED = "rehydration_failed"


#: The single policy table: which durable terminal outcome each refusal ends
#: an attempt with. Values are the persisted ``terminal_outcome`` vocabulary,
#: so widening it is a tenant migration, not an edit here.
#:
#: Three entries are deliberately NOT ``infrastructure_failure``, which reads as
#: "transient, try again":
#:
#: * ``MANIFEST_CONFLICT`` is the concurrency signal -- a second tick landed on
#:   this playbook's frozen evidence and the loser analysed nothing. It clears
#:   on its own when the bundle moves at the next UTC day boundary.
#: * ``SLOT_PINNED`` made no provider call: the question's durable row is owned
#:   by a same-day sibling job. Reported as infrastructure it was
#:   indistinguishable from the provider fault that stranded the row.
#: * ``BUNDLE_SCHEMA_UNSUPPORTED`` cannot be cured by any retry on any host:
#:   the bundle was frozen by an image whose manifest derivation differs.
#:
#: ``IDENTITY_BOUND`` reaching the runner's top level is a conflict
#: (``infrastructure_failure``); the attempt loop converts the recoverable case
#: into ``SLOT_PINNED`` itself.
TERMINAL_REASON_BY_REFUSAL: dict[OpenWorldRefusal, str] = {
    OpenWorldRefusal.LEASE_LOST: "infrastructure_failure",
    OpenWorldRefusal.MANIFEST_CONFLICT: "evidence_manifest_conflict",
    OpenWorldRefusal.IDENTITY_BOUND: "infrastructure_failure",
    OpenWorldRefusal.SLOT_PINNED: "invocation_slot_pinned",
    OpenWorldRefusal.INVOCATION_CONFLICT: "infrastructure_failure",
    OpenWorldRefusal.AUTHORITY_STALE: "analyst_unqualified",
    OpenWorldRefusal.INTEGRITY: "infrastructure_failure",
    OpenWorldRefusal.GOVERNANCE_CHANGED: "governance_invalidated",
    OpenWorldRefusal.GOVERNANCE_INVALID: "governance_invalidated",
    OpenWorldRefusal.TARGET_STALE: "stale_incumbent",
    OpenWorldRefusal.BUNDLE_SCHEMA_UNSUPPORTED: "bundle_schema_unsupported",
    OpenWorldRefusal.EVIDENCE_INTEGRITY: "infrastructure_failure",
    OpenWorldRefusal.REHYDRATION_FAILED: "infrastructure_failure",
}


class OpenWorldRefusalError(Exception):
    """Base of every exception that carries an :class:`OpenWorldRefusal`.

    A mixin beside each class's existing base (``StorageError``,
    ``ValueError``), so every ``except`` that matched it before still does.
    """

    refusal: ClassVar[OpenWorldRefusal]


def terminal_reason_for(refusal: OpenWorldRefusal) -> str:
    """The durable terminal outcome a refusal ends an attempt with."""
    return TERMINAL_REASON_BY_REFUSAL[refusal]


@dataclass(frozen=True)
class OpenWorldInvocationBinding:
    """The durable identity every invocation call rechecks, in RPC key names."""

    job_id: int
    analyst_input_identity: str
    component_identity_digest: str
    component_role: str
    manifest_role: str | None
    user_id: str | None
    session_id: str | None
    bundle_digest: str
    outcome_contract_digest: str
    qualification_suite_digest: str | None
    qualification_result_digest: str
    trajectory_digest: str | None
    candidate_artifact_digest: str | None
    candidate_analyst_input_identity: str | None

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ExpectedProvenance:
    """What a prepared invocation promises it will be answered by."""

    expected_provider: str
    expected_model_revision: str
    input_digest: str

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CompletionRecord:
    """What a completed invocation observed. ``usage_json`` is a JSON string."""

    output_json: str
    output_digest: str
    observed_provider: str
    observed_model_revision: str
    called_at: int
    latency_ms: int
    usage_json: str | None

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


__all__ = [
    "TERMINAL_REASON_BY_REFUSAL",
    "CompletionRecord",
    "ExpectedProvenance",
    "OpenWorldInvocationBinding",
    "OpenWorldRefusal",
    "OpenWorldRefusalError",
    "terminal_reason_for",
]
