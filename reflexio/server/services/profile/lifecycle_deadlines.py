"""Controlled explicit-deadline execution through the native storage owner.

Uses existing storage scopes, writer admission/transaction ownership,
row locks and content-free hard-delete events. Creates no connection pool,
scheduler or background work. Deadlines use each row's own persisted authority;
explicit forget uses an exact source-authorized CURRENT version from the native
write plan. Attention ancestry is deliberately not deletion authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.server.services.deferred_learning_plan import (
    ProfilePayloadErasure,
    ProfileWritePlan,
)
from reflexio.server.services.profile.lifecycle import (
    KEY,
    metadata,
    storage_deadline,
    whole_epoch,
)

# Source-authoritative persisted payload fields exposed by native reads. Keep
# this explicit projection in step with UserProfile; schema-coverage tests require
# review of additions. Derived vectors/expanded search terms are not returned by
# native decision reads and do not supply erasure authority. Native hard-delete
# still removes them with the profile row.
PAYLOAD_FIELDS = (
    "profile_id",
    "user_id",
    "content",
    "last_modified_timestamp",
    "generated_from_request_id",
    "profile_time_to_live",
    "expiration_timestamp",
    "custom_features",
    "source",
    "status",
    "extractor_names",
    "tags",
    "source_interaction_ids",
    "source_span",
    "notes",
    "reader_angle",
    "merged_into",
    "superseded_by",
)
DERIVED_FIELDS = frozenset({"embedding", "expanded_terms"})


def profile_version(row: dict[str, Any]) -> str:
    """Canonical typed source payload, identical at compile and locked read.

    Decode native JSON/array representations and use the same enum/default
    projection as UserProfile reads. Never substitute a modification timestamp
    for content/provenance equality: supported updates can share that timestamp.
    """
    value = {key: row[key] for key in PAYLOAD_FIELDS if key in row}
    for key in ("custom_features", "extractor_names", "tags", "source_interaction_ids"):
        if isinstance(value.get(key), str):
            value[key] = json.loads(value[key])
    value["source_interaction_ids"] = value.get("source_interaction_ids") or []
    value["source"] = value.get("source") or ""
    payload = UserProfile.model_validate(value).model_dump(
        mode="json", include=set(PAYLOAD_FIELDS)
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DeadlineJob:
    profile_id: str
    due: int
    version: str


def job_for(row: dict[str, Any]) -> DeadlineJob | None:
    value = dict(row)
    if isinstance(value.get("custom_features"), str):
        value["custom_features"] = json.loads(value["custom_features"])
    features = value.get("custom_features") or {}
    if KEY not in features:
        return None
    lifecycle = metadata(
        UserProfile.model_validate(
            {
                **value,
                **{
                    field: json.loads(value[field])
                    for field in ("extractor_names", "tags", "source_interaction_ids")
                    if isinstance(value.get(field), str)
                },
            }
        )
    )
    raw = lifecycle["explicit_storage_deadline"]
    if raw is None:
        return None
    due = storage_deadline(raw).epoch
    return DeadlineJob(row["profile_id"], due, profile_version(row))


class NativeLifecycleDeadlineStore:
    """Exact subject/project authority with bounded keyset discovery."""

    def __init__(self, storage: Any, subject_id: str, project_id: str = ""):
        if not subject_id:
            raise ValueError("Deadline execution requires a subject")
        self.storage = storage
        self.subject_id = subject_id
        self.project_id = project_id

    def _rows(
        self,
        *,
        after_profile_id: str = "",
        profile_id: str | None = None,
        lock: bool = False,
    ) -> list[dict[str, Any]]:
        return self.storage.profile_lifecycle_rows(
            self.subject_id,
            self.project_id,
            after_profile_id=after_profile_id,
            profile_id=profile_id,
            lock=lock,
        )

    def delete(self, expected: DeadlineJob, now: int) -> dict[str, Any]:
        """Version-CAS, row lock, deletion and event share one transaction."""
        whole_epoch(now)
        joined = self.storage.in_commit_scope
        with self.storage.commit_scope():
            rows = self._rows(profile_id=expected.profile_id, lock=True)
            current = job_for(rows[0]) if rows else None
            if current != expected or expected.due > now:
                return {
                    "profile_id": expected.profile_id,
                    "status": "stale_or_not_due",
                }
            count = self.storage.delete_profile_lifecycle_payload(
                self.subject_id,
                self.project_id,
                expected.profile_id,
                reason="explicit_retention_deadline",
            )
            if count != 1:
                raise ValueError(
                    "Locked deadline deletion did not remove exactly one row"
                )
        return {
            "profile_id": expected.profile_id,
            "version": expected.version,
            "due_epoch": expected.due,
            "execution_epoch": now,
            "status": "delete_staged" if joined else "hard_deleted",
        }

    def erase_exact(
        self, instruction: ProfilePayloadErasure, *, request_id: str
    ) -> dict[str, Any]:
        """Erase only an explicitly authorized seen CURRENT version or fail closed."""
        if (
            instruction.user_id != self.subject_id
            or not request_id
            or instruction.request_id != request_id
            or not isinstance(instruction.source_quote, str)
            or not instruction.source_quote.strip()
            or not instruction.source_profile_id
        ):
            raise ValueError(
                "Payload erasure lacks scoped source authority and request lineage"
            )
        with self.storage.commit_scope():
            rows = self._rows(profile_id=instruction.profile_id, lock=True)
            if (
                len(rows) != 1
                or rows[0]["status"] is not None
                or profile_version(rows[0]) != instruction.expected_version
            ):
                raise ValueError(
                    "Explicit forget target changed since source-authorized compilation"
                )
            count = self.storage.delete_profile_lifecycle_payload(
                self.subject_id,
                self.project_id,
                instruction.profile_id,
                reason="source_authorized_explicit_forget",
            )
            if count != 1:
                raise ValueError(
                    "Explicit forget did not erase exactly one locked payload"
                )
        return {
            "profile_id": instruction.profile_id,
            "status": "erasure_staged",
            "request_id": request_id,
        }

    def purge_due(
        self, now: int, *, checkpoint: Callable[[], None] | None = None
    ) -> list[dict[str, Any]]:
        """Include all statuses; never mistake an expiry filter for erasure."""
        whole_epoch(now)
        after = ""
        audit = []
        while True:
            if checkpoint is not None:
                checkpoint()
            rows = self._rows(after_profile_id=after)
            if not rows:
                return audit
            for row in rows:
                if row["profile_id"] <= after:
                    raise ValueError("Deadline scan did not advance its keyset")
                after = row["profile_id"]
                job = job_for(row)
                if job is not None and job.due <= now:
                    audit.append(self.delete(job, now))


def persist_with_deadlines[T](
    store: NativeLifecycleDeadlineStore, persist: Callable[[], T], now: int
) -> tuple[T, list[dict[str, Any]]]:
    """Publish and enforce own-row authority in the native commit scope.

    Deadline sweeps use only each row's persisted explicit deadline. The
    publication callback may separately apply source-authorized exact erasures;
    neither operation infers permission from broad reconciliation lineage.
    """
    whole_epoch(now)
    with store.storage.commit_scope():
        audit = store.purge_due(now)
        result = persist()
        audit.extend(store.purge_due(now))
    return result, audit


def publish_with_lifecycle[T](
    store: NativeLifecycleDeadlineStore,
    plan: ProfileWritePlan,
    persist: Callable[[ProfileWritePlan], T],
    now: int,
) -> tuple[T, list[dict[str, Any]]]:
    """One native publication transaction; ordinary corrections remain soft."""
    whole_epoch(now)
    if plan.user_id != store.subject_id:
        raise ValueError("Lifecycle publication has a foreign owner")
    instructions = {item.profile_id: item for item in plan.erasure_instructions}
    if len(instructions) != len(plan.erasure_instructions):
        raise ValueError("Conflicting or duplicated payload erasure instructions")
    if not set(instructions).issubset(plan.superseded_ids):
        raise ValueError("Payload erasure must bind exact seen replaced profile IDs")
    if plan.superseded_ids and not plan.request_id:
        raise ValueError("Profile retirement requires request lineage")
    if set(instructions).intersection(p.profile_id for p in plan.new_profiles):
        raise ValueError("Payload erasure cannot target its replacement")
    erasure_audit = []

    def publish() -> T:
        return persist(
            replace(
                plan,
                superseded_ids=[
                    i for i in plan.superseded_ids if i not in instructions
                ],
            )
        )

    with store.storage.commit_scope():
        # Validate the exact seen version before a deadline sweep can remove
        # that same target. A missing or changed target still aborts every
        # effect; no deadline deletion can masquerade as a successful forget.
        erasure_audit.extend(
            store.erase_exact(instructions[identity], request_id=plan.request_id)
            for identity in sorted(instructions)
        )
        result, deadline_audit = persist_with_deadlines(store, publish, now)
    return result, [*erasure_audit, *deadline_audit]
