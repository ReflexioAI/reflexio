"""Explicit profile retention authority and native storage-owned execution.

Persisted profiles are the durable jobs. Scheduling belongs to the server's
existing scheduler; this module creates no thread, connection or execution pool.
Estimated and unknown lifetimes never authorize hard deletion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

KEYS = ("rsi_lifetime_v8", "rsi_lifetime_v7", "rsi_lifetime_v6")
ANCESTORS_KEY = "rsi_deadline_ancestors_v8"
VERSION_FIELDS = (
    "profile_id",
    "user_id",
    "content",
    "last_modified_timestamp",
    "expiration_timestamp",
    "source_span",
    "custom_features",
)


def profile_version(row: dict[str, Any]) -> str:
    """Fingerprint the same fields in compiled UserProfiles and locked DB rows.

    Status/lineage are excluded because native supersede updates those fields.
    Storage separately checks the owner and project under its write lock.
    """
    value = {key: row.get(key) for key in VERSION_FIELDS}
    value["custom_features"] = value["custom_features"] or {}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def due_epoch(features: Any) -> int | None:
    """Decode machine authority, preserving inclusive integer-second boundaries."""
    if features is None:
        return None
    if not isinstance(features, dict):
        raise ValueError("Profile lifetime metadata must be an object")
    values = [features[key] for key in KEYS if key in features]
    if not values:
        return None
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("Profile lifetime authority must be typed")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("Conflicting profile lifetime authorities")
    value = values[0]
    if value.get("basis") != "explicit_retention":
        return None
    deadline, boundary = value.get("logical_deadline"), value.get("boundary")
    if type(deadline) is not int or boundary not in {"stop_at", "through"}:
        raise ValueError("Explicit profile retention needs exact deadline and boundary")
    return deadline + (boundary == "through")


@dataclass(frozen=True)
class DeadlineJob:
    profile_id: str
    due: int
    version: str
    user_id: str
    project_id: str = ""


def job_for(row: dict[str, Any]) -> DeadlineJob | None:
    due = due_epoch(row.get("custom_features"))
    if due is None:
        return None
    return DeadlineJob(
        profile_id=row["profile_id"],
        due=due,
        version=profile_version(row),
        user_id=row["user_id"],
        project_id=row.get("project_id", ""),
    )


@dataclass(frozen=True)
class AncestorBinding:
    version: str
    due: int
    lifetime: dict[str, Any]


def ancestor_bindings(
    profiles: Sequence[Any], superseded_ids: Sequence[str], *, user_id: str
) -> dict[str, AncestorBinding]:
    """Accept only compiler-authored versions of actually retired owned rows."""
    if not user_id:
        raise ValueError("Deadline publication requires an explicit owner")
    bindings: dict[str, AncestorBinding] = {}
    for profile in profiles:
        row = profile if isinstance(profile, dict) else profile.model_dump(mode="json")
        if row.get("user_id") != user_id:
            raise ValueError("Deadline publication profile has a foreign owner")
        features = row.get("custom_features") or {}
        if not isinstance(features, dict):
            raise ValueError("Profile lifetime metadata must be an object")
        refs = features.get(ANCESTORS_KEY, [])
        if not isinstance(refs, list):
            raise ValueError("Deadline ancestor bindings must be explicit records")
        due = due_epoch(features)
        if refs and due is None:
            raise ValueError("Ancestor erasure requires actual explicit retention")
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref) != {"profile_id", "version"}
                or ref["profile_id"] not in superseded_ids
                or not isinstance(ref["version"], str)
                or len(ref["version"]) != 64
            ):
                raise ValueError(
                    "Deadline ancestor must bind an actual retired native row"
                )
            if due is None:
                raise ValueError("Ancestor erasure requires actual explicit retention")
            identity = ref["profile_id"]
            lifetime = next(features[key] for key in KEYS if key in features)
            item = AncestorBinding(ref["version"], due, dict(lifetime))
            previous = bindings.get(identity)
            if previous is not None and previous.version != item.version:
                raise ValueError("Conflicting deadline ancestor versions")
            if previous is None or item.due < previous.due:
                bindings[identity] = item
    return bindings


def _whole_seconds(now: int) -> None:
    if type(now) is not int:
        raise ValueError("Deadline clock must be whole seconds")


class ProfileExplicitRetentionMixin:
    """Shared policy over backend-owned locks, transactions and deletion APIs.

    Backend hooks must use the concrete storage's existing transaction owner.
    No public operation infers erasure authority from prose, expiry or lineage.
    Inactive copies remain eligible when their own persisted authority is due.
    """

    commit_scope: Any
    in_commit_scope: Any

    def _profile_deadline_project_id(self) -> str:
        raise NotImplementedError

    def _profile_deadline_rows(
        self,
        *,
        user_id: str | None,
        after_profile_id: str,
        limit: int,
        profile_id: str | None = None,
        lock: bool = False,
        metadata_only: bool = False,
        candidate_mode: str | None = None,
        now: int | None = None,
        earliest: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _profile_deadline_update_features(
        self, row: dict[str, Any], features: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def _profile_deadline_delete(self, row: dict[str, Any]) -> int:
        raise NotImplementedError

    def list_profile_deadline_jobs(
        self,
        *,
        user_id: str | None = None,
        after_profile_id: str = "",
        limit: int = 256,
        now: int | None = None,
    ) -> list[DeadlineJob]:
        """Keyset-page persisted authority, optionally selecting only due jobs.

        Every DB read is bounded. Empty pages, not short pages, end discovery;
        unrelated/estimated rows do not consume the returned job limit.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Deadline page limit must be between 1 and 1000")
        if user_id is not None and not user_id:
            raise ValueError("Deadline owner must be nonempty when supplied")
        if now is not None:
            _whole_seconds(now)
        project_id = self._profile_deadline_project_id()
        jobs = []
        cursor = after_profile_id
        while len(jobs) < limit:
            rows = self._profile_deadline_rows(
                user_id=user_id,
                after_profile_id=cursor,
                limit=min(256, limit - len(jobs)),
                candidate_mode="all",
                now=now,
            )
            if not rows:
                break
            for row in rows:
                identity = row["profile_id"]
                if identity <= cursor:
                    raise ValueError("Deadline discovery did not advance its keyset")
                cursor = identity
                if row.get("project_id", "") != project_id:
                    raise ValueError("Deadline discovery returned a foreign project")
                if user_id is not None and row["user_id"] != user_id:
                    raise ValueError("Deadline discovery returned a foreign owner")
                job = job_for(row)
                if job is not None and (now is None or job.due <= now):
                    jobs.append(job)
        return jobs

    def next_profile_deadline(self, *, user_id: str | None = None) -> int | None:
        """Validate uncertain candidates and the indexed regular minimum.

        The SQL projection is only candidate selection. Ambiguous JSON/legacy
        equality and wide SQLite integers remain in the sentinel bucket; their
        Python-accepted deadlines compete with the regular minimum, rather than
        accidentally hiding it. Invalid authority still fails explicitly.
        """
        if user_id is not None and not user_id:
            raise ValueError("Deadline owner must be nonempty when supplied")
        project_id = self._profile_deadline_project_id()
        earliest = None
        cursor = ""
        while True:
            rows = self._profile_deadline_rows(
                user_id=user_id,
                after_profile_id=cursor,
                limit=256,
                metadata_only=True,
                candidate_mode="uncertain",
            )
            if not rows:
                break
            for row in rows:
                if row["profile_id"] <= cursor:
                    raise ValueError("Deadline discovery did not advance its keyset")
                cursor = row["profile_id"]
                self._check_deadline_identity(row, project_id, user_id)
                due = due_epoch(row.get("custom_features"))
                if due is not None and (earliest is None or due < earliest):
                    earliest = due
        rows = self._profile_deadline_rows(
            user_id=user_id,
            after_profile_id="",
            limit=1,
            metadata_only=True,
            candidate_mode="regular",
            earliest=True,
        )
        if rows:
            row = rows[0]
            self._check_deadline_identity(row, project_id, user_id)
            due = due_epoch(row.get("custom_features"))
            if due is None:
                raise ValueError("Regular deadline projection lost its authority")
            if earliest is None or due < earliest:
                earliest = due
        return earliest

    @staticmethod
    def _check_deadline_identity(
        row: dict[str, Any], project_id: str, user_id: str | None
    ) -> None:
        if row.get("project_id", "") != project_id:
            raise ValueError("Deadline discovery returned a foreign project")
        if user_id is not None and row["user_id"] != user_id:
            raise ValueError("Deadline discovery returned a foreign owner")

    def delete_profile_deadline(
        self, job: DeadlineJob, now: int, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Version CAS and hard-delete event share the native writer transaction."""
        _whole_seconds(now)
        if user_id is not None and job.user_id != user_id:
            raise ValueError("Deadline job has a foreign owner")
        if not job.user_id or job.project_id != self._profile_deadline_project_id():
            raise ValueError("Deadline job has a foreign owner or project")
        joined = self.in_commit_scope
        with self.commit_scope():
            rows = self._profile_deadline_rows(
                user_id=job.user_id,
                after_profile_id="",
                limit=1,
                profile_id=job.profile_id,
                lock=True,
            )
            current = job_for(rows[0]) if rows else None
            if current != job or job.due > now:
                return {"profile_id": job.profile_id, "status": "stale_or_not_due"}
            if self._profile_deadline_delete(rows[0]) != 1:
                raise ValueError(
                    "Locked due profile deletion did not commit exactly one row"
                )
        return {
            "profile_id": job.profile_id,
            "version": job.version,
            "due_epoch": job.due,
            "execution_epoch": now,
            "lag_seconds": now - job.due,
            "status": "delete_staged" if joined else "hard_deleted",
        }

    def bind_profile_deadline_ancestors(
        self, profiles: Sequence[Any], superseded_ids: Sequence[str], *, user_id: str
    ) -> list[dict[str, Any]]:
        """Stage exact old-copy constraints before publishing replacement rows."""
        bindings = ancestor_bindings(profiles, superseded_ids, user_id=user_id)
        project_id = self._profile_deadline_project_id()
        audit = []
        with self.commit_scope():
            for identity, item in sorted(bindings.items()):
                rows = self._profile_deadline_rows(
                    user_id=user_id,
                    after_profile_id="",
                    limit=1,
                    profile_id=identity,
                    lock=True,
                )
                if (
                    not rows
                    or rows[0]["user_id"] != user_id
                    or rows[0].get("project_id", "") != project_id
                    or profile_version(rows[0]) != item.version
                ):
                    raise ValueError(
                        "Deadline ancestor changed since authorized compilation"
                    )
                old = rows[0]
                previous = due_epoch(old.get("custom_features"))
                if previous is not None and previous <= item.due:
                    audit.append(
                        {"profile_id": identity, "status": "earlier_constraint_kept"}
                    )
                    continue
                features = dict(old.get("custom_features") or {})
                for key in KEYS:
                    features.pop(key, None)
                features[KEYS[0]] = dict(item.lifetime)
                self._profile_deadline_update_features(old, features)
                audit.append(
                    {
                        "profile_id": identity,
                        "status": "constraint_staged",
                        "due_epoch": item.due,
                        "expected_version": item.version,
                    }
                )
        return audit

    def purge_profile_deadlines(
        self,
        now: int,
        *,
        user_id: str | None = None,
        batch_size: int = 256,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run persisted-state catch-up from a managed idle tick or publication.

        The scheduler may bound successful/stale job attempts per tick with limit;
        publication leaves it unset to remove every due owned copy atomically.
        Failures propagate to the existing scheduler/publication failure owner.
        """
        _whole_seconds(now)
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Deadline work limit must be positive")
        audit: list[dict[str, Any]] = []
        cursor = ""
        while limit is None or len(audit) < limit:
            count = batch_size if limit is None else min(batch_size, limit - len(audit))
            jobs = self.list_profile_deadline_jobs(
                user_id=user_id, after_profile_id=cursor, limit=count, now=now
            )
            if not jobs:
                return audit
            for job in jobs:
                audit.append(self.delete_profile_deadline(job, now, user_id=user_id))
                cursor = job.profile_id
        return audit


def persist_with_deadlines[T](
    storage: Any, write_plan: Any, persist: Callable[[], T], now: int
) -> tuple[T, list[dict[str, Any]]]:
    """Bind old versions, publish and purge overdue copies in one native commit.

    The write plan carries bindings through durable outcome encode/decode. No
    process-local compiler state is required when resuming a cached plan.
    """
    _whole_seconds(now)
    audit = []
    user_id = write_plan.user_id if write_plan is not None else None
    with storage.commit_scope():
        if write_plan is not None:
            audit.extend(
                storage.bind_profile_deadline_ancestors(
                    write_plan.new_profiles, write_plan.superseded_ids, user_id=user_id
                )
            )
        result = persist()
        audit.extend(storage.purge_profile_deadlines(now, user_id=user_id))
    return result, audit
