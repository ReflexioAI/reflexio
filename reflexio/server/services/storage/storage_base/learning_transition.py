# ruff: noqa: S608 -- Identifiers come from the fixed storage adapter registry.
"""One dataset policy authority and subordinate project admission witnesses.

The root is installed by actual dataset provisioning or a qualified upgrade,
never by ordinary SDK admission. Project witnesses inherit its policy; they do
not choose a phase or floor. All runtime locks follow user-work -> root ->
project -> execution rows. Administrative closure must use the same root writer.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from reflexio.profile_authoring_protocol import (
    AUTHORING_POLICIES,
    PROFILE_AUTHORING_POLICY,
)
from reflexio.server.services.storage.error import ExecutionInputBusyError

from ._extraction_stream import StreamSQL, decoded

if TYPE_CHECKING:
    from .transition_closure import ProjectTransitionClosure

LEARNING_TRANSITION_PREFIX = "__reflexio_learning_transition_v1__:"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_OPERATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class LearningTransitionBlockedError(ExecutionInputBusyError):
    """The durable dataset authority refuses this learning admission."""


class LearningTransitionIntegrityError(ValueError):
    """Stored state or input violates the typed control contract."""


class LearningTransitionConflictError(ValueError):
    """An immutable nonce/version/operation expectation changed."""


def _hex(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _version(value: Any) -> bool:
    return type(value) is int and value >= 0


def _operation(value: Any) -> None:
    if not isinstance(value, str) or not _OPERATION.fullmatch(value):
        raise LearningTransitionIntegrityError("Invalid learning transition operation")


def _selection(value: str | None) -> None:
    if value is not None and not _hex(value, _SHA):
        raise LearningTransitionIntegrityError("Invalid learning transition digest")


class _Fingerprint:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type] -- concrete children are immutable dataclasses.

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class LearningPolicyAuthority(_Fingerprint):
    schema_version: int
    authority_nonce: str
    generation: int
    phase: Literal["open", "sealed", "armed"]
    active_policy: str
    target_policy: str | None
    operation_id: str | None
    initialization_kind: Literal["first_dataset", "qualified_upgrade"]
    initialization_nonce: str
    provenance_sha256: str
    bootstrap: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or not _hex(self.authority_nonce, _NONCE)
            or not _version(self.generation)
            or self.phase not in ("open", "sealed", "armed")
            or self.active_policy not in AUTHORING_POLICIES
            or self.initialization_kind not in ("first_dataset", "qualified_upgrade")
            or not _hex(self.initialization_nonce, _NONCE)
            or not _hex(self.provenance_sha256, _SHA)
            or type(self.bootstrap) is not bool
        ):
            raise LearningTransitionIntegrityError("Invalid learning policy authority")
        if self.phase == "open":
            if (
                self.target_policy is not None
                or self.operation_id is not None
                or self.bootstrap
            ):
                raise LearningTransitionIntegrityError(
                    "Open authority retains transition state"
                )
        else:
            _operation(self.operation_id)
            if self.target_policy not in AUTHORING_POLICIES:
                raise LearningTransitionIntegrityError(
                    "Invalid authority target policy"
                )

    @classmethod
    def from_dict(cls, raw: Any) -> LearningPolicyAuthority:
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise LearningTransitionIntegrityError(
                "Invalid learning policy authority fields"
            )
        return cls(**raw)


@dataclass(frozen=True)
class LearningTransitionState(_Fingerprint):
    schema_version: int
    org_id: str
    project_id: str
    nonce: str
    revision: int
    authority_nonce: str
    authority_generation: int
    operation_id: str | None
    selection_sha256: str | None
    closure_sha256: str | None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 3
            or not isinstance(self.org_id, str)
            or not self.org_id.strip()
            or not isinstance(self.project_id, str)
            or not _hex(self.nonce, _NONCE)
            or not _version(self.revision)
            or not _hex(self.authority_nonce, _NONCE)
            or not _version(self.authority_generation)
        ):
            raise LearningTransitionIntegrityError("Invalid learning project witness")
        if self.operation_id is not None:
            _operation(self.operation_id)
        _selection(self.selection_sha256)
        _selection(self.closure_sha256)
        if self.selection_sha256 is not None and (
            self.operation_id is None or self.closure_sha256 is not None
        ):
            raise LearningTransitionIntegrityError(
                "Invalid project grant/closure binding"
            )

    @classmethod
    def from_dict(
        cls, raw: Any, *, org_id: str, project_id: str
    ) -> LearningTransitionState:
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise LearningTransitionIntegrityError(
                "Invalid learning project witness fields"
            )
        result = cls(**raw)
        if result.org_id != org_id or result.project_id != project_id:
            raise LearningTransitionIntegrityError("Learning transition scope mismatch")
        return result


@dataclass(frozen=True)
class LearningTransitionAdmission(_Fingerprint):
    authority: LearningPolicyAuthority
    project: LearningTransitionState | None

    @property
    def mode(self) -> str:
        return self.authority.phase

    @property
    def policy_floor(self) -> str:
        return self.authority.active_policy

    @property
    def target_policy(self) -> str | None:
        return self.authority.target_policy

    @property
    def operation_id(self) -> str | None:
        return self.authority.operation_id

    @property
    def selection_sha256(self) -> str | None:
        return self.project.selection_sha256 if self.project else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority.to_dict(),
            "project": self.project.to_dict() if self.project else None,
        }


def require_public_operation_key(key: str) -> None:
    if key.startswith(LEARNING_TRANSITION_PREFIX):
        raise LearningTransitionIntegrityError("Reserved learning transition namespace")


def learning_transition_key(org_id: str, project_id: str) -> str:
    scope = json.dumps([org_id, project_id], separators=(",", ":"), ensure_ascii=False)
    return LEARNING_TRANSITION_PREFIX + hashlib.sha256(scope.encode()).hexdigest()


def _read_authority(db: StreamSQL) -> LearningPolicyAuthority | None:
    raw = db.read_learning_policy_authority()
    return None if raw is None else LearningPolicyAuthority.from_dict(raw)


def _read(
    db: StreamSQL, key: str, org: str, project: str
) -> LearningTransitionState | None:
    rows = db.query(
        f"SELECT r.operation_state FROM {db.table('_operation_state')} r WHERE r.service_name=? AND {db.project_filter()} LIMIT 1",
        (key, project),
    )
    if not rows:
        return None
    try:
        raw = decoded(rows[0]["operation_state"])
    except (ValueError, TypeError) as error:
        raise LearningTransitionIntegrityError(
            "Invalid learning project witness encoding"
        ) from error
    return LearningTransitionState.from_dict(raw, org_id=org, project_id=project)


def _write(
    db: StreamSQL, key: str, state: LearningTransitionState, *, missing: bool
) -> None:
    payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    with db.learning_transition_write_protocol():
        if missing:
            db.insert_learning_transition(key, state.project_id, payload)
        else:
            rows = db.query(
                f"UPDATE {db.table('_operation_state')} AS r SET operation_state=?,updated_at=? WHERE r.service_name=? AND {db.project_filter()} RETURNING service_name",
                (payload, datetime.now(UTC).isoformat(), key, state.project_id),
            )
            if len(rows) != 1:
                raise LearningTransitionConflictError(
                    "Learning project witness disappeared"
                )


def _require_authority(db: StreamSQL) -> LearningPolicyAuthority:
    state = _read_authority(db)
    if state is None:
        raise LearningTransitionBlockedError(
            "Dataset learning policy authority requires provisioning or qualified upgrade"
        )
    return state


def _expected_authority(
    state: LearningPolicyAuthority | None, nonce: str, generation: int
) -> None:
    if not _hex(nonce, _NONCE) or not _version(generation):
        raise LearningTransitionIntegrityError("Invalid authority expectation")
    if (
        state is None
        or state.authority_nonce != nonce
        or state.generation != generation
    ):
        raise LearningTransitionConflictError("Learning policy authority changed")


def _bound(state: LearningTransitionState, authority: LearningPolicyAuthority) -> bool:
    return (
        state.authority_nonce == authority.authority_nonce
        and state.authority_generation == authority.generation
        and state.operation_id == authority.operation_id
    )


def _provision_project_db(
    db: StreamSQL, authority: LearningPolicyAuthority, *, org_id: str, project_id: str
) -> LearningTransitionState:
    if (
        not isinstance(org_id, str)
        or not org_id.strip()
        or not isinstance(project_id, str)
    ):
        raise LearningTransitionIntegrityError("Invalid learning transition scope")
    project = db.learning_transition_project(project_id)
    key = learning_transition_key(org_id, project)
    db.lock_learning_transition(key)
    current = _read(db, key, org_id, project)
    if current is not None and current.authority_nonce != authority.authority_nonce:
        raise LearningTransitionIntegrityError(
            "Project witness belongs to another dataset authority"
        )
    if current is not None and (
        current.authority_generation > authority.generation
        or (
            current.authority_generation == authority.generation
            and current.operation_id != authority.operation_id
        )
    ):
        raise LearningTransitionIntegrityError(
            "Project witness contradicts dataset authority"
        )
    if current is not None and _bound(current, authority):
        return current
    state = LearningTransitionState(
        3,
        org_id,
        project,
        current.nonce if current else uuid.uuid4().hex,
        current.revision + 1 if current else 0,
        authority.authority_nonce,
        authority.generation,
        authority.operation_id,
        None,
        None,
    )
    _write(db, key, state, missing=current is None)
    return state


def fence_learning_transition_db(
    db: StreamSQL, *, org_id: str, project_id: str, selection_sha256: str | None = None
) -> LearningTransitionAdmission:
    """Join the caller writer after user ownership, before execution-row locks."""
    _selection(selection_sha256)
    db.lock_learning_policy_authority(exclusive=False)
    authority = _require_authority(db)
    if authority.active_policy != PROFILE_AUTHORING_POLICY:
        raise LearningTransitionBlockedError(
            "Runtime source does not match dataset policy authority"
        )
    if authority.phase == "armed":
        raise LearningTransitionBlockedError("Dataset policy transition is armed")
    state = _provision_project_db(db, authority, org_id=org_id, project_id=project_id)
    if authority.phase == "sealed" and (
        selection_sha256 is None or state.selection_sha256 != selection_sha256
    ):
        raise LearningTransitionBlockedError(
            "Dataset transition lacks the exact selected-work grant"
        )
    return LearningTransitionAdmission(authority, state)


def _change_learning_policy_authority_db(
    db: StreamSQL,
    *,
    action: Literal["seal", "arm", "activate", "abort"],
    operation_id: str,
    expected_nonce: str,
    expected_generation: int,
    target_policy: str | None = None,
) -> LearningPolicyAuthority:
    """Private lifecycle seam for the exclusive provisioning/release owner.

    The owner MUST derive full roster/closure on this same writer before arm or
    activate. This helper is not a public proof or a boolean-based readiness API.
    """
    _operation(operation_id)
    db.lock_learning_policy_authority(exclusive=True)
    current = _require_authority(db)
    _expected_authority(current, expected_nonce, expected_generation)
    if action == "seal":
        if current.phase != "open" or target_policy not in AUTHORING_POLICIES:
            raise LearningTransitionConflictError(
                "Policy authority cannot seal this transition"
            )
        state = replace(
            current,
            generation=current.generation + 1,
            phase="sealed",
            operation_id=operation_id,
            target_policy=target_policy,
        )
    else:
        if current.operation_id != operation_id:
            raise LearningTransitionConflictError("Policy authority operation changed")
        if action == "arm" and current.phase == "sealed":
            state = replace(current, generation=current.generation + 1, phase="armed")
        elif (
            action == "activate"
            and current.phase == "armed"
            and current.target_policy == PROFILE_AUTHORING_POLICY
        ):
            state = replace(
                current,
                generation=current.generation + 1,
                phase="open",
                active_policy=PROFILE_AUTHORING_POLICY,
                target_policy=None,
                operation_id=None,
                bootstrap=False,
            )
        elif action == "abort" and current.phase == "sealed" and not current.bootstrap:
            state = replace(
                current,
                generation=current.generation + 1,
                phase="open",
                target_policy=None,
                operation_id=None,
            )
        else:
            raise LearningTransitionConflictError(
                "Policy authority lifecycle refuses transition"
            )
    db.write_learning_policy_authority(
        json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":")),
        missing=False,
    )
    return state


class LearningTransitionStore:
    org_id: str
    in_commit_scope: Any
    commit_scope: Any
    _stream_sql: Any
    _independent_writer_metadata_sql: Any
    _learning_policy_authority_sql: Any

    def read_learning_policy_authority(self) -> LearningPolicyAuthority | None:
        with self._stream_sql(read_only=True) as db:
            return _read_authority(db)

    def read_learning_transition(self) -> LearningTransitionAdmission | None:
        with self._stream_sql(read_only=True) as db:
            authority = _read_authority(db)
            if authority is None:
                return None
            project = db.learning_transition_project()
            return LearningTransitionAdmission(
                authority,
                _read(
                    db,
                    learning_transition_key(self.org_id, project),
                    self.org_id,
                    project,
                ),
            )

    def fence_learning_transition(
        self, *, selection_sha256: str | None = None
    ) -> LearningTransitionAdmission:
        if not self.in_commit_scope:
            raise LearningTransitionIntegrityError(
                "Learning transition fence requires a commit scope"
            )
        with self._stream_sql() as db:
            return fence_learning_transition_db(
                db,
                org_id=self.org_id,
                project_id=db.learning_transition_project(),
                selection_sha256=selection_sha256,
            )

    def provision_learning_scope(self) -> LearningTransitionAdmission:
        with self.commit_scope(), self._stream_sql() as db:
            db.lock_learning_policy_authority(exclusive=False)
            authority = _require_authority(db)
            if (
                authority.phase == "open"
                and authority.active_policy != PROFILE_AUTHORING_POLICY
            ):
                raise LearningTransitionBlockedError(
                    "Runtime source cannot provision another active policy"
                )
            return LearningTransitionAdmission(
                authority,
                _provision_project_db(
                    db,
                    authority,
                    org_id=self.org_id,
                    project_id=db.learning_transition_project(),
                ),
            )

    @contextmanager
    def _learning_policy_authority_scope(
        self,
    ) -> Iterator[tuple[StreamSQL, LearningPolicyAuthority | None]]:
        with self._learning_policy_authority_sql() as db:
            db.lock_learning_policy_authority(exclusive=True)
            yield db, _read_authority(db)

    def grant_learning_transition(
        self,
        operation_id: str,
        *,
        expected_authority_nonce: str,
        expected_authority_generation: int,
        expected_nonce: str | None,
        expected_revision: int | None,
        selection_sha256: str | None,
    ) -> LearningTransitionAdmission:
        _operation(operation_id)
        _selection(selection_sha256)
        with self.commit_scope(), self._stream_sql() as db:
            db.lock_learning_policy_authority(exclusive=False)
            authority = _require_authority(db)
            _expected_authority(
                authority, expected_authority_nonce, expected_authority_generation
            )
            if authority.phase != "sealed" or authority.operation_id != operation_id:
                raise LearningTransitionConflictError(
                    "Dataset policy authority is not draining this operation"
                )
            project = db.learning_transition_project()
            key = learning_transition_key(self.org_id, project)
            db.lock_learning_transition(key)
            current = _read(db, key, self.org_id, project)
            if expected_nonce is None and expected_revision is None:
                if current is not None:
                    raise LearningTransitionConflictError(
                        "Learning project witness already exists"
                    )
            else:
                if not _hex(expected_nonce, _NONCE) or not _version(expected_revision):
                    raise LearningTransitionIntegrityError(
                        "Invalid project witness expectation"
                    )
                if (
                    current is None
                    or current.nonce != expected_nonce
                    or current.revision != expected_revision
                ):
                    raise LearningTransitionConflictError(
                        "Learning project witness changed"
                    )
            if (
                current is not None
                and current.authority_nonce != authority.authority_nonce
            ):
                raise LearningTransitionIntegrityError(
                    "Project witness belongs to another dataset authority"
                )
            if current is not None and (
                current.authority_generation > authority.generation
                or (
                    current.authority_generation == authority.generation
                    and current.operation_id != authority.operation_id
                )
            ):
                raise LearningTransitionIntegrityError(
                    "Project witness contradicts dataset authority"
                )
            state = LearningTransitionState(
                3,
                self.org_id,
                project,
                current.nonce if current else uuid.uuid4().hex,
                current.revision + 1 if current else 0,
                authority.authority_nonce,
                authority.generation,
                operation_id,
                selection_sha256,
                None,
            )
            _write(db, key, state, missing=current is None)
            return LearningTransitionAdmission(authority, state)

    def read_learning_transition_closure(
        self,
    ) -> tuple[LearningTransitionAdmission, ProjectTransitionClosure]:
        from .transition_closure import derive_project_transition_closure_db

        with self._independent_writer_metadata_sql() as db:
            db.lock_learning_policy_authority(exclusive=False)
            authority = _require_authority(db)
            if authority.phase != "armed":
                raise LearningTransitionBlockedError(
                    "Dataset policy transition is not armed"
                )
            project = db.learning_transition_project()
            key = learning_transition_key(self.org_id, project)
            before = _read(db, key, self.org_id, project)
            if before is not None and (
                not _bound(before, authority) or before.selection_sha256 is not None
            ):
                raise LearningTransitionBlockedError(
                    "Project closure witness is stale or retains a selection grant"
                )
            closure = derive_project_transition_closure_db(
                db, org_id=self.org_id, project_id=project
            )
            if (
                _read_authority(db) != authority
                or _read(db, key, self.org_id, project) != before
            ):
                raise LearningTransitionConflictError(
                    "Learning transition changed during inspection"
                )
            return LearningTransitionAdmission(authority, before), closure


def _record_learning_transition_closure_db(
    db: StreamSQL,
    *,
    authority: LearningPolicyAuthority,
    org_id: str,
    project_id: str,
) -> tuple[LearningTransitionState, ProjectTransitionClosure]:
    """Derive one project witness on the exclusive root owner's same writer.

    The Enterprise owner must additionally prove the complete routed roster and
    reconcile org-level shared work. A local closure is never a fleet receipt.
    """
    from .transition_closure import derive_project_transition_closure_db

    if (
        authority.phase not in ("sealed", "armed")
        or _require_authority(db) != authority
    ):
        raise LearningTransitionConflictError("Closure authority is not held")
    state = _provision_project_db(db, authority, org_id=org_id, project_id=project_id)
    if state.selection_sha256 is not None:
        raise LearningTransitionBlockedError("Project selection grant must be revoked")
    closure = derive_project_transition_closure_db(
        db, org_id=org_id, project_id=project_id
    )
    if not closure.ready or closure.blockers:
        raise LearningTransitionBlockedError("Project execution closure is incomplete")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "authority": authority.fingerprint(),
                "org_id": org_id,
                "project_id": project_id,
                "counts": closure.counts,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    result = replace(state, revision=state.revision + 1, closure_sha256=fingerprint)
    _write(db, learning_transition_key(org_id, project_id), result, missing=False)
    return result, closure
