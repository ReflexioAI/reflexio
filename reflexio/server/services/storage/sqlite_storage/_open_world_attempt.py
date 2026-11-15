"""The open-world attempt store on SQLite: authority, invocations, context,
artifacts and the session compaction checkpoint.

The ``OpenWorldStoragePort`` members an attempt runs through after its bundle
is frozen (``OpenWorldAttemptStorage`` and ``session_compaction_checkpoint``).
Each call runs in one ``BEGIN IMMEDIATE`` transaction and first rechecks what
authorizes it: the job's live lease and fence, the target's current version
and the frozen governance subjects. A replay of something already stored is
checked the same way before it returns the stored row.

The semantics mirror the PostgreSQL implementation of the same port, whose
checks run in stored functions; this module restates them in Python, with the
same refusal codes and messages, and ``_open_world_rule_evidence`` restates the
artifact checks. The shared contract
(``reflexio.test_support.open_world_attempt_contract``) holds both to the same
answers. What SQLite answers differently, and why:

* **Context bytes.** ``context_json`` is canonical JSON. PostgreSQL stores the
  ``jsonb`` text rendering. Both store the SHA-256 of the bytes they hold, and
  every reader compares the parsed content, never the bytes across backends.
* **Roles.** The online ``confirmation`` analyst has no SQLite counterpart, so
  the invocation table does not admit that role; a prepare naming it is
  refused at the role check, as PostgreSQL refuses it without a confirmation
  qualification.
* **Trigger source roles.** V2 gate/builder calls have separate method-bound
  identities; they carry no invented manifest session. Their request bytes bind
  the prior admitted Content result and exact source frame.
* **Authority shape.** Only the rule-evidence method authority is admitted:
  the table rejects the retired held-out authority shape.
* **Projects.** A SQLite database is one project, so the compaction reuse read
  takes a complete checkpoint of the same identity from any job.
"""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from typing import Any, NoReturn

from pydantic import ValidationError

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES,
    OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA,
    RULE_EVIDENCE_POLICY_ID,
)
from reflexio.server.services.offline_tuner.open_world_freeze_validation import (
    canonical_artifact_payload,
    freeze_validation_identity,
    validate_artifact_request,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    CompletionRecord,
    ExpectedProvenance,
    OpenWorldInvocationBinding,
    OpenWorldRefusal,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND,
    OPEN_WORLD_CANDIDATE_ARTIFACT_KIND,
    OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldAttemptContextRow,
    OpenWorldInvocationConflictError,
    OpenWorldInvocationLeaseConflictError,
    OpenWorldInvocationRow,
    OpenWorldQualificationAuthorityRow,
)
from reflexio.server.services.storage.error import OptimizationArtifactIntegrityError

from . import _open_world_rule_evidence as rule
from ._base import SQLiteStorageBase
from ._open_world_trigger import (
    AUTHORITY_V2,
    POLICY_V2,
    SQLiteTriggerV2Mixin,
)
from .playbook._optimization import _row_to_playbook_optimization_artifact

#: The tables this module owns; ``rule_evidence_method_ready`` probes them.
ATTEMPT_TABLES = (
    "offline_tuner_open_world_qualification_authority",
    "offline_tuner_open_world_attempt_context",
    "offline_tuner_open_world_invocations",
    "offline_tuner_session_compactions",
)

_AUTHORITY_SCHEMA = "offline-tuner-rule-evidence-authority-v1"
_COMPONENT_IDENTITY_SCHEMA = "offline-tuner-open-world-component-identity-v1"
_QUALIFICATION_CHECKS = {
    "exact_edit": True,
    "abstention": True,
    "invalid_reference_refused": True,
    "invalid_clause_refused": True,
    "global_edit_without_local_gap": True,
}
_MANIFEST_ROLES = ("discovery", "held_out", "safety_only")
_INVOCATION_IDENTITY_FIELDS = (
    "manifest_role",
    "user_id",
    "session_id",
    "qualification_result_digest",
    "trajectory_digest",
    "candidate_artifact_digest",
    "candidate_analyst_input_identity",
)


def _refuse(refusal: OpenWorldRefusal, message: str) -> NoReturn:
    rule.refuse(refusal, message)


def _blank(value: object) -> bool:
    """``COALESCE(btrim(value), '') = ''``: ``btrim`` strips spaces only."""
    return not isinstance(value, str) or value.strip(" ") == ""


def component_digest(identity: object) -> str:
    """``rule_evidence_component_digest``: a component identity's digest."""
    if not isinstance(identity, dict) or set(identity) != rule.COMPONENT_IDENTITY_KEYS:
        rule.integrity("rule-evidence component identity is invalid")
    return rule.text_digest(
        rule.canonical_text({**identity, "schema_version": _COMPONENT_IDENTITY_SCHEMA})
    )


def _authority_row(row: sqlite3.Row) -> OpenWorldQualificationAuthorityRow:
    return OpenWorldQualificationAuthorityRow(
        epoch=int(row["epoch"]),
        state=row["state"],
        authority_digest=str(row["authority_digest"]),
        discovery_component_identity_digest=row["discovery_component_identity_digest"],
        discovery_qualification_suite_digest=row[
            "discovery_qualification_suite_digest"
        ],
        discovery_qualification_result_digest=row[
            "discovery_qualification_result_digest"
        ],
        held_out_component_identity_digest=row["held_out_component_identity_digest"],
        held_out_qualification_suite_digest=row["held_out_qualification_suite_digest"],
        held_out_qualification_result_digest=row[
            "held_out_qualification_result_digest"
        ],
        candidate_generator_identity_digest=row["candidate_generator_identity_digest"],
        candidate_generator_authorization_digest=row[
            "candidate_generator_authorization_digest"
        ],
        unavailable_reason=row["unavailable_reason"],
        updated_at=int(row["updated_at"]),
        method_authority_json=row["method_authority_json"],
    )


def _invocation_row(row: sqlite3.Row | dict[str, Any]) -> OpenWorldInvocationRow:
    usage = row["usage_json"]
    return OpenWorldInvocationRow(
        job_id=int(row["job_id"]),
        analyst_input_identity=str(row["analyst_input_identity"]),
        state=row["state"],
        component_identity_digest=str(row["component_identity_digest"]),
        input_digest=str(row["input_digest"]),
        component_role=row["component_role"],
        manifest_role=row["manifest_role"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        qualification_result_digest=str(row["qualification_result_digest"]),
        trajectory_digest=row["trajectory_digest"],
        candidate_artifact_digest=row["candidate_artifact_digest"],
        candidate_analyst_input_identity=row["candidate_analyst_input_identity"],
        output_json=row["output_json"],
        output_digest=row["output_digest"],
        observed_provider=row["observed_provider"],
        observed_model_revision=row["observed_model_revision"],
        called_at=row["called_at"],
        latency_ms=row["latency_ms"],
        usage_json=None if usage is None else str(usage),
    )


def _context_row(row: sqlite3.Row) -> OpenWorldAttemptContextRow:
    return OpenWorldAttemptContextRow(
        job_id=int(row["job_id"]),
        bundle_digest=str(row["bundle_digest"]),
        outcome_contract_digest=str(row["outcome_contract_digest"]),
        context_json=str(row["context_json"]),
        context_digest=str(row["context_digest"]),
        created_at=int(row["created_at"]),
        qualification_authority_epoch=int(row["qualification_authority_epoch"]),
        qualification_authority_digest=str(row["qualification_authority_digest"]),
    )


def _usage_text(usage: object) -> str | None:
    """Usage as stored: canonical JSON, or ``None`` for JSON ``null``."""
    if usage is None:
        return None
    return rule.canonical_text(usage)


class SQLiteOpenWorldAttemptMixin(SQLiteTriggerV2Mixin):
    """``OpenWorldAttemptStorage`` and the compaction checkpoint on SQLite."""

    conn: sqlite3.Connection
    org_id: str
    _lock: Any
    commit_scope: Any
    _evidence_job_locked: Any
    _validate_evidence_target_locked: Any
    _validate_evidence_governance_locked: Any

    # ------------------------------------------------------------------
    # Qualification authority
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def rule_evidence_method_ready(self) -> bool:
        """True only when current tables and v2 checked vocabularies exist."""
        with self._lock:
            definitions = {
                row["name"]: row["sql"]
                for row in self.conn.execute(
                    "SELECT name,sql FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        return (
            set(ATTEMPT_TABLES) <= definitions.keys()
            and all(
                role in definitions.get("offline_tuner_open_world_invocations", "")
                for role in ("'trigger_gate'", "'trigger_builder'")
            )
            and all(
                kind in definitions.get("playbook_optimization_artifacts", "")
                for kind in (
                    "'open_world_content_result'",
                    "'open_world_trigger_result'",
                )
            )
        )

    @SQLiteStorageBase.handle_exceptions
    def reconcile_rule_evidence_authority(
        self, *, authority_json: str, authority_digest: str, now: int
    ) -> OpenWorldQualificationAuthorityRow:
        """Make the qualified rule-evidence method the current authority.

        An exact replay returns the current row unchanged; any other valid
        authority replaces it under the next epoch.

        Raises:
            OpenWorldQualificationAuthorityStaleError: The authority is not the
                canonical, self-consistent rule-evidence method authority.
        """
        value = self._validated_method_authority(authority_json, authority_digest)
        diagnostic, proposal = value["diagnostic"], value["proposal"]
        discovery = component_digest(diagnostic["component_identity"])
        candidate = component_digest(proposal["component_identity"])
        authorization = rule.json_digest(
            {
                "contract_type": "open_world_pinned_generator_authorization",
                "schema_version": "v1",
                "component_identity": {
                    **proposal["component_identity"],
                    "schema_version": _COMPONENT_IDENTITY_SCHEMA,
                },
            }
        )
        if authorization != rule.text(rule.get(proposal, "authorization_digest")):
            _refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "rule-evidence proposal authorization is invalid",
            )
        columns = {
            "authority_digest": authority_digest,
            "method_authority_json": authority_json,
            "discovery_component_identity_digest": discovery,
            "discovery_qualification_suite_digest": diagnostic["suite_digest"],
            "discovery_qualification_result_digest": diagnostic["result_digest"],
            "candidate_generator_identity_digest": candidate,
            "candidate_generator_authorization_digest": authorization,
        }
        with self._lock, self.commit_scope():
            existing = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_qualification_authority"
                " WHERE singleton = 1"
            ).fetchone()
            if (
                existing is not None
                and existing["state"] == "active"
                and all(existing[name] == value for name, value in columns.items())
                and existing["held_out_component_identity_digest"] is None
            ):
                return _authority_row(existing)
            epoch = 1 if existing is None else int(existing["epoch"]) + 1
            self.conn.execute(
                "INSERT OR REPLACE INTO offline_tuner_open_world_qualification_authority"
                " (singleton, epoch, state, authority_digest,"
                " discovery_component_identity_digest,"
                " discovery_qualification_suite_digest,"
                " discovery_qualification_result_digest,"
                " candidate_generator_identity_digest,"
                " candidate_generator_authorization_digest, unavailable_reason,"
                " updated_at, method_authority_json)"
                " VALUES (1, ?, 'active', ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    epoch,
                    authority_digest,
                    discovery,
                    diagnostic["suite_digest"],
                    diagnostic["result_digest"],
                    candidate,
                    authorization,
                    now,
                    authority_json,
                ),
            )
            row = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_qualification_authority"
                " WHERE singleton = 1"
            ).fetchone()
        return _authority_row(row)

    @staticmethod
    def _validated_method_authority(
        authority_json: str, authority_digest: str
    ) -> dict[str, Any]:
        """The decoded authority, or a stale-authority refusal."""

        def invalid() -> Any:
            _refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "rule-evidence qualification authority is invalid",
            )

        try:
            value = json.loads(authority_json)
        except json.JSONDecodeError:
            return invalid()
        diagnostic = rule.get(value, "diagnostic")
        proposal = rule.get(value, "proposal")
        receipt = rule.get(diagnostic, "qualification_receipt")
        try:
            canonical = rule.canonical_text(value)
            receipt_digest = rule.json_digest(receipt)
        except OptimizationArtifactIntegrityError:
            return invalid()
        if (
            not isinstance(value, dict)
            or canonical != authority_json
            or rule.json_digest(value) != authority_digest
            or (
                rule.text(rule.get(value, "schema_version")),
                rule.text(rule.get(value, "policy_id")),
            )
            not in {
                (_AUTHORITY_SCHEMA, RULE_EVIDENCE_POLICY_ID),
                (AUTHORITY_V2, POLICY_V2),
            }
            or set(value)
            != (
                {"schema_version", "policy_id", "diagnostic", "proposal"}
                | (
                    {"trigger_maintenance"}
                    if value.get("schema_version") == AUTHORITY_V2
                    else set()
                )
            )
            or rule.text_at(diagnostic, "component_identity", "component_role")
            != "discovery"
            or rule.text_at(proposal, "component_identity", "component_role")
            != "candidate"
            or not rule.is_hex64(rule.text(rule.get(diagnostic, "suite_digest")))
            or rule.text(rule.get(receipt, "suite_digest"))
            != rule.text(rule.get(diagnostic, "suite_digest"))
            or rule.get(receipt, "passed") is not True
            or rule.canonical_text(rule.get(receipt, "checks"))
            != rule.canonical_text(_QUALIFICATION_CHECKS)
            or set(receipt) - {"suite_digest", "checks", "passed"}
            or receipt_digest != rule.text(rule.get(diagnostic, "result_digest"))
        ):
            return invalid()
        if value["schema_version"] == AUTHORITY_V2:
            from ._open_world_trigger import validate_trigger_authority

            validate_trigger_authority(value)
        return value

    @SQLiteStorageBase.handle_exceptions
    def load_open_world_qualification_authority(
        self,
    ) -> OpenWorldQualificationAuthorityRow | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_qualification_authority"
                " WHERE singleton = 1"
            ).fetchone()
        return None if row is None else _authority_row(row)

    # ------------------------------------------------------------------
    # Invocations
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def prepare_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        expected: ExpectedProvenance,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow:
        """Persist one immutable prepared analyst-input intent, or replay it.

        Raises:
            OpenWorldInvocationIntegrityError: The identity is not canonical.
            OpenWorldInvocationLeaseConflictError: The lease is not current.
            OpenWorldInvocationIdentityBoundError: The identity is stored with
                another binding, another job's included.
            OpenWorldInvocationConflictError: Another identity holds this
                question's slot, or the binding does not match the frozen
                bundle, context or candidate.
            OpenWorldInvocationManifestError: The session is not in the frozen
                manifest.
        """
        payload = {**binding.payload(), **expected.payload()}
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(binding.job_id, fence=fence, now=now)
            if (
                not rule.is_hex64(payload["analyst_input_identity"])
                or not rule.is_hex64(payload["component_identity_digest"])
                or not rule.is_hex64(payload["input_digest"])
                or not rule.is_hex64(payload["qualification_result_digest"])
                or _blank(payload["expected_provider"])
                or _blank(payload["expected_model_revision"])
                or _blank(payload["component_role"])
            ):
                rule.integrity("canonical invocation identity is required")
            job = self._attempt_job_locked(binding.job_id)
            if job is None:
                rule.integrity("open-world optimizer job is required")
            self._require_live_lease(job, fence=fence, now=now)
            context = self._assert_attempt_authorized_locked(
                job, payload["bundle_digest"], payload["outcome_contract_digest"]
            )
            self._assert_role_identity(
                context,
                payload["component_role"],
                payload["component_identity_digest"],
                payload["qualification_suite_digest"],
                payload["qualification_result_digest"],
            )
            self._assert_prepared_slot_locked(job, context, payload)
            existing = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_invocations"
                " WHERE analyst_input_identity = ?",
                (payload["analyst_input_identity"],),
            ).fetchone()
            if existing is not None:
                if (
                    existing["job_id"] != job.job_id
                    or existing["component_identity_digest"]
                    != payload["component_identity_digest"]
                    or existing["expected_provider"] != payload["expected_provider"]
                    or existing["expected_model_revision"]
                    != payload["expected_model_revision"]
                    or existing["input_digest"] != payload["input_digest"]
                    or existing["component_role"] != payload["component_role"]
                    or any(
                        existing[name] != payload[name]
                        for name in _INVOCATION_IDENTITY_FIELDS
                    )
                ):
                    _refuse(
                        OpenWorldRefusal.IDENTITY_BOUND,
                        "analyst input identity is already bound",
                    )
                return _invocation_row(existing)
            try:
                self.conn.execute(
                    "INSERT INTO offline_tuner_open_world_invocations ("
                    " analyst_input_identity, job_id, component_identity_digest,"
                    " expected_provider, expected_model_revision, input_digest,"
                    " component_role, manifest_role, user_id, session_id,"
                    " qualification_result_digest, trajectory_digest,"
                    " candidate_artifact_digest, candidate_analyst_input_identity,"
                    " prepared_at) VALUES"
                    " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        payload["analyst_input_identity"],
                        job.job_id,
                        payload["component_identity_digest"],
                        payload["expected_provider"],
                        payload["expected_model_revision"],
                        payload["input_digest"],
                        payload["component_role"],
                        *(payload[name] for name in _INVOCATION_IDENTITY_FIELDS[:3]),
                        payload["qualification_result_digest"],
                        *(payload[name] for name in _INVOCATION_IDENTITY_FIELDS[4:]),
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE" in str(exc):
                    rule.conflict("logical invocation already bound")
                rule.integrity("invocation row violates its constraints")
            row = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_invocations"
                " WHERE analyst_input_identity = ?",
                (payload["analyst_input_identity"],),
            ).fetchone()
        return _invocation_row(row)

    def _assert_prepared_slot_locked(
        self,
        job: PlaybookOptimizationJob,
        context: sqlite3.Row,
        payload: dict[str, Any],
    ) -> None:
        """The per-role session and candidate binding a prepare must carry."""
        role, manifest_role = payload["component_role"], payload["manifest_role"]
        session_fields = (payload["user_id"], payload["session_id"])
        if role == "discovery" and manifest_role == "discovery":
            if (
                any(_blank(value) for value in session_fields)
                or not rule.is_hex64(payload["trajectory_digest"])
                or payload["candidate_artifact_digest"] is not None
                or payload["candidate_analyst_input_identity"] is not None
            ):
                rule.integrity("diagnostic source identity is incomplete")
            self._assert_manifest_session(context, manifest_role, payload)
        elif role == "held_out":
            if (
                manifest_role not in ("held_out", "safety_only")
                or any(_blank(value) for value in session_fields)
                or not rule.is_hex64(payload["trajectory_digest"])
                or not rule.is_hex64(payload["candidate_artifact_digest"])
                or not rule.is_hex64(payload["candidate_analyst_input_identity"])
            ):
                rule.integrity("manifest session identity is incomplete")
            self._assert_manifest_session(context, manifest_role, payload)
            self._assert_candidate_locked(
                job.job_id,
                payload["candidate_artifact_digest"],
                payload["candidate_analyst_input_identity"],
            )
        elif any(
            payload[name] is not None for name in _INVOCATION_IDENTITY_FIELDS[:3]
        ) or any(payload[name] is not None for name in _INVOCATION_IDENTITY_FIELDS[4:]):
            rule.integrity("manifest session identity is incomplete")

    @SQLiteStorageBase.handle_exceptions
    def complete_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        completion: CompletionRecord,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow:
        """Attach the one observed output and provenance to a prepared intent.

        A completed row is immutable: the same completion replays it, any
        other bytes or provenance conflict.
        """
        try:
            usage = (
                None
                if completion.usage_json is None
                else json.loads(completion.usage_json)
            )
        except json.JSONDecodeError as exc:
            raise OptimizationArtifactIntegrityError(
                "open-world invocation usage is not valid JSON"
            ) from exc
        payload = {**binding.payload(), **completion.payload()}
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(binding.job_id, fence=fence, now=now)
            output = payload["output_json"]
            if (
                not rule.is_hex64(payload["analyst_input_identity"])
                or not rule.is_hex64(payload["component_identity_digest"])
                or output is None
                or not rule.is_hex64(payload["output_digest"])
                or _blank(payload["observed_provider"])
                or _blank(payload["observed_model_revision"])
                or payload["called_at"] is None
                or payload["latency_ms"] is None
                or payload["latency_ms"] < 0
                or not rule.is_hex64(payload["qualification_result_digest"])
            ):
                rule.integrity("canonical invocation completion is required")
            if payload["output_digest"] != rule.text_digest(output):
                rule.integrity("output digest does not match output bytes")
            job = self._attempt_job_locked(binding.job_id)
            if job is None:
                rule.refuse(OpenWorldRefusal.LEASE_LOST, "lease is no longer current")
            self._require_live_lease(job, fence=fence, now=now)
            row = self._invocation_of_job_locked(binding)
            context = self._assert_attempt_authorized_locked(
                job, payload["bundle_digest"], payload["outcome_contract_digest"]
            )
            self._assert_role_identity(
                context,
                payload["component_role"],
                payload["component_identity_digest"],
                payload["qualification_suite_digest"],
                payload["qualification_result_digest"],
            )
            if (
                row["component_identity_digest"] != payload["component_identity_digest"]
                or row["component_role"] != payload["component_role"]
                or row["expected_provider"] != payload["observed_provider"]
                or row["expected_model_revision"] != payload["observed_model_revision"]
                or any(
                    row[name] != payload[name] for name in _INVOCATION_IDENTITY_FIELDS
                )
            ):
                rule.conflict("component identity does not match invocation")
            self._assert_stored_slot_locked(job, context, row)
            if row["state"] == "completed":
                if (
                    row["output_json"] != output
                    or row["output_digest"] != (payload["output_digest"])
                ):
                    rule.conflict("output bytes conflict")
                stored_usage = (
                    None if row["usage_json"] is None else json.loads(row["usage_json"])
                )
                if (
                    row["observed_provider"] != payload["observed_provider"]
                    or row["observed_model_revision"]
                    != payload["observed_model_revision"]
                    or row["called_at"] != payload["called_at"]
                    or row["latency_ms"] != payload["latency_ms"]
                    or stored_usage != usage
                ):
                    rule.conflict("observed provenance conflict")
                return _invocation_row(row)
            self.conn.execute(
                "UPDATE offline_tuner_open_world_invocations"
                " SET state = 'completed', output_json = ?, output_digest = ?,"
                " observed_provider = ?, observed_model_revision = ?,"
                " called_at = ?, latency_ms = ?, usage_json = ?, completed_at = ?"
                " WHERE analyst_input_identity = ?",
                (
                    output,
                    payload["output_digest"],
                    payload["observed_provider"],
                    payload["observed_model_revision"],
                    payload["called_at"],
                    payload["latency_ms"],
                    _usage_text(usage),
                    now,
                    payload["analyst_input_identity"],
                ),
            )
            stored = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_invocations"
                " WHERE analyst_input_identity = ?",
                (payload["analyst_input_identity"],),
            ).fetchone()
        return _invocation_row(stored)

    @SQLiteStorageBase.handle_exceptions
    def load_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow:
        """Load one invocation after rechecking every binding it was made under."""
        payload = binding.payload()
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(
                binding.job_id, fence=fence, now=now, write=False
            )
            job = self._attempt_job_locked(binding.job_id)
            if job is None:
                rule.refuse(OpenWorldRefusal.LEASE_LOST, "lease is no longer current")
            self._require_live_lease(job, fence=fence, now=now)
            row = self._invocation_of_job_locked(binding)
            context = self._assert_attempt_authorized_locked(
                job, payload["bundle_digest"], payload["outcome_contract_digest"]
            )
            self._assert_role_identity(
                context,
                payload["component_role"],
                payload["component_identity_digest"],
                payload["qualification_suite_digest"],
                payload["qualification_result_digest"],
            )
            if (
                row["component_identity_digest"] != payload["component_identity_digest"]
                or row["component_role"] != payload["component_role"]
                or any(
                    row[name] != payload[name] for name in _INVOCATION_IDENTITY_FIELDS
                )
            ):
                rule.conflict("component identity does not match invocation")
            self._assert_stored_slot_locked(job, context, row)
        return _invocation_row(row)

    def _invocation_of_job_locked(
        self, binding: OpenWorldInvocationBinding
    ) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_invocations"
            " WHERE analyst_input_identity = ?",
            (binding.analyst_input_identity,),
        ).fetchone()
        if row is None or row["job_id"] != binding.job_id:
            rule.refuse(OpenWorldRefusal.LEASE_LOST, "lease is no longer current")
        return row

    def _assert_stored_slot_locked(
        self, job: PlaybookOptimizationJob, context: sqlite3.Row, row: sqlite3.Row
    ) -> None:
        if row["component_role"] == "discovery" and row["manifest_role"] == "discovery":
            self._assert_manifest_session(context, row["manifest_role"], dict(row))
        elif row["component_role"] == "held_out":
            self._assert_manifest_session(context, row["manifest_role"], dict(row))
            self._assert_candidate_locked(
                job.job_id,
                row["candidate_artifact_digest"],
                row["candidate_analyst_input_identity"],
            )

    # ------------------------------------------------------------------
    # Attempt context
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def bind_open_world_attempt_context(
        self,
        *,
        job_id: int,
        bundle_digest: str,
        outcome_contract_digest: str,
        authority_epoch: int,
        authority_digest: str,
        fence: int,
        now: int,
    ) -> OpenWorldAttemptContextRow:
        """Bind this attempt's one immutable authorization context.

        The manifest (every frozen session's trajectory digest) is derived from
        the stored bundle, not accepted. An exact replay returns the stored
        row; any drift -- another bundle, contract or authority -- raises.
        """
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(job_id, fence=fence, now=now)
            if (
                not rule.is_hex64(bundle_digest)
                or not rule.is_hex64(outcome_contract_digest)
                or not isinstance(authority_epoch, int)
                or authority_epoch <= 0
                or not rule.is_hex64(authority_digest)
            ):
                rule.integrity("canonical attempt context identity is required")
            job = self._attempt_job_locked(job_id)
            if job is None:
                rule.integrity("open-world optimizer job is required")
            self._require_live_lease(job, fence=fence, now=now)
            authority = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_qualification_authority"
                " WHERE singleton = 1"
            ).fetchone()
            if (
                authority is None
                or authority["state"] != "active"
                or authority["epoch"] != authority_epoch
                or authority["authority_digest"] != authority_digest
            ):
                _refuse(
                    OpenWorldRefusal.AUTHORITY_STALE,
                    "current qualification authority is missing, unavailable, or stale",
                )
            bundle = self._bundle_of_locked(
                job,
                bundle_digest,
                "attempt context does not bind the exact frozen evidence bundle",
            )
            if rule.text_at(bundle, "target", "success_definition_digest") != (
                outcome_contract_digest
            ):
                rule.conflict(
                    "attempt context does not bind the exact frozen outcome contract"
                )
            manifest = sorted(
                (
                    {
                        "manifest_role": role,
                        "user_id": rule.text(rule.get(session, "user_id")),
                        "session_id": rule.text(rule.get(session, "session_id")),
                        "trajectory_digest": rule.text(
                            rule.get(session, "trajectory_digest")
                        ),
                    }
                    for role in _MANIFEST_ROLES
                    for session in _sessions(rule.path(bundle, role, "sessions"))
                ),
                key=lambda entry: (
                    entry["manifest_role"],
                    entry["user_id"] or "",
                    entry["session_id"] or "",
                ),
            )
            context = {
                "contract_type": "open_world_attempt_context",
                "schema_version": "v2",
                "bundle_digest": bundle_digest,
                "outcome_contract_digest": outcome_contract_digest,
                "manifest": manifest,
                "qualification_authority": {
                    "epoch": authority["epoch"],
                    "authority_digest": authority["authority_digest"],
                },
                "discovery": {
                    "component_identity_digest": authority[
                        "discovery_component_identity_digest"
                    ],
                    "qualification_suite_digest": authority[
                        "discovery_qualification_suite_digest"
                    ],
                    "qualification_result_digest": authority[
                        "discovery_qualification_result_digest"
                    ],
                },
                "held_out": {
                    "component_identity_digest": authority[
                        "held_out_component_identity_digest"
                    ],
                    "qualification_suite_digest": authority[
                        "held_out_qualification_suite_digest"
                    ],
                    "qualification_result_digest": authority[
                        "held_out_qualification_result_digest"
                    ],
                },
                "candidate": {
                    "generator_identity_digest": authority[
                        "candidate_generator_identity_digest"
                    ],
                    "authorization_digest": authority[
                        "candidate_generator_authorization_digest"
                    ],
                },
            }
            method_value = json.loads(authority["method_authority_json"])
            if method_value.get("schema_version") == AUTHORITY_V2:
                for role, name in (
                    ("trigger_gate", "gate"),
                    ("trigger_builder", "builder"),
                ):
                    stage = method_value["trigger_maintenance"][name]
                    context[role] = {
                        "component_identity_digest": component_digest(
                            stage["component_identity"]
                        ),
                        "qualification_suite_digest": None,
                        "qualification_result_digest": stage["authorization_digest"],
                    }
            context_json = rule.canonical_text(context)
            context_digest = rule.text_digest(context_json)
            existing = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_attempt_context WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing is not None:
                if existing["context_digest"] != context_digest:
                    rule.conflict(
                        "durable attempt context conflicts with a different "
                        "authorization under current qualification authority"
                    )
                return _context_row(existing)
            self.conn.execute(
                "INSERT INTO offline_tuner_open_world_attempt_context ("
                " job_id, bundle_digest, outcome_contract_digest, manifest_json,"
                " discovery_component_identity_digest,"
                " discovery_qualification_suite_digest,"
                " discovery_qualification_result_digest,"
                " held_out_component_identity_digest,"
                " held_out_qualification_suite_digest,"
                " held_out_qualification_result_digest,"
                " candidate_generator_identity_digest,"
                " candidate_generator_authorization_digest,"
                " qualification_authority_epoch, qualification_authority_digest,"
                " context_json, context_digest, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    bundle_digest,
                    outcome_contract_digest,
                    rule.canonical_text(manifest),
                    authority["discovery_component_identity_digest"],
                    authority["discovery_qualification_suite_digest"],
                    authority["discovery_qualification_result_digest"],
                    authority["held_out_component_identity_digest"],
                    authority["held_out_qualification_suite_digest"],
                    authority["held_out_qualification_result_digest"],
                    authority["candidate_generator_identity_digest"],
                    authority["candidate_generator_authorization_digest"],
                    authority["epoch"],
                    authority["authority_digest"],
                    context_json,
                    context_digest,
                    now,
                ),
            )
            row = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_attempt_context WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _context_row(row)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def write_open_world_artifact(
        self,
        *,
        job_id: int,
        artifact_kind: str,
        content_json: str,
        content_digest: str,
        fence: int,
        now: int,
    ) -> PlaybookOptimizationArtifact:
        """Persist the one artifact of ``artifact_kind`` for the leased job.

        The content is authorized against the completed invocations and the
        frozen bundle before anything is compared or written; a stored
        artifact is never rewritten, and a replay of its exact bytes returns it.
        """
        payload, canonical = canonical_artifact_payload(content_json)
        if sha256(canonical.encode()).hexdigest() != content_digest:
            raise OptimizationArtifactIntegrityError(
                "open-world artifact content/digest mismatch"
            )
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(job_id, fence=fence, now=now)
            self._authorize_artifact_locked(job_id, artifact_kind, payload)
            existing = self._artifact_of_locked(job_id, artifact_kind)
            if existing is not None:
                try:
                    validate_artifact_request(
                        existing, content_json=canonical, content_digest=content_digest
                    )
                except OptimizationArtifactIntegrityError as exc:
                    raise OpenWorldInvocationConflictError(str(exc)) from exc
                return existing
            job = self._attempt_job_locked(job_id)
            if (
                job is None
                or job.status not in ("pending", "running")
                or job.lease_fence != fence
                or job.lease_expires_at is None
                or job.lease_expires_at <= now
            ):
                rule.refuse(
                    OpenWorldRefusal.LEASE_LOST,
                    "optimizer job lease is no longer current",
                )
            try:
                cursor = self.conn.execute(
                    "INSERT INTO playbook_optimization_artifacts"
                    " (job_id, artifact_kind, content_json, content_digest,"
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (job_id, artifact_kind, canonical, content_digest, now, now),
                )
            except sqlite3.IntegrityError:
                rule.integrity("optimizer artifact violates its constraints")
            row = self.conn.execute(
                "SELECT * FROM playbook_optimization_artifacts WHERE artifact_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return _row_to_playbook_optimization_artifact(row)

    @SQLiteStorageBase.handle_exceptions
    def read_open_world_artifact(
        self, *, job_id: int, artifact_kind: str, fence: int, now: int
    ) -> PlaybookOptimizationArtifact | None:
        """Read one artifact after rechecking the lease and its own authority.

        The decision of an abstained job stays readable after its lease ends;
        every other read needs the live lease. The stored bytes are checked
        against their digest (by the row model) and re-authorized before they
        are returned.
        """
        with self._lock, self.commit_scope():
            job = self._revalidate_attempt_job_locked(
                job_id,
                fence=fence,
                now=now,
                write=False,
                allow_terminal_abstain=(
                    artifact_kind == OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND
                ),
            )
            # The row model rejects a stored digest that does not name the
            # canonical content, so what is returned is the digested bytes.
            artifact = self._artifact_of_locked(job_id, artifact_kind)
            if artifact is None:
                return None
            payload, _canonical = canonical_artifact_payload(artifact.content_json)
            _assert_terminal_decision_matches_job(job, artifact_kind, payload)
            self._authorize_artifact_locked(job_id, artifact_kind, payload)
        return artifact

    def _artifact_of_locked(
        self, job_id: int, artifact_kind: str
    ) -> PlaybookOptimizationArtifact | None:
        row = self.conn.execute(
            "SELECT * FROM playbook_optimization_artifacts"
            " WHERE job_id = ? AND artifact_kind = ?",
            (job_id, artifact_kind),
        ).fetchone()
        return None if row is None else _row_to_playbook_optimization_artifact(row)

    # ------------------------------------------------------------------
    # Session compaction checkpoint
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def session_compaction_checkpoint(
        self,
        *,
        job_id: int,
        identity: str,
        fence: int,
        now: int,
        receipt: dict[str, Any] | None = None,
        protocol: str | None = None,
        expected_digest: str | None = None,
    ) -> dict[str, Any] | None:
        """Read or advance a bounded checkpoint under the live job lease.

        The shared protocol is compare-and-swap: a write must name the digest
        of the stored record (``None`` for none), and advances only along
        ``validate_advance``. The rolling protocol never regresses its batch
        count and is immutable once complete. Every record is at most
        ``ROW_BYTES`` of canonical JSON.
        """
        from reflexio.server.services.offline_tuner.evidence.compaction import (
            CompactionReceipt,
        )
        from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
            PROTOCOL,
            ROW_BYTES,
            SharedReceipt,
            validate_advance,
        )

        if protocol not in (None, PROTOCOL):
            raise OptimizationArtifactIntegrityError(
                "unknown compaction checkpoint protocol"
            )
        record_type = SharedReceipt if protocol == PROTOCOL else CompactionReceipt
        if not rule.is_hex64(identity):
            raise OptimizationArtifactIntegrityError("invalid compaction identity")
        if receipt is not None:
            try:
                receipt = record_type.model_validate(receipt).model_dump(
                    mode="json", exclude_none=True
                )
            except ValidationError as exc:
                raise OptimizationArtifactIntegrityError(
                    "invalid compaction receipt"
                ) from exc
            if receipt["identity"] != identity:
                raise OptimizationArtifactIntegrityError(
                    "compaction receipt identity mismatch"
                )
        content = (
            canonical_json_bytes(receipt).decode() if receipt is not None else None
        )
        if content is not None and len(content.encode()) > ROW_BYTES:
            raise OptimizationArtifactIntegrityError("compaction receipt exceeds bound")
        with self._lock, self.commit_scope():
            self._revalidate_attempt_job_locked(
                job_id,
                fence=fence,
                now=now,
                write=receipt is not None or protocol == PROTOCOL,
            )
            existing = self.conn.execute(
                "SELECT content_json, content_digest, batches, complete"
                " FROM offline_tuner_session_compactions"
                " WHERE job_id = ? AND compaction_identity = ?",
                (job_id, identity),
            ).fetchone()
            if existing is None and receipt is None and protocol is None:
                # A complete general compaction is reusable by another job;
                # the compactor checks its full source identity on rehydration.
                existing = self.conn.execute(
                    "SELECT content_json, content_digest, batches, complete"
                    " FROM offline_tuner_session_compactions"
                    " WHERE compaction_identity = ? AND complete = 1"
                    " ORDER BY job_id DESC LIMIT 1",
                    (identity,),
                ).fetchone()
            if (
                protocol == PROTOCOL
                and receipt is not None
                and expected_digest
                != (existing["content_digest"] if existing is not None else None)
            ):
                raise OpenWorldInvocationConflictError(
                    "shared checkpoint compare-and-swap failed"
                )
            if existing is not None:
                if (
                    sha256(existing["content_json"].encode()).hexdigest()
                    != (existing["content_digest"])
                ):
                    raise OptimizationArtifactIntegrityError(
                        "compaction checkpoint digest mismatch"
                    )
                try:
                    stored = record_type.model_validate_json(
                        existing["content_json"]
                    ).model_dump(mode="json", exclude_none=True)
                except ValidationError as exc:
                    raise OptimizationArtifactIntegrityError(
                        "invalid stored compaction receipt"
                    ) from exc
                if (
                    stored["identity"] != identity
                    or stored["batches"] != existing["batches"]
                    or stored["complete"] != bool(existing["complete"])
                ):
                    raise OptimizationArtifactIntegrityError(
                        "stored compaction receipt column mismatch"
                    )
                if receipt is None:
                    return stored
                if protocol == PROTOCOL:
                    try:
                        validate_advance(
                            SharedReceipt.model_validate(stored),
                            SharedReceipt.model_validate(receipt),
                        )
                    except ValueError as exc:
                        raise OpenWorldInvocationConflictError(
                            "invalid shared checkpoint transition"
                        ) from exc
                else:
                    if (
                        receipt["batches"] == stored["batches"]
                        and receipt["compaction"] != stored["compaction"]
                    ):
                        raise OpenWorldInvocationConflictError(
                            "compaction finalization changed checkpoint evidence"
                        )
                    if (
                        existing["complete"]
                        or receipt["batches"] < existing["batches"]
                        or (
                            receipt["batches"] == existing["batches"]
                            and not receipt["complete"]
                        )
                    ):
                        if content != existing["content_json"]:
                            raise OpenWorldInvocationConflictError(
                                "compaction checkpoint cannot regress or change "
                                "after completion"
                            )
                        return receipt
            if receipt is not None and content is not None:
                self.conn.execute(
                    "INSERT INTO offline_tuner_session_compactions (job_id,"
                    " compaction_identity, batches, complete, content_json,"
                    " content_digest) VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT (job_id, compaction_identity) DO UPDATE"
                    " SET batches = excluded.batches, complete = excluded.complete,"
                    " content_json = excluded.content_json,"
                    " content_digest = excluded.content_digest",
                    (
                        job_id,
                        identity,
                        receipt["batches"],
                        int(receipt["complete"]),
                        content,
                        sha256(content.encode()).hexdigest(),
                    ),
                )
            return receipt

    # ------------------------------------------------------------------
    # Revalidation; every helper runs under the lock, inside a transaction
    # ------------------------------------------------------------------

    def _attempt_job_locked(self, job_id: int) -> PlaybookOptimizationJob | None:
        job = self._evidence_job_locked(job_id)
        if job is None or job.optimizer_kind != OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND:
            return None
        return job

    @staticmethod
    def _require_live_lease(
        job: PlaybookOptimizationJob, *, fence: int, now: int
    ) -> None:
        if (
            job.status not in ("pending", "running")
            or job.lease_owner is None
            or job.lease_fence != fence
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        ):
            rule.refuse(OpenWorldRefusal.LEASE_LOST, "lease is no longer current")

    def _revalidate_attempt_job_locked(
        self,
        job_id: int,
        *,
        fence: int,
        now: int,
        write: bool = True,
        allow_terminal_abstain: bool = False,
    ) -> PlaybookOptimizationJob:
        """Recheck lease and fence, tenant, target version and governance.

        Every fenced read and write calls this first, replays included. An
        abstained job's decision stays readable (``allow_terminal_abstain``,
        reads only); nothing else passes without the live lease.
        """
        job = self._attempt_job_locked(job_id)
        if job is None:
            raise OptimizationArtifactIntegrityError(
                "open-world optimizer job does not exist"
            )
        lease_is_not_current = (
            job.status not in ("pending", "running")
            or job.lease_owner is None
            or job.lease_fence != fence
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        )
        terminal_abstain_is_readable = (
            allow_terminal_abstain
            and not write
            and job.status == "skipped"
            and job.stage == "abstained"
            and job.terminal_outcome in OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES
        )
        if lease_is_not_current and not terminal_abstain_is_readable:
            raise OpenWorldInvocationLeaseConflictError("lease is no longer current")
        fingerprint, subject_epochs = freeze_validation_identity(
            job, org_id=self.org_id
        )
        self._validate_evidence_target_locked(
            job.target_id, expected_fingerprint=fingerprint
        )
        self._validate_evidence_governance_locked(subject_epochs)
        return job

    def _bundle_of_locked(
        self, job: PlaybookOptimizationJob, bundle_digest: object, message: str
    ) -> Any:
        """The frozen bundle ``bundle_digest`` names for ``job``, decoded."""
        try:
            metadata = json.loads(job.metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if bundle_digest != rule.text_at(
            metadata, "open_world_evidence", "bundle_digest"
        ):
            rule.conflict(message)
        row = self.conn.execute(
            "SELECT content_json FROM playbook_optimization_artifacts"
            " WHERE job_id = ? AND artifact_kind = ? AND content_digest = ?",
            (job.job_id, OPEN_WORLD_EVIDENCE_ARTIFACT_KIND, bundle_digest),
        ).fetchone()
        if row is None:
            rule.conflict(message)
        return rule.decode(row["content_json"], message)

    def _assert_attempt_authorized_locked(
        self,
        job: PlaybookOptimizationJob,
        bundle_digest: object,
        outcome_contract_digest: object,
    ) -> sqlite3.Row:
        """``assert_offline_tuner_open_world_authorized``: the job's context."""
        if not rule.is_hex64(bundle_digest) or not rule.is_hex64(
            outcome_contract_digest
        ):
            rule.integrity("canonical invocation identity is required")
        bundle = self._bundle_of_locked(
            job,
            bundle_digest,
            "invocation does not bind the exact frozen evidence bundle",
        )
        if rule.text_at(bundle, "target", "success_definition_digest") != (
            outcome_contract_digest
        ):
            rule.conflict("invocation does not bind the exact frozen outcome contract")
        context = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_attempt_context WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        if context is None:
            _refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "invocation has no current qualification authority binding",
            )
        authority = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_qualification_authority"
            " WHERE singleton = 1"
        ).fetchone()
        if (
            authority is None
            or authority["state"] != "active"
            or authority["epoch"] != context["qualification_authority_epoch"]
            or authority["authority_digest"]
            != context["qualification_authority_digest"]
        ):
            _refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "invocation qualification authority is stale or unavailable",
            )
        context_json = rule.decode(
            context["context_json"],
            "invocation does not bind the exact durable attempt context",
        )
        if (
            context["bundle_digest"] != bundle_digest
            or context["outcome_contract_digest"] != outcome_contract_digest
            or json.loads(context["manifest_json"])
            != rule.get(context_json, "manifest")
            or context["context_digest"] != rule.text_digest(context["context_json"])
        ):
            rule.conflict("invocation does not bind the exact durable attempt context")
        return context

    def _assert_role_identity(
        self,
        context: sqlite3.Row,
        role: object,
        component_identity_digest: object,
        qualification_suite_digest: object,
        qualification_result_digest: object,
    ) -> None:
        """``assert_offline_tuner_open_world_role_identity``."""
        if role == "discovery":
            expected = (
                context["discovery_component_identity_digest"],
                context["discovery_qualification_suite_digest"],
                context["discovery_qualification_result_digest"],
            )
        elif role == "held_out":
            expected = (
                context["held_out_component_identity_digest"],
                context["held_out_qualification_suite_digest"],
                context["held_out_qualification_result_digest"],
            )
        elif role in ("trigger_gate", "trigger_builder"):
            stage = self._bound_trigger_component(context, str(role))
            expected = (
                component_digest(stage["component_identity"]),
                None,
                stage["authorization_digest"],
            )
        elif role == "candidate":
            expected = (
                context["candidate_generator_identity_digest"],
                None,
                context["candidate_generator_authorization_digest"],
            )
        else:
            rule.conflict("invocation does not bind the exact authorized analyst role")
        if (
            component_identity_digest,
            qualification_suite_digest,
            qualification_result_digest,
        ) != expected:
            rule.conflict(
                "invocation does not bind the exact authorized analyst qualification"
            )

    @staticmethod
    def _assert_manifest_session(
        context: sqlite3.Row, manifest_role: object, binding: dict[str, Any]
    ) -> None:
        """``assert_offline_tuner_open_world_manifest_session``."""
        entries = json.loads(context["manifest_json"])
        members = [
            entry
            for entry in entries
            if entry.get("manifest_role") == manifest_role
            and entry.get("user_id") == binding["user_id"]
            and entry.get("session_id") == binding["session_id"]
        ]
        if not members:
            rule.refuse(
                OpenWorldRefusal.MANIFEST_CONFLICT,
                "frozen manifest membership is required",
            )
        if not any(
            entry.get("trajectory_digest") == binding["trajectory_digest"]
            for entry in members
        ):
            rule.conflict("invocation does not bind the exact frozen trajectory")

    def _assert_candidate_locked(
        self, job_id: int, candidate_digest: object, candidate_identity: object
    ) -> None:
        """``assert_offline_tuner_open_world_candidate``."""
        candidate = self._artifact_of_locked(job_id, OPEN_WORLD_CANDIDATE_ARTIFACT_KIND)
        if candidate is None or candidate.content_digest != candidate_digest:
            rule.conflict("invocation does not bind the exact durable candidate")
        content = json.loads(candidate.content_json)
        if (
            rule.text_at(content, "payload", "invocation", "analyst_input_identity")
            != candidate_identity
        ):
            rule.conflict("invocation does not bind the exact durable candidate")

    # ------------------------------------------------------------------
    # Rule-evidence artifact authorization
    # ------------------------------------------------------------------

    def _assert_receipt_locked(
        self,
        *,
        job_id: int,
        bundle_digest: str,
        outcome_contract_digest: str,
        prepared: object,
        invocation: object,
        observed: object,
        role: str,
    ) -> sqlite3.Row:
        """``assert_rule_evidence_receipt``: a receipt is its durable invocation.

        Storage recomputes the component digest, the request digest and the
        analyst-input identity from the receipt's preimages, so a receipt whose
        identity was not derived from what was asked is refused.
        """
        binding = rule.get(prepared, "binding")
        expected = rule.get(prepared, "expected")
        component = rule.get(prepared, "component_identity")
        question = rule.get(prepared, "analyst_input_identity")
        request_json = rule.text(rule.get(prepared, "request_json"))
        attempt = 1
        if isinstance(question, dict) and "attempt" in question:
            value = question["attempt"]
            if (
                rule.json_type(value) != "number"
                or not isinstance(value, int)
                or not 1 <= value <= 2_147_483_647
            ):
                rule.integrity(
                    "rule-evidence invocation attempt must be a positive integer"
                )
            attempt = value
        job = self._evidence_job_locked(job_id)
        if job is None:
            rule.integrity("open-world optimizer job is required")
        context = self._assert_attempt_authorized_locked(
            job, bundle_digest, outcome_contract_digest
        )
        row = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_invocations"
            " WHERE analyst_input_identity = ?",
            (rule.text(rule.get(binding, "analyst_input_identity")),),
        ).fetchone()
        request = rule.decode(request_json, "rule-evidence request is not JSON")
        if (
            row is None
            or row["state"] != "completed"
            or row["job_id"] != job_id
            or row["component_role"] != role
            or rule.text(rule.get(binding, "job_id")) != str(job_id)
            or rule.text(rule.get(binding, "bundle_digest")) != bundle_digest
            or rule.text(rule.get(binding, "outcome_contract_digest"))
            != outcome_contract_digest
            or rule.text(rule.get(binding, "component_role")) != role
            or component_digest(component) != row["component_identity_digest"]
            or rule.text(rule.get(expected, "expected_provider"))
            != row["expected_provider"]
            or rule.text(rule.get(expected, "expected_model_revision"))
            != row["expected_model_revision"]
            or rule.text(rule.get(component, "provider")) != row["expected_provider"]
            or rule.text(rule.get(component, "immutable_model_revision"))
            != row["expected_model_revision"]
            or rule.text(rule.get(component, "component_role")) != role
            or rule.text(rule.get(expected, "input_digest")) != row["input_digest"]
            or rule.text_digest(request_json or "") != row["input_digest"]
            or rule.canonical_text(request) != request_json
            or rule.text(rule.get(request, "bundle_digest")) != bundle_digest
            or rule.text(rule.get(request, "task_context_digest"))
            != outcome_contract_digest
            or rule.text(rule.get(request, "component_identity_digest"))
            != row["component_identity_digest"]
            or rule.json_digest(
                rule.get(request, "output_schema")
                if role in ("trigger_gate", "trigger_builder")
                else rule.schema_template(rule.get(request, "output_schema"))
            )
            != rule.text(rule.get(component, "output_schema_digest"))
            or rule.text_digest(rule.text(rule.get(request, "system")) or "")
            != rule.text(rule.get(component, "prompt_template_digest"))
            or rule.text(rule.get(binding, "qualification_result_digest"))
            != row["qualification_result_digest"]
            or rule.text(rule.get(binding, "manifest_role")) != row["manifest_role"]
            or rule.text(rule.get(binding, "user_id")) != row["user_id"]
            or rule.text(rule.get(binding, "session_id")) != row["session_id"]
            or rule.text(rule.get(binding, "trajectory_digest"))
            != row["trajectory_digest"]
            or rule.text(rule.get(binding, "candidate_artifact_digest")) is not None
            or rule.text(rule.get(binding, "candidate_analyst_input_identity"))
            is not None
        ):
            rule.conflict("rule-evidence receipt does not bind its completed request")
        self._assert_role_identity(
            context,
            role,
            row["component_identity_digest"],
            rule.text(rule.get(binding, "qualification_suite_digest")),
            row["qualification_result_digest"],
        )
        preimage = {
            **rule.without(question, "attempt"),
            "schema_version": "offline-tuner-open-world-analyst-input-identity-"
            + ("v1" if attempt == 1 else "v2"),
        }
        if attempt > 1:
            preimage["attempt"] = attempt
        usage = row["usage_json"]
        expected_invocation = {
            "analyst_input_identity": row["analyst_input_identity"],
            "component_identity_digest": row["component_identity_digest"],
            "output_digest": row["output_digest"],
            "observed_provider": row["observed_provider"],
            "observed_model_revision": row["observed_model_revision"],
            "called_at": row["called_at"],
            "latency_ms": row["latency_ms"],
            "usage_json": None if usage is None else str(usage),
        }
        if (
            rule.json_digest(preimage) != row["analyst_input_identity"]
            or rule.text(rule.get(question, "component_identity_digest"))
            != row["component_identity_digest"]
            or rule.text(rule.get(question, "qualification_result_digest"))
            != row["qualification_result_digest"]
            or rule.text(rule.get(question, "rendered_input_digest"))
            != row["input_digest"]
            or rule.text(rule.get(question, "outcome_contract_digest"))
            != outcome_contract_digest
            or rule.text(rule.get(question, "role")) != role
            or rule.canonical_text(invocation)
            != rule.canonical_text(expected_invocation)
        ):
            rule.conflict("rule-evidence receipt provenance changed")
        output = {
            "schema_version": "v2"
            if role in ("trigger_gate", "trigger_builder")
            else "v1",
            "contract_type": {
                "discovery": "rule_evidence_diagnostic",
                "candidate": "rule_evidence_proposal",
                "trigger_gate": "rule_evidence_trigger_gate",
                "trigger_builder": "rule_evidence_trigger_builder",
            }[role],
            "payload": observed,
        }
        if (
            json.loads(row["output_json"]) != output
            or rule.text_digest(row["output_json"]) != row["output_digest"]
            or rule.canonical_text(output) != row["output_json"]
        ):
            rule.conflict("rule-evidence receipt observed output changed")
        if role not in ("trigger_gate", "trigger_builder"):
            rule.assert_schema_domains(
                rule.get(request, "output_schema"), rule.get(request, "payload"), role
            )
        if role == "discovery":
            self._assert_manifest_session(context, "discovery", dict(row))
            if (
                row["manifest_role"] != "discovery"
                or rule.text(rule.get(observed, "user_id")) != row["user_id"]
                or rule.text(rule.get(observed, "session_id")) != row["session_id"]
                or rule.text_at(request, "payload", "user_id") != row["user_id"]
                or rule.text_at(request, "payload", "session_id") != row["session_id"]
            ):
                rule.integrity("rule-evidence source scope changed")
        return row

    def _authorize_artifact_locked(
        self, job_id: int, artifact_kind: str, content: object
    ) -> None:
        """``assert_offline_tuner_open_world_artifact_authorized``.

        A discovery memo must be the completed diagnostics of exactly the
        frozen discovery sessions; a candidate the exact edit its completed
        proposal resolves to over those diagnostics; a decision must bind the
        stored candidate, or abstain with a completed proposal that edits
        nothing. An accepted decision re-enters once as its bound candidate.
        """
        if rule.text(rule.get(content, "schema_version")) == "v2":
            self._authorize_v2_artifact_locked(job_id, artifact_kind, content)
            return
        memo_authorized = False
        while True:
            if artifact_kind == OPEN_WORLD_EVIDENCE_ARTIFACT_KIND:
                return
            job = self._evidence_job_locked(job_id)
            bundle_row = self.conn.execute(
                "SELECT content_json FROM playbook_optimization_artifacts"
                " WHERE job_id = ? AND artifact_kind = ?",
                (job_id, OPEN_WORLD_EVIDENCE_ARTIFACT_KIND),
            ).fetchone()
            bundle = (
                None if bundle_row is None else json.loads(bundle_row["content_json"])
            )
            payload = rule.get(content, "payload")
            if (
                job is None
                or rule.text(rule.get(bundle, "schema_version"))
                != OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA
                or rule.text(rule.get(content, "schema_version")) != "v1"
                or rule.json_type(payload) != "object"
            ):
                rule.integrity("unsupported rule-evidence artifact schema")
            task_context = rule.get(bundle, "task_context")
            if rule.null_or_absent(task_context) or rule.json_digest(
                task_context
            ) != rule.text_at(bundle, "target", "success_definition_digest"):
                rule.integrity("task context digest changed")
            incumbent = self.conn.execute(
                "SELECT content, user_id, trigger FROM user_playbooks"
                " WHERE user_playbook_id = ?",
                (job.target_id,),
            ).fetchone()
            if incumbent is None:
                rule.integrity("rule-evidence incumbent missing")
            context = self._assert_attempt_authorized_locked(
                job,
                rule.text(rule.get(payload, "bundle_digest")),
                rule.text_at(bundle, "target", "success_definition_digest"),
            )
            authority = self.conn.execute(
                "SELECT * FROM offline_tuner_open_world_qualification_authority"
                " WHERE singleton = 1"
            ).fetchone()
            method = None if authority is None else authority["method_authority_json"]
            if (
                authority is None
                or method is None
                or rule.json_digest(json.loads(method)) != authority["authority_digest"]
                or rule.text(
                    rule.get(
                        rule.decode(method, "method authority is not JSON"), "policy_id"
                    )
                )
                not in (RULE_EVIDENCE_POLICY_ID, POLICY_V2)
            ):
                _refuse(
                    OpenWorldRefusal.AUTHORITY_STALE,
                    "rule-evidence authority is unavailable",
                )
            if (
                artifact_kind != OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND
                and rule.text(
                    rule.get(
                        rule.decode(method, "method authority is not JSON"), "policy_id"
                    )
                )
                == POLICY_V2
            ):
                rule.integrity(
                    "v2 authority requires explicit v2 preparation artifacts"
                )
            if artifact_kind == OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND:
                self._authorize_memo_locked(
                    job, content, bundle, incumbent, context, authority
                )
                return
            discovery = self._artifact_of_locked(
                job_id, OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND
            )
            discovery_content = (
                None if discovery is None else json.loads(discovery.content_json)
            )
            if (
                discovery is None
                or rule.text(rule.get(payload, "diagnostics_artifact_digest"))
                != discovery.content_digest
                or rule.text(rule.get(payload, "diagnostic_population_digest"))
                != rule.text_at(
                    discovery_content, "payload", "diagnostic_population_digest"
                )
            ):
                rule.conflict("rule-evidence artifact does not bind diagnostics")
            if not memo_authorized:
                self._authorize_artifact_locked(
                    job_id, OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND, discovery_content
                )
            if artifact_kind == OPEN_WORLD_CANDIDATE_ARTIFACT_KIND:
                self._authorize_candidate_locked(
                    job, content, bundle, context, discovery_content
                )
                return
            if (
                artifact_kind != OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND
                or rule.text(rule.get(content, "contract_type"))
                != "rule_evidence_decision"
                or rule.text(rule.get(payload, "policy_id")) != RULE_EVIDENCE_POLICY_ID
            ):
                rule.integrity("unsupported rule-evidence decision policy")
            candidate = self._artifact_of_locked(
                job_id, OPEN_WORLD_CANDIDATE_ARTIFACT_KIND
            )
            if rule.text_at(payload, "decision", "decision") == "abstain":
                reason = rule.text_at(payload, "decision", "reason")
                if (
                    rule.text(rule.get(payload, "candidate_digest")) is not None
                    or candidate is not None
                    or (
                        reason is not None
                        and reason
                        not in ("no_grounded_hypothesis", "evidence_view_incomplete")
                    )
                ):
                    rule.integrity("invalid rule-evidence abstention")
                receipt = rule.get(payload, "proposal_receipt")
                if rule.json_type(receipt) != "object":
                    rule.integrity("abstention requires actual tuner audit receipt")
                proposal = rule.get(receipt, "observed_output")
                self._assert_receipt_locked(
                    job_id=job.job_id,
                    bundle_digest=context["bundle_digest"],
                    outcome_contract_digest=context["outcome_contract_digest"],
                    prepared=rule.get(receipt, "prepared"),
                    invocation=rule.get(receipt, "invocation"),
                    observed=proposal,
                    role="candidate",
                )
                request = rule.get(
                    rule.decode(
                        rule.text_at(receipt, "prepared", "request_json"),
                        "rule-evidence request is not JSON",
                    ),
                    "payload",
                )
                rule.assert_proposal_context(
                    request, rule.get(discovery_content, "payload"), bundle
                )
                edit = rule.resolve_audit(
                    rule.get(request, "original_source"), proposal
                )
                if edit["rule_audit"]["edits"]:
                    rule.integrity("global audit abstention carries edits")
                return
            if rule.get(payload, "proposal_receipt") is not None:
                rule.integrity(
                    "accepted decision proposal receipt belongs in exact candidate"
                )
            if (
                rule.text_at(payload, "decision", "decision") != "accept"
                or candidate is None
                or rule.text(rule.get(payload, "candidate_digest"))
                != candidate.content_digest
            ):
                rule.conflict("decision does not bind exact candidate")
            # Authorize the bound candidate as a direct candidate write would,
            # without authorizing the memo a second time.
            artifact_kind = OPEN_WORLD_CANDIDATE_ARTIFACT_KIND
            content = json.loads(candidate.content_json)
            memo_authorized = True

    def _authorize_memo_locked(
        self,
        job: PlaybookOptimizationJob,
        content: object,
        bundle: object,
        incumbent: sqlite3.Row,
        context: sqlite3.Row,
        authority: sqlite3.Row,
    ) -> None:
        payload = rule.get(content, "payload")
        sessions = rule.get(payload, "sessions")
        frozen = rule.path(bundle, "discovery", "sessions")
        if (
            rule.text(rule.get(content, "contract_type")) != "rule_evidence_diagnostics"
            or rule.text(rule.get(payload, "job_id")) != str(job.job_id)
            or rule.text(rule.get(payload, "task_context_digest"))
            != context["outcome_contract_digest"]
            or rule.text(rule.get(payload, "authority_epoch"))
            != str(authority["epoch"])
            or rule.text(rule.get(payload, "authority_digest"))
            != authority["authority_digest"]
            or rule.json_type(sessions) != "array"
            or not isinstance(frozen, list)
            or len(sessions) != len(frozen)
            or not 1 <= len(sessions) <= 5
        ):
            rule.integrity("diagnostic artifact population is invalid")
        live_content = str(incumbent["content"])
        population: list[Any] = []
        for index, session in enumerate(sessions):
            manifest = frozen[index]
            row = self._assert_receipt_locked(
                job_id=job.job_id,
                bundle_digest=context["bundle_digest"],
                outcome_contract_digest=context["outcome_contract_digest"],
                prepared=rule.get(session, "prepared"),
                invocation=rule.get(session, "invocation"),
                observed=rule.get(session, "observed_output"),
                role="discovery",
            )
            if any(
                rule.text(rule.get(session, name))
                != rule.text(rule.get(manifest, name))
                or row[name] != rule.text(rule.get(session, name))
                for name in ("user_id", "session_id", "trajectory_digest")
            ):
                rule.integrity("diagnostic artifact source population changed")
            trajectory_json = rule.get(session, "trajectory_json")
            if (
                rule.json_type(trajectory_json) != "string"
                or rule.text_digest(trajectory_json) != row["trajectory_digest"]
            ):
                rule.integrity("diagnostic original trajectory preimage changed")
            request_json = rule.text_at(session, "prepared", "request_json")
            request = rule.get(
                rule.decode(request_json, "rule-evidence request is not JSON"),
                "payload",
            )
            messages = rule.get(request, "messages")
            spans = rule.path(session, "source_document", "spans")
            if (
                rule.text_at(session, "source_document", "user_id") != row["user_id"]
                or rule.text_at(session, "source_document", "session_id")
                != row["session_id"]
                or rule.text_at(session, "source_document", "trajectory_digest")
                != row["trajectory_digest"]
                or rule.text_at(session, "source_document", "manifest_role")
                != "discovery"
                or not isinstance(spans, list)
                or not isinstance(messages, list)
                or len(spans) != len(messages)
            ):
                rule.integrity("diagnostic source document changed")
            trajectory = json.loads(trajectory_json)
            for span_index, span in enumerate(spans):
                if (
                    rule.text(rule.get(span, "span_id"))
                    != row["trajectory_digest"]
                    + ":"
                    + rule.json_digest(
                        {
                            **rule.without(span, "span_id"),
                            "schema_version": "open-world-evidence-span-v2",
                        }
                    )
                    or rule.text(rule.get(span, "user_id")) != row["user_id"]
                    or rule.text(rule.get(span, "session_id")) != row["session_id"]
                    or rule.text(rule.get(span, "manifest_role")) != "discovery"
                    or rule.text(rule.get(span, "tenant_id"))
                    != rule.text_at(bundle, "target", "tenant_id")
                    or rule.get(messages, span_index)
                    != {
                        "turn": span_index,
                        "role": rule.text(rule.get(span, "speaker")),
                        "content": rule.text(rule.get(span, "immutable_text")),
                    }
                ):
                    rule.integrity("diagnostic original source span changed")
                rule.assert_source_span(span, trajectory)
            resolved = rule.resolve(rule.get(session, "observed_output"), live_content)
            assessment = resolved["rule_assessment"]
            turns = [rule.get(message, "turn") for message in messages]
            for reference in rule.slots(
                rule.path(
                    session,
                    "observed_output",
                    "rule_assessment",
                    "counterevidence_refs",
                ),
                rule.text(rule.get(session, "session_id")),
                rule.counter_slot_count(request_json),
                False,
            ):
                if reference["turn"] not in turns:
                    rule.integrity("target counterevidence is outside frozen session")
            if (
                rule.get(request, "incumbent")
                != {
                    "id": job.target_id,
                    "user_id": incumbent["user_id"],
                    "version": rule.text_at(
                        bundle, "target", "incumbent_full_version_fingerprint"
                    ),
                    "trigger": incumbent["trigger"],
                    "content_clauses": rule.catalog(live_content),
                }
                or rule.text_at(request, "incumbent", "user_id")
                != rule.text(rule.get(manifest, "user_id"))
                or rule.text(rule.get(request, "agent_context"))
                != rule.text_at(bundle, "task_context", "agent_context")
                or rule.text(rule.get(request, "success_definition"))
                != rule.text_at(bundle, "task_context", "success_definition")
                or rule.get(request, "available_tools")
                != rule.path(bundle, "task_context", "available_tools")
                or rule.text(rule.get(request, "tool_context_status"))
                != rule.text_at(bundle, "task_context", "tool_context_status")
                or not any(
                    rule.text(rule.get(message, "role")) == "agent"
                    for message in messages
                )
            ):
                rule.integrity(
                    "diagnostic request does not bind incumbent and task context"
                )
            for observation in [*resolved["observations"], assessment]:
                evidence = observation["evidence_refs"]
                if not 1 <= len(evidence) <= 6:
                    rule.integrity("diagnostic source references required")
                for reference in evidence:
                    if (
                        reference["session_id"]
                        != rule.text(rule.get(manifest, "session_id"))
                        or reference["turn"] not in turns
                    ):
                        rule.integrity(
                            "diagnostic source reference is outside frozen session"
                        )
            quote = assessment["incumbent_quote"]
            disposition = assessment["disposition"]
            if quote != "" and quote not in live_content:
                rule.integrity("diagnostic incumbent quote absent")
            if disposition in ("rule_gap", "adherence_failure") and _blank(quote):
                rule.integrity("diagnostic rule quote is required")
            observations = rule.path(session, "observed_output", "observations")
            if disposition == "rule_gap" and (
                quote not in live_content
                or quote in live_content[live_content.find(quote) + 1 :]
                or not any(
                    rule.text(rule.get(item, "kind")) == "problem"
                    for item in (observations if isinstance(observations, list) else [])
                )
            ):
                rule.integrity("rule gap requires unique clause and observed problem")
            missing = rule.path(session, "observed_output", "missing_information")
            if disposition == "unknown" and isinstance(missing, list) and not missing:
                rule.integrity(
                    "unknown diagnostic requires missing deciding information"
                )
            population.append(row["output_digest"])
        if rule.json_digest(
            {
                "contract_type": "rule_evidence_diagnostic_population",
                "schema_version": "v1",
                "payload": population,
            }
        ) != rule.text(rule.get(payload, "diagnostic_population_digest")):
            rule.conflict("diagnostic population digest changed")

    def _authorize_candidate_locked(
        self,
        job: PlaybookOptimizationJob,
        content: object,
        bundle: object,
        context: sqlite3.Row,
        discovery_content: object,
    ) -> None:
        payload = rule.get(content, "payload")
        if (
            rule.text(rule.get(content, "contract_type")) != "rule_evidence_candidate"
            or rule.text(rule.get(payload, "job_id")) != str(job.job_id)
            or rule.text(rule.get(payload, "task_context_digest"))
            != context["outcome_contract_digest"]
            or rule.text(rule.get(payload, "incumbent_full_version_fingerprint"))
            != rule.text_at(bundle, "target", "incumbent_full_version_fingerprint")
        ):
            rule.integrity("rule-evidence candidate schema changed")
        proposal = rule.get(payload, "observed_output")
        self._assert_receipt_locked(
            job_id=job.job_id,
            bundle_digest=context["bundle_digest"],
            outcome_contract_digest=context["outcome_contract_digest"],
            prepared=rule.get(payload, "prepared"),
            invocation=rule.get(payload, "invocation"),
            observed=proposal,
            role="candidate",
        )
        request = rule.get(
            rule.decode(
                rule.text_at(payload, "prepared", "request_json"),
                "rule-evidence request is not JSON",
            ),
            "payload",
        )
        rule.assert_proposal_context(
            request, rule.get(discovery_content, "payload"), bundle
        )
        incumbent = rule.text_at(request, "original_source", "incumbent", "content")
        edit = rule.resolve_audit(rule.get(request, "original_source"), proposal)
        revised = edit["revised_content"]
        if (
            not edit["rule_audit"]["edits"]
            or rule.text(rule.get(payload, "original_clause")) != incumbent
            or rule.text(rule.get(payload, "replacement")) != revised
            or rule.text(rule.get(payload, "revised_content")) != revised
            or rule.get(payload, "rule_audit") != edit["rule_audit"]
            or rule.text_at(payload, "content_diff", "incumbent_content_digest")
            != rule.text_digest(incumbent or "")
            or rule.text_at(payload, "content_diff", "candidate_content_digest")
            != rule.text_digest(revised)
            or _blank(rule.text_at(payload, "content_diff", "unified_diff"))
        ):
            rule.integrity("candidate is not the source-linked atomic global edit")
        sessions = rule.path(discovery_content, "payload", "sessions")
        citations: list[dict[str, Any]] = []
        for item in edit["rule_audit"]["edits"]:
            for reference in item["evidence_refs"]:
                session = next(
                    (
                        candidate
                        for candidate in (
                            sessions if isinstance(sessions, list) else []
                        )
                        if rule.text(rule.get(candidate, "session_id"))
                        == rule.text(reference["session_id"])
                    ),
                    None,
                )
                turn = reference["turn"]
                span = (
                    rule.path(session, "source_document", "spans", turn)
                    if isinstance(turn, int) and not isinstance(turn, bool)
                    else None
                )
                if rule.null_or_absent(span) or span is None:
                    rule.integrity("selected gap source span absent")
                immutable_text = rule.text(rule.get(span, "immutable_text")) or ""
                citation: dict[str, Any] = {
                    "span_id": rule.text(rule.get(span, "span_id")),
                    "exact_quote": rule.text(rule.get(span, "immutable_text")),
                    "start_offset": 0,
                    "end_offset": len(immutable_text),
                }
                citation = {
                    **citation,
                    "citation_id": rule.json_digest(
                        {
                            **citation,
                            "schema_version": "open-world-verified-citation-v1",
                        }
                    ),
                    "manifest_role": "discovery",
                    "user_id": rule.text(rule.get(session, "user_id")),
                    "session_id": rule.text(rule.get(session, "session_id")),
                    "trajectory_digest": rule.text(
                        rule.get(session, "trajectory_digest")
                    ),
                }
                if citation not in citations:
                    citations.append(citation)
        if rule.get(payload, "verified_citations") != citations:
            rule.integrity("candidate must inherit exact selected-gap source citations")


def _sessions(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _assert_terminal_decision_matches_job(
    job: PlaybookOptimizationJob, artifact_kind: str, payload: object
) -> None:
    """A terminal job's decision must be the abstention its outcome names."""
    if artifact_kind != OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND:
        return
    if not (
        job.status == "skipped"
        or job.stage == "abstained"
        or job.terminal_outcome is not None
    ):
        return
    decision = rule.path(payload, "payload", "decision")
    if not (
        job.status == "skipped"
        and job.stage == "abstained"
        and job.terminal_outcome in OPEN_WORLD_ABSTAIN_TERMINAL_OUTCOMES
        and isinstance(decision, dict)
        and decision.get("decision") == "abstain"
        and decision.get("reason") == job.terminal_outcome
    ):
        raise OpenWorldInvocationConflictError("decision terminal outcome conflict")


__all__ = ["ATTEMPT_TABLES", "SQLiteOpenWorldAttemptMixin", "component_digest"]
