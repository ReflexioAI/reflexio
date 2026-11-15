"""Independent durable v2 source/receipt authorization for SQLite.

The active runner submits results; this owner reconstructs their source and
replays saved outputs without providers before admitting immutable artifacts.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
from typing import Any, NoReturn

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    PROFILE,
    TriggerSourceFrame,
    run_trigger_maintenance,
)

from . import _open_world_rule_evidence as rule

POLICY_V2 = "evidence-grounded-playbook-revision-v2"
AUTHORITY_V2 = "offline-tuner-rule-evidence-authority-v2"
CONTENT_KIND = "open_world_content_result"
TRIGGER_KIND = "open_world_trigger_result"
COMMON = frozenset(
    {
        "job_id",
        "bundle_digest",
        "task_context_digest",
        "incumbent_full_version_fingerprint",
        "authority_epoch",
        "authority_digest",
        "diagnostics_artifact_digest",
        "diagnostic_population_digest",
    }
)
EXTRA = {
    CONTENT_KIND: frozenset(
        {
            "prepared",
            "invocation",
            "observed_output",
            "original_content",
            "final_content",
            "rule_audit",
            "content_diff",
        }
    ),
    TRIGGER_KIND: frozenset(
        {
            "content_result_artifact_digest",
            "source_frame",
            "source_frame_digest",
            "profile",
            "gate_receipt",
            "builder_receipt",
            "resolved_result",
        }
    ),
    "open_world_candidate": frozenset(
        {
            "content_result_artifact_digest",
            "trigger_result_artifact_digest",
            "revised_content",
            "revised_trigger",
            "content_diff",
            "change_fields",
        }
    ),
    "open_world_attempt_decision": frozenset(
        {
            "policy_id",
            "content_result_artifact_digest",
            "trigger_result_artifact_digest",
            "candidate_digest",
            "decision",
        }
    ),
}
CONTRACTS = {
    CONTENT_KIND: "rule_evidence_content_result",
    TRIGGER_KIND: "rule_evidence_trigger_result",
    "open_world_candidate": "rule_evidence_candidate",
    "open_world_attempt_decision": "rule_evidence_decision",
}


def trigger_authorization(component: dict[str, Any]) -> str | None:
    return rule.json_digest(
        {
            "schema_version": "rule-evidence-trigger-component-authority-v1",
            "role": component["component_role"],
            "component_identity": component,
        }
    )


def validate_trigger_authority(value: dict[str, Any]) -> None:
    """Verify exact per-role authorization and the new structural receipt."""
    from reflexio.server.services.offline_tuner.evidence.open_world_models import (
        TRIGGER_QUALIFICATION_CHECK_NAMES,
    )

    from ._open_world_attempt import component_digest

    def invalid() -> NoReturn:
        rule.refuse(
            rule.OpenWorldRefusal.AUTHORITY_STALE,
            "Trigger qualification authority is invalid",
        )

    maintenance = value.get("trigger_maintenance")
    if (
        not isinstance(maintenance, dict)
        or set(maintenance)
        != {
            "profile",
            "gate",
            "builder",
            "qualification_receipt",
            "qualification_result_digest",
        }
        or maintenance["profile"] != PROFILE
    ):
        invalid()
    receipt = maintenance["qualification_receipt"]
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"suite_digest", "checks", "passed"}
        or not rule.is_hex64(receipt["suite_digest"])
        or receipt["passed"] is not True
        or not isinstance(receipt["checks"], dict)
        or set(receipt["checks"]) != set(TRIGGER_QUALIFICATION_CHECK_NAMES)
        or any(
            receipt["checks"][name] is not True
            for name in TRIGGER_QUALIFICATION_CHECK_NAMES
        )
        or rule.json_digest(receipt) != maintenance["qualification_result_digest"]
    ):
        invalid()
    for name, role in (("gate", "trigger_gate"), ("builder", "trigger_builder")):
        stage = maintenance[name]
        if (
            not isinstance(stage, dict)
            or set(stage) != {"component_identity", "authorization_digest"}
            or rule.text_at(stage, "component_identity", "component_role") != role
        ):
            invalid()
        component_digest(stage["component_identity"])
        if (
            trigger_authorization(stage["component_identity"])
            != stage["authorization_digest"]
        ):
            invalid()


class SQLiteTriggerV2Mixin:
    """Backend-owned v2 verification using only durable rows and typed replay."""

    conn: sqlite3.Connection
    _evidence_job_locked: Any
    _assert_attempt_authorized_locked: Any
    _artifact_of_locked: Any
    _authorize_artifact_locked: Any
    _assert_receipt_locked: Any

    def _bound_trigger_component(
        self, context: sqlite3.Row, role: str
    ) -> dict[str, Any]:
        authority = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_qualification_authority WHERE singleton=1"
        ).fetchone()
        if (
            authority is None
            or authority["state"] != "active"
            or authority["authority_digest"]
            != context["qualification_authority_digest"]
            or authority["epoch"] != context["qualification_authority_epoch"]
        ):
            rule.refuse(
                rule.OpenWorldRefusal.AUTHORITY_STALE,
                "Trigger method authority changed",
            )
        method = rule.decode(
            authority["method_authority_json"], "Trigger authority is not JSON"
        )
        if (
            rule.get(method, "schema_version") != AUTHORITY_V2
            or rule.get(method, "policy_id") != POLICY_V2
            or rule.json_digest(method) != authority["authority_digest"]
        ):
            rule.refuse(
                rule.OpenWorldRefusal.AUTHORITY_STALE,
                "Trigger method authority is unavailable",
            )
        return rule.path(
            method,
            "trigger_maintenance",
            "gate" if role == "trigger_gate" else "builder",
        )

    def _v2_common_locked(
        self, job_id: int, kind: str, content: object, verified: set[tuple[str, str]]
    ) -> tuple[dict[str, Any], Any, Any, sqlite3.Row, sqlite3.Row, Any]:
        if (
            kind not in CONTRACTS
            or not isinstance(content, dict)
            or set(content) != {"contract_type", "schema_version", "payload"}
            or content.get("schema_version") != "v2"
            or content.get("contract_type") != CONTRACTS.get(kind)
        ):
            rule.integrity("unsupported v2 rule-evidence artifact schema")
        payload = content.get("payload")
        if not isinstance(payload, dict) or set(payload) != COMMON | EXTRA[kind]:
            rule.integrity("v2 rule-evidence artifact fields changed")
        job = self._evidence_job_locked(job_id)
        evidence = self._artifact_of_locked(job_id, "open_world_evidence_bundle")
        discovery = self._artifact_of_locked(job_id, "open_world_discovery_memo")
        if job is None or evidence is None or discovery is None:
            rule.conflict("v2 artifact prerequisite is missing")
        bundle = json.loads(evidence.content_json)
        context = self._assert_attempt_authorized_locked(
            job,
            evidence.content_digest,
            rule.text_at(bundle, "target", "success_definition_digest"),
        )
        authority = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_qualification_authority WHERE singleton=1"
        ).fetchone()
        if (
            authority is None
            or rule.text_at(json.loads(authority["method_authority_json"]), "policy_id")
            != POLICY_V2
        ):
            rule.refuse(
                rule.OpenWorldRefusal.AUTHORITY_STALE,
                "v2 artifact requires v2 authority",
            )
        incumbent = self.conn.execute(
            "SELECT content,trigger,user_id FROM user_playbooks WHERE user_playbook_id=?",
            (job.target_id,),
        ).fetchone()
        if incumbent is None:
            rule.integrity("v2 incumbent is missing")
        diagnostic = json.loads(discovery.content_json)
        expected = {
            "job_id": job_id,
            "bundle_digest": evidence.content_digest,
            "task_context_digest": context["outcome_contract_digest"],
            "incumbent_full_version_fingerprint": rule.text_at(
                bundle, "target", "incumbent_full_version_fingerprint"
            ),
            "authority_epoch": authority["epoch"],
            "authority_digest": authority["authority_digest"],
            "diagnostics_artifact_digest": discovery.content_digest,
            "diagnostic_population_digest": rule.text_at(
                diagnostic, "payload", "diagnostic_population_digest"
            ),
        }
        if (
            type(payload["job_id"]) is not int
            or type(payload["authority_epoch"]) is not int
            or any(payload[key] != value for key, value in expected.items())
        ):
            rule.conflict("v2 artifact source identity changed")
        memo_key = ("open_world_discovery_memo", discovery.content_digest)
        if memo_key not in verified:
            self._authorize_artifact_locked(
                job_id, "open_world_discovery_memo", diagnostic
            )
            verified.add(memo_key)
        return payload, job, bundle, context, incumbent, diagnostic

    def _checked_v2_dependency(
        self, job_id: int, kind: str, digest: object, verified: set[tuple[str, str]]
    ) -> dict[str, Any]:
        artifact = self._artifact_of_locked(job_id, kind)
        if artifact is None or artifact.content_digest != digest:
            rule.conflict("v2 artifact prerequisite digest changed")
        content = json.loads(artifact.content_json)
        key = (kind, artifact.content_digest)
        if key not in verified:
            self._authorize_v2_artifact_locked(job_id, kind, content, verified)
            verified.add(key)
        return content["payload"]

    def _authorize_v2_artifact_locked(
        self,
        job_id: int,
        kind: str,
        content: object,
        verified: set[tuple[str, str]] | None = None,
    ) -> None:
        # This memo is local to one locked transaction, never retained on storage.
        if verified is None:
            verified = set()
        payload, job, bundle, context, incumbent, diagnostic = self._v2_common_locked(
            job_id, kind, content, verified
        )
        if kind == CONTENT_KIND:
            self._authorize_content_result_locked(
                payload, job, bundle, context, incumbent, diagnostic
            )
            return
        content_result = self._checked_v2_dependency(
            job_id, CONTENT_KIND, payload["content_result_artifact_digest"], verified
        )
        if kind == TRIGGER_KIND:
            self._authorize_trigger_result_locked(
                payload, job, context, incumbent, content_result
            )
            return
        trigger_result = self._checked_v2_dependency(
            job_id, TRIGGER_KIND, payload["trigger_result_artifact_digest"], verified
        )
        if (
            trigger_result["content_result_artifact_digest"]
            != payload["content_result_artifact_digest"]
        ):
            rule.conflict("final candidate names different preparation results")
        revision = trigger_result["resolved_result"]["revision"]
        changes = [
            name
            for name, old, new in (
                ("content", incumbent["content"], revision["revised_content"]),
                ("trigger", incumbent["trigger"], revision["revised_trigger"]),
            )
            if old != new
        ]
        if kind == "open_world_candidate":
            if (
                not changes
                or payload["revised_content"] != revision["revised_content"]
                or payload["revised_trigger"] != revision["revised_trigger"]
                or payload["content_diff"] != content_result["content_diff"]
                or payload["change_fields"] != changes
            ):
                rule.integrity(
                    "candidate is not the authorized final playbook revision"
                )
            return
        if payload["policy_id"] != POLICY_V2:
            rule.integrity("v2 decision policy changed")
        candidate = self._artifact_of_locked(job_id, "open_world_candidate")
        if not changes:
            if (
                candidate is not None
                or payload["candidate_digest"] is not None
                or payload["decision"]
                != {"decision": "abstain", "reason": "no_grounded_hypothesis"}
            ):
                rule.integrity("invalid final-version abstention")
            return
        if (
            candidate is None
            or candidate.content_digest != payload["candidate_digest"]
            or payload["decision"] != {"decision": "accept"}
        ):
            rule.conflict("v2 decision does not bind final candidate")
        candidate_payload = self._checked_v2_dependency(
            job_id, "open_world_candidate", payload["candidate_digest"], verified
        )
        if any(
            candidate_payload[name] != payload[name]
            for name in (
                "content_result_artifact_digest",
                "trigger_result_artifact_digest",
            )
        ):
            rule.conflict("v2 decision preparation bindings changed")

    def _authorize_content_result_locked(
        self,
        payload: dict[str, Any],
        job: Any,
        bundle: Any,
        context: sqlite3.Row,
        incumbent: sqlite3.Row,
        diagnostic: Any,
    ) -> None:
        self._assert_receipt_locked(
            job_id=job.job_id,
            bundle_digest=context["bundle_digest"],
            outcome_contract_digest=context["outcome_contract_digest"],
            prepared=payload["prepared"],
            invocation=payload["invocation"],
            observed=payload["observed_output"],
            role="candidate",
        )
        request = rule.decode(
            rule.text_at(payload, "prepared", "request_json"),
            "Content request is not JSON",
        )["payload"]
        rule.assert_proposal_context(request, diagnostic["payload"], bundle)
        resolved = rule.resolve_audit(
            request["original_source"], payload["observed_output"]
        )
        final_content = resolved["revised_content"]
        diff = {
            "incumbent_content_digest": rule.text_digest(incumbent["content"]),
            "candidate_content_digest": rule.text_digest(final_content),
            "unified_diff": "".join(
                difflib.unified_diff(
                    incumbent["content"].splitlines(keepends=True),
                    final_content.splitlines(keepends=True),
                    fromfile="incumbent",
                    tofile="candidate",
                )
            ),
        }
        if (
            payload["original_content"] != incumbent["content"]
            or payload["final_content"] != final_content
            or payload["rule_audit"] != resolved["rule_audit"]
            or payload["content_diff"] != diff
        ):
            rule.integrity("Content result is not its completed source-linked audit")

    def _authorize_trigger_result_locked(
        self,
        payload: dict[str, Any],
        job: Any,
        context: sqlite3.Row,
        incumbent: sqlite3.Row,
        content_result: dict[str, Any],
    ) -> None:
        frame = TriggerSourceFrame(
            incumbent_id=job.target_id,
            owner_id=incumbent["user_id"],
            incumbent_version=payload["incumbent_full_version_fingerprint"],
            evidence_bundle_digest=context["bundle_digest"],
            final_content_artifact_digest=payload["content_result_artifact_digest"],
            original_content=incumbent["content"],
            original_trigger=incumbent["trigger"],
            final_content=content_result["final_content"],
        )
        if (
            rule.canonical_text(payload["source_frame"])
            != rule.canonical_text(frame.model_dump(mode="json"))
            or payload["source_frame_digest"] != frame.digest
            or payload["profile"] != PROFILE
        ):
            rule.conflict("Trigger source frame changed durable Content result")
        used: set[str] = set()

        def replay(
            system: str,
            actor_payload: dict[str, object],
            schema: type[StrictStructuredOutput],
        ) -> StrictStructuredOutput:
            role = (
                "trigger_gate"
                if schema.__name__ == "TriggerGate"
                else "trigger_builder"
            )
            field = "gate_receipt" if role == "trigger_gate" else "builder_receipt"
            receipt = payload[field]
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {"prepared", "invocation", "observed_output"}
                or role in used
            ):
                rule.integrity("Trigger stage requires one actual receipt")
            used.add(role)
            self._assert_receipt_locked(
                job_id=job.job_id,
                bundle_digest=context["bundle_digest"],
                outcome_contract_digest=context["outcome_contract_digest"],
                prepared=receipt["prepared"],
                invocation=receipt["invocation"],
                observed=receipt["observed_output"],
                role=role,
            )
            prepared_request = rule.decode(
                rule.text_at(receipt, "prepared", "request_json"),
                "Trigger request is not JSON",
            )
            expected_request = {
                "bundle_digest": context["bundle_digest"],
                "task_context_digest": context["outcome_contract_digest"],
                "component_identity_digest": receipt["invocation"][
                    "component_identity_digest"
                ],
                "system": system,
                "payload": actor_payload,
                "output_schema": schema.model_json_schema(),
                "source_frame_digest": frame.digest,
                "content_result_artifact_digest": payload[
                    "content_result_artifact_digest"
                ],
            }
            if rule.canonical_text(prepared_request) != rule.canonical_text(
                expected_request
            ):
                rule.conflict("Trigger receipt is not the exact source-only request")
            return schema.model_validate(receipt["observed_output"])

        try:
            resolved = run_trigger_maintenance(frame, replay)
        except (ValueError, TypeError, KeyError) as exc:
            rule.integrity(f"Trigger resolution failed: {exc}")
        if (
            rule.canonical_text(resolved.model_dump(mode="json"))
            != rule.canonical_text(payload["resolved_result"])
            or ("trigger_gate" not in used and payload["gate_receipt"] is not None)
            or (
                "trigger_builder" not in used and payload["builder_receipt"] is not None
            )
        ):
            rule.integrity(
                "Trigger result differs from its completed source-only calls"
            )
