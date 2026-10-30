"""Durable provider quota and immutable step receipts for one extraction window.

The window executor owns publication. This coordinator only stages work in the
existing parent agent run and never writes a learning or a billing event.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.input_budget import (
    ComputationEvidenceChangedError,
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    OutputTruncatedRequestError,
    ProviderInputOverflowError,
    check_request_input_budget,
    extraction_provider_guard,
    extraction_request_identity,
    extraction_usage_observer,
    extraction_use_model_capacity,
)
from reflexio.server.llm.token_accounting import RunTokenTotals, sum_trace_tokens
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.extraction.input_preparation import (
    PreparedExtractionInput,
    SourceSpan,
    repartition_extraction_input,
    resolve_evidence_span,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedRecord,
    ProtectedRecordUpdate,
)
from reflexio.server.services.storage.storage_base import AgentRunRecord, BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowAttemptBudgetExceededError,
    OverflowCheckpointConflictError,
    Window,
    validate_extraction_overflow_manifest,
)

if TYPE_CHECKING:
    from reflexio.server.services.extraction.profile_package_preparation import (
        ProfilePackagePreparation,
    )


class OverflowSuspendedError(ProviderRequestGuardError):
    """Saved work needs another worker turn, without completing the window."""


def legacy_observed_usage(payload: dict[str, Any] | None) -> dict[str, int] | None:
    """Retain observed old trace counts without inventing historical calls."""
    if payload is None or "trace" not in payload:
        return None
    trace = ToolLoopTrace.model_validate(payload["trace"])
    if not any(
        turn.prompt_tokens is not None or turn.completion_tokens is not None
        for turn in trace.turns
    ):
        return None
    totals = asdict(sum_trace_tokens(trace))
    if any(type(value) is not int or value < 0 for value in totals.values()):
        raise InputBudgetConfigurationError("Invalid observed historical usage")
    return totals


# Dispatches of one byte-identical request allowed after its responses were
# cut off by the output limit. Reasoning models truncate stochastically: the
# same request often finishes on a later dispatch (each further one succeeded
# about a fifth of the time in production), so a few are worth spending before
# the window is held instead of exhausting the rest of its attempt budget.
MAX_TRUNCATED_REQUEST_DISPATCHES = 4


current_overflow_coordinator: ContextVar[OverflowCoordinator | None] = ContextVar(
    "reflexio_overflow_coordinator", default=None
)


class OverflowCoordinator:
    """One leased parent; independent retries reload its monotonic journal."""

    def __init__(
        self,
        storage: BaseStorage,
        window: Window,
        token: str,
        manifest: dict[str, Any],
        *,
        source_digest: str,
        context_limits: Mapping[str, int],
        turn_seconds: float = 240,
        prepared: PreparedExtractionInput | None = None,
        agent_context: str = "",
    ):
        validate_extraction_overflow_manifest(manifest)
        if manifest.get("source_digest") != source_digest:
            raise OverflowCheckpointConflictError("Invalid extraction coordinator")
        if manifest["version"] == 2:
            from reflexio.server.services.extraction.overflow_protocol import (
                validate_package_scope,
            )

            try:
                validate_package_scope(
                    manifest,
                    {
                        "org_id": storage.org_id,
                        "project_id": window.project_id,
                        "user_id": window.user_id,
                        "window_id": window.window_id,
                        "kind": window.kind,
                    },
                )
            except ValueError as exc:
                raise OverflowCheckpointConflictError(
                    "Invalid package coordinator scope"
                ) from exc
            if prepared is not None:
                raise OverflowCheckpointConflictError(
                    "Package cannot impersonate legacy chunks"
                )
        if turn_seconds <= 0:
            raise ValueError("Extraction turn budget must be positive")
        legacy = manifest.get("legacy_observed_usage")
        if legacy is not None and (
            not isinstance(legacy, dict)
            or set(legacy) != set(asdict(RunTokenTotals()))
            or any(type(value) is not int or value < 0 for value in legacy.values())
        ):
            raise InputBudgetConfigurationError("Invalid observed historical usage")
        self.storage, self.window, self.token = storage, window, token
        self.source_digest = source_digest
        self.context_limits = dict(context_limits)
        if prepared is not None and prepared.source_digest != source_digest:
            raise OverflowCheckpointConflictError(
                "Extraction preparation binding changed"
            )
        self.prepared = prepared
        self.agent_context = agent_context
        self.record = ProtectedRecord.model_validate(
            manifest.get("checkpoint", {}).get("protected_record", {})
        )
        self.candidate_models: dict[str, ModelProvenance | None] = {}
        self.carried_source_spans: tuple[SourceSpan, ...] | None = None
        self.candidate_evidence_refs: dict[str, tuple[str, ...]] = {}
        self.candidate_records: dict[str, ProtectedRecord] = {}
        self.review_models: dict[str, ModelProvenance] = {}
        self.candidate_historical_matches: dict[str, tuple[int, ...]] = {}
        self.saved_run: AgentRunRecord | None = None
        self._manifest = copy.deepcopy(manifest)
        self._deadline = time.monotonic() + turn_seconds
        self._lock = threading.RLock()
        # A copied extraction worker context has its own active attempt. A
        # response cannot accidentally account against another thread's call.
        self._active_attempt: ContextVar[str | None] = ContextVar(
            "reflexio_overflow_active_attempt", default=None
        )
        self._attempt_stage: ContextVar[tuple[str, int, list[str]] | None] = ContextVar(
            "reflexio_overflow_attempt_stage", default=None
        )
        self._profile_request: ContextVar[
            tuple[ProfilePackagePreparation, str, list[dict[str, Any]], str] | None
        ] = ContextVar("reflexio_profile_package_request", default=None)

    @contextmanager
    def profile_request(
        self,
        preparation: ProfilePackagePreparation,
        stage: str,
        messages: list[dict[str, Any]],
        *,
        receipt_key: str,
    ) -> Iterator[list[str]]:
        """Bind one authenticated logical request before native quota admission.

        The consumer owns the seed/continuation. This scope freezes its complete
        raw messages; native fallback and repair guards use that same seed.
        """
        protocol = self.package_protocol
        if (
            self.window.kind != "profile"
            or preparation.contract_digest != protocol["contract_digest"]
            or preparation.compression.witness != protocol["plan_digest"]
            or preparation.compression.source_digest != self.source_digest
            or preparation.snapshot["consumer"] != protocol["consumer"]
            or stage not in {"gate", "agent", "dedup"}
            or len(receipt_key) != 64
            or any(char not in "0123456789abcdef" for char in receipt_key)
            or self._profile_request.get() is not None
        ):
            raise InputBudgetConfigurationError("Profile package request changed")
        self.validate_current_parent()
        binding = self._profile_request.set(
            (preparation, stage, copy.deepcopy(messages), receipt_key)
        )
        attempt_stage = {
            "gate": "gate",
            "agent": "profile_agent",
            "dedup": "profile_dedup",
        }[stage]
        try:
            with self.provider_scope(), self.attempt_stage(attempt_stage) as attempts:
                yield attempts
        finally:
            self._profile_request.reset(binding)

    def validate_profile_attempts(
        self, stage: str, identities: Any, receipt_key: str
    ) -> None:
        """Only real attempts for this exact logical step can support its result."""
        protocol = self.package_protocol
        attempts = {attempt["id"]: attempt for attempt in self._manifest["attempts"]}
        bindings = self.checkpoint.get("profile_attempt_bindings", {})
        if (
            stage not in {"gate", "profile_agent", "profile_dedup"}
            or not isinstance(identities, list)
            or not identities
            or len(identities) > protocol["stage_caps"].get(stage, 0)
            or any(not isinstance(identity, str) for identity in identities)
            or len(set(identities)) != len(identities)
            or any(
                identity not in attempts
                or attempts[identity]["stage"] != stage
                or bindings.get(identity) != receipt_key
                for identity in identities
            )
        ):
            raise InputBudgetConfigurationError(
                "Profile result lacks bound paid attempts"
            )

    @staticmethod
    def initial_manifest(source_digest: str) -> dict[str, Any]:
        if not source_digest:
            raise ValueError("Extraction source witness is required")
        return {
            "version": 1,
            "revision": 0,
            "source_digest": source_digest,
            "attempt_limit": 128,
            "attempts": [],
            "checkpoint": {"stage": "prepare", "completed_steps": {}},
        }

    @staticmethod
    def initial_package_manifest(
        source_digest: str, input_protocol: dict[str, Any]
    ) -> dict[str, Any]:
        """Construct only; admission belongs to an explicit window executor."""
        manifest = {
            **OverflowCoordinator.initial_manifest(source_digest),
            "version": 2,
            "preparation_version": 4,
            "input_protocol": copy.deepcopy(input_protocol),
        }
        validate_extraction_overflow_manifest(manifest)
        return manifest

    @contextmanager
    def attempt_stage(
        self, stage: str, *, minimum_remaining_attempts: int = 0
    ) -> Iterator[list[str]]:
        """Attribute physical dispatches, retaining stage quota across restarts.

        ``minimum_remaining_attempts`` is the required additional SAME-stage
        work after each proposed dispatch. Cache hits reserve nothing. The
        caller can bind the yielded actual IDs to its completed node receipt.
        """
        if (
            self._manifest["version"] != 2
            or stage not in self._manifest["input_protocol"]["stage_caps"]
            or type(minimum_remaining_attempts) is not int
            or minimum_remaining_attempts < 0
            or self._attempt_stage.get() is not None
        ):
            raise InputBudgetConfigurationError("Invalid package attempt stage")
        identities: list[str] = []
        token = self._attempt_stage.set((stage, minimum_remaining_attempts, identities))
        try:
            yield identities
        finally:
            self._attempt_stage.reset(token)

    def require_full_source_coverage(self) -> None:
        """Cached plans and fresh plans obey the same publication barrier."""
        prepared = self.require_prepared()
        expected = [self.candidate_key(chunk.text) for chunk in prepared.chunks]
        if (
            not expected
            or self.checkpoint.get("source_coverage") != expected
            or self.checkpoint.get("stage")
            not in {"extracted", "gated_false", "recovered"}
        ):
            raise ProviderRequestGuardError(
                "Extraction cannot publish without verified full source coverage"
            )

    def billing_input_text(self) -> str:
        """Count original primary fields once, independent of provider retries."""
        fields: dict[tuple[int, str, str], list[str]] = {}
        for chunk in self.require_prepared().chunks:
            for span in chunk.spans:
                fields.setdefault(
                    (span.interaction_id, span.request_id, span.field), []
                ).append(span.text)
        return "\n".join("".join(pieces) for pieces in fields.values())

    def verified_adopted_record(self, index: int) -> ProtectedRecord:
        """Revalidate separate evidence under the original legacy run contract."""
        self._require_adopted_contract()
        record = ProtectedRecord.model_validate(
            self.checkpoint.get("candidate_evidence", {}).get(f"C{index}", {})
        )
        return self._verify_adopted_record(record)

    def _require_adopted_contract(self) -> None:
        run = self.saved_run
        schema = (
            "StructuredProfilesOutput"
            if self.window.kind == "profile"
            else "StructuredPlaybookList"
        )
        if (
            run is None
            or run.committed_output is None
            or run.id != f"window:{self.window.window_id}"
            or run.binding.org_id != self.storage.org_id
            or run.binding.user_id != self.window.user_id
            or run.binding.extractor_kind != self.window.kind
            or self.checkpoint.get("extraction_policy")
            or run.generation_request_snapshot.get("output_schema_name")
            not in ((schema, None) if self.window.kind == "profile" else (schema,))
        ):
            raise InputBudgetConfigurationError(
                "Candidate lacks a scoped legacy adoption contract"
            )
        prepared = self.require_prepared()
        coverage = [self.candidate_key(chunk.text) for chunk in prepared.chunks]
        if self.checkpoint.get("protected_coverage") != coverage:
            raise ProviderRequestGuardError(
                "Adopted candidate lacks full original scan"
            )

    def _verify_adopted_record(self, record: ProtectedRecord) -> ProtectedRecord:
        run = self.saved_run
        if run is None:
            raise InputBudgetConfigurationError("Saved adoption source is unavailable")
        spans = tuple(
            span
            for chunk in self.require_prepared().chunks
            for span in (*chunk.neighbors, *chunk.spans)
        )
        checked = ProtectedRecord().merge(
            ProtectedRecordUpdate(evidence=list(record.evidence)), visible_spans=spans
        )
        if not checked.evidence or any(
            resolve_evidence_span(ref, spans).interaction_id
            not in run.binding.source_interaction_ids
            for item in checked.evidence
            for ref in item.evidence_refs
        ):
            raise ProviderRequestGuardError(
                "Adopted candidate lacks original attributed quotations"
            )
        return checked

    def is_verified_adopted_playbook(self, candidate: UserPlaybook) -> bool:
        """Only this window's converted legacy candidates enter its normal reviewer."""
        run = self.saved_run
        record = self.candidate_records.get(self.playbook_key(candidate))
        if (
            run is None
            or record is None
            or self.window.kind != "playbook"
            or run.generation_request_snapshot.get("output_schema_name")
            != "StructuredPlaybookList"
            or self.checkpoint.get("extraction_policy")
            or candidate.user_id != run.binding.user_id
            or candidate.request_id != run.binding.request_id
            or candidate.agent_version != run.binding.agent_version
            or candidate.source_interaction_ids != run.binding.source_interaction_ids
        ):
            return False
        self.require_full_source_coverage()
        self._require_adopted_contract()
        self._verify_adopted_record(record)
        return True

    @property
    def package_protocol(self) -> dict[str, Any]:
        """Return the frozen package descriptor without exposing mutable state."""
        if self._manifest["version"] != 2:
            raise InputBudgetConfigurationError("Extraction is not package protocol")
        return copy.deepcopy(self._manifest["input_protocol"])

    def validate_current_parent(self) -> None:
        """Fence cache-only results using the same authority checks as dispatch."""
        with self._lock:
            self.storage.validate_extraction_overflow(
                self.window,
                self.token,
                expected_revision=self._manifest["revision"],
                source_digest=self.source_digest,
                expected_input_protocol=self._manifest.get("input_protocol"),
            )

    @property
    def checkpoint(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._manifest.get("checkpoint", {}))

    @property
    def token_totals(self) -> RunTokenTotals:
        """Only observed usage, including earlier turns; unknown calls stay unknown."""
        totals = RunTokenTotals()
        with self._lock:
            if self._manifest.get("legacy_observed_usage") is not None:
                totals.add(**self._manifest["legacy_observed_usage"])
            for attempt in self._manifest["attempts"]:
                if attempt["usage"] is not None:
                    totals.add(**attempt["usage"])
        return totals

    def require_remaining_work(self, minimum_attempts: int) -> None:
        if type(minimum_attempts) is not int or minimum_attempts < 0:
            raise ValueError("Remaining work must be a nonnegative integer")
        with self._lock:
            if len(self._manifest["attempts"]) + minimum_attempts > 128:
                raise OverflowAttemptBudgetExceededError(
                    "Remaining extraction work exceeds provider attempt budget"
                )

    def _advance(self, **change: Any) -> None:
        with self._lock:
            if self._manifest["version"] == 2:
                change["expected_input_protocol"] = self._manifest["input_protocol"]
            self._manifest = self.storage.checkpoint_extraction_overflow(
                self.window,
                self.token,
                expected_revision=self._manifest["revision"],
                source_digest=self.source_digest,
                **change,
            )

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._advance(checkpoint=copy.deepcopy(checkpoint))

    def step_key(self, stage: str, inputs: Any) -> str:
        """Bind reuse to source, presentation, prompt/config and protected context."""
        encoded = json.dumps(
            [self.source_digest, stage, inputs],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def completed_step(self, key: str) -> dict[str, Any] | None:
        return self.checkpoint.get("completed_steps", {}).get(key)

    def require_prepared(self) -> PreparedExtractionInput:
        if self.prepared is None:
            raise OverflowCheckpointConflictError(
                "Extraction preparation is unavailable"
            )
        return self.prepared

    def remember_historical_playbooks(self, playbooks: list[UserPlaybook]) -> None:
        from reflexio.server.services.playbook.components.overflow_evidence import (
            historical_playbook_digest,
        )

        if not playbooks:
            return
        checkpoint = self.checkpoint
        witnesses = checkpoint.setdefault("historical_playbooks", {})
        changed = False
        for playbook in playbooks:
            if (
                not playbook.user_playbook_id
                or playbook.user_id != self.window.user_id
                or playbook.status is not None
                or playbook.merged_into is not None
                or playbook.superseded_by is not None
                or playbook.retired_at is not None
                or (
                    self.saved_run is not None
                    and self.saved_run.binding.agent_version != playbook.agent_version
                )
            ):
                raise InputBudgetConfigurationError(
                    "Historical playbook scope or status changed"
                )
            key = str(playbook.user_playbook_id)
            digest = historical_playbook_digest(playbook)
            if key in witnesses and witnesses[key] != digest:
                # Fails this pass. The saved witness is now permanently stale,
                # so the worker discards it and the next pass starts empty.
                raise ComputationEvidenceChangedError(
                    "Historical playbook changed after checkpoint"
                )
            if key not in witnesses:
                witnesses[key] = digest
                changed = True
        if len(witnesses) > 128:
            raise InputBudgetConfigurationError(
                "Historical playbook witness limit exceeded"
            )
        if changed:
            self.save_checkpoint(checkpoint)

    def register_extraction_child(self, chunk_text: str, key: str) -> None:
        checkpoint = self.checkpoint
        children = checkpoint.setdefault("extraction_children", {})
        identity = self.candidate_key(chunk_text)
        contract = {"key": key, "policy": checkpoint.get("extraction_policy")}
        previous = children.get(identity)
        if previous is not None and previous != contract:
            raise InputBudgetConfigurationError("Extraction child contract changed")
        if previous is None:
            children[identity] = contract
            self.save_checkpoint(checkpoint)

    def unfinished_extraction_chunks(
        self, prepared: PreparedExtractionInput, *, start: int = 0
    ) -> int:
        """Count validated child receipts as completed, never as future calls."""
        checkpoint = self.checkpoint
        policy = checkpoint.get("extraction_policy")
        children = checkpoint.get("extraction_children", {})
        receipts = checkpoint.get("completed_steps", {})
        remaining = 0
        for chunk in prepared.chunks[start:]:
            contract = children.get(self.candidate_key(chunk.text), {})
            if (
                not policy
                or contract.get("policy") != policy
                or contract.get("key") not in receipts
            ):
                remaining += 1
        return remaining

    @staticmethod
    def candidate_key(candidate: Any) -> str:
        if hasattr(candidate, "model_dump"):
            candidate = candidate.model_dump(mode="json")
        encoded = json.dumps(
            candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def candidate_model(self, candidate: Any) -> ModelProvenance | None:
        raw = self.checkpoint.get("candidate_models", {}).get(
            self.candidate_key(candidate)
        )
        return ModelProvenance(**raw) if raw is not None else None

    def playbook_key(self, candidate: Any) -> str:
        # New playbooks have database id=0 until persistence. Bind their view
        # to immutable rule/source data rather than that shared placeholder id.
        return self.candidate_key(
            {
                "kind": "playbook",
                "user_id": candidate.user_id,
                "agent_version": candidate.agent_version,
                "request_id": candidate.request_id,
                "content": candidate.content,
                "trigger": candidate.trigger,
                "rationale": candidate.rationale,
                "source_interaction_ids": candidate.source_interaction_ids,
                "source_span_digest": hashlib.sha256(
                    (candidate.source_span or "").encode()
                ).hexdigest(),
            }
        )

    def complete_step(self, key: str, output: Any, provenance: ModelProvenance) -> None:
        with self._lock:
            checkpoint = self.checkpoint
            steps = checkpoint.setdefault("completed_steps", {})
            receipt = {"output": output, "model_provenance": asdict(provenance)}
            if self._manifest["version"] == 2:
                receipt["kind"] = "model_output"
            if key in steps:
                if steps[key] != receipt:
                    raise OverflowCheckpointConflictError(
                        "Completed extraction step is immutable"
                    )
                return
            if len(steps) >= 128:
                raise OverflowAttemptBudgetExceededError(
                    "Extraction checkpoint capacity exhausted"
                )
            steps[key] = receipt
            if checkpoint.get("active_child", {}).get("key") == key:
                checkpoint.pop("active_child")
            self.save_checkpoint(checkpoint)

    def complete_selection_step(self, key: str, receipt: dict[str, Any]) -> None:
        """Save a host-normalized selection without inventing authored provenance."""
        if self._manifest["version"] != 2:
            raise InputBudgetConfigurationError(
                "Selection receipts require package protocol"
            )
        with self._lock:
            checkpoint = self.checkpoint
            steps = checkpoint["completed_steps"]
            if key in steps:
                if steps[key] != receipt:
                    raise OverflowCheckpointConflictError(
                        "Completed source selection is immutable"
                    )
                return
            steps[key] = copy.deepcopy(receipt)
            validate_extraction_overflow_manifest(
                {**self._manifest, "checkpoint": checkpoint}
            )
            self.save_checkpoint(checkpoint)

    def _request_fingerprint(self, params: dict[str, Any]) -> str:
        request = {
            key: params.get(key)
            for key in (
                "model",
                "messages",
                "tools",
                "tool_choice",
                "response_format",
                "max_tokens",
                "max_completion_tokens",
                "api_base",
                "base_url",
            )
        }
        if request["response_format"] is not None and hasattr(
            request["response_format"], "model_json_schema"
        ):
            request["response_format"] = request["response_format"].model_json_schema()
        return self.candidate_key(request)

    def _guard(
        self,
        params: dict[str, Any],
        hard_timeout: float,
        _ladder: tuple[str, ...],
        _structured: bool,
    ) -> None:
        check_request_input_budget(
            params,
            context_limits=self.context_limits,
            prompt_target=None if extraction_use_model_capacity.get() else 65536,
        )
        fingerprint = self._request_fingerprint(params)
        if (
            self.checkpoint.get("repartition", {}).get("rejected_request_fingerprint")
            == fingerprint
        ):
            raise InputBudgetExceededError(
                "An unchanged rejected request cannot be resent"
            )
        if (
            self.checkpoint.get("truncated_requests", {}).get(fingerprint, 0)
            >= MAX_TRUNCATED_REQUEST_DISPATCHES
        ):
            raise OutputTruncatedRequestError(
                "An unchanged request truncated by its output limit cannot be resent"
            )
        # Don't start a call that can outlive this turn. Each fallback/repair
        # reaches this boundary again, so a late rung yields without dispatch.
        if hard_timeout <= 0 or hard_timeout + 5 > self._deadline - time.monotonic():
            raise OverflowSuspendedError("Extraction turn needs continuation")
        attempt_id = uuid.uuid4().hex
        reservation: dict[str, Any] = {"reserve_attempt_id": attempt_id}
        stage = self._attempt_stage.get()
        if self._manifest["version"] == 2:
            if stage is None:
                raise InputBudgetConfigurationError(
                    "Package dispatch lacks stage attribution"
                )
            if stage[0] != "compress":
                ladder_name = {
                    "gate": "gate",
                    "profile_agent": "agent",
                    "profile_dedup": "dedup",
                    "playbook_intent": "intent",
                }[stage[0]]
                expected_ladder = self._manifest["input_protocol"]["consumer"][
                    "ladders"
                ][ladder_name]
                if _ladder != tuple(expected_ladder):
                    raise InputBudgetConfigurationError(
                        "Package consumer model ladder changed"
                    )
                if self.window.kind == "profile":
                    binding = self._profile_request.get()
                    if binding is None or binding[1] != ladder_name:
                        raise InputBudgetConfigurationError(
                            "Profile package dispatch lacks its request contract"
                        )
                    binding[0].check_request(
                        ladder_name, binding[2], params, _ladder, _structured
                    )
            reservation.update(
                reserve_attempt_stage=stage[0], minimum_remaining_attempts=stage[1]
            )
        try:
            with self._lock:
                binding = self._profile_request.get()
                if self._manifest["version"] == 2 and binding is not None:
                    checkpoint = self.checkpoint
                    checkpoint.setdefault("profile_attempt_bindings", {})[
                        attempt_id
                    ] = binding[3]
                    reservation["checkpoint"] = checkpoint
                self._advance(**reservation)
        except Exception as exc:
            raise ProviderRequestGuardError(
                "Durable extraction attempt reservation failed"
            ) from exc
        self._active_attempt.set(attempt_id)
        if stage is not None and self._manifest["version"] == 2:
            stage[2].append(attempt_id)
        from reflexio.server.services.extraction.overflow_presentation import (
            current_overflow_presentation,
        )

        presentation = current_overflow_presentation.get()
        if self._manifest["version"] == 2:
            extraction_request_identity.set(
                {
                    "attempt_id": attempt_id,
                    "request_fingerprint": fingerprint,
                    "protocol_version": 2,
                }
            )
            return
        extraction_request_identity.set(
            {
                "attempt_id": attempt_id,
                "request_fingerprint": fingerprint,
                "chunk_index": presentation.chunk.index if presentation else None,
                "chunk_digest": self.candidate_key(presentation.chunk.text)
                if presentation
                else None,
            }
        )

    def repartition_rejected_input(self, error: ProviderInputOverflowError) -> None:
        """Freeze one changed source plan; preserve attempts and immutable receipts."""
        if self._manifest["version"] == 2:
            raise InputBudgetExceededError(
                "Package rejection cannot use legacy repartition"
            ) from error
        checkpoint = self.checkpoint
        prepared = self.require_prepared()
        identity = error.identity
        index = identity.get("chunk_index")
        if (
            checkpoint.get("repartition") is not None
            or checkpoint.get("active_child") is not None
            or checkpoint.get("waiting_child") is not None
            or type(index) is not int
            or not 0 <= index < len(prepared.chunks)
            or identity.get("chunk_digest")
            != self.candidate_key(prepared.chunks[index].text)
            or not isinstance(identity.get("request_fingerprint"), str)
            or not any(
                attempt["id"] == identity.get("attempt_id")
                for attempt in self._manifest["attempts"]
            )
        ):
            raise InputBudgetExceededError(
                "Rejected input cannot be safely repartitioned"
            ) from error
        split = len(prepared.chunks[index].spans) // 2
        changed = repartition_extraction_input(
            prepared, chunk_index=index, split_after=split
        )
        history_count = sum(map(len, checkpoint.get("historical_chunks", {}).values()))
        if len(changed.chunks) + history_count > 48:
            raise InputBudgetExceededError(
                "Repartition and history exceed chunk capacity"
            ) from error
        self.require_remaining_work(self.unfinished_extraction_chunks(changed) + 3)
        old_coverage = [self.candidate_key(chunk.text) for chunk in prepared.chunks]
        new_coverage = [self.candidate_key(chunk.text) for chunk in changed.chunks]
        if checkpoint.get("protected_coverage") == old_coverage:
            # Both new children are exact subdivisions of the already-scanned
            # original. Its validated record and source references stay intact.
            checkpoint["protected_coverage"] = new_coverage
        checkpoint["repartition"] = {
            "count": 1,
            "chunk_index": index,
            "split_after": split,
            "original_chunk_digest": identity["chunk_digest"],
            "chunk_digests": new_coverage,
            "rejected_request_fingerprint": identity["request_fingerprint"],
            "rejected_attempt_id": identity["attempt_id"],
        }
        checkpoint.pop("source_coverage", None)
        checkpoint["stage"] = "prepare"
        with self.storage.commit_scope():
            self.storage.fence_user_extraction(self.window.user_id, self.token)
            self.storage.validate_extraction_source_digest(
                self.window, self.source_digest
            )
            self.storage.validate_extraction_answers(
                checkpoint.get("answer_witnesses", {})
            )
            self.storage.validate_extraction_answer_dependencies(
                self.window, checkpoint.get("answer_dependencies", {})
            )
            self.storage.validate_extraction_historical_playbooks(
                self.window, checkpoint.get("historical_playbooks", {})
            )
            self.save_checkpoint(checkpoint)

    def _observe(
        self,
        provenance: ModelProvenance,
        totals: RunTokenTotals,
        *,
        finish_reason: str | None = None,
    ) -> None:
        attempt_id = self._active_attempt.get()
        if attempt_id is None:
            raise ProviderRequestGuardError("Provider usage lacks an active attempt")
        change: dict[str, Any] = {
            "observed_attempt": {
                "id": attempt_id,
                "usage": asdict(totals),
                "model_provenance": asdict(provenance),
            }
        }
        fingerprint = (extraction_request_identity.get() or {}).get(
            "request_fingerprint"
        )
        with self._lock:
            if finish_reason == "length" and isinstance(fingerprint, str):
                # Counted in the attempt's own journal write, so no later turn
                # can dispatch the same request before the count is durable.
                checkpoint = self.checkpoint
                truncated = checkpoint.setdefault("truncated_requests", {})
                truncated[fingerprint] = truncated.get(fingerprint, 0) + 1
                change["checkpoint"] = checkpoint
            self._advance(**change)
        self._active_attempt.set(None)

    @contextmanager
    def provider_scope(self) -> Iterator[None]:
        coordinator_token = current_overflow_coordinator.set(self)
        guard_token = extraction_provider_guard.set(self._guard)
        observer_token = extraction_usage_observer.set(self._observe)
        identity_token = extraction_request_identity.set(None)
        try:
            yield
        finally:
            extraction_request_identity.reset(identity_token)
            extraction_usage_observer.reset(observer_token)
            extraction_provider_guard.reset(guard_token)
            current_overflow_coordinator.reset(coordinator_token)
