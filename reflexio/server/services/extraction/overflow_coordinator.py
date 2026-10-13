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
from typing import Any

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    ProviderInputOverflowError,
    check_request_input_budget,
    extraction_provider_guard,
    extraction_request_identity,
    extraction_usage_observer,
)
from reflexio.server.llm.token_accounting import RunTokenTotals, sum_trace_tokens
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.extraction.input_preparation import (
    PreparedExtractionInput,
    repartition_extraction_input,
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
        if (
            manifest.get("version") != 1
            or manifest.get("source_digest") != source_digest
            or manifest.get("attempt_limit") != 128
            or type(manifest.get("revision")) is not int
            or not isinstance(manifest.get("attempts"), list)
        ):
            raise OverflowCheckpointConflictError("Invalid extraction coordinator")
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
        by_ref = {span.evidence_ref: span for span in spans}
        if not checked.evidence or any(
            by_ref[ref].interaction_id not in run.binding.source_interaction_ids
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
                raise InputBudgetConfigurationError(
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
        check_request_input_budget(params, context_limits=self.context_limits)
        fingerprint = self._request_fingerprint(params)
        if (
            self.checkpoint.get("repartition", {}).get("rejected_request_fingerprint")
            == fingerprint
        ):
            raise InputBudgetExceededError(
                "An unchanged rejected request cannot be resent"
            )
        # Don't start a call that can outlive this turn. Each fallback/repair
        # reaches this boundary again, so a late rung yields without dispatch.
        if hard_timeout <= 0 or hard_timeout + 5 > self._deadline - time.monotonic():
            raise OverflowSuspendedError("Extraction turn needs continuation")
        attempt_id = uuid.uuid4().hex
        try:
            self._advance(reserve_attempt_id=attempt_id)
        except Exception as exc:
            raise ProviderRequestGuardError(
                "Durable extraction attempt reservation failed"
            ) from exc
        self._active_attempt.set(attempt_id)
        from reflexio.server.services.extraction.overflow_presentation import (
            current_overflow_presentation,
        )

        presentation = current_overflow_presentation.get()
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

    def _observe(self, provenance: ModelProvenance, totals: RunTokenTotals) -> None:
        attempt_id = self._active_attempt.get()
        if attempt_id is None:
            raise ProviderRequestGuardError("Provider usage lacks an active attempt")
        self._advance(
            observed_attempt={
                "id": attempt_id,
                "usage": asdict(totals),
                "model_provenance": asdict(provenance),
            }
        )
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
