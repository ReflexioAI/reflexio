"""Retrieved-learning relevance/impact evaluator.

Judges every learning that was retrieved and injected into a session
(``Interaction.retrieved_learnings``) with two LLM judge families:

- **relevance**: does the learning apply to the session's task/response?
- **impact**: relative to the agent's definition of success, did applying it
  move the response toward success, away from it, or not materially change it
  (counterfactual judgment from the observed transcript)?

Plain class colocated with the group-evaluation runner (no
``BaseGenerationService`` machinery — this is a judge that writes to its own
table, like the shadow-comparison judge). Consumes the bounded storage
snapshot; never loads full ``Interaction`` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from reflexio.models.api_schema.domain import RetrievedLearningEvaluationResult
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.service_utils import (
    log_llm_messages,
    log_model_response,
    slice_content_by_tokens,
)
from reflexio.server.services.storage.storage_base.retrieved_learning_state import (
    CANONICAL_RETRIEVED_KINDS,
    BoundedRetrievedLearningSnapshot,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)

RETRIEVED_LEARNING_RELEVANCE_PROMPT_ID = "retrieved_learning_relevance"
RETRIEVED_LEARNING_IMPACT_PROMPT_ID = "retrieved_learning_impact"

# Bounded-input constants (code constants by design — no env knobs).
MAX_CANONICAL_CANDIDATES = 1_000
CANDIDATES_PER_CHUNK = 25
LEARNING_BODY_TOKEN_LIMIT = 256
TRANSCRIPT_TOKEN_LIMIT = 8_000


class RetrievedLearningRelevanceVerdict(StrictStructuredOutput):
    """One relevance verdict, keyed by the echoed opaque ``learning_ref``."""

    learning_ref: str = Field(
        description="Exact echo of the learning_ref shown in the prompt,"
        " e.g. 'profile:abc123'"
    )
    is_relevant: bool = Field(
        description="Whether this learning applies to the user's task and the"
        " agent response in this session"
    )
    relevance_reason: str = Field(
        description="Why this learning is or is not relevant to the session"
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class RetrievedLearningRelevanceOutput(StrictStructuredOutput):
    """Relevance judge output: exactly one verdict per listed learning."""

    verdicts: list[RetrievedLearningRelevanceVerdict] = Field(
        description="Exactly one verdict per learning listed in the prompt"
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class RetrievedLearningImpactVerdict(StrictStructuredOutput):
    """One impact verdict, keyed by the echoed opaque ``learning_ref``."""

    learning_ref: str = Field(
        description="Exact echo of the learning_ref shown in the prompt,"
        " e.g. 'user_playbook:42'"
    )
    impact: Literal["positive", "negative", "neutral"] = Field(
        description="Whether applying this learning plausibly improved,"
        " harmed, or did not materially change the agent response"
    )
    impact_reason: str = Field(
        description="Counterfactual reasoning for the impact judgment"
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class RetrievedLearningImpactOutput(StrictStructuredOutput):
    """Impact judge output: exactly one verdict per listed learning."""

    verdicts: list[RetrievedLearningImpactVerdict] = Field(
        description="Exactly one verdict per learning listed in the prompt"
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


@dataclass
class LearningCandidate:
    """One resolved historically attached learning.

    ``title`` comes from the resolved original row (playbook name; empty for
    profiles) and is shown to the judges only — it is never persisted.
    """

    kind: str
    learning_id: str
    title: str
    content: str
    trigger: str

    @property
    def learning_ref(self) -> str:
        return f"{self.kind}:{self.learning_id}"


@dataclass
class RetrievedLearningEvaluationRun:
    """Evaluator output handed to the storage replacement CAS.

    ``outcome`` is ``"evaluated"`` when at least one usable verdict exists (or
    the candidate set is legitimately empty) and ``"failed"`` when no judge
    produced any usable verdict or a bound was exceeded — the runner then
    preserves the prior snapshot via ``finish_retrieved_learning_evaluation_run``.
    """

    outcome: Literal["evaluated", "failed"]
    proposed_status: Literal["complete", "degraded"] = "complete"
    rows: list[RetrievedLearningEvaluationResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RetrievedLearningEvaluator:
    """Judge retrieved learnings for session relevance and response impact."""

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        agent_context: str = "",
        success_definition: str = "",
    ):
        """Initialize the evaluator.

        Args:
            request_context (RequestContext): Storage + prompt manager.
            llm_client (LiteLLMClient): Unified LLM client.
            agent_context (str): Context about the agent, injected into both
                judge prompts.
            success_definition (str): The agent's definition of success (from
                ``AgentSuccessConfig.success_definition_prompt``), injected into
                the impact judge only so impact is judged relative to what the
                org actually considers success. Empty means the impact judge
                falls back to general task helpfulness.
        """
        self.request_context = request_context
        self.client = llm_client
        self.agent_context = agent_context
        self.success_definition = success_definition

        config = self.request_context.configurator.get_config()
        llm_config = config.llm_config if config else None
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        self.model_name = resolve_model_name(
            ModelRole.EVALUATION,
            site_var_value=site_var.get("default_evaluate_model_name"),
            config_override=llm_config.generation_model_name if llm_config else None,
            api_key_config=config.api_key_config if config else None,
        )

    # ===============================
    # public methods
    # ===============================

    def evaluate(
        self,
        user_id: str,
        session_id: str,
        agent_version: str,
        snapshot: BoundedRetrievedLearningSnapshot,
    ) -> RetrievedLearningEvaluationRun:
        """Evaluate every attached learning whose original row still exists.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.
            agent_version (str): Version requested by the runner.
            snapshot (BoundedRetrievedLearningSnapshot): Bounded projection
                loaded by storage.

        Returns:
            RetrievedLearningEvaluationRun: Proposed rows + status, or a
            failure with a sanitized reason.
        """
        diagnostics: dict[str, Any] = {
            "invalid_ref_count": 0,
            "failed_relevance_chunks": 0,
            "failed_impact_chunks": 0,
        }
        if snapshot.attachment_limit_exceeded:
            diagnostics["error_type"] = "attachment_limit_exceeded"
            return RetrievedLearningEvaluationRun(
                outcome="failed", diagnostics=diagnostics
            )

        refs = self._collect_canonical_refs(snapshot, diagnostics)
        if len(refs) > MAX_CANONICAL_CANDIDATES:
            # Never truncate silently: zero LLM calls, preserve prior snapshot.
            diagnostics["error_type"] = "candidate_limit_exceeded"
            return RetrievedLearningEvaluationRun(
                outcome="failed", diagnostics=diagnostics
            )

        candidates = self._resolve_candidates(user_id, refs, diagnostics)
        diagnostics["candidate_count"] = len(candidates)
        if not candidates:
            # No attached or eligible refs: zero LLM calls; the storage
            # replacement clears prior rows and records not_applicable.
            return RetrievedLearningEvaluationRun(
                outcome="evaluated",
                proposed_status="complete",
                rows=[],
                diagnostics=diagnostics,
            )

        transcript = self._format_transcript(snapshot)
        relevance: dict[str, RetrievedLearningRelevanceVerdict] = {}
        impact: dict[str, RetrievedLearningImpactVerdict] = {}
        chunks = [
            candidates[i : i + CANDIDATES_PER_CHUNK]
            for i in range(0, len(candidates), CANDIDATES_PER_CHUNK)
        ]
        for chunk in chunks:
            chunk_refs = {c.learning_ref for c in chunk}
            relevance_output = self._run_judge(
                RETRIEVED_LEARNING_RELEVANCE_PROMPT_ID,
                RetrievedLearningRelevanceOutput,
                transcript,
                chunk,
            )
            if relevance_output is None:
                diagnostics["failed_relevance_chunks"] += 1
            else:
                for verdict in relevance_output.verdicts:
                    if verdict.learning_ref in chunk_refs:
                        relevance[verdict.learning_ref] = verdict
            impact_output = self._run_judge(
                RETRIEVED_LEARNING_IMPACT_PROMPT_ID,
                RetrievedLearningImpactOutput,
                transcript,
                chunk,
            )
            if impact_output is None:
                diagnostics["failed_impact_chunks"] += 1
            else:
                for verdict in impact_output.verdicts:
                    if verdict.learning_ref in chunk_refs:
                        impact[verdict.learning_ref] = verdict

        if not relevance and not impact:
            diagnostics["error_type"] = "all_judges_failed"
            return RetrievedLearningEvaluationRun(
                outcome="failed", diagnostics=diagnostics
            )

        # Build rows by iterating canonical input candidates — never LLM
        # output — so every candidate appears exactly once and unknown output
        # cannot be inserted.
        created_at = snapshot.earliest_request_created_at or 0
        rows: list[RetrievedLearningEvaluationResult] = []
        for candidate in candidates:
            relevance_verdict = relevance.get(candidate.learning_ref)
            impact_verdict = impact.get(candidate.learning_ref)
            rows.append(
                RetrievedLearningEvaluationResult(
                    user_id=user_id,
                    session_id=session_id,
                    agent_version=agent_version,
                    kind=candidate.kind,  # type: ignore[arg-type]
                    learning_id=candidate.learning_id,
                    is_relevant=(
                        relevance_verdict.is_relevant if relevance_verdict else None
                    ),
                    relevance_reason=(
                        relevance_verdict.relevance_reason if relevance_verdict else ""
                    ),
                    impact=impact_verdict.impact if impact_verdict else None,
                    impact_reason=(
                        impact_verdict.impact_reason if impact_verdict else ""
                    ),
                    created_at=created_at,
                )
            )
        degraded = (
            diagnostics["failed_relevance_chunks"] > 0
            or diagnostics["failed_impact_chunks"] > 0
        )
        return RetrievedLearningEvaluationRun(
            outcome="evaluated",
            proposed_status="degraded" if degraded else "complete",
            rows=rows,
            diagnostics=diagnostics,
        )

    # ===============================
    # candidate construction
    # ===============================

    @staticmethod
    def _collect_canonical_refs(
        snapshot: BoundedRetrievedLearningSnapshot,
        diagnostics: dict[str, Any],
    ) -> dict[tuple[str, str], None]:
        """Dedupe ``(kind, learning_id)`` refs in chronological order.

        Returns:
            dict: ``(kind, learning_id) -> None`` used as an ordered set.
        """
        refs: dict[tuple[str, str], None] = {}
        for interaction in snapshot.interactions:
            for kind, learning_id in interaction.refs:
                if kind not in CANONICAL_RETRIEVED_KINDS:
                    diagnostics["invalid_ref_count"] += 1
                    continue
                refs.setdefault((kind, learning_id), None)
        return refs

    def _resolve_candidates(
        self,
        user_id: str,
        refs: dict[tuple[str, str], None],
        diagnostics: dict[str, Any],
    ) -> list[LearningCandidate]:
        """Resolve attached refs to their original rows; skip missing rows.

        Lifecycle status and agent-playbook approval do not affect historical
        evaluation. Learning content/trigger/title come from the original row;
        unresolvable refs are skipped, never followed to a successor.

        Resolution is best-effort, not a durability guarantee: row retention
        (``RETENTION_TARGETS``) evicts tombstoned rows first and
        ``gc_expired_tombstones`` hard-deletes aged ones, so a learning that was
        genuinely served can age out and then resolve to nothing. That is a
        silent skip by design — an old session simply judges fewer learnings.
        """
        storage = self.request_context.storage
        if storage is None:
            return []
        profile_ids = [lid for (kind, lid) in refs if kind == "profile"]
        user_playbook_ids: list[int] = []
        agent_playbook_ids: list[int] = []
        for kind, lid in refs:
            if kind not in ("user_playbook", "agent_playbook"):
                continue
            try:
                parsed = int(lid)
            except ValueError:
                diagnostics["invalid_ref_count"] += 1
                continue
            if parsed <= 0:
                diagnostics["invalid_ref_count"] += 1
                continue
            if kind == "user_playbook":
                user_playbook_ids.append(parsed)
            else:
                agent_playbook_ids.append(parsed)

        resolved: dict[tuple[str, str], LearningCandidate] = {}
        if profile_ids:
            for profile in storage.get_profiles_by_ids(
                user_id, profile_ids, include_inactive=True
            ):
                resolved[("profile", profile.profile_id)] = LearningCandidate(
                    kind="profile",
                    learning_id=profile.profile_id,
                    title="",
                    content=profile.content,
                    trigger="",
                )
        if user_playbook_ids:
            for playbook in storage.get_user_playbooks_by_ids(
                user_id, user_playbook_ids, include_inactive=True
            ):
                key = ("user_playbook", str(playbook.user_playbook_id))
                resolved[key] = LearningCandidate(
                    kind="user_playbook",
                    learning_id=str(playbook.user_playbook_id),
                    title=playbook.playbook_name,
                    content=playbook.content,
                    trigger=playbook.trigger or "",
                )
        for playbook in storage.get_agent_playbooks_by_ids(
            agent_playbook_ids,
            include_inactive=True,
        ):
            key = ("agent_playbook", str(playbook.agent_playbook_id))
            resolved[key] = LearningCandidate(
                kind="agent_playbook",
                learning_id=str(playbook.agent_playbook_id),
                title=playbook.playbook_name,
                content=playbook.content,
                trigger=playbook.trigger or "",
            )
        # Preserve first-seen order of the attached refs.
        return [resolved[key] for key in refs if key in resolved]

    # ===============================
    # judging
    # ===============================

    @staticmethod
    def _format_transcript(snapshot: BoundedRetrievedLearningSnapshot) -> str:
        lines = [
            f"{interaction.role}: {interaction.content}"
            for interaction in snapshot.interactions
            if interaction.role or interaction.content
        ]
        return slice_content_by_tokens("\n".join(lines), TRANSCRIPT_TOKEN_LIMIT)

    def _learnings_payload(self, chunk: list[LearningCandidate]) -> str:
        import json

        return json.dumps(
            [
                {
                    "learning_ref": c.learning_ref,
                    "kind": c.kind,
                    "title": c.title,
                    "content": slice_content_by_tokens(
                        c.content, LEARNING_BODY_TOKEN_LIMIT
                    ),
                    "trigger": slice_content_by_tokens(
                        c.trigger, LEARNING_BODY_TOKEN_LIMIT
                    ),
                }
                for c in chunk
            ],
            indent=2,
        )

    def _run_judge[
        TJudgeOutput: (RetrievedLearningRelevanceOutput, RetrievedLearningImpactOutput)
    ](
        self,
        prompt_id: str,
        output_model: type[TJudgeOutput],
        transcript: str,
        chunk: list[LearningCandidate],
    ) -> TJudgeOutput | None:
        """Run one judge call for one chunk, with one bounded semantic repair.

        A response is valid only when its verdict refs exactly equal the
        chunk's input refs (no duplicates, no missing, no unknown). One
        corrective retry naming only the coverage error; after the second
        failure the chunk is marked failed.
        """
        expected_refs = {c.learning_ref for c in chunk}
        variables: dict[str, Any] = {
            "agent_context_prompt": self.agent_context,
            "interactions": transcript,
            "learnings": self._learnings_payload(chunk),
        }
        # Only the impact judge is anchored to the definition of success;
        # relevance stays success-agnostic ("does this learning apply?").
        if prompt_id == RETRIEVED_LEARNING_IMPACT_PROMPT_ID:
            variables["success_definition_prompt"] = self.success_definition
        prompt = self.request_context.prompt_manager.render_prompt(prompt_id, variables)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            log_llm_messages(logger, f"Retrieved-learning judge {prompt_id}", messages)
            response = self.client.generate_chat_response(
                messages=messages,
                model=self.model_name,
                response_format=output_model,
            )
            if not isinstance(response, output_model):
                logger.warning(
                    "event=retrieved_learning_eval_judge_bad_response prompt=%s"
                    " attempt=%d type=%s",
                    prompt_id,
                    attempt,
                    type(response).__name__,
                )
                return None
            log_model_response(logger, f"Judge {prompt_id} response", response)
            coverage_error = _verdict_coverage_error(
                [v.learning_ref for v in response.verdicts], expected_refs
            )
            if coverage_error is None:
                return response
            if attempt == 0:
                logger.info(
                    "event=retrieved_learning_eval_chunk_retry prompt=%s reason=%s",
                    prompt_id,
                    coverage_error,
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": response.model_dump_json()},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response had a coverage error:"
                            f" {coverage_error}. Return exactly one verdict per"
                            " learning_ref listed in the original prompt — no"
                            " duplicates, no omissions, no other refs."
                        ),
                    },
                ]
        return None


def _verdict_coverage_error(refs: list[str], expected: set[str]) -> str | None:
    """Name the coverage error for a judge response, or None when exact."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    unknown: set[str] = set()
    for ref in refs:
        if ref in seen:
            duplicates.add(ref)
        seen.add(ref)
        if ref not in expected:
            unknown.add(ref)
    missing = expected - seen
    problems: list[str] = []
    if missing:
        problems.append(f"missing refs {sorted(missing)}")
    if duplicates:
        problems.append(f"duplicate refs {sorted(duplicates)}")
    if unknown:
        problems.append(f"unknown refs {sorted(unknown)}")
    return "; ".join(problems) if problems else None
