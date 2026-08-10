"""Sequential re-review orchestration for persisted user playbooks."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from reflexio.models.api_schema.domain.entities import (
    LineageContext,
    ReviewUserPlaybookEdit,
    ReviewUserPlaybookResult,
    ReviewUserPlaybooksRequest,
    ReviewUserPlaybooksResponse,
    UserPlaybook,
)
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.components.reviewer import (
    CandidateReviewDecision,
    PlaybookCandidateEvidenceError,
    PlaybookCandidateReviewer,
)
from reflexio.server.services.playbook.playbook_edit_apply import apply_playbook_edit
from reflexio.server.services.playbook.playbook_service_utils import (
    is_evidence_validated,
)
from reflexio.server.services.playbook.review_window import (
    PlaybookReviewWindowError,
    reconstruct_playbook_review_window,
)

logger = logging.getLogger(__name__)

# The reviewer's lineage actor. Every row this service creates or supersedes is
# attributable to it, so an operator can reconstruct a run from lineage alone
# when the HTTP response is lost (apply mode runs in the background).
REVIEW_ACTOR = "playbook_review"

_FAILED_MSG = "Review failed; see server logs for the cause"


def new_review_run_id() -> str:
    """Mint the id that correlates one review run's lineage events."""
    return f"playbook_review_{uuid.uuid4().hex}"


class _PlaybookReviewRaceError(RuntimeError):
    """A reviewed incumbent stopped being current before its apply."""


@dataclass
class _ReviewedPlaybook:
    original: UserPlaybook
    result: ReviewUserPlaybookResult
    survivor: UserPlaybook | None


class UserPlaybookReviewService:
    """Review persisted playbooks and optionally apply each decision in order."""

    def __init__(
        self,
        *,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
    ) -> None:
        self.request_context = request_context
        self.client = llm_client
        if request_context.storage is None:
            raise RuntimeError("Storage not configured")
        self.storage = request_context.storage

    def _review_window(
        self,
        playbook: UserPlaybook,
    ) -> list[RequestInteractionDataModel]:
        if not playbook.user_id:
            raise PlaybookReviewWindowError(
                f"User playbook {playbook.user_playbook_id} has no owning user_id"
            )
        if not is_evidence_validated(playbook):
            raise PlaybookReviewWindowError(
                f"User playbook {playbook.user_playbook_id} lacks validated evidence"
            )

        generation_run = self.storage.get_latest_finalized_agent_run_for_request(
            org_id=self.request_context.org_id,
            extractor_kind="playbook",
            user_id=playbook.user_id,
            request_id=playbook.request_id,
        )
        if generation_run is None or not generation_run.binding.source_interaction_ids:
            raise PlaybookReviewWindowError(
                f"User playbook {playbook.user_playbook_id} has no complete "
                "generation-window provenance"
            )

        # The extraction run records the complete prompt window. The playbook row
        # stores only the evidence cited by that candidate and may also retain
        # cited IDs from older rows consumed by consolidation. Review needs both:
        # full original chronology plus every persisted evidence unit.
        source_ids = list(
            dict.fromkeys(
                [
                    *generation_run.binding.source_interaction_ids,
                    *playbook.source_interaction_ids,
                ]
            )
        )
        return reconstruct_playbook_review_window(
            storage=self.storage,
            source_interaction_ids=source_ids,
            user_id=playbook.user_id,
            subject=f"User playbook {playbook.user_playbook_id}",
        )

    def _existing_playbooks(
        self,
        consolidator: PlaybookConsolidator,
        candidates: list[UserPlaybook],
        simulated_archived_ids: set[int],
    ) -> list[UserPlaybook]:
        first = candidates[0]
        candidate_ids = {playbook.user_playbook_id for playbook in candidates}
        return [
            playbook
            for playbook in consolidator.retrieve_existing_playbooks(
                candidates,
                user_id=first.user_id,
                agent_version=first.agent_version,
            )
            if playbook.user_playbook_id not in candidate_ids
            and playbook.user_playbook_id not in simulated_archived_ids
        ]

    @staticmethod
    def _public_result(
        playbook: UserPlaybook,
        decision: CandidateReviewDecision,
    ) -> ReviewUserPlaybookResult:
        edit = None
        public_decision: Literal["accept", "edit", "reject"] = (
            "edit" if decision.decision == "revise" else decision.decision
        )
        if decision.revision is not None:
            edit = ReviewUserPlaybookEdit(
                content=decision.revision.content,
                trigger=decision.revision.trigger,
                rationale=decision.revision.rationale,
            )
        return ReviewUserPlaybookResult(
            user_playbook_id=playbook.user_playbook_id,
            decision=public_decision,
            reason_code=decision.reason_code,
            reason=decision.reason,
            edit=edit,
        )

    @staticmethod
    def _skipped_result(
        playbook: UserPlaybook,
        exc: PlaybookReviewWindowError,
    ) -> _ReviewedPlaybook:
        # The reason text is this service's own fail-closed message about the
        # row's provenance, never a raw storage/model error, so it is safe to
        # return to the caller.
        return _ReviewedPlaybook(
            original=playbook,
            result=ReviewUserPlaybookResult(
                user_playbook_id=playbook.user_playbook_id,
                decision="skip",
                reason_code="evidence_unavailable",
                reason=str(exc),
            ),
            survivor=None,
        )

    def _review_each(
        self,
        selected: list[UserPlaybook],
    ) -> Iterator[_ReviewedPlaybook]:
        if not selected:
            return
        root_config = self.request_context.configurator.get_config()
        playbook_config = root_config.user_playbook_extractor_config
        if playbook_config is None:
            raise ValueError("User playbook extraction is not configured")
        reviewer = PlaybookCandidateReviewer(
            request_context=self.request_context,
            llm_client=self.client,
        )
        if not reviewer.is_enabled():
            raise ValueError("No active user-playbook review prompt is configured")
        consolidator = PlaybookConsolidator(
            request_context=self.request_context,
            llm_client=self.client,
            dedup_config=playbook_config.deduplication_config,
        )
        tool_context = ""
        if root_config.tool_can_use:
            tool_context = "\n".join(
                f"{tool.tool_name}: {tool.tool_description}"
                for tool in root_config.tool_can_use
            )

        simulated_archived_ids: set[int] = set()
        for playbook in selected:
            candidates = [playbook]
            try:
                window = self._review_window(playbook)
            except PlaybookReviewWindowError as exc:
                # Selection is "newest current rows in a window", which cannot
                # know whether a row is still reviewable. One unreviewable row
                # must not cost the rest of the batch.
                logger.info(
                    "event=playbook_review_skipped user_playbook_id=%d reason=%s",
                    playbook.user_playbook_id,
                    exc,
                )
                yield self._skipped_result(playbook, exc)
                continue
            try:
                outcome = reviewer.decide(
                    candidates=candidates,
                    request_interaction_data_models=window,
                    existing_playbooks=self._existing_playbooks(
                        consolidator,
                        candidates,
                        simulated_archived_ids,
                    ),
                    agent_context=self.request_context.configurator.get_agent_context(),
                    playbook_definition=(
                        playbook_config.extraction_definition_prompt or ""
                    ).strip(),
                    tool_context=tool_context,
                )
            except PlaybookCandidateEvidenceError as exc:
                unavailable = PlaybookReviewWindowError(str(exc))
                logger.info(
                    "event=playbook_review_skipped user_playbook_id=%d reason=%s",
                    playbook.user_playbook_id,
                    unavailable,
                )
                yield self._skipped_result(playbook, unavailable)
                continue
            survivors = reviewer.apply_decisions(
                candidates=candidates,
                outcome=outcome,
            )
            decision = outcome.output.decisions[0]
            survivor = None if decision.decision == "reject" else survivors[0]
            reviewed = _ReviewedPlaybook(
                original=playbook,
                result=self._public_result(playbook, decision),
                survivor=survivor,
            )
            yield reviewed
            if reviewed.result.decision in ("edit", "reject"):
                # Apply mode has removed this row from CURRENT before the
                # generator resumes. Report mode mirrors that transition.
                simulated_archived_ids.add(playbook.user_playbook_id)

    @staticmethod
    def _successor(
        reviewed: _ReviewedPlaybook,
        *,
        created_at: int,
    ) -> UserPlaybook:
        if reviewed.survivor is None:
            raise ValueError("Edit decision is missing its reviewed survivor")
        return reviewed.survivor.model_copy(
            update={
                "user_playbook_id": 0,
                # Preserve the extraction request that owns the full historical
                # window. The review run remains attributable through the revise
                # lineage event below; replacing this with ``run_id`` would make
                # the successor impossible to review again from full provenance.
                "request_id": reviewed.original.request_id,
                "created_at": created_at,
                # CURRENT, not PENDING: retrieval reads user playbooks with
                # ``status_filter=[None]``, so a PENDING replacement would take
                # the guidance out of service until an unrelated global upgrade.
                "status": None,
                "source": reviewed.original.source,
                "embedding": [],
                "expanded_terms": None,
                "tags": None,
                "merged_into": None,
                "superseded_by": None,
            }
        )

    def _apply_edit(self, reviewed: _ReviewedPlaybook, *, run_id: str) -> None:
        successor = self._successor(reviewed, created_at=int(time.time()))
        # All fallible model/embedding work happens before the transaction opens
        # (``apply_playbook_edit`` is called with ``skip_embedding=True``), so a
        # later playbook never rolls this decision back.
        self.storage.precompute_user_playbook_embeddings([successor])
        successor_id = apply_playbook_edit(
            self.storage,
            incumbent_id=reviewed.original.user_playbook_id,
            new_playbook=successor,
            source=reviewed.original.source or REVIEW_ACTOR,
            request_id=run_id,
            skip_embedding=True,
            revise_context=LineageContext(
                op_kind="revise",
                actor=REVIEW_ACTOR,
                source_ids=[str(value) for value in successor.source_interaction_ids],
                request_id=run_id,
                reason="Replacement created by persisted playbook review",
            ),
        )
        if successor_id == -1:
            raise _PlaybookReviewRaceError(
                "The reviewed playbook stopped being current before apply; "
                "that decision was not applied"
            )
        reviewed.result.applied = True
        reviewed.result.successor_user_playbook_id = successor_id

    def _apply_reject(self, reviewed: _ReviewedPlaybook) -> None:
        user_id = reviewed.original.user_id
        if not user_id or not self.storage.archive_user_playbook_by_id(
            user_id=user_id,
            user_playbook_id=reviewed.original.user_playbook_id,
        ):
            raise _PlaybookReviewRaceError(
                "The reviewed playbook stopped being current before apply; "
                "that decision was not applied"
            )
        reviewed.result.applied = True

    def _apply_one(self, reviewed: _ReviewedPlaybook, *, run_id: str) -> None:
        if reviewed.result.decision == "accept":
            reviewed.result.applied = True
        elif reviewed.result.decision == "edit":
            self._apply_edit(reviewed, run_id=run_id)
        elif reviewed.result.decision == "reject":
            self._apply_reject(reviewed)
        # "skip" was never reviewed; it writes nothing and stays applied=False.

    @staticmethod
    def _response(
        *,
        success: bool,
        report_only: bool,
        run_id: str,
        selected_count: int,
        results: list[ReviewUserPlaybookResult],
        msg: str,
    ) -> ReviewUserPlaybooksResponse:
        return ReviewUserPlaybooksResponse(
            success=success,
            report_only=report_only,
            run_id=run_id,
            selected_count=selected_count,
            accepted_count=sum(item.decision == "accept" for item in results),
            edited_count=sum(item.decision == "edit" for item in results),
            rejected_count=sum(item.decision == "reject" for item in results),
            skipped_count=sum(item.decision == "skip" for item in results),
            results=results,
            msg=msg,
        )

    def run(
        self,
        request: ReviewUserPlaybooksRequest,
        *,
        run_id: str | None = None,
    ) -> ReviewUserPlaybooksResponse:
        """Review newest-first and optionally commit each completed decision."""
        run_id = run_id or new_review_run_id()
        results: list[ReviewUserPlaybookResult] = []
        selected_count = 0
        try:
            selected = self.storage.get_user_playbooks(
                limit=request.top_k,
                status_filter=[None],
                start_time=int(request.start_time.timestamp()),
                end_time=int(request.end_time.timestamp()),
            )
            selected_count = len(selected)
            for reviewed in self._review_each(selected):
                results.append(reviewed.result)
                if not request.report_only:
                    self._apply_one(reviewed, run_id=run_id)
            return self._response(
                success=True,
                report_only=request.report_only,
                run_id=run_id,
                selected_count=selected_count,
                results=results,
                msg=f"Reviewed {len(results)} user playbook(s)",
            )
        except Exception:
            # The caller gets progress counts only: the exception text can carry
            # storage/model internals, so it belongs in the log, not the API.
            logger.exception(
                "event=playbook_review_failed run_id=%s reviewed=%d selected=%d",
                run_id,
                len(results),
                selected_count,
            )
            committed = sum(result.applied for result in results)
            progress = (
                f"Reviewed {len(results)} of {selected_count} selected playbook(s)"
            )
            if not request.report_only:
                progress += f"; {committed} decision(s) remain committed"
            return self._response(
                success=False,
                report_only=request.report_only,
                run_id=run_id,
                selected_count=selected_count,
                results=results,
                msg=f"{progress}: {_FAILED_MSG}",
            )
