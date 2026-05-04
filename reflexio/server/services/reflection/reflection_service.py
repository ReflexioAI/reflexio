"""Sliding-window reflection: critique-and-revise of cited memories.

Invoked from ``GenerationService.run`` after the publish has saved its
interactions and before the extractor pool spins up. Mirrors the
extractor pattern: window of size ``window_size`` (global), advanced
every ``stride_size`` interactions per ``OperationStateManager``
bookmark. When the gate is open and at least one Assistant interaction
in the window cites a *current* user playbook / user profile row, the
service asks an LLM whether any of those cited rows should be replaced
in light of how they were applied across the window. Replacements
archive the cited row and insert a new current row with copied identity
and metadata.

Runs *before* extraction on purpose: extractors then read the
post-reflection state when computing existing-data context.

Failure isolation:
- Per-decision apply errors are caught locally so one bad decision
  cannot block the others.
- Top-level failures (extractor LLM call) are caught and logged; the
  bookmark is *not* advanced so the next publish retries the window.
- The caller (``GenerationService.run``) wraps reflection in its own
  try/except so a reflection bug never breaks the publish.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import (
    Citation,
    Interaction,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.reflection.reflection_extractor import (
    ReflectionExtractor,
)
from reflexio.server.services.reflection.reflection_service_utils import (
    REFLECTION_OPERATION_NAME,
    ReflectionDecision,
    ReflectionResult,
    ReflectionServiceRequest,
)

if TYPE_CHECKING:
    from reflexio.models.api_schema.internal_schema import (
        RequestInteractionDataModel,
    )
    from reflexio.server.api_endpoints.request_context import RequestContext

logger = logging.getLogger(__name__)


class ReflectionService:
    """Sliding-window reflection step.

    Args:
        request_context (RequestContext): Provides storage, prompt
            manager, and configurator.
        llm_client (LiteLLMClient): Shared LLM client.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
    ):
        self.request_context = request_context
        self.client = llm_client

    def run(self, request: ReflectionServiceRequest) -> ReflectionResult:
        """Execute one reflection pass.

        Always returns a ``ReflectionResult`` describing what happened.
        Routine failure modes (gate closed, no citations, missing rows,
        LLM error) do not raise. Storage exceptions propagate.
        """
        result = ReflectionResult()
        config = self.request_context.configurator.get_config()
        reflection_config = config.reflection_config if config else None
        if reflection_config is not None and not reflection_config.enabled:
            return result

        storage = self.request_context.storage
        if storage is None:
            return result

        window_size = config.window_size if config else 10
        stride_size = config.stride_size if config else 5
        sources = [request.source] if request.source else None

        mgr = OperationStateManager(
            storage,  # type: ignore[arg-type]
            self.request_context.org_id,
            REFLECTION_OPERATION_NAME,
        )

        # Gate: how many new interactions since last bookmark?
        _, new_models = mgr.get_extractor_state_with_new_interactions(
            REFLECTION_OPERATION_NAME,
            user_id=request.user_id,
            sources=sources,
        )
        new_count = sum(len(m.interactions) for m in new_models)
        if new_count < stride_size:
            return result
        result.gate_open = True

        # Pull window of last window_size interactions for the user.
        window_models, _flat = storage.get_last_k_interactions_grouped(
            user_id=request.user_id,
            k=window_size,
            sources=sources,
        )
        window_interactions = _flatten(window_models)
        if not window_interactions:
            mgr.update_extractor_bookmark(
                REFLECTION_OPERATION_NAME,
                processed_interactions=[],
                user_id=request.user_id,
            )
            return result

        citations = _collect_citations(window_interactions)
        result.cited_count = len(citations)
        if not citations:
            # Advance the bookmark — we did examine this window.
            mgr.update_extractor_bookmark(
                REFLECTION_OPERATION_NAME,
                processed_interactions=window_interactions,
                user_id=request.user_id,
            )
            return result

        cited_profiles, cited_playbooks, missing = self._resolve_cited_rows(
            user_id=request.user_id,
            citations=citations,
        )
        result.skipped_count = missing
        result.considered_count = len(cited_profiles) + len(cited_playbooks)
        if result.considered_count == 0:
            mgr.update_extractor_bookmark(
                REFLECTION_OPERATION_NAME,
                processed_interactions=window_interactions,
                user_id=request.user_id,
            )
            return result

        agent_context = (config.agent_context_prompt or "") if config else ""
        model_override = reflection_config.model if reflection_config else None
        extractor = ReflectionExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            agent_context=agent_context,
            model_override=model_override,
        )

        try:
            output = extractor.run(
                window_interactions=window_interactions,
                cited_profiles=cited_profiles,
                cited_user_playbooks=cited_playbooks,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "event=reflection_extractor_failed user_id=%s error_type=%s error=%s",
                request.user_id,
                type(exc).__name__,
                exc,
            )
            # Don't advance bookmark — let the next publish retry.
            return result

        result.ran = True
        profiles_by_id = {p.profile_id: p for p in cited_profiles}
        playbooks_by_id = {p.user_playbook_id: p for p in cited_playbooks}

        for decision in output.decisions:
            if decision.action == "no_change":
                result.no_change_count += 1
                continue
            try:
                applied = self._apply_replace(
                    request=request,
                    decision=decision,
                    profiles_by_id=profiles_by_id,
                    playbooks_by_id=playbooks_by_id,
                )
            except Exception as exc:  # noqa: BLE001 — per-decision isolation
                result.failed_count += 1
                logger.warning(
                    "event=reflection_apply_failed kind=%s target_id=%s "
                    "error_type=%s error=%s",
                    decision.target_kind,
                    decision.target_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            if applied:
                result.replaced_count += 1
            else:
                result.skipped_count += 1

        mgr.update_extractor_bookmark(
            REFLECTION_OPERATION_NAME,
            processed_interactions=window_interactions,
            user_id=request.user_id,
        )

        logger.info(
            "event=reflection_done user_id=%s gate_open=%s ran=%s "
            "cited=%d considered=%d no_change=%d replaced=%d "
            "skipped=%d failed=%d",
            request.user_id,
            result.gate_open,
            result.ran,
            result.cited_count,
            result.considered_count,
            result.no_change_count,
            result.replaced_count,
            result.skipped_count,
            result.failed_count,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_cited_rows(
        self,
        *,
        user_id: str,
        citations: list[Citation],
    ) -> tuple[list[UserProfile], list[UserPlaybook], int]:
        """Resolve citations to current rows; count missing/archived.

        Citations whose target row has already been archived (e.g. by
        the deduplicator earlier in the publish flow, or by a prior
        reflection pass) are silently skipped — only ``status IS NULL``
        rows are sent to the LLM. Uses ``get_profiles_by_ids`` /
        ``get_user_playbooks_by_ids`` so storage filters server-side
        rather than scanning every row for the user.

        Returns:
            Tuple of (cited_profiles, cited_playbooks, missing_count).
        """
        storage = self.request_context.storage
        wanted_profile_ids: set[str] = set()
        wanted_playbook_ids: set[int] = set()
        for c in citations:
            if c.kind == "profile":
                wanted_profile_ids.add(c.real_id)
            elif c.kind == "playbook":
                try:
                    wanted_playbook_ids.add(int(c.real_id))
                except (TypeError, ValueError):
                    continue

        cited_profiles: list[UserProfile] = []
        if wanted_profile_ids:
            cited_profiles = storage.get_profiles_by_ids(  # type: ignore[union-attr]
                user_id=user_id,
                profile_ids=list(wanted_profile_ids),
                status_filter=[None],
            )

        cited_playbooks: list[UserPlaybook] = []
        if wanted_playbook_ids:
            cited_playbooks = storage.get_user_playbooks_by_ids(  # type: ignore[union-attr]
                user_id=user_id,
                user_playbook_ids=list(wanted_playbook_ids),
                status_filter=[None],
            )

        found = len(cited_profiles) + len(cited_playbooks)
        missing = max(
            0,
            len(wanted_profile_ids) + len(wanted_playbook_ids) - found,
        )
        return cited_profiles, cited_playbooks, missing

    def _apply_replace(
        self,
        *,
        request: ReflectionServiceRequest,
        decision: ReflectionDecision,
        profiles_by_id: dict[str, UserProfile],
        playbooks_by_id: dict[int, UserPlaybook],
    ) -> bool:
        if decision.action != "replace" or not decision.new_content:
            return False
        if decision.target_kind == "profile":
            cited = profiles_by_id.get(decision.target_id)
            if cited is None:
                return False
            return self._replace_profile(request, decision, cited)
        if decision.target_kind == "playbook":
            try:
                target_id = int(decision.target_id)
            except (TypeError, ValueError):
                return False
            cited = playbooks_by_id.get(target_id)
            if cited is None:
                return False
            return self._replace_playbook(request, decision, cited)
        return False

    def _replace_profile(
        self,
        request: ReflectionServiceRequest,
        decision: ReflectionDecision,
        cited: UserProfile,
    ) -> bool:
        """Insert the replacement profile, then archive the cited row.

        Insert-first ordering means that if ``add_user_profile`` raises,
        the cited row stays current and the per-decision exception
        handler reports ``failed_count``. Only after the new row is
        durable do we flip the cited row to ARCHIVED — and if *that*
        fails we log at ERROR rather than silently dropping the user's
        data, leaving a transient duplicate that downstream dedup can
        clean up.
        """
        storage = self.request_context.storage
        if storage is None:
            return False
        now_ts = int(datetime.now(UTC).timestamp())
        new_profile = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id=cited.user_id,
            content=decision.new_content or cited.content,
            last_modified_timestamp=now_ts,
            generated_from_request_id=cited.generated_from_request_id,
            profile_time_to_live=(
                decision.new_profile_time_to_live or cited.profile_time_to_live
            ),
            custom_features=cited.custom_features,
            source=cited.source,
            status=None,
            extractor_names=cited.extractor_names,
        )
        storage.add_user_profile(cited.user_id, [new_profile])
        try:
            archived = storage.archive_profile_by_id(
                user_id=request.user_id, profile_id=cited.profile_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "event=reflection_archive_after_insert_failed kind=profile "
                "cited_id=%s new_id=%s error_type=%s error=%s",
                cited.profile_id,
                new_profile.profile_id,
                type(exc).__name__,
                exc,
            )
            return True
        if not archived:
            logger.error(
                "event=reflection_archive_after_insert_noop kind=profile "
                "cited_id=%s new_id=%s",
                cited.profile_id,
                new_profile.profile_id,
            )
        return True

    def _replace_playbook(
        self,
        request: ReflectionServiceRequest,
        decision: ReflectionDecision,
        cited: UserPlaybook,
    ) -> bool:
        """Insert the replacement playbook, then archive the cited row.

        Same insert-first ordering as ``_replace_profile``: if insert
        fails, the cited row stays current; if the post-insert archive
        fails, log at ERROR and accept a transient duplicate rather
        than losing data.
        """
        storage = self.request_context.storage
        if storage is None:
            return False
        owning_user_id = cited.user_id or request.user_id
        new_playbook = UserPlaybook(
            user_playbook_id=0,  # auto-assigned by storage
            user_id=owning_user_id,
            agent_version=cited.agent_version or request.agent_version,
            request_id=cited.request_id,
            playbook_name=cited.playbook_name,
            content=decision.new_content or cited.content,
            trigger=(
                decision.new_trigger
                if decision.new_trigger is not None
                else cited.trigger
            ),
            rationale=(
                decision.new_rationale
                if decision.new_rationale is not None
                else cited.rationale
            ),
            blocking_issue=cited.blocking_issue,
            status=None,
            source=cited.source,
            source_interaction_ids=[],
        )
        storage.save_user_playbooks([new_playbook])
        try:
            archived = storage.archive_user_playbook_by_id(
                user_id=owning_user_id,
                user_playbook_id=cited.user_playbook_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "event=reflection_archive_after_insert_failed kind=playbook "
                "cited_id=%s new_id=%s error_type=%s error=%s",
                cited.user_playbook_id,
                new_playbook.user_playbook_id,
                type(exc).__name__,
                exc,
            )
            return True
        if not archived:
            logger.error(
                "event=reflection_archive_after_insert_noop kind=playbook "
                "cited_id=%s new_id=%s",
                cited.user_playbook_id,
                new_playbook.user_playbook_id,
            )
        return True


def _flatten(
    request_models: list[RequestInteractionDataModel],
) -> list[Interaction]:
    """Flatten grouped request models into a flat list, oldest first."""
    out: list[Interaction] = []
    for rm in request_models:
        out.extend(rm.interactions)
    out.sort(key=lambda i: i.created_at)
    return out


def _collect_citations(interactions: list[Interaction]) -> list[Citation]:
    """Pull distinct citations off Assistant interactions."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for i in interactions:
        if i.role != "Assistant":
            continue
        for c in i.citations:
            key = (c.kind, c.real_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out
