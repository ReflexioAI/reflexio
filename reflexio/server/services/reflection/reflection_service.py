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
from dataclasses import dataclass
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
from reflexio.server.tracing import sentry_tags

if TYPE_CHECKING:
    from reflexio.models.api_schema.internal_schema import (
        RequestInteractionDataModel,
    )
    from reflexio.server.api_endpoints.request_context import RequestContext

logger = logging.getLogger(__name__)

# Fallback per-pass revision cap when no ReflectionConfig is available in
# the apply path. Mirrors ReflectionConfig.max_revisions_per_pass default.
_DEFAULT_MAX_REVISIONS_PER_PASS = 8


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

        # Post-horizon filter: only send citations with enough follow-up context.
        post_horizon_size = (
            reflection_config.post_horizon_size if reflection_config else 3
        )
        eligible = _filter_citations_by_horizon(
            citations=citations,
            window=window_interactions,
            post_horizon_size=post_horizon_size,
            stride_size=stride_size,
        )
        deferred_count = len(citations) - len(eligible)
        result.skipped_count += deferred_count
        if not eligible:
            mgr.update_extractor_bookmark(
                REFLECTION_OPERATION_NAME,
                processed_interactions=window_interactions,
                user_id=request.user_id,
            )
            return result

        eligible_citations = [e.citation for e in eligible]
        horizon_by_key = {
            (e.citation.kind, e.citation.real_id): e.has_full_horizon for e in eligible
        }
        cited_profiles, cited_playbooks, missing = self._resolve_cited_rows(
            user_id=request.user_id,
            citations=eligible_citations,
        )
        result.skipped_count += missing
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
                horizon_by_key=horizon_by_key,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            with sentry_tags(
                subsystem="reflection",
                op="extractor",
                org_id=self.request_context.org_id,
                user_id=request.user_id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=reflection_extractor_failed user_id=%s",
                    request.user_id,
                )
            # Don't advance bookmark — let the next publish retry.
            return result

        result.ran = True
        profiles_by_id = {p.profile_id: p for p in cited_profiles}
        playbooks_by_id = {p.user_playbook_id: p for p in cited_playbooks}

        max_revisions_per_pass = (
            reflection_config.max_revisions_per_pass
            if reflection_config is not None
            else _DEFAULT_MAX_REVISIONS_PER_PASS
        )

        for decision in output.decisions:
            if not _is_revision(decision):
                result.no_change_count += 1
                continue
            # Per-pass cap: once we've applied max_revisions_per_pass
            # revisions, skip any further revision-intent decisions.
            if result.revised_count >= max_revisions_per_pass:
                result.capped_count += 1
                continue
            try:
                self._validate_decision(decision, profiles_by_id, playbooks_by_id)
                applied = self._apply_revision(
                    request=request,
                    decision=decision,
                    profiles_by_id=profiles_by_id,
                    playbooks_by_id=playbooks_by_id,
                )
            except Exception as exc:  # noqa: BLE001 — per-decision isolation
                result.failed_count += 1
                with sentry_tags(
                    subsystem="reflection",
                    op="apply_decision",
                    org_id=self.request_context.org_id,
                    user_id=request.user_id,
                    target_kind=decision.target_kind,
                    target_id=decision.target_id,
                    error_type=type(exc).__name__,
                ):
                    logger.exception(
                        "event=reflection_apply_failed kind=%s target_id=%s",
                        decision.target_kind,
                        decision.target_id,
                    )
                continue
            if applied:
                result.revised_count += 1
                # Field-derivable granular counters (no mode label exists).
                if decision.new_trigger is not None:
                    result.trigger_revised_count += 1
                if decision.new_content is not None:
                    result.content_revised_count += 1
                if decision.new_profile_time_to_live is not None:
                    result.ttl_changed_count += 1
            else:
                result.skipped_count += 1

        mgr.update_extractor_bookmark(
            REFLECTION_OPERATION_NAME,
            processed_interactions=window_interactions,
            user_id=request.user_id,
        )

        logger.info(
            "event=reflection_done user_id=%s gate_open=%s ran=%s "
            "cited=%d considered=%d no_change=%d revised=%d "
            "trigger_revised=%d content_revised=%d ttl_changed=%d capped=%d "
            "skipped=%d failed=%d",
            request.user_id,
            result.gate_open,
            result.ran,
            result.cited_count,
            result.considered_count,
            result.no_change_count,
            result.revised_count,
            result.trigger_revised_count,
            result.content_revised_count,
            result.ttl_changed_count,
            result.capped_count,
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

    def _validate_decision(
        self,
        decision: ReflectionDecision,
        profiles_by_id: dict[str, UserProfile],  # noqa: ARG002 - kept for signature parity with _apply_revision
        playbooks_by_id: dict[int, UserPlaybook],
    ) -> None:
        """Raise if a playbook content rewrite omits ``new_rationale``.

        Flip is LLM-reported, not derived: the ``memory_reflection`` prompt
        instructs the model to set ``new_rationale`` whenever it changes a
        rule's orientation (a flip), and to set it on substance rewrites of
        playbook content as well. So any playbook revision that sets
        ``new_content`` must carry a ``new_rationale`` — this preserves the
        flip-requires-rationale audit-trail spirit without re-deriving
        polarity from wording. Trigger-only / TTL-only revisions are exempt.

        Per-decision try/except in the caller catches these and counts
        them as failed_count.

        Args:
            decision (ReflectionDecision): The decision to validate.
            profiles_by_id (dict[str, UserProfile]): Resolved profile
                rows keyed by profile_id.
            playbooks_by_id (dict[int, UserPlaybook]): Resolved playbook
                rows keyed by user_playbook_id.

        Raises:
            ValueError: When a playbook ``new_content`` revision omits
                ``new_rationale``.
        """
        if decision.target_kind == "profile":
            return
        # Playbook
        try:
            target_id = int(decision.target_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"playbook target_id not an int: {decision.target_id!r}"
            ) from exc
        cited = playbooks_by_id.get(target_id)
        if cited is None:
            return  # apply step will mark as skipped
        if decision.new_content is not None and not decision.new_rationale:
            raise ValueError("playbook content revision must include new_rationale")

    def _apply_revision(
        self,
        *,
        request: ReflectionServiceRequest,
        decision: ReflectionDecision,
        profiles_by_id: dict[str, UserProfile],
        playbooks_by_id: dict[int, UserPlaybook],
    ) -> bool:
        """Apply a revision decision.

        Args:
            request (ReflectionServiceRequest): The current service
                request (for user_id / agent_version).
            decision (ReflectionDecision): The revision to apply.
            profiles_by_id (dict[str, UserProfile]): Resolved profile
                rows keyed by profile_id.
            playbooks_by_id (dict[int, UserPlaybook]): Resolved playbook
                rows keyed by user_playbook_id.

        Returns:
            bool: ``applied``. ``False`` means the target row could not be
            resolved (target archived between resolve and apply, etc.).

        Flip is no longer derived here: orientation changes are LLM-reported
        via the rewritten ``new_content`` + ``new_rationale`` the prompt
        emits, not inferred from wording at apply time.
        """
        if decision.target_kind == "profile":
            cited_p = profiles_by_id.get(decision.target_id)
            if cited_p is None:
                return False
            return self._replace_profile(request, decision, cited_p)
        # Playbook
        try:
            target_id = int(decision.target_id)
        except (TypeError, ValueError):
            return False
        cited_pb = playbooks_by_id.get(target_id)
        if cited_pb is None:
            return False
        return self._replace_playbook(request, decision, cited_pb)

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
        _log_edit_magnitude(
            kind="profile",
            target_id=cited.profile_id,
            old_content=cited.content,
            new_content=new_profile.content,
        )
        storage.add_user_profile(cited.user_id, [new_profile])
        try:
            archived = storage.archive_profile_by_id(
                user_id=request.user_id, profile_id=cited.profile_id
            )
        except Exception as exc:  # noqa: BLE001
            with sentry_tags(
                subsystem="reflection",
                op="archive_after_insert",
                kind="profile",
                org_id=self.request_context.org_id,
                user_id=cited.user_id,
                cited_id=cited.profile_id,
                new_id=new_profile.profile_id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=reflection_archive_after_insert_failed kind=profile "
                    "cited_id=%s new_id=%s",
                    cited.profile_id,
                    new_profile.profile_id,
                )
            return True
        if not archived:
            with sentry_tags(
                subsystem="reflection",
                op="archive_after_insert_noop",
                kind="profile",
                org_id=self.request_context.org_id,
                user_id=cited.user_id,
                cited_id=cited.profile_id,
                new_id=new_profile.profile_id,
            ):
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
            status=None,
            source=cited.source,
            source_interaction_ids=list(cited.source_interaction_ids),
        )
        _log_edit_magnitude(
            kind="playbook",
            target_id=str(cited.user_playbook_id),
            old_content=cited.content,
            new_content=new_playbook.content,
        )
        # Flip is LLM-reported, not derived: when the model changes a rule's
        # orientation it rewrites ``new_content`` and names the motivating
        # failure/observation in ``new_rationale`` (per the memory_reflection
        # prompt). A rewrite-with-rationale is the LLM-reported flip/revision
        # signal — log it for observability without re-deriving polarity.
        if decision.new_content is not None and decision.new_rationale:
            logger.info(
                "reflection.content_revision playbook_id=%s "
                'content_excerpt="%s" prior_excerpt="%s" rationale_excerpt="%s"',
                cited.user_playbook_id,
                (new_playbook.content or "")[:120].replace('"', "'"),
                (cited.content or "")[:120].replace('"', "'"),
                (decision.new_rationale or "")[:120].replace('"', "'"),
            )
        storage.save_user_playbooks([new_playbook])
        try:
            archived = storage.archive_user_playbook_by_id(
                user_id=owning_user_id,
                user_playbook_id=cited.user_playbook_id,
            )
        except Exception as exc:  # noqa: BLE001
            with sentry_tags(
                subsystem="reflection",
                op="archive_after_insert",
                kind="playbook",
                org_id=self.request_context.org_id,
                user_id=owning_user_id,
                cited_id=cited.user_playbook_id,
                new_id=new_playbook.user_playbook_id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=reflection_archive_after_insert_failed kind=playbook "
                    "cited_id=%s new_id=%s",
                    cited.user_playbook_id,
                    new_playbook.user_playbook_id,
                )
            return True
        if not archived:
            with sentry_tags(
                subsystem="reflection",
                op="archive_after_insert_noop",
                kind="playbook",
                org_id=self.request_context.org_id,
                user_id=owning_user_id,
                cited_id=cited.user_playbook_id,
                new_id=new_playbook.user_playbook_id,
            ):
                logger.error(
                    "event=reflection_archive_after_insert_noop kind=playbook "
                    "cited_id=%s new_id=%s",
                    cited.user_playbook_id,
                    new_playbook.user_playbook_id,
                )
        return True


_PROFILE_REVISION_FIELDS: tuple[str, ...] = (
    "new_content",
    "new_profile_time_to_live",
)
_PLAYBOOK_REVISION_FIELDS: tuple[str, ...] = (
    "new_content",
    "new_trigger",
    "new_rationale",
)


_REVISION_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "profile": _PROFILE_REVISION_FIELDS,
    "playbook": _PLAYBOOK_REVISION_FIELDS,
}


def _log_edit_magnitude(
    *,
    kind: str,
    target_id: str,
    old_content: str | None,
    new_content: str | None,
) -> None:
    """Log a cheap edit-magnitude signal for one applied revision.

    The magnitude is the content-size delta (new length minus old length)
    in characters — a coarse proxy for how large the revision is, useful
    for offline regularization analysis without storing diffs.

    Args:
        kind (str): ``"profile"`` or ``"playbook"``.
        target_id (str): Id of the cited row being replaced.
        old_content (str | None): Cited row content before the revision.
        new_content (str | None): Replacement row content after the revision.
    """
    old_len = len(old_content or "")
    new_len = len(new_content or "")
    logger.info(
        "event=reflection_edit_magnitude target_kind=%s target_id=%s "
        "old_len=%d new_len=%d delta=%d",
        kind,
        target_id,
        old_len,
        new_len,
        new_len - old_len,
    )


def _is_revision(decision: ReflectionDecision) -> bool:
    """Return True iff a revision field relevant to the target kind is set.

    Splits the revision-field set by ``target_kind`` so a profile decision
    is not classified as a revision purely because a playbook-only field
    (e.g. ``new_trigger``) happens to be populated. This prevents
    unnecessary replace/archive churn that would otherwise produce no
    effective change.

    Args:
        decision (ReflectionDecision): The decision to inspect.

    Returns:
        bool: True if at least one kind-relevant revision field is non-None.
    """
    fields = _REVISION_FIELDS_BY_KIND.get(
        decision.target_kind, _PLAYBOOK_REVISION_FIELDS
    )
    return any(getattr(decision, f) is not None for f in fields)


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


@dataclass(frozen=True)
class _EligibleCitation:
    """A citation that passed the post-horizon filter.

    Attributes:
        citation (Citation): The citation itself.
        position (int): 0-indexed position in the window where it
            appeared (earliest occurrence — see deduplication note in
            ``_filter_citations_by_horizon``).
        has_full_horizon (bool): True iff at least
            ``post_horizon_size`` interactions follow the citation in
            the window. False indicates a ``last_chance`` judgment
            with weaker evidence.
    """

    citation: Citation
    position: int
    has_full_horizon: bool


def _filter_citations_by_horizon(
    citations: list[Citation],
    window: list[Interaction],
    post_horizon_size: int,
    stride_size: int,
) -> list[_EligibleCitation]:
    """Filter citations to those that have enough post-citation context.

    For each unique ``(kind, real_id)``, finds the **earliest** occurrence
    in the window (maximizes follow-up turns) and decides:

    - ``after_count >= post_horizon_size`` → eligible, ``has_full_horizon=True``.
    - ``position < stride_size`` → eligible (last-chance, about to fall
      out of the window next stride), ``has_full_horizon=False``.
    - otherwise → deferred (excluded from this pass).

    Only Assistant-role interactions contribute citations to consider —
    user-role interactions are skipped even if their ``citations`` list is
    populated.

    Args:
        citations (list[Citation]): Distinct citations collected from
            assistant turns in the window.
        window (list[Interaction]): The reflection window, oldest first.
        post_horizon_size (int): Minimum follow-up turns required for a
            full-horizon judgment. Zero disables the horizon check.
        stride_size (int): Used to detect citations about to fall out.

    Returns:
        list[_EligibleCitation]: Citations to send to the LLM, each
        paired with its window position and horizon flag.
    """
    seen: dict[tuple[str, str], int] = {}
    for idx, interaction in enumerate(window):
        if interaction.role != "Assistant":
            continue
        for c in interaction.citations:
            key = (c.kind, c.real_id)
            seen.setdefault(key, idx)  # earliest occurrence only

    citation_by_key = {(c.kind, c.real_id): c for c in citations}

    out: list[_EligibleCitation] = []
    for key, position in seen.items():
        cite = citation_by_key.get(key)
        if cite is None:
            continue
        after_count = len(window) - position - 1
        if post_horizon_size <= 0 or after_count >= post_horizon_size:
            out.append(
                _EligibleCitation(
                    citation=cite, position=position, has_full_horizon=True
                )
            )
        elif position < stride_size:
            out.append(
                _EligibleCitation(
                    citation=cite, position=position, has_full_horizon=False
                )
            )
    return out
