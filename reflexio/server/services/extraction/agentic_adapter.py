"""Adapter wiring ``AgenticExtractionService`` into the classic publish flow.

The classic ``GenerationService.run`` expects a pair of generation services
(profile + playbook) it can fan out in parallel. The agentic orchestrator is
a single service that returns vetted ``VettedProfile`` / ``VettedPlaybook``
values without persistence.

This module provides ``AgenticExtractionRunner`` — a thin wrapper that:

1. Applies the same ``_cheap_should_run_reject`` pre-filter the classic
   path uses (honouring ``force_extraction``).
2. Renders the scoped interactions into a transcript string and runs
   the 6-reader / 2-critic / lazy-reconciler orchestrator.
3. Converts vetted items into ``UserProfile`` / ``UserPlaybook`` with
   identifiers, timestamps, and ``source`` filled in.
4. Runs the classic ``ProfileDeduplicator`` (when its feature flag is
   enabled) before persisting profiles — matches classic behaviour.
5. Runs the classic ``PlaybookDeduplicator`` (same feature flag) before
   persisting playbooks, and deletes superseded rows after successful save.
6. Persists profiles + playbooks via the existing storage APIs.
7. Triggers ``PlaybookAggregator`` for every configured playbook with an
   aggregation_config, unless ``skip_aggregation`` was set on the
   publish request.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import (
    NEVER_EXPIRES_TIMESTAMP,
    DeleteUserProfileRequest,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Request
from reflexio.server.services.base_generation_service import _cheap_should_run_reject
from reflexio.server.services.extraction.agentic_extraction_service import (
    AgenticExtractionService,
)
from reflexio.server.services.extraction.critics import VettedPlaybook, VettedProfile
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_deduplicator import PlaybookDeduplicator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.profile.profile_deduplicator import ProfileDeduplicator
from reflexio.server.services.service_utils import format_sessions_to_history_string
from reflexio.server.site_var.feature_flags import is_deduplicator_enabled

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import Interaction
    from reflexio.models.api_schema.service_schemas import PublishUserInteractionRequest
    from reflexio.models.config_schema import Config
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTL handling
# ---------------------------------------------------------------------------

# Seconds per ProfileTimeToLive literal. "infinity" is handled via
# NEVER_EXPIRES_TIMESTAMP and therefore has no entry here.
_TTL_SECONDS: dict[str, int] = {
    "one_day": 86_400,
    "one_week": 7 * 86_400,
    "one_month": 30 * 86_400,
    "one_quarter": 90 * 86_400,
    "one_year": 365 * 86_400,
}


def _compute_expiration(ttl: str, now_ts: int) -> int:
    """Map a ``time_to_live`` literal to an absolute expiration timestamp.

    Args:
        ttl (str): One of the six ``ProfileTimeToLive`` literal values.
        now_ts (int): Reference timestamp to add the TTL offset onto.

    Returns:
        int: ``NEVER_EXPIRES_TIMESTAMP`` when ``ttl == "infinity"``,
        otherwise ``now_ts + seconds``.
    """
    if ttl == "infinity":
        return NEVER_EXPIRES_TIMESTAMP
    return now_ts + _TTL_SECONDS[ttl]


# ---------------------------------------------------------------------------
# Request shim for the orchestrator's duck-typed Protocol
# ---------------------------------------------------------------------------


@dataclass
class _ReqShim:
    """Satisfies the ``_HasExtractionInputs`` Protocol on ``AgenticExtractionService``."""

    user_id: str
    sessions: str


# ---------------------------------------------------------------------------
# Vetted -> User converters
# ---------------------------------------------------------------------------


def _vetted_to_user_profile(
    vp: VettedProfile,
    *,
    user_id: str,
    request_id: str,
    source: str | None,
    now_ts: int,
) -> UserProfile:
    """Convert a ``VettedProfile`` into a persistable ``UserProfile``."""
    return UserProfile(
        profile_id=str(uuid.uuid4()),
        user_id=user_id,
        content=vp.content,
        last_modified_timestamp=now_ts,
        generated_from_request_id=request_id,
        profile_time_to_live=ProfileTimeToLive(vp.time_to_live),
        expiration_timestamp=_compute_expiration(vp.time_to_live, now_ts),
        source=source,
        extractor_names=["agentic"],
        source_span=vp.source_span,
        notes=vp.notes,
        reader_angle=vp.reader_angle,
    )


def _vetted_to_user_playbook(
    vpb: VettedPlaybook,
    *,
    user_id: str,
    request_id: str,
    agent_version: str,
    source: str | None,
    now_ts: int,
) -> UserPlaybook:
    """Convert a ``VettedPlaybook`` into a persistable ``UserPlaybook``."""
    return UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version=agent_version,
        request_id=request_id,
        created_at=now_ts,
        content=vpb.content or "",
        trigger=vpb.trigger,
        rationale=vpb.rationale,
        source=source,
        source_span=vpb.source_span,
        notes=vpb.notes,
        reader_angle=vpb.reader_angle,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AgenticExtractionRunner:
    """Wrap ``AgenticExtractionService`` so it mirrors the classic publish contract.

    Args:
        llm_client (LiteLLMClient): Configured LLM client for readers / critics
            / reconciler / deduplicator / aggregator.
        request_context (RequestContext): Provides ``storage`` + ``prompt_manager``
            + ``configurator``.
        org_id (str): Organisation ID, used for feature-flag checks and
            downstream aggregator wiring.
        output_pending_status (bool): Mirror the classic
            ``ProfileGenerationService.output_pending_status`` flag so rerun
            flows can surface pending profiles consistently.
    """

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        org_id: str,
        output_pending_status: bool = False,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage
        self.org_id = org_id
        self.output_pending_status = output_pending_status
        self.service = AgenticExtractionService(
            llm_client=llm_client, request_context=request_context
        )

    def run(
        self,
        *,
        publish_request: PublishUserInteractionRequest,
        request_id: str,
        new_interactions: list[Interaction],
        new_request: Request,
        config: Config,
    ) -> list[str]:
        """Run agentic extraction + dedup + aggregation and persist.

        Args:
            publish_request (PublishUserInteractionRequest): The original
                publish request — ``source``, ``agent_version``,
                ``force_extraction``, ``skip_aggregation`` are read from it.
            request_id (str): Per-publish UUID assigned by ``GenerationService.run``.
            new_interactions (list[Interaction]): Interactions persisted for
                this publish, used for both the pre-filter and transcript.
            new_request (Request): The ``Request`` row just persisted; used
                to synthesise the precheck ``RequestInteractionDataModel``.
            config (Config): Resolved top-level config. ``user_playbook_extractor_configs``
                drive the aggregator loop.

        Returns:
            list[str]: Non-fatal warnings to surface back to the caller.
        """
        warnings: list[str] = []
        session_data_models = self._build_session_data_models(
            new_interactions=new_interactions, new_request=new_request
        )

        # (1) Pre-filter — cheap reject for sessions with no learnable signal.
        if not publish_request.force_extraction:
            reason = _cheap_should_run_reject(session_data_models)
            if reason is not None:
                logger.info(
                    "agentic pre-filter rejected: reason=%s identifier=%s",
                    reason,
                    publish_request.user_id,
                )
                return warnings

        # (2) Run the orchestrator against the rendered transcript.
        sessions_str = format_sessions_to_history_string(session_data_models)
        result = self.service.run(
            _ReqShim(user_id=publish_request.user_id, sessions=sessions_str)
        )
        if result.skipped_reason:
            logger.info("agentic extraction skipped: %s", result.skipped_reason)
            return warnings

        # (3) Convert VettedProfile / VettedPlaybook into persistable shapes.
        now_ts = int(datetime.now(UTC).timestamp())
        source = publish_request.source or None
        new_profiles = [
            _vetted_to_user_profile(
                vp,
                user_id=publish_request.user_id,
                request_id=request_id,
                source=source,
                now_ts=now_ts,
            )
            for vp in result.profiles
        ]
        new_playbooks = [
            _vetted_to_user_playbook(
                vpb,
                user_id=publish_request.user_id,
                request_id=request_id,
                agent_version=publish_request.agent_version,
                source=source,
                now_ts=now_ts,
            )
            for vpb in result.playbooks
        ]

        # (4) Profile dedup — matches classic when the feature flag is on.
        existing_ids_to_delete: list[str] = []
        if new_profiles and is_deduplicator_enabled(self.org_id):
            deduplicator = ProfileDeduplicator(
                request_context=self.request_context, llm_client=self.client
            )
            try:
                (
                    new_profiles,
                    existing_ids_to_delete,
                    _superseded,
                ) = deduplicator.deduplicate(
                    new_profiles, publish_request.user_id, request_id
                )
                logger.info(
                    "Agentic dedup: %d profiles retained, %d superseded IDs to delete",
                    len(new_profiles),
                    len(existing_ids_to_delete),
                )
            except Exception as e:  # noqa: BLE001 - dedup failures degrade gracefully
                logger.warning(
                    "agentic profile deduplicator failed: %s: %s",
                    type(e).__name__,
                    e,
                )
                warnings.append(f"profile deduplicator failed: {e}")

        # Apply source + status to the deduplicated set (classic parity).
        for p in new_profiles:
            p.source = source
            p.status = Status.PENDING if self.output_pending_status else None

        # (5) Persist profiles + delete superseded, if storage is configured.
        if self.storage is None:
            logger.warning("agentic runner has no storage; skipping persistence")
            return warnings

        if new_profiles:
            self.storage.add_user_profile(publish_request.user_id, new_profiles)
        for pid in existing_ids_to_delete:
            try:
                self.storage.delete_user_profile(
                    DeleteUserProfileRequest(
                        user_id=publish_request.user_id, profile_id=pid
                    )
                )
            except Exception as e:  # noqa: BLE001 - degrade gracefully on delete
                warnings.append(f"delete superseded profile {pid} failed: {e}")

        # (6a) Playbook dedup — matches classic's PlaybookGenerationService._process_results.
        playbook_ids_to_delete: list[int] = []
        if new_playbooks and is_deduplicator_enabled(self.org_id):
            new_playbooks, playbook_ids_to_delete = self._run_playbook_dedup(
                new_playbooks=new_playbooks,
                publish_request=publish_request,
                request_id=request_id,
                config=config,
                warnings=warnings,
            )

        # (6b) Apply status to the deduplicated playbook set (classic parity).
        for pb in new_playbooks:
            pb.status = Status.PENDING if self.output_pending_status else None

        # (6c) Persist playbooks, then delete superseded IDs only on successful save.
        if new_playbooks:
            try:
                self.storage.save_user_playbooks(new_playbooks)
                if playbook_ids_to_delete:
                    try:
                        deleted = self.storage.delete_user_playbooks_by_ids(
                            playbook_ids_to_delete
                        )
                        logger.info("Deleted %d superseded user playbook(s)", deleted)
                    except Exception as e:  # noqa: BLE001 - degrade gracefully
                        warnings.append(f"delete superseded playbooks failed: {e}")
            except Exception as e:  # noqa: BLE001 - save failures surface as warnings
                logger.warning(
                    "agentic save_user_playbooks failed: %s: %s",
                    type(e).__name__,
                    e,
                )
                warnings.append(f"save_user_playbooks failed: {e}")

        # (7) Playbook aggregation — mirrors classic's per-config loop.
        if new_playbooks and not publish_request.skip_aggregation:
            self._run_aggregation(
                config=config, publish_request=publish_request, warnings=warnings
            )

        return warnings

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_session_data_models(
        *, new_interactions: list[Interaction], new_request: Request
    ) -> list[RequestInteractionDataModel]:
        """Wrap this publish's interactions in a single-element batch for the precheck."""
        return [
            RequestInteractionDataModel(
                session_id=new_request.session_id or "",
                request=new_request,
                interactions=list(new_interactions),
            )
        ]

    def _run_playbook_dedup(
        self,
        *,
        new_playbooks: list[UserPlaybook],
        publish_request: PublishUserInteractionRequest,
        request_id: str,
        config: Config,
        warnings: list[str],
    ) -> tuple[list[UserPlaybook], list[int]]:
        """Run the classic ``PlaybookDeduplicator`` on this publish's playbooks.

        Mirrors ``PlaybookGenerationService._process_results`` at
        ``playbook_generation_service.py:271-305``: pulls ``dedup_config`` from
        the first extractor config that has one, wraps the list as the
        ``list[list[UserPlaybook]]`` the deduplicator expects, and returns
        the deduplicated playbooks plus IDs of superseded existing rows the
        caller should delete after a successful save.

        Failures degrade gracefully: the original ``new_playbooks`` are
        returned unchanged and the error is appended to ``warnings``.
        """
        dedup_config = next(
            (
                c.deduplication_config
                for c in (config.user_playbook_extractor_configs or [])
                if c.deduplication_config
            ),
            None,
        )
        try:
            deduplicator = PlaybookDeduplicator(
                request_context=self.request_context,
                llm_client=self.client,
                dedup_config=dedup_config,
            )
            deduped, ids_to_delete = deduplicator.deduplicate(
                [new_playbooks],
                request_id,
                publish_request.agent_version,
                user_id=publish_request.user_id,
            )
            logger.info(
                "Agentic playbook dedup: %d playbooks retained, %d superseded IDs to delete",
                len(deduped),
                len(ids_to_delete),
            )
            # Classic falls back to the original list when deduper returns
            # nothing; mirror that safety net.
            retained = deduped or new_playbooks
            return retained, ids_to_delete
        except Exception as e:  # noqa: BLE001 - dedup failures degrade gracefully
            logger.warning(
                "agentic playbook deduplicator failed: %s: %s",
                type(e).__name__,
                e,
            )
            warnings.append(f"playbook deduplicator failed: {e}")
            return new_playbooks, []

    def _run_aggregation(
        self,
        *,
        config: Config,
        publish_request: PublishUserInteractionRequest,
        warnings: list[str],
    ) -> None:
        """Run ``PlaybookAggregator`` for every configured playbook with an ``aggregation_config``."""
        for pb_cfg in config.user_playbook_extractor_configs or []:
            if not getattr(pb_cfg, "aggregation_config", None):
                continue
            try:
                aggregator = PlaybookAggregator(
                    llm_client=self.client,
                    request_context=self.request_context,
                    agent_version=publish_request.agent_version,
                )
                aggregator.run(
                    PlaybookAggregatorRequest(
                        agent_version=publish_request.agent_version,
                        playbook_name=pb_cfg.extractor_name,
                    )
                )
            except Exception as e:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "agentic aggregation failed for %s: %s: %s",
                    pb_cfg.extractor_name,
                    type(e).__name__,
                    e,
                )
                warnings.append(f"aggregation failed for {pb_cfg.extractor_name}: {e}")
