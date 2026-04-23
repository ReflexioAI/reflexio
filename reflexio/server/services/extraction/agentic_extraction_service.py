"""AgenticExtractionService — 6-reader + 2-critic + lazy-reconciler orchestrator.

Phase 3 landing: the service runs three profile-angle readers and three
playbook-angle readers in parallel, then parallel critics for each lane, and
finally a reconciler only when critics raised cross-entity flags. The service
returns the vetted lanes without persisting to storage — Phase 6 wires this
output into the classic profile/playbook adapters and dedup pipelines.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from reflexio.server.services.extraction.critics import (
    CrossEntityFlag,
    PlaybookCritic,
    ProfileCritic,
    Reconciler,
    VettedPlaybook,
    VettedProfile,
    summarize,
)
from reflexio.server.services.extraction.readers import (
    PlaybookReader,
    ProfileReader,
    ReaderInputs,
)

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class _HasExtractionInputs(Protocol):
    """Duck-typed request for ``AgenticExtractionService.run``.

    Attributes:
        user_id (str): User the extraction is for.
        sessions (str): Rendered transcript string fed to the readers.
    """

    user_id: str
    sessions: str


@dataclass
class ExtractionResult:
    """Outcome of one AgenticExtractionService.run call.

    Attributes:
        profiles (list[VettedProfile]): Profile items that survived critic + reconciler.
        playbooks (list[VettedPlaybook]): Playbook items that survived critic + reconciler.
        skipped_reason (str | None): Set when the run bailed out early
            (e.g. missing prerequisites). ``None`` for successful runs.
    """

    profiles: list[VettedProfile] = field(default_factory=list)
    playbooks: list[VettedPlaybook] = field(default_factory=list)
    skipped_reason: str | None = None

    @classmethod
    def skipped(cls, reason: str) -> ExtractionResult:
        """Build a skipped result with an explanation string."""
        return cls(profiles=[], playbooks=[], skipped_reason=reason)


class AgenticExtractionService:
    """Agentic extraction orchestrator wired into the backend dispatcher.

    Construction matches ``ProfileGenerationService`` so ``build_extraction_service``
    can swap the two transparently: both accept ``llm_client`` and
    ``request_context`` as keyword arguments.

    Args:
        llm_client (LiteLLMClient): Configured LLM client for all agent calls.
        request_context (RequestContext): Request context providing
            ``storage`` and ``prompt_manager``.
        reader_workers (int): ThreadPool workers for the 6 parallel readers.
            Capped at 6 (one per angle).
        critic_workers (int): ThreadPool workers for the 2 parallel critics.
    """

    PROFILE_ANGLES: tuple[str, str, str] = ("facts", "context", "temporal")
    PLAYBOOK_ANGLES: tuple[str, str, str] = ("behavior", "trigger", "rationale")

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        reader_workers: int = 6,
        critic_workers: int = 2,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage
        self.prompt_manager = request_context.prompt_manager
        self._reader_workers = min(reader_workers, 6)
        self._critic_workers = min(critic_workers, 2)

    def run(self, request: _HasExtractionInputs) -> ExtractionResult:
        """Execute the full 6+2+reconciler pipeline for one request.

        Args:
            request: Object providing ``user_id`` and ``sessions`` attributes.

        Returns:
            ExtractionResult: Vetted profile and playbook lists, or a
            skipped-reason result when inputs are missing.
        """
        sessions = getattr(request, "sessions", None)
        if not sessions:
            return ExtractionResult.skipped("no sessions to extract")

        reader_inputs = ReaderInputs(sessions=sessions)
        profile_cands, playbook_cands = self._run_readers(reader_inputs)

        vetted_profiles, profile_flags = self._run_profile_critic(
            profile_cands, playbook_cands
        )
        vetted_playbooks, playbook_flags = self._run_playbook_critic(
            playbook_cands, profile_cands
        )

        all_flags = [*profile_flags, *playbook_flags]
        if all_flags:
            vetted_profiles, vetted_playbooks = self._run_reconciler(
                vetted_profiles, vetted_playbooks, all_flags
            )

        return ExtractionResult(
            profiles=list(vetted_profiles), playbooks=list(vetted_playbooks)
        )

    # ---------------- phase helpers ---------------- #

    def _run_readers(self, inputs: ReaderInputs) -> tuple[list[Any], list[Any]]:
        """Run all 6 angle readers in parallel; return (profile_cands, playbook_cands)."""
        with ThreadPoolExecutor(max_workers=self._reader_workers) as pool:
            profile_futs = [
                pool.submit(
                    ProfileReader(
                        angle,  # type: ignore[arg-type]
                        client=self.client,
                        prompt_manager=self.prompt_manager,
                    ).read,
                    inputs,
                )
                for angle in self.PROFILE_ANGLES
            ]
            playbook_futs = [
                pool.submit(
                    PlaybookReader(
                        angle,  # type: ignore[arg-type]
                        client=self.client,
                        prompt_manager=self.prompt_manager,
                    ).read,
                    inputs,
                )
                for angle in self.PLAYBOOK_ANGLES
            ]
            profile_cands = [c for f in profile_futs for c in _safe_result(f)]
            playbook_cands = [c for f in playbook_futs for c in _safe_result(f)]
        return profile_cands, playbook_cands

    def _run_profile_critic(
        self,
        profile_cands: list[Any],
        playbook_cands: list[Any],
    ) -> tuple[list[VettedProfile], list[CrossEntityFlag]]:
        critic = ProfileCritic(client=self.client, prompt_manager=self.prompt_manager)
        return critic.review(profile_cands, summarize(playbook_cands))

    def _run_playbook_critic(
        self,
        playbook_cands: list[Any],
        profile_cands: list[Any],
    ) -> tuple[list[VettedPlaybook], list[CrossEntityFlag]]:
        critic = PlaybookCritic(client=self.client, prompt_manager=self.prompt_manager)
        return critic.review(playbook_cands, summarize(profile_cands))

    def _run_reconciler(
        self,
        vetted_profiles: list[VettedProfile],
        vetted_playbooks: list[VettedPlaybook],
        flags: list[CrossEntityFlag],
    ) -> tuple[list[VettedProfile], list[VettedPlaybook]]:
        reconciler = Reconciler(client=self.client, prompt_manager=self.prompt_manager)
        return reconciler.resolve(vetted_profiles, vetted_playbooks, flags)


def _safe_result(fut: Future, *, timeout: float = 30.0) -> list[Any]:
    """Return a future's list-typed result or empty list on failure.

    Reader exceptions should not kill the whole extraction — they degrade
    recall for that angle, but other angles may still produce candidates.
    """
    try:
        return fut.result(timeout=timeout)
    except Exception as e:
        logger.warning("reader future failed: %s: %s", type(e).__name__, e)
        return []
