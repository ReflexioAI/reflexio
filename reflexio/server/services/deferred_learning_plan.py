"""Write-plan dataclasses for the durable-learning compute/persist split (gate b).

Compute resolves these in-memory plans issuing **no** learning DB write; persist
applies them inside one short fenced ``commit_scope``. This module starts with the
extractor stride-bookmark advance and grows additional write-plan dataclasses in the
later gate-(b) tasks (profile/playbook/reflection/generation/deferred plans).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import (
        UserPlaybook,
        UserProfile,
    )
    from reflexio.models.api_schema.service_schemas import Interaction
    from reflexio.server.llm.token_accounting import RunTokenTotals
    from reflexio.server.services.base_generation_service import PreparedGenerationRun
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionDecision,
        ReflectionResult,
        ReflectionServiceRequest,
    )


@dataclass(frozen=True)
class ExtractorBookmarkAdvance:
    """The extractor stride-bookmark advance, deferred out of the extractor (F1).

    The extractor no longer self-advances its bookmark inside ``run()`` (a DB
    write). It emits this on the ``ExtractionOutcome`` instead, so the advance is
    applied later — inside the persist fence on the durable path, or right after
    result-processing on the synchronous ``.run()`` path — keeping the bookmark
    advance atomic with the row writes it corresponds to.
    """

    extractor_name: str
    processed_interactions: list[Interaction]
    # str | None (not str): the playbook extractor runs org-level with
    # user_id=None (PlaybookGenerationServiceConfig.user_id is Optional), which
    # maps to an unscoped bookmark key. update_extractor_bookmark accepts None.
    user_id: str | None


@dataclass
class ReflectionWritePlan:
    """Resolved reflection write-plan for the compute/persist split (F5, V2).

    ``ReflectionService.compute`` produces this issuing **no** learning DB write;
    ``persist`` applies the whole cap/validate/CAS apply loop verbatim inside the
    fence, then advances the reflection stride-bookmark; ``emit_side_effects``
    fires the post-commit billing + aggregation triggers. The apply loop stays a
    single monolith (no per-revision ``ResolvedReflectionRevision`` dataclass).

    Attributes:
        request: The originating reflection request (user_id / request_id /
            agent_version).
        result: The reflection result. Compute seeds it with the gate/citation
            counters resolved so far; persist's apply loop mutates it in place
            (revised/failed/skipped/... counters) as it applies each decision.
        decisions: ``ReflectionOutput.decisions`` to apply — ``[]`` on the four
            early bookmark-advance paths (nothing to apply, only advance).
        profiles_by_id: Cited profile rows resolved in compute, keyed by
            ``profile_id`` (the apply loop's ``_apply_revision`` input).
        playbooks_by_id: Cited playbook rows resolved in compute, keyed by
            ``user_playbook_id``.
        max_revisions_per_pass: Per-pass revision cap applied inside the loop.
        bookmark_interactions: The window the reflection stride-bookmark advances
            over in persist (``[]`` on the empty-window advance path).
        advance_bookmark: ``True`` on all five bookmark-advance paths (the four
            early-advance returns + the normal end). Persist advances the
            reflection bookmark iff this is ``True``.
        record_learnings: ``True`` only on the normal-end path — gates the
            post-commit ``_record_learnings_generated`` billing event in emit.
        replacement_profiles: Replacement ``UserProfile`` rows whose embeddings
            were precomputed in compute (V2), keyed by the **cited** profile_id
            they replace. Persist inserts these with ``skip_embedding=True`` so
            no embedding runs inside the fence. Empty on non-compute (direct)
            ``_replace_*`` calls, which then build+embed the row themselves.
        replacement_playbooks: Same, keyed by the cited ``user_playbook_id``.
        aggregation_successor_ids: Playbook successor ids collected during the
            persist apply loop; their aggregation trigger is dispatched in
            ``emit_side_effects`` (post-commit) so it never fires on a
            rolled-back job.
    """

    request: ReflectionServiceRequest
    result: ReflectionResult
    decisions: list[ReflectionDecision]
    profiles_by_id: dict[str, UserProfile]
    playbooks_by_id: dict[int, UserPlaybook]
    max_revisions_per_pass: int
    bookmark_interactions: list[Interaction]
    advance_bookmark: bool
    record_learnings: bool
    replacement_profiles: dict[str, UserProfile] = field(default_factory=dict)
    replacement_playbooks: dict[int, UserPlaybook] = field(default_factory=dict)
    aggregation_successor_ids: list[int] = field(default_factory=list)


@dataclass
class GenerationComputePlan:
    """Resolved compute output of one ``BaseGenerationService`` run (gate b).

    ``compute_generation`` runs the prepare gate + extractor + dedup/embedding
    resolution (``_resolve_write_plan``) and drives the ``agent_run`` rows to
    their terminal state (``_finalize_extraction_runs`` — agent_run only, §4.3),
    issuing **no** learning DB write. ``persist_generation`` applies
    ``write_plan`` + the extractor bookmark advance inside the fence;
    ``emit_generation_side_effects`` fires the post-commit telemetry + billing.

    The billing inputs (``extraction_run_ids`` / ``token_totals`` /
    ``generated_count`` / ``prepared``) are **snapshotted at compute time** so
    the fence-crossing emit reads this plan rather than the reused service
    instance's mutable ``_last_*`` accumulators (purity contract, plan §File
    Structure). See ``emit_generation_side_effects`` for the single-use-instance
    invariant that also keeps the money helper's ``self._last_*`` reads safe.

    Attributes:
        prepared: The prepared generation run (identifier / extractor_name /
            extractor_config), reused by emit for telemetry + billing input.
        generated_count: Learnings produced by this extraction run.
        write_plan: The resolved write-plan (``ProfileWritePlan`` /
            ``PlaybookWritePlan`` in Tasks 6-7, a ``_LegacyItems`` shim marker
            until then) or ``None`` when the extractor produced nothing.
        bookmark_advance: The deferred extractor stride-bookmark advance (F1),
            applied inside the persist scope. ``None`` when the extractor
            produced no output or the service has no stride bookmark.
        generation_start: ``perf_counter`` captured at compute start; emit reads
            it for the ``generation_succeeded`` ``duration_ms`` (parity).
        extraction_run_ids: Snapshot of the run's ``agent_run`` ids.
        token_totals: Snapshot of the run's LLM token totals (billing cost
            facet), or ``None`` when the extractor reported none.
    """

    prepared: PreparedGenerationRun[Any]
    generated_count: int
    write_plan: Any
    bookmark_advance: ExtractorBookmarkAdvance | None
    generation_start: float
    extraction_run_ids: list[str]
    token_totals: RunTokenTotals | None
