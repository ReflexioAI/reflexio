"""Write-plan dataclasses for the durable-learning compute/persist split (gate b).

Compute resolves these in-memory plans issuing **no** learning DB write; persist
applies them inside one short fenced ``commit_scope``. This module starts with the
extractor stride-bookmark advance and grows additional write-plan dataclasses in the
later gate-(b) tasks (profile/playbook/generation/deferred plans).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import (
        LineageContext,
        UserPlaybook,
        UserProfile,
    )
    from reflexio.models.api_schema.service_schemas import Interaction
    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.token_accounting import RunTokenTotals
    from reflexio.server.services.base_generation_service import (
        BaseGenerationService,
        PreparedGenerationRun,
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
class ProfileWritePlan:
    """Resolved profile write-plan for the compute/persist split (gate b).

    ``ProfileGenerationService._resolve_write_plan`` produces this in compute
    (dedup 2nd-LLM call + existing-row reads + source/status assignment +
    precomputed ``.embedding`` via ``precompute_profile_embeddings``), issuing
    **no** learning DB write. ``_persist_write_plan`` applies it inside the
    fence: ``add_user_profile(..., skip_embedding=True)`` (embeddings already
    set) then ``supersede_profiles_by_ids``.

    Attributes:
        user_id: Owning user id for the row writes.
        request_id: Generation request id — the lineage key
            ``supersede_profiles_by_ids`` records. Compute drops any supersede
            ids when this is empty (unreconstructable), so persist only ever
            supersedes with a non-empty request_id.
        new_profiles: New profile rows with ``source``/``status`` set and
            ``.embedding`` precomputed in compute.
        superseded_ids: Existing profile ids to soft-supersede on the dedup
            path.
    """

    user_id: str
    request_id: str
    new_profiles: list[UserProfile]
    superseded_ids: list[str]
    lineage_contexts: list[LineageContext] = field(default_factory=list)


@dataclass
class PlaybookWritePlan:
    """Resolved playbook write-plan for the compute/persist split (gate b).

    ``PlaybookGenerationService._resolve_write_plan`` produces this in compute
    (``dedupe_and_drop_empty`` + the deduplicator's 2nd-LLM call + existing-row
    reads + source/status assignment + precomputed ``.embedding`` via
    ``precompute_user_playbook_embeddings``), issuing **no** learning DB write.
    ``_persist_write_plan`` applies it inside the fence:
    ``save_user_playbooks(..., skip_embedding=True)`` (which assigns survivor
    ids) then ``_apply_consolidation_lineage`` (which MUST see those ids, so it
    runs AFTER the save).

    The off-thread schedulers (``_enqueue_user_playbook_optimization`` +
    ``_trigger_playbook_aggregation``) are NOT part of persist — they fire
    post-commit in ``emit_generation_side_effects`` (durable / ``.run()`` path)
    or right after persist in the permanent ``_finalize_extracted_items``
    wrapper (synchronous resume/manual path). ``output_pending_status`` /
    ``skip_aggregation`` are snapshotted here so the scheduler dispatch reads the
    plan rather than the reused service instance.

    Attributes:
        request_id: Generation request id — the lineage key
            ``_apply_consolidation_lineage`` records on merges/supersedes.
        output_pending_status: Whether the run emits PENDING rows (rerun mode);
            when True the aggregation trigger is suppressed.
        skip_aggregation: Whether aggregation is skipped (extract-only); when
            True the aggregation trigger is suppressed.
        new_playbooks: New playbook rows with ``source``/``status`` set and
            ``.embedding`` precomputed in compute. Survivor ids are assigned by
            ``save_user_playbooks`` in persist, before lineage reads them.
        superseded_ids: ALL archived existing ids (merge sources + leftovers)
            routed through ``_apply_consolidation_lineage``.
        merge_groups: ``(survivor_index_into_new_playbooks, source_existing_ids)``
            per dedup merge group.
    """

    request_id: str
    output_pending_status: bool
    skip_aggregation: bool
    new_playbooks: list[UserPlaybook]
    superseded_ids: list[int]
    merge_groups: list[tuple[int, list[int]]]
    lineage_contexts: list[LineageContext] = field(default_factory=list)
    consolidation_provenance: ModelProvenance | None = None


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


@dataclass
class DeferredLearningPlan:
    """One durable-learning job's resolved compute output (gate b, Task 8).

    ``GenerationService.compute_deferred_learning`` acquires the same-user guard
    (F4), then runs profile + playbook compute holding **no**
    ``commit_scope``, and assembles this plan. It carries each present half as a
    ``(held_service_instance, its_compute_plan)`` pair so
    ``persist_deferred_learning`` can apply the fence-critical writes and
    ``emit_deferred_learning_side_effects`` can fire the post-commit telemetry on
    the very instances that produced the plans (the single-use-instance invariant
    that keeps the money helpers' ``self._last_*`` reads valid — see
    ``BaseGenerationService.emit_generation_side_effects``).

    Attributes:
        request_id: The job's generation request id (lineage key).
        user_id: The user the job learns for; also the per-user F4 lock scope.
        agent_version: Resolved agent version — carried so the post-commit
            ``schedule_tagging`` dispatch in emit has it (the plan is all emit
            receives).
        lock_acquired: ``False`` when the F4 same-user guard denied this job
            (another durable job for the same user is mid-flight). On ``False``
            both plan halves are ``None`` and NO LLM/compute ran — the
            worker (Task 9) must leave the job reclaimable and must NOT
            ``complete_learning_job`` it.
        profile: ``(ProfileGenerationService, GenerationComputePlan)`` when the
            profile extractor produced a plan, else ``None``.
        playbook: ``(PlaybookGenerationService, GenerationComputePlan)`` when the
            playbook extractor produced a plan, else ``None``.
        warnings: Best-effort per-half compute failures (mirrors
            ``GenerationServiceResult.warnings``) — a failed half is dropped from
            the plan, the others still persist.
    """

    request_id: str
    user_id: str
    agent_version: str
    lock_acquired: bool
    profile: tuple[BaseGenerationService, GenerationComputePlan] | None
    playbook: tuple[BaseGenerationService, GenerationComputePlan] | None
    warnings: list[str] = field(default_factory=list)
