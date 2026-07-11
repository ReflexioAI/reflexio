"""Phase-A characterization backstops for the BaseGenerationService decomposition.

These are the money-critical, MRO-critical guards pinned BEFORE any mixin move
(Tier-1b decomposition). They must stay green byte-for-byte through Phase B — a
red here means a real behavior/contract regression, never "relax the test".

Three concerns, one per top-level test:

1. ``test_billing_drain_ordering_terminal_emitters_and_double_bill_guard``
   (SINK-1 / SINK-3 / F3) — drives ``run`` over a >=2-item FIFO queue and spies
   the TERMINAL money emitters (``record_learnings_generated`` /
   ``record_extraction_tokens`` in ``reflexio.server.billing_meter``) plus
   ``_finalize_extraction_runs`` on ONE ordered call-recorder. Pins:
     (a) per item, ``record_extraction_tokens`` fires once, LAST, AFTER finalize;
     (b) the ``:820-821`` per-run reset is the double-bill guard — item 2 returns a
         NON-``ExtractionOutcome`` so ``_last_token_totals`` is not overwritten and
         item 2 bills ZERO provider tokens (not item 1's);
     (c) a failed extraction emits NO learning billing;
     (d) INV-3 — the Precheck->Billing seam: each item bills its OWN precheck
         window, not a prior item's;
     (e) the ``:392`` / ``:319`` exception-swallow WARNING was NOT logged.

2. ``test_subclass_surface_mro_and_abcness`` (SINK-2) — the ~30-name inheritance
   contract resolves on each of the 3 subclasses + the base, the
   ``__abstractmethods__`` frozenset is EXACTLY the day-0 set, and the base stays
   un-instantiable.

3. ``test_finalize_vs_fail_atomicity`` (R6) — a ``_process_results`` failure inside
   ``_run_generation``'s try marks the runs finalization-failed (NOT a completed
   finalize) and NEVER reaches billing (money-only-on-success).
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    ExtractorExecutionError,
)
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.usage_metrics import (
    UsageEvent,
    configure_usage_event_recorder,
)

_MODULE_LOGGER = "reflexio.server.services.base_generation_service"
_ORG_ID = "base_gen_char_org"

# The exact day-0 abstract set. Captured verbatim at HEAD (24bd943). Phase B must
# NOT shrink this — a smaller set means an @abstractmethod leaked onto a plain
# (non-ABCMeta) mixin and silently dropped from __abstractmethods__ (SINK-2).
_EXPECTED_ABSTRACTMETHODS = frozenset(
    {
        "_create_extractor",
        "_get_base_service_name",
        "_get_lock_scope_id",
        "_get_service_name",
        "_load_extractor_config",
        "_load_generation_service_config",
        "_process_results",
        "_should_track_in_progress",
        # Added by gate (b) Task 8: the durable compute/persist split's
        # item-finalization seam. Intentional API-surface growth (NOT a
        # decomposition leak) — the 3 concrete subclasses each implement both.
        "_resolve_write_plan",
        "_persist_write_plan",
    }
)

# The ~30-name inheritance contract each subclass resolves via MRO. Every name
# here must stay resolvable on all 3 subclasses + the base through Phase B; a
# name that stops resolving means a mixin move broke the composed surface.
_CONTRACT_NAMES = (
    # abstract methods (declared on the base, implemented by subclasses)
    *_EXPECTED_ABSTRACTMETHODS,
    # orchestration (stays on the base)
    "run",
    "_run_generation",
    "_prepare_generation_run",
    "_finalize_extracted_items",
    "_create_state_manager",
    "_serialize_request_for_queue",
    "_deserialize_request_from_queue",
    # config-filter helpers
    "_filter_extractor_config_by_service_config",
    "_filter_config_by_stride",
    "_get_extractor_state_service_name",
    # usage/billing
    "_usage_pipeline",
    "_usage_context",
    "_record_generation_event",
    "_extraction_input_text",
    "_record_billing_learning_events",
    "_count_generated_results",
    "EMITS_LEARNING_BILLING",
    # extraction-run lifecycle
    "_execute_extractor",
    "_fail_active_extraction_runs",
    "_finalize_extraction_runs",
    "_mark_extraction_runs_finalization_failed",
    # should-run precheck
    "_should_run_before_extraction",
    "_build_should_run_prompt",
    "_collect_scoped_interactions_for_precheck",
    "_get_precheck_interaction_query_kwargs",
    "_resolve_should_run_model",
    # batch/rerun
    "_run_batch_with_progress",
    "run_rerun",
    "_pre_process_rerun",
    # status-change (public + hooks)
    "run_upgrade",
    "run_downgrade",
    "_has_items_with_status",
    "_delete_items_by_status",
    "_update_items_status",
    "_create_status_change_response",
)


# ---------------------------------------------------------------------------
# Harness: a concrete BaseGenerationService subclass with controllable extractor
# results, mirroring the fixture style of test_generation_billing_emission.py.
# ---------------------------------------------------------------------------


class _CharRequest(BaseModel):
    """Pydantic request so the queue drain round-trips via model_validate."""

    user_id: str
    request_id: str
    source: str = "api"
    auto_run: bool = True


class _RecordingExtractor:
    """Returns a pre-set per-user result, or raises a pre-set exception."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def run(self) -> Any:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _CharService(BaseGenerationService):  # type: ignore[type-arg]
    """Profile-shaped generation service with per-user extractor outputs.

    ``EMITS_LEARNING_BILLING=True`` so the ② Learning terminal emitters fire.
    ``_build_should_run_prompt`` returns None so the should-run gate stashes its
    precheck window (INV-3 seam) WITHOUT making an LLM call.
    """

    EMITS_LEARNING_BILLING: bool = True

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        results_by_user: dict[str, Any],
        *,
        track: bool = True,
    ) -> None:
        super().__init__(llm_client, request_context)
        self._results_by_user = results_by_user
        self._track = track

    def _load_extractor_config(self):
        return ProfileExtractorConfig(
            extractor_name="char_extractor",
            extraction_definition_prompt="Extract durable preferences.",
        )

    def _load_generation_service_config(self, request):
        return request

    def _create_extractor(self, extractor_config, service_config):
        return _RecordingExtractor(self._results_by_user[service_config.user_id])

    def _get_service_name(self) -> str:
        return "profile_generation_char"

    def _get_base_service_name(self) -> str:
        return "profile_generation_char"

    def _process_results(self, results):
        pass

    # Legacy write-in-compute path (mirrors AgentSuccessEvaluationService): route
    # through the permanent _finalize_extracted_items wrapper so this helper stays
    # instantiable after gate (b) made _resolve_write_plan/_persist_write_plan
    # abstract, preserving the finalize-before-fail atomicity these tests pin.
    def _resolve_write_plan(self, results):
        for result in results:
            if result:
                self._finalize_extracted_items(result)
        return

    def _persist_write_plan(self, plan):
        return

    def _should_track_in_progress(self) -> bool:
        return self._track

    def _get_lock_scope_id(self, request):
        return getattr(request, "user_id", None)

    def _build_should_run_prompt(self, scoped_config, session_data_models):
        # Proceed without an LLM call — but only AFTER
        # _collect_scoped_interactions_for_precheck has stashed the window.
        return None


class _FakeStateManager:
    """Deterministic FIFO drain: hands back ``queue`` entries then None."""

    def __init__(self, queue: list[dict]) -> None:
        self._queue = list(queue)

    def acquire_lock(self, request_id, *, scope_id=None, payload=None) -> bool:  # noqa: ARG002
        return True

    def is_cancellation_requested(self) -> bool:
        return False

    def release_lock_pop_queue(self, request_id, *, scope_id=None):  # noqa: ARG002
        return self._queue.pop(0) if self._queue else None

    def clear_lock(self, *, scope_id=None) -> None:
        pass


def _request_context(storage: Any) -> RequestContext:
    """Build a RequestContext wired to ``storage`` with a fixed profile Config."""
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = _ORG_ID
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="char_extractor",
            extraction_definition_prompt="Extract durable preferences.",
        ),
    )
    ctx.configurator.get_agent_context.return_value = "char agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


def _build_sqlite_storage(db_path: str) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        return SQLiteStorage(org_id=_ORG_ID, db_path=db_path)


def _seed_user(
    storage: SQLiteStorage,
    *,
    user_id: str,
    request_id: str,
    interaction_id: int,
    content: str,
) -> None:
    storage.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            created_at=1_000,
            source="api",
            agent_version="v1",
            session_id=request_id,
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=interaction_id,
            user_id=user_id,
            request_id=request_id,
            created_at=1_000,
            role="User",
            content=content,
        )
    )


def _llm_client() -> LiteLLMClient:
    return LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))


# ---------------------------------------------------------------------------
# Test 1: billing drain-ordering + terminal-emitter + double-bill + no-swallow.
# ---------------------------------------------------------------------------

# user1 is seeded with markedly MORE text than user2 so their billing_input_tokens
# differ — this is what makes INV-3 (assertion d) observable.
_USER1_CONTENT = (
    "Please remember that I always prefer dark-mode terminals, tabs over spaces, "
    "concise answers with no emojis, and that my staging database lives on port "
    "5433. These preferences are durable and important for my whole workflow."
)
_USER2_CONTENT = "Also note that I strongly prefer polars over pandas for dataframes."


def test_billing_drain_ordering_terminal_emitters_and_double_bill_guard(
    tmp_path, monkeypatch, caplog
):
    """SINK-1/3, F3: terminal emitters, ordering, double-bill guard, INV-3, no-swallow.

    Drives ``run`` over a 2-item FIFO queue. Item 1's extractor returns an
    ``ExtractionOutcome`` carrying real provider token_totals; item 2 returns a
    plain list (NON-``ExtractionOutcome``) so ``_last_token_totals`` is NOT
    overwritten at ``:991`` — proving the ``:820-821`` reset is the double-bill guard.
    """
    # The gate must actually run so it stashes each item's precheck window
    # (INV-3). MOCK_LLM_RESPONSE=true would short-circuit the gate.
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)

    storage = _build_sqlite_storage(str(tmp_path / "drain.db"))
    _seed_user(
        storage,
        user_id="user1",
        request_id="req_user1",
        interaction_id=1,
        content=_USER1_CONTENT,
    )
    _seed_user(
        storage,
        user_id="user2",
        request_id="req_user2",
        interaction_id=2,
        content=_USER2_CONTENT,
    )

    item1_totals = RunTokenTotals(prompt_tokens=111, completion_tokens=222)
    service = _CharService(
        _llm_client(),
        _request_context(storage),
        results_by_user={
            # Item 1: real ExtractionOutcome -> writes _last_token_totals at :991.
            "user1": ExtractionOutcome.completed(
                [{"learning": "dark mode"}],
                run_id=None,
                token_totals=item1_totals,
            ),
            # Item 2: NON-ExtractionOutcome -> _last_token_totals stays reset (None).
            "user2": [{"learning": "polars"}],
        },
    )
    # Deterministic 2-item drain: holder=user1, one queued entry=user2.
    service._create_state_manager = lambda: _FakeStateManager(  # type: ignore[method-assign]
        [
            {
                "request_id": "req_user2",
                "payload": {
                    "user_id": "user2",
                    "request_id": "req_user2",
                    "source": "api",
                    "auto_run": True,
                },
            }
        ]
    )

    # ONE ordered call-recorder across the two terminal money emitters + finalize.
    recorder = Mock()
    caplog.set_level(logging.WARNING, logger=_MODULE_LOGGER)
    with (
        patch(
            "reflexio.server.billing_meter.record_learnings_generated",
            recorder.learnings,
        ),
        patch(
            "reflexio.server.billing_meter.record_extraction_tokens",
            recorder.tokens,
        ),
    ):
        # finalize is a no-op here (empty run_ids); spying it captures ordering.
        service._finalize_extraction_runs = recorder.finalize  # type: ignore[method-assign]
        service.run(
            _CharRequest(
                user_id="user1", request_id="req_user1", source="api", auto_run=True
            )
        )

    # (a) Per item, the trio fires in the exact order finalize -> learnings ->
    # tokens; record_extraction_tokens is LAST and AFTER _finalize_extraction_runs.
    ordered_names = [c[0] for c in recorder.mock_calls]
    assert ordered_names == [
        "finalize",
        "learnings",
        "tokens",
        "finalize",
        "learnings",
        "tokens",
    ], f"unexpected money-emit ordering: {ordered_names}"

    tok_calls = recorder.tokens.call_args_list
    assert len(tok_calls) == 2, "record_extraction_tokens must fire once per item"
    by_req = {c.kwargs["request_id"]: c.kwargs for c in tok_calls}
    assert set(by_req) == {"req_user1", "req_user2"}

    # (b) Double-bill guard: item 1 bills its real provider tokens; item 2 — whose
    # extractor returned a non-ExtractionOutcome — bills ZERO provider tokens
    # because the :820-821 reset cleared _last_token_totals before item 2 ran.
    assert by_req["req_user1"]["prompt_tokens"] == 111
    assert by_req["req_user1"]["completion_tokens"] == 222
    assert by_req["req_user2"]["prompt_tokens"] == 0
    assert by_req["req_user2"]["completion_tokens"] == 0

    # (d) INV-3 (Precheck->Billing seam): each item bills ITS OWN precheck window.
    # user1 was seeded with much more text than user2, so their input-token bases
    # differ; if item 2 had reused item 1's stashed window they would be equal.
    bi1 = by_req["req_user1"]["billing_input_tokens"]
    bi2 = by_req["req_user2"]["billing_input_tokens"]
    assert bi1 > 0 and bi2 > 0, f"expected non-zero input bases, got {bi1}, {bi2}"
    assert bi1 > bi2, (
        "item 1 (longer window) must bill more input tokens than item 2; "
        f"got bi1={bi1}, bi2={bi2} — INV-3 window fidelity broken"
    )

    # learnings_generated also fires once per successful item.
    assert len(recorder.learnings.call_args_list) == 2

    # (e) No-swallow: the :392 / :319 exception-swallow WARNINGs must NOT appear —
    # a clean billing path never logs them, so their presence would mean a botched
    # move silently stopped billing.
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "_record_billing_learning_events failed" in m
        or "billing learning events not emitted" in m
        or "_extraction_input_text failed" in m
        for m in warnings
    ), f"unexpected billing-swallow warning(s): {warnings}"

    # (c) A FAILED extraction emits NO ② Learning billing (money-only-on-success).
    fail_service = _CharService(
        _llm_client(),
        _request_context(MagicMock()),
        results_by_user={"user3": RuntimeError("extractor boom")},
        track=False,  # no queue drain — a single _run_generation via run()
    )
    fail_recorder = Mock()
    with (
        patch(
            "reflexio.server.billing_meter.record_learnings_generated",
            fail_recorder.learnings,
        ),
        patch(
            "reflexio.server.billing_meter.record_extraction_tokens",
            fail_recorder.tokens,
        ),
        pytest.raises(ExtractorExecutionError),
    ):
        fail_service.run(
            _CharRequest(
                user_id="user3", request_id="req3", source="api", auto_run=False
            )
        )
    fail_recorder.learnings.assert_not_called()
    fail_recorder.tokens.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: subclass-surface / MRO / ABC-ness (SINK-2).
# ---------------------------------------------------------------------------


def _all_subject_classes():
    from reflexio.server.services.agent_success_evaluation.service import (
        AgentSuccessEvaluationService,
    )
    from reflexio.server.services.playbook.service import PlaybookGenerationService
    from reflexio.server.services.profile.service import ProfileGenerationService

    return [
        BaseGenerationService,
        ProfileGenerationService,
        PlaybookGenerationService,
        AgentSuccessEvaluationService,
    ]


def test_subclass_surface_mro_and_abcness():
    """The ~30-name contract resolves everywhere; the abstract set is EXACT.

    Guards the decomposition's central structural invariant: every contract name
    stays resolvable via MRO on each of the 3 subclasses + the base, the
    ``__abstractmethods__`` frozenset is byte-identical to the day-0 set, and the
    base stays un-instantiable. A shrink of the frozenset means an
    ``@abstractmethod`` leaked onto a plain mixin (SINK-2).
    """
    for cls in _all_subject_classes():
        missing = [name for name in _CONTRACT_NAMES if not hasattr(cls, name)]
        assert not missing, f"{cls.__name__} is missing contract names: {missing}"

    # EXACT equality — do NOT relax if Phase B turns this red.
    assert frozenset(BaseGenerationService.__abstractmethods__) == (
        _EXPECTED_ABSTRACTMETHODS
    ), (
        "BaseGenerationService.__abstractmethods__ drifted from the day-0 set: "
        f"{sorted(BaseGenerationService.__abstractmethods__)}"
    )

    # The 3 concrete subclasses must be fully concrete (no leftover abstracts).
    from reflexio.server.services.agent_success_evaluation.service import (
        AgentSuccessEvaluationService,
    )
    from reflexio.server.services.playbook.service import PlaybookGenerationService
    from reflexio.server.services.profile.service import ProfileGenerationService

    for cls in (
        ProfileGenerationService,
        PlaybookGenerationService,
        AgentSuccessEvaluationService,
    ):
        assert frozenset(cls.__abstractmethods__) == frozenset(), (
            f"{cls.__name__} unexpectedly has abstract methods: "
            f"{sorted(cls.__abstractmethods__)}"
        )

    # The base ABC must remain un-instantiable.
    with pytest.raises(TypeError):
        BaseGenerationService(MagicMock(), MagicMock())  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Test 3: finalize-vs-fail atomicity (R6).
# ---------------------------------------------------------------------------


class _ProcessFailsService(_CharService):
    """A service whose _process_results raises inside _run_generation's try."""

    def _process_results(self, results):
        raise ValueError("process boom")


def test_finalize_vs_fail_atomicity(monkeypatch):
    """R6: a _process_results failure marks runs finalization-failed, not finalized.

    Inside ``_run_generation``'s try (``:827-833``), a ``_process_results`` failure
    must call ``_mark_extraction_runs_finalization_failed(exc)``, must NOT let
    ``_finalize_extraction_runs`` complete, and must NEVER reach
    ``_record_billing_learning_events`` (money only on success).
    """
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")  # bypass the gate deterministically

    service = _ProcessFailsService(
        _llm_client(),
        _request_context(MagicMock()),
        # A truthy, non-ExtractionOutcome result so _process_results is invoked.
        results_by_user={"user_r6": [{"learning": "boom"}]},
        track=False,
    )

    finalize_spy = Mock()
    mark_failed_spy = Mock()
    billing_spy = Mock()
    service._finalize_extraction_runs = finalize_spy  # type: ignore[method-assign]
    service._mark_extraction_runs_finalization_failed = mark_failed_spy  # type: ignore[method-assign]
    service._record_billing_learning_events = billing_spy  # type: ignore[method-assign]

    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    try:
        # auto_run=False -> the gate is bypassed; _process_results then raises,
        # which becomes ExtractorExecutionError? No: the ValueError is raised
        # inside the inner try, re-raised, caught by the outer handler, and (not
        # being ExtractorExecutionError) swallowed after recording the failure.
        service.run(
            _CharRequest(
                user_id="user_r6", request_id="req_r6", source="api", auto_run=False
            )
        )
    finally:
        configure_usage_event_recorder(None)

    # The finalization-failed path took over...
    assert mark_failed_spy.call_count == 1
    (exc_arg,) = mark_failed_spy.call_args[0]
    assert isinstance(exc_arg, ValueError)
    assert "process boom" in str(exc_arg)

    # ...instead of a completed finalize, and billing was NEVER reached.
    finalize_spy.assert_not_called()
    billing_spy.assert_not_called()

    # The inner re-raise reached the outer handler: generation_failed recorded,
    # generation_succeeded never recorded.
    names = {e.event_name for e in events}
    assert "generation_failed" in names
    assert "generation_succeeded" not in names
