"""Publication contracts through the real agent writer and resume worker.

These regressions exercise the unshipped owner refactor
and its shared atomic per-output publisher in place of run-wide finalization.
Only model/dedup results and best-effort dispatch are scripted; ownership,
output envelopes, learning writes, receipts and dependency resolution are real.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import Interaction
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.overflow_answers import answer_witness
from reflexio.server.services.extraction.resumable_agent import (
    run_resumable_extraction_agent,
)
from reflexio.server.services.extraction.resume_worker import (
    ExtractionResumeWorker,
    _request_interaction_models_from_ids,
)
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList,
    build_playbook_prompt_context,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.service import ProfileGenerationServiceConfig
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
)
from reflexio.server.usage_metrics import (
    UsageEvent,
    UsageEventDeliveryStatus,
    configure_usage_event_recorder,
)
from tests.server.services.extraction import test_resume_worker as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
pytestmark = pytest.mark.integration


_REAL_PRIOR_KNOWLEDGE = ExtractionResumeWorker._messages_with_prior_knowledge
PublicationCase = tuple[
    SQLiteStorage, ExtractionResumeWorker, str, list[str], list[UsageEvent]
]


@pytest.fixture(params=["profile", "playbook"])
def publication(
    request, storage, request_context, monkeypatch
) -> Iterator[PublicationCase]:
    base._seed_interactions(storage)
    base._seed_ready_run(storage)
    storage._insert_interaction(
        Interaction(
            interaction_id=3,
            user_id="user_1",
            request_id="request_1",
            role="user",
            content="Require approval before deployment.",
            created_at=1002,
        )
    )
    storage._execute(
        "UPDATE _agent_runs SET source_interaction_ids=?, window_end_interaction_id=3 WHERE id='run_1'",
        ("[1,2,3]",),
    )
    kind = request.param
    request_context.configurator.get_config().user_playbook_extractor_config = (
        PlaybookConfig(extraction_definition_prompt="Extract deployment procedures.")
    )
    request_context.prompt_manager = PromptManager()
    if kind == "playbook":
        storage._execute(
            "UPDATE _agent_runs SET extractor_kind='playbook' WHERE id='run_1'"
        )
    worker = ExtractionResumeWorker(
        request_context=request_context, llm_client=MagicMock()
    )
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_messages_with_prior_knowledge",
        lambda _self, **kw: kw["messages"],
    )
    for path in (
        "reflexio.server.services.profile.components.consolidator.ProfileConsolidator",
        "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
    ):
        consolidator = MagicMock()
        consolidator.return_value.deduplicate.side_effect = (
            lambda items, *_args, **_kw: (items, [], [])
        )
        consolidator.return_value.model_provenance = None
        consolidator.return_value.consolidated_output_indices = set()
        consolidator.return_value.retrieve_existing_playbooks.return_value = []
        if "PlaybookConsolidator" in path:
            consolidator.return_value.deduplicate.side_effect = (
                lambda groups, *_args, **_kw: (
                    [item for group in groups for item in group],
                    [],
                    [],
                )
            )
        monkeypatch.setattr(path, consolidator)
    monkeypatch.setattr(
        PlaybookCandidateReviewer,
        "review",
        lambda _self, *, candidates, **_kw: candidates,
    )
    dispatches: list[str] = []
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_schedule_finalized_tagging",
        lambda _self, _run: dispatches.append("tagging"),
    )
    monkeypatch.setattr(
        PlaybookGenerationService,
        "_dispatch_playbook_schedulers",
        lambda *_args: dispatches.append("playbook"),
    )
    events: list[UsageEvent] = []

    def accept(event):
        events.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(accept)
    yield storage, worker, kind, dispatches, events


@pytest.fixture
def shared_worker(publication: PublicationCase) -> DurableLearningWorker:
    context = publication[1].request_context
    return DurableLearningWorker(lambda _org: context)


def _question(storage):
    prior = storage.get_pending_tool_call("ptc_1")
    assert prior is not None
    question = "Which approval is required?"
    storage.create_pending_tool_call(
        replace(
            prior,
            id="ptc_2",
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            result=None,
            resolved_at=None,
            valid_until=None,
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=question
            ),
        )
    )


def _loop_result(kwargs, kind, storage, *, nonempty, pending=()):
    field = "profiles" if kind == "profile" else "playbooks"
    entry: dict[str, Any] = {"content": "Deployment requires approval."}
    if kind == "profile":
        entry["time_to_live"] = "infinity"
    else:
        entry.update(
            trigger="when deploying", rationale="Required deployment procedure"
        )
        sources = _request_interaction_models_from_ids(
            storage, [1, 2, 3], fallback_source="api", fallback_agent_version="v1"
        )
        context = build_playbook_prompt_context(sources, label_turns=True)
        entry.update(
            evidence_kind="preference",
            evidence_refs=[
                ref
                for ref, unit in context.evidence_units.items()
                if unit.interaction_id == 3
            ],
        )
    output = kwargs["response_format"].model_validate(
        {field: [entry] if nonempty else []}
    )
    validator = kwargs.get("structured_output_validator")
    if validator is not None:
        assert not validator(output), (
            "Scripted output must satisfy the real evidence validator"
        )
    return ToolLoopResult(
        ctx=kwargs["ctx"],
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        structured_output=output,
        pending_tool_call_ids=list(pending),
        max_steps_remaining=8,
    )


def _learnings(storage, kind):
    return (
        storage.get_user_profile("user_1")
        if kind == "profile"
        else storage.get_user_playbooks(user_id="user_1", agent_version="v1")
    )


def _seed_prior_answer(storage):
    dependency = storage.get_pending_tool_call("ptc_1")
    assert dependency is not None
    prior = replace(
        dependency,
        id="prior-answer",
        user_id="other-user",
        dedup_key="prior-approval",
        question_text="Which release approval applies?",
        result={"answer": "Release approval from finance is required."},
        embedding=[1.0] * 512,
    )
    storage.create_pending_tool_call(prior)
    return dependency, prior


@pytest.mark.parametrize("injection", ["prior_context", "cached_tool_result"])
def test_answer_input_is_captured_when_supplied_to_model(
    shared_worker: DurableLearningWorker, publication, monkeypatch, injection
):
    storage, worker, kind, _, _ = publication
    dependency, prior = _seed_prior_answer(storage)
    expected = {
        record.id: build_scope_hash(answer_witness(asdict(record)))
        for record in [dependency, prior]
    }
    monkeypatch.setattr(storage, "_get_embedding", lambda *_args, **_kw: [1.0] * 512)
    if injection == "prior_context":
        monkeypatch.setattr(
            ExtractionResumeWorker,
            "_messages_with_prior_knowledge",
            _REAL_PRIOR_KNOWLEDGE,
        )

    def complete(**kwargs):
        if injection == "prior_context":
            assert "Release approval from finance is required." in str(
                kwargs["messages"]
            )
        else:
            supplied = kwargs["registry"].handle_outcome(
                "attach_pending_info_request",
                json.dumps({"pending_tool_call_id": prior.id}),
                kwargs["ctx"],
            )
            assert supplied.result["result"] == prior.result
        storage.update_resolved_pending_tool_call_result(
            "prior-answer",
            result={"answer": "A different approval applies now."},
            resolved_at=datetime.now(UTC),
            valid_for_seconds=3600,
        )
        return _loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.committed_output is not None
    publication = result.committed_output["publication"]
    assert set(publication["dependency_inputs"]) == {"ptc_1"}
    assert publication["answer_inputs"] == expected
    changed = storage.get_pending_tool_call("prior-answer")
    assert changed is not None
    assert (
        build_scope_hash(answer_witness(asdict(changed)))
        != publication["answer_inputs"]["prior-answer"]
    )
    assert "finance" not in str(publication)
    assert [
        d.pending_tool_call_id for d in storage.list_run_tool_dependencies("run_1")
    ] == ["ptc_1"]


def test_initial_agent_context_and_writer_share_answer_capture(
    publication, monkeypatch
):
    import hashlib

    from reflexio.server.services.extraction.input_preparation import (
        source_witness_bytes,
    )

    storage, worker, kind, _, _ = publication
    source_request = storage.get_request("request_1")
    assert source_request is not None
    expected_sources = {
        "project_id": "",
        "sources": {
            str(item.interaction_id): {
                "digest": hashlib.sha256(
                    source_witness_bytes(source_request, item)
                ).hexdigest(),
                "origin": "request",
            }
            for item in storage.get_interactions_by_ids([1, 2, 3])
        },
    }
    dependency, prior = _seed_prior_answer(storage)
    expected = {
        record.id: build_scope_hash(answer_witness(asdict(record)))
        for record in [dependency, prior]
    }
    monkeypatch.setattr(storage, "_get_embedding", lambda *_args, **_kw: [1.0] * 512)
    search = storage.search_prior_tool_calls

    def admitted_search(**kwargs):
        from reflexio.server.services.durable_learning.user_lease import (
            _require_user_work_owner,
        )

        _require_user_work_owner(storage).fence(storage)
        return search(**kwargs)

    monkeypatch.setattr(storage, "search_prior_tool_calls", admitted_search)

    model_calls = []

    def complete(**kwargs):
        model_calls.append(1)
        assert "Release approval from finance is required." in str(kwargs["messages"])
        assert "Use AWS ECS." in str(kwargs["messages"])
        storage._execute(
            "UPDATE requests SET session_id='changed-in-initial-model' WHERE request_id='request_1'"
        )
        return _loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    config = worker.request_context.configurator.get_config()
    service_config = (
        ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="request_1",
            source="api",
            auto_run=False,
            force_extraction=True,
        )
        if kind == "profile"
        else PlaybookGenerationServiceConfig(
            user_id="user_1",
            request_id="request_1",
            source="api",
            agent_version="v1",
            auto_run=False,
            force_extraction=True,
        )
    )
    service_config.extraction_window_id = "answer-capture-window"

    def start():
        return run_resumable_extraction_agent(
            request_context=worker.request_context,
            client=worker.client,
            extractor_kind=kind,
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
            request_interaction_data_models=_request_interaction_models_from_ids(
                storage, [1, 2, 3], fallback_source="api", fallback_agent_version="v1"
            ),
            extractor_config=config.profile_extractor_config
            if kind == "profile"
            else config.user_playbook_extractor_config,
            service_config=service_config,
            agent_context="Deployment guidance",
            messages=[{"role": "user", "content": "Extract the deployment rule"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput
            if kind == "profile"
            else StructuredReferencedExtractedPlaybookList,
            log_label="synthetic_answer_capture",
        )

    result = start()
    saved = storage.get_agent_run(result.run_id)
    assert saved is not None and saved.committed_output is not None
    assert saved.committed_output["publication"]["answer_inputs"] == expected
    assert saved.committed_output["publication"]["dependency_inputs"] == {}
    assert saved.committed_output["publication"]["source_inputs"] == expected_sources

    def no_lookup(**_kwargs):
        raise AssertionError(
            "Saved output replay must not retrieve fresh prior answers"
        )

    monkeypatch.setattr(storage, "search_prior_tool_calls", no_lookup)
    replay = start()
    assert replay.output == result.output
    assert (
        storage.get_agent_run(result.run_id).committed_output == saved.committed_output
    )
    assert model_calls == [1]


def test_uninterrupted_publication_writes_and_bills_once(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    """Control: the same fixture must reach real successful persistence."""
    storage, worker, kind, _, events = publication
    calls = []

    def complete(**kwargs):
        calls.append(1)
        return _loop_result(kwargs, kind, storage, nonempty=True)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert len(calls) == 1
    rows = _learnings(storage, kind)
    assert [row.content for row in rows] == ["Deployment requires approval."]
    entity = "profile" if kind == "profile" else "user_playbook"
    ids = [
        str(row.profile_id if kind == "profile" else row.user_playbook_id)
        for row in rows
    ]
    assert (
        storage.get_agent_run_finalization_receipt(run_id="run_1", entity_type=entity)
        == ids
    )
    assert [
        d.consumed_at is not None for d in storage.list_run_tool_dependencies("run_1")
    ] == [True]
    assert [e.event_key for e in events if e.event_name == "learnings_generated"] == [
        f"learn:{entity}:{ids[0]}"
    ]


@pytest.mark.parametrize("nonempty", [False, True])
def test_publication_crash_rolls_back_learning_receipt_and_answer_consumption(
    shared_worker: DurableLearningWorker, publication, monkeypatch, nonempty
):
    storage, worker, kind, dispatches, events = publication
    calls = []

    def complete(**kwargs):
        calls.append(1)
        return _loop_result(kwargs, kind, storage, nonempty=nonempty)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )

    crashes = []

    def crashing(consume):
        def consume_then_crash(*args, **kwargs):
            consume(*args, **kwargs)
            crashes.append(1)
            raise RuntimeError("controlled crash after actual answer consumption")

        return consume_then_crash

    # Keep the fault at actual consumption as the new publisher replaces the
    # old run-wide seam. Neither mock publishes or consumes anything itself.
    for name in ("consume_run_tool_dependencies", "consume_agent_output_dependencies"):
        monkeypatch.setattr(storage, name, crashing(getattr(storage, name)))

    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZATION_FAILED
    assert len(calls) == 1
    assert len(crashes) == 1, "The fault must execute after actual consumption"
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.committed_output is not None
    assert set(saved.committed_output["publication"]["dependency_inputs"]) == {"ptc_1"}
    actual = {
        "receipt": storage.get_agent_run_finalization_receipt(
            run_id="run_1",
            entity_type="profile" if kind == "profile" else "user_playbook",
        ),
        "consumed": [
            d.consumed_at is not None
            for d in storage.list_run_tool_dependencies("run_1")
        ],
        "learnings": [row.content for row in _learnings(storage, kind)],
        "dispatches": dispatches,
        "billed": [
            event.event_key
            for event in events
            if event.event_name == "learnings_generated"
        ],
    }
    assert actual == {
        "receipt": None,
        "consumed": [False],
        "learnings": [],
        "dispatches": [],
        "billed": [],
    }


def test_second_answer_publishes_a_new_output_in_the_same_run(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    storage, worker, kind, _, _ = publication
    _question(storage)
    calls = []

    def complete(**kwargs):
        calls.append(1)
        first = len(calls) == 1
        if first:
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
            )
        return _loop_result(
            kwargs,
            kind,
            storage,
            nonempty=not first,
            pending=["ptc_2"] if first else [],
        )

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZED_PENDING_TOOL
    assert first.committed_output is not None
    first_output = first.committed_output["publication"]["id"]
    entity = "profile" if kind == "profile" else "user_playbook"
    assert (
        storage.get_agent_run_finalization_receipt(run_id="run_1", entity_type=entity)
        == []
    )
    storage.resolve_pending_tool_call(
        "ptc_2",
        result={"answer": "Approval required"},
        resolved_at=datetime.now(UTC),
        valid_for_seconds=3600,
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZED
    assert second.committed_output is not None
    assert len(calls) == 2
    assert first_output != second.committed_output["publication"]["id"]
    assert set(second.committed_output["publication"]["dependency_inputs"]) == {"ptc_2"}
    rows = _learnings(storage, kind)
    assert [row.content for row in rows] == ["Deployment requires approval."]
    second_ids = [
        str(row.profile_id if kind == "profile" else row.user_playbook_id)
        for row in rows
    ]
    assert {
        d.pending_tool_call_id: d.consumed_at is not None
        for d in storage.list_run_tool_dependencies("run_1")
    } == {"ptc_1": True, "ptc_2": True}
    # The publisher's required read-only observation seam is per output, not
    # the old run-wide lookup. It is deliberately not implemented yet: these
    # are retained red contracts, not passing feature or migration gates.
    assert (
        storage.get_agent_output_finalization_receipt(
            run_id="run_1", output_id=first_output, entity_type=entity
        )
        == []
    )
    assert (
        storage.get_agent_output_finalization_receipt(
            run_id="run_1",
            output_id=second.committed_output["publication"]["id"],
            entity_type=entity,
        )
        == second_ids
    )


@pytest.mark.parametrize("resolved_before_attachment", [False, True])
def test_late_answer_remains_unconsumed_and_immediately_runnable(
    shared_worker: DurableLearningWorker,
    publication,
    monkeypatch,
    resolved_before_attachment,
):
    storage, worker, kind, _, _ = publication
    _question(storage)

    def complete(**kwargs):
        if not resolved_before_attachment:
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
            )
        storage.resolve_pending_tool_call(
            "ptc_2",
            result={"answer": "Arrived during compute"},
            resolved_at=datetime.now(UTC),
            valid_for_seconds=3600,
        )
        if resolved_before_attachment:
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
            )
            dependency = next(
                d
                for d in storage.list_run_tool_dependencies("run_1")
                if d.pending_tool_call_id == "ptc_2"
            )
            assert (
                dependency.pending_tool_call_id == "ptc_2"
                and dependency.resolved_at is None
            )
        return _loop_result(kwargs, kind, storage, nonempty=False, pending=["ptc_2"])

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.committed_output is not None
    assert set(result.committed_output["publication"]["dependency_inputs"]) == {"ptc_1"}
    consumed = {
        d.pending_tool_call_id: d.consumed_at is not None
        for d in storage.list_run_tool_dependencies("run_1")
    }
    assert (result.status, consumed) == (
        AgentRunStatus.RESUME_READY,
        {"ptc_1": True, "ptc_2": False},
    )

    assert result.pending_tool_call_ids == ["ptc_2"]
    answer = storage.get_pending_tool_call("ptc_2")
    dependency = next(
        d
        for d in storage.list_run_tool_dependencies("run_1")
        if d.pending_tool_call_id == "ptc_2"
    )
    assert answer is not None and dependency.resolved_at == answer.resolved_at
    assert [answer.id for answer in worker._load_resolved_tool_calls(result)] == [
        "ptc_2"
    ]


@pytest.mark.parametrize("failure_phase", ["prior_context", "dependency", "model"])
def test_answer_refusal_charges_only_after_preparation(
    shared_worker: DurableLearningWorker, publication, monkeypatch, failure_phase
):
    storage, worker, kind, _, _ = publication
    dependency, prior = _seed_prior_answer(storage)
    model_calls = []
    expired = replace(dependency, valid_until=datetime.now(UTC) - timedelta(seconds=1))
    # At the limit, preparation failure must still be retryable; a physical model
    # entry consumes the last attempt even if a tool then refuses stale input.
    config = worker.request_context.configurator.get_config()
    config.pending_tool_call_config.max_resume_attempts = 1
    if failure_phase == "prior_context":
        monkeypatch.setattr(storage, "_get_embedding", lambda *_a, **_kw: [1.0] * 512)
        monkeypatch.setattr(
            ExtractionResumeWorker,
            "_messages_with_prior_knowledge",
            _REAL_PRIOR_KNOWLEDGE,
        )
        getter = storage.get_pending_tool_call

        def corrected_after_search(call_id):
            record = getter(call_id)
            return (
                replace(record, result={"answer": "Corrected after search"})
                if call_id == prior.id
                else record
            )

        monkeypatch.setattr(storage, "get_pending_tool_call", corrected_after_search)
    elif failure_phase == "dependency":
        monkeypatch.setattr(
            ExtractionResumeWorker,
            "_load_resolved_tool_calls",
            lambda _self, _run: [expired],
        )

    def model(**_kwargs):
        from reflexio.server.services.extraction.overflow_answers import (
            pin_extraction_answer,
        )

        model_calls.append(1)
        # Simulate the same capture refusal during an already-entered tool loop.
        pin_extraction_answer(expired)
        raise AssertionError("Expired answer must be refused")

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None
    charged = int(failure_phase == "model")
    assert len(model_calls) == charged
    assert result.resume_attempts == charged
    assert result.status == (
        AgentRunStatus.FAILED if charged else AgentRunStatus.RESUME_READY
    )
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.resume_attempts == charged
    assert saved.committed_output is None
    assert [d.consumed_at for d in storage.list_run_tool_dependencies("run_1")] == [
        None
    ]


def test_preparation_readiness_preserves_previous_model_attempts(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    storage, worker, _kind, _, _ = publication
    storage._execute("UPDATE _agent_runs SET resume_attempts=3 WHERE id='run_1'", ())

    def unavailable(_self, **_kwargs):
        raise ReadinessUnavailableError("Preparation unavailable")

    monkeypatch.setattr(
        ExtractionResumeWorker, "_messages_with_prior_knowledge", unavailable
    )
    model = MagicMock(side_effect=AssertionError("Must not enter model"))
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.RESUME_READY
    assert result.resume_attempts == 3
    assert storage.get_agent_run("run_1").resume_attempts == 3
    model.assert_not_called()


@pytest.mark.parametrize("withdrawal", [None, "logical_claim", "user_lease"])
@pytest.mark.parametrize("tool_name", ["ask_human", "attach_pending_info_request"])
def test_in_model_attachment_cannot_mutate_after_owner_withdrawal(
    shared_worker: DurableLearningWorker,
    publication,
    monkeypatch,
    withdrawal,
    tool_name,
):
    storage, worker, kind, _, _ = publication
    if tool_name == "ask_human" and kind == "profile":
        pytest.skip("Profile extraction exposes only the attachment tool")
    _question(storage)
    entered = []
    tool_errors = []

    def model(**kwargs):
        entered.append(1)
        if withdrawal == "logical_claim":
            storage._execute(
                "UPDATE _agent_runs SET claimed_by='replacement-worker' WHERE id='run_1'",
                (),
            )
        elif withdrawal == "user_lease":
            storage._execute(
                "UPDATE learning_work SET lease_until=0 WHERE org_id=? AND user_id='user_1'",
                (storage.org_id,),
            )
        try:
            kwargs["registry"].handle_outcome(
                tool_name,
                json.dumps(
                    {"pending_tool_call_id": "ptc_2"}
                    if tool_name == "attach_pending_info_request"
                    else {"question": "Which synthetic deployment should we use?"}
                ),
                kwargs["ctx"],
            )
        except Exception as exc:
            # Persisted effects are the oracle even if errors are normalized.
            tool_errors.append(type(exc).__name__)
        return _loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert entered == [1]
    dependencies = {
        d.pending_tool_call_id for d in storage.list_run_tool_dependencies("run_1")
    }
    if withdrawal is None:
        assert len(dependencies) == 2
        if tool_name == "attach_pending_info_request":
            assert dependencies == {"ptc_1", "ptc_2"}
    else:
        assert dependencies == {"ptc_1"}, "Stale execution may not attach work"
        assert not any(
            row.question_text == "Which synthetic deployment should we use?"
            for row in storage.list_pending_tool_calls()
        ), "Stale execution may not create an orphan question"
    saved = storage.get_agent_run("run_1")
    assert saved is not None
    if withdrawal is None:
        assert tool_errors == [], "Control must exercise the real available tool"
        assert saved.committed_output is not None
    else:
        assert saved.committed_output is None
    assert saved.resume_attempts == 1, "Already-entered model work stays charged"


@pytest.mark.parametrize("tool_name", ["ask_human", "attach_pending_info_request"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_busy_pending_tool_aborts_real_loop_and_defers_resume(
    shared_worker: DurableLearningWorker, publication, monkeypatch, tool_name, wrapped
):
    from types import SimpleNamespace

    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.services.storage.error import ExecutionInputBusyError

    storage, worker, kind, dispatches, events = publication
    if tool_name == "ask_human" and kind == "profile":
        pytest.skip("Profile extraction exposes only the attachment tool")
    _question(storage)
    calls = []

    def completion(**_kwargs):
        calls.append(1)
        assert len(calls) == 1, (
            "Busy tools must stop before another physical model call"
        )
        return SimpleNamespace(
            value=SimpleNamespace(
                content=None,
                tool_calls=[
                    {
                        "id": "synthetic_tool_call",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                {"question": "Which synthetic pending target?"}
                                if tool_name == "ask_human"
                                else {"pending_tool_call_id": "ptc_2"}
                            ),
                        },
                    }
                ],
            ),
            provenance=ModelProvenance("gpt-4o-mini", "test"),
        )

    monkeypatch.setattr(
        LiteLLMClient,
        "get_completion_model_ladder",
        lambda _self, **_kw: ["gpt-4o-mini"],
    )
    monkeypatch.setattr(
        LiteLLMClient,
        "generate_chat_response_with_provenance",
        lambda _self, **kwargs: completion(**kwargs),
    )
    refusals = []

    def busy(*_args, **_kwargs):
        refusals.append(1)
        error = ExecutionInputBusyError("synthetic input held by callback")
        if wrapped:
            raise RuntimeError("explicit storage wrapper") from error
        raise error

    monkeypatch.setattr(storage, "_lock_pending_tool_execution_inputs", busy)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert calls == [1] and refusals == [1]
    assert result is not None and result.status == AgentRunStatus.RESUME_READY
    assert result.last_error == "ExecutionInputBusyError"
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.committed_output is None
    assert saved.resume_attempts == 1, "Already entered model work remains charged"
    assert saved.finalization_attempts == 0
    assert {
        d.pending_tool_call_id for d in storage.list_run_tool_dependencies("run_1")
    } == {"ptc_1"}
    assert dispatches == [] and _learnings(storage, kind) == []
    assert not any(e.event_name == "learnings_generated" for e in events)


@pytest.mark.parametrize("fallback", [False, True])
def test_new_output_freezes_full_source_inputs_before_model(
    shared_worker: DurableLearningWorker, publication, monkeypatch, fallback
):
    import hashlib

    from reflexio.server.services.extraction.input_preparation import (
        source_witness_bytes,
    )

    storage, worker, kind, _, _ = publication
    getter = storage.get_request
    if fallback:
        monkeypatch.setattr(storage, "get_request", lambda _identifier: None)
    models = _request_interaction_models_from_ids(
        storage, [1, 2, 3], fallback_source="api", fallback_agent_version="v1"
    )
    expected = {
        "project_id": "",
        "sources": {
            str(item.interaction_id): {
                "digest": hashlib.sha256(
                    source_witness_bytes(model.request, item)
                ).hexdigest(),
                "origin": "fallback" if fallback else "request",
                **({"fallback_first_interaction_id": 1} if fallback else {}),
            }
            for model in models
            for item in model.interactions
        },
    }
    entered = []

    def complete(**kwargs):
        entered.append(1)
        # Both a request-only replacement and a new real row after fallback
        # must not rewrite which source the already-entered model saw.
        storage._execute(
            "UPDATE requests SET session_id='changed-during-model' WHERE request_id='request_1'"
        )
        monkeypatch.setattr(storage, "get_request", getter)
        return _loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert entered == [1]
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.committed_output is not None
    assert saved.committed_output["publication"].get("source_inputs") == expected


def test_publication_refuses_borrowed_writer_before_preparation_or_external_effects(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    storage, worker, kind, dispatches, events = publication
    calls = []

    def complete(**kwargs):
        calls.append(1)
        return _loop_result(kwargs, kind, storage, nonempty=True)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    publish = ExtractionResumeWorker._publish_retained_output

    def borrowed_writer(active_worker, run):
        with storage.commit_scope():
            publish(active_worker, run)
            raise RuntimeError("controlled rollback of caller-owned writer")

    monkeypatch.setattr(
        ExtractionResumeWorker, "_publish_retained_output", borrowed_writer
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert calls == [1]
    assert [
        event for event in events if event.event_name == "learnings_generated"
    ] == [], "No financial delivery may escape a caller's rollback"
    assert dispatches == []
    assert _learnings(storage, kind) == []
    assert result is not None and result.status == AgentRunStatus.FINALIZATION_FAILED


def test_answer_expiry_after_terminal_status_write_rolls_back_publication(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    storage, worker, kind, dispatches, events = publication
    used = storage.get_pending_tool_call("ptc_1")
    assert used is not None
    short_lived = storage.update_resolved_pending_tool_call_result(
        used.id,
        result=used.result,
        resolved_at=datetime.now(UTC),
        valid_for_seconds=10,
    )
    assert short_lived is not None and short_lived.valid_until is not None
    expired_at = short_lived.valid_until.timestamp() + 1
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: _loop_result(kw, kind, storage, nonempty=True),
    )
    update = storage.update_agent_run_status
    transitions = []

    def update_then_expire(run_id, status, **kwargs):
        result = update(run_id, status, **kwargs)
        if status == AgentRunStatus.FINALIZED:
            transitions.append(status)
            monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _db: expired_at)
        return result

    monkeypatch.setattr(storage, "update_agent_run_status", update_then_expire)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert transitions == [AgentRunStatus.FINALIZED]
    assert result is not None and result.status == AgentRunStatus.FINALIZATION_FAILED
    assert _learnings(storage, kind) == [] and dispatches == []
    assert [
        event for event in events if event.event_name == "learnings_generated"
    ] == []
    assert all(
        d.consumed_at is None for d in storage.list_run_tool_dependencies("run_1")
    )
    assert (
        storage.get_agent_run_finalization_receipt(
            run_id="run_1",
            entity_type="profile" if kind == "profile" else "user_playbook",
        )
        is None
    )


@pytest.mark.parametrize(
    "delivery_status",
    [
        UsageEventDeliveryStatus.FAILED,
        UsageEventDeliveryStatus.REJECTED,
        UsageEventDeliveryStatus.UNKNOWN,
    ],
)
def test_delivery_failure_keeps_execution_terminal_and_retries_only_immutable_facts(
    shared_worker: DurableLearningWorker, publication, monkeypatch, delivery_status
):
    from reflexio.server.services.extraction.publication import _deliver

    storage, worker, kind, _, events = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    calls = []

    def complete(**kwargs):
        calls.append(1)
        return _loop_result(kwargs, kind, storage, nonempty=True)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    configure_usage_event_recorder(lambda _event: delivery_status)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.finalization_attempts == 0 and calls == [1]
    assert len(_learnings(storage, kind)) == 1
    assert result.committed_output is not None
    output_id = result.committed_output["publication"]["id"]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(db, result.id, output_id)
    assert receipt is not None and receipt.delivery_state == "pending"
    assert receipt.financial_facts is not None

    def unavailable(*_args, **_kwargs):
        raise AssertionError(
            "Receipt delivery must not read execution sources or mutable model config"
        )

    for name in ("get_agent_run", "get_request", "get_interactions_by_ids"):
        monkeypatch.setattr(storage, name, unavailable)
    monkeypatch.setattr(worker.request_context.configurator, "get_config", unavailable)
    configure_usage_event_recorder(
        lambda event: events.append(event) or UsageEventDeliveryStatus.APPENDED
    )
    _deliver(storage, receipt)
    with storage._stream_sql(read_only=True) as db:
        accepted = storage._get_output_publication_receipt(db, result.id, output_id)
    assert accepted is not None and accepted.delivery_state == "accepted"
    assert accepted.financial_facts is None and calls == [1]
    assert (
        len(events) == 1
        and events[0].created_at == receipt.financial_facts["created_at"]
    )


def test_publication_preparation_uses_private_retained_output_not_mutable_caller(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    storage, worker, kind, _, _ = publication
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: _loop_result(kw, kind, storage, nonempty=True),
    )
    decode = ExtractionResumeWorker._items_from_committed_output
    mutations = []

    def mutate_caller_then_decode(active_worker, frame):
        original = active_worker._active_claim
        assert original is not None and original.committed_output is not None
        field = "profiles" if kind == "profile" else "playbooks"
        original.committed_output["output"][field][0]["content"] = (
            "An unchecked caller replaced the result."
        )
        mutations.append(1)
        return decode(active_worker, frame)

    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_items_from_committed_output",
        mutate_caller_then_decode,
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert mutations == [1]
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert [row.content for row in _learnings(storage, kind)] == [
        "Deployment requires approval."
    ]


def test_postcommit_index_failure_does_not_reopen_committed_publication(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    storage, worker, kind, dispatches, events = publication
    calls = []
    flushed = []

    def complete(**kwargs):
        calls.append(1)
        return _loop_result(kwargs, kind, storage, nonempty=True)

    def fail_index(*_args, **_kwargs):
        flushed.append(1)
        raise RuntimeError("controlled deferred index failure after primary commit")

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    monkeypatch.setattr(storage, "_flush_index_op", fail_index)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert flushed, "The actual deferred index path must have run"
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.status == AgentRunStatus.FINALIZED
    assert result == saved
    assert calls == [1] and len(_learnings(storage, kind)) == 1
    assert dispatches
    entity = "profile" if kind == "profile" else "user_playbook"
    ids = storage.get_agent_run_finalization_receipt(run_id="run_1", entity_type=entity)
    assert ids is not None and len(ids) == 1
    assert [
        event.event_key for event in events if event.event_name == "learnings_generated"
    ] == [f"learn:{entity}:{ids[0]}"]


def test_postcommit_dispatch_failure_preserves_other_obligations(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from reflexio.server.services.extraction.publication import PreparedPublication

    storage, worker, kind, dispatches, events = publication
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kwargs: _loop_result(kwargs, kind, storage, nonempty=True),
    )

    def fail_dispatch(_prepared):
        dispatches.append("attempted_dispatch")
        raise RuntimeError("controlled postcommit scheduler outage")

    monkeypatch.setattr(PreparedPublication, "dispatch", fail_dispatch)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert dispatches == ["attempted_dispatch", "tagging"]
    assert len(_learnings(storage, kind)) == 1
    assert (
        len([event for event in events if event.event_name == "learnings_generated"])
        == 1
    )


@pytest.mark.parametrize("resolved", [False, True])
def test_unused_answer_expiry_during_status_write_stops_advertising_work(
    shared_worker: DurableLearningWorker, publication, monkeypatch, resolved
):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    storage, worker, kind, _, _ = publication
    _question(storage)
    expires_at = datetime.now(UTC) + timedelta(seconds=10)
    storage._execute(
        "UPDATE _pending_tool_calls SET expires_at=? WHERE id='ptc_2'",
        (expires_at.isoformat(),),
    )

    def complete(**kwargs):
        storage.attach_run_tool_dependency(
            RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
        )
        if resolved:
            storage.resolve_pending_tool_call(
                "ptc_2",
                result={"answer": "Arrived after model inputs froze"},
                resolved_at=expires_at - timedelta(seconds=10),
                valid_for_seconds=10,
            )
        return _loop_result(kwargs, kind, storage, nonempty=True, pending=["ptc_2"])

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    write_status = storage.update_agent_run_status
    observed = []
    initial_status = (
        AgentRunStatus.RESUME_READY
        if resolved
        else AgentRunStatus.FINALIZED_PENDING_TOOL
    )

    def expire_after_status(run_id, status, **kwargs):
        result = write_status(run_id, status, **kwargs)
        if status == initial_status:
            observed.append(status)
            monkeypatch.setattr(
                SQLiteStreamSQL, "now", lambda _db: expires_at.timestamp() + 1
            )
        return result

    monkeypatch.setattr(storage, "update_agent_run_status", expire_after_status)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert observed == [initial_status], (
        "Control must exercise the final status-write boundary"
    )
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.pending_tool_call_ids == []
    assert storage.get_agent_run(result.id) == result
    assert len(_learnings(storage, kind)) == 1
    dependencies = {
        d.pending_tool_call_id: d for d in storage.list_run_tool_dependencies(result.id)
    }
    assert dependencies["ptc_1"].consumed_at is not None
    assert dependencies["ptc_2"].consumed_at is None
    assert (
        shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 0
    ), "Expired unused input must not keep reentering the queue"
