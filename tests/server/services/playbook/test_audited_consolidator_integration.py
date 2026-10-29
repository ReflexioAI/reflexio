"""Real stored provenance, consolidator dispatch and LiteLLM request seam."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import pytest

from reflexio.models.api_schema.domain.entities import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.config_schema import (
    DeduplicationConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.agent_run_records import (
    source_interaction_digest,
)
from reflexio.server.services.extraction.input_preparation import SourceSpan
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
    current_overflow_coordinator,
)
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord
from reflexio.server.services.playbook import input_capture
from reflexio.server.services.playbook.components import reconciliation
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.components.reconciliation import (
    AuditedConsolidationAbstainedError,
    Reconciler,
)
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.server.services.playbook.review_window import PlaybookReviewWindowError
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from tests.server.services.playbook.test_audited_reconciliation import (
    payload as reconciliation_payload,
)
from tests.server.services.playbook.test_audited_reconciliation import (
    replacement as reconciliation_replacement,
)
from tests.server.services.test_durable_window_pipeline import (
    _overflow_coordinator,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_durable_pipeline,
)

pytestmark = pytest.mark.integration
durable_pipeline = imported_durable_pipeline


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(
        org_id="audited-consolidator", db_path=str(tmp_path / "audit.db")
    )


@pytest.fixture
def consolidator(storage, tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    context = RequestContext(org_id=storage.org_id, storage_base_dir=str(tmp_path))
    context.storage = storage
    context.prompt_manager = MagicMock()
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o", fallback_models=[]))
    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as manager:
        manager.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-4o"
        }
        yield PlaybookConsolidator(
            context, client, DeduplicationConfig(strategy="audited")
        )


def source(storage, *, owner="owner", agent_version="v1", expert=False):
    request = Request(
        request_id="evidence",
        user_id=owner,
        session_id="session",
        agent_version=agent_version,
        source="agent",
    )
    storage.add_request(request)
    interaction = Interaction(
        user_id=owner,
        request_id=request.request_id,
        content="Check authorization before submitting.",
        role="Assistant" if expert else "User",
        expert_content="Report authorization result." if expert else "",
    )
    storage.add_user_interaction(owner, interaction)
    # Some backend facades do not mutate the caller's input ID.
    return storage.get_interactions_by_request_ids([request.request_id])[
        0
    ].interaction_id


def row(*, ids, persisted_id=0, status=None):
    return UserPlaybook(
        user_playbook_id=persisted_id,
        user_id="owner",
        agent_version="v1",
        request_id="generation",
        trigger="before submitting",
        content="Check authorization.",
        source_interaction_ids=ids,
        status=status,
        source="agent",
        reader_angle="operator",
    )


def response(answer):
    return litellm.ModelResponse(
        model="gpt-4o",
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(answer)},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )


@pytest.mark.parametrize(
    "mode", ["disabled", "untargeted", "selected", "failed", "factory_failed"]
)
def test_exact_input_capture_is_lazy_detached_and_fail_open(
    consolidator, storage, monkeypatch, caplog, mode
):
    interaction_id = source(storage, expert=True)
    incumbent = row(ids=[interaction_id], persisted_id=123)
    candidate = row(ids=[interaction_id])
    ref = {"session_id": "evidence", "turn": interaction_id}
    snapshots = []
    calls = []
    original_builder = input_capture.build_consolidator_input_snapshot
    build_count = 0

    def build(**kwargs):
        nonlocal build_count
        build_count += 1
        if mode == "factory_failed":
            raise RuntimeError("private source text must stay out of logs")
        return original_builder(**kwargs)

    monkeypatch.setattr(
        "reflexio.server.services.playbook.components.reconciliation."
        "build_consolidator_input_snapshot",
        build,
    )

    def observe(**kwargs):
        assert kwargs["storage"] is storage
        assert kwargs["user_id"] == "owner"
        assert kwargs["generation_request_id"] == "generation"
        assert calls == []  # Selection precedes the method and audit outcomes.
        if mode == "untargeted":
            return
        snapshot = kwargs["snapshot_factory"]()
        snapshots.append(snapshot)
        detached = snapshot.as_dict()
        detached["source_witnesses"].clear()
        decoded = json.loads(detached["payload_json"])
        decoded["new"][0]["content"] = "observer mutation must stay detached"
        if mode == "failed":
            raise RuntimeError("private source text and credential must not appear")

    monkeypatch.setattr(
        input_capture,
        "get_service",
        lambda _key: None if mode == "disabled" else SimpleNamespace(observe=observe),
    )
    answers = iter(
        [
            {
                "decisions": [
                    {
                        "kind": "remove",
                        "new_ids": ["NEW-0"],
                        "existing_ids": [],
                        "reason": "Existing advice already covers the candidate.",
                        "evidence_refs": [ref],
                    }
                ]
            },
            {"findings": []},
        ]
    )

    def complete(params, _timeout):
        calls.append(params)
        return response(next(answers))

    with patch.object(
        consolidator.client, "_completion_with_hard_timeout", side_effect=complete
    ):
        result = consolidator.deduplicate(
            [[candidate]],
            generation_request_id="generation",
            agent_version="v1",
            user_id="owner",
            existing_playbooks=[incumbent],
        )
    assert result == ([], [], [])
    assert len(calls) == 2
    assert build_count == (1 if mode in {"selected", "failed", "factory_failed"} else 0)
    method_text = calls[0]["messages"][1]["content"]
    assert json.loads(method_text)["new"][0]["content"] == candidate.content
    if snapshots:
        snapshot = snapshots[0]
        assert snapshot.payload_json == method_text
        assert (
            snapshot.payload_sha256
            == hashlib.sha256(method_text.encode("utf-8")).hexdigest()
        )
        original = storage.get_interactions_by_ids([interaction_id])[0]
        assert snapshot.as_dict()["source_witnesses"] == [
            {
                "interaction_id": interaction_id,
                "request_id": original.request_id,
                "digest": source_interaction_digest(original),
            }
        ]
        assert snapshot.as_dict()["existing_playbook_witnesses"] == [
            {
                "user_playbook_id": 123,
                "trigger": incumbent.trigger,
                "content": incumbent.content,
                "rationale": incumbent.rationale,
                "source_interaction_ids": incumbent.source_interaction_ids,
            }
        ]
        assert set(snapshot.as_dict()["identity"]) == {
            "agent_version",
            "model_name",
            "runtime_sha256",
            "policy_sha256",
        }
        assert (
            snapshot.runtime_sha256
            == hashlib.sha256(Path(reconciliation.__file__).read_bytes()).hexdigest()
        )
        assert snapshot.model_name == consolidator.model_name
    assert "private source text" not in caplog.text
    if mode in {"failed", "factory_failed"}:
        assert "event=consolidator_input_capture_failed" in caplog.text


def test_input_is_captured_before_failed_audited_decision(
    consolidator, storage, monkeypatch
):
    interaction_id = source(storage)
    snapshots = []

    def observe(**kwargs):
        snapshots.append(kwargs["snapshot_factory"]())

    def fail(_engine, payload, _references):
        assert snapshots[0].payload_json == json.dumps(payload, ensure_ascii=False)
        raise AuditedConsolidationAbstainedError("scripted decision failure")

    monkeypatch.setattr(
        input_capture, "get_service", lambda _key: SimpleNamespace(observe=observe)
    )
    monkeypatch.setattr(
        "reflexio.server.services.playbook.components.reconciliation._run_audited",
        fail,
    )
    with pytest.raises(AuditedConsolidationAbstainedError):
        consolidator.deduplicate(
            [[row(ids=[interaction_id])]],
            generation_request_id="generation",
            agent_version="v1",
            user_id="owner",
            existing_playbooks=[row(ids=[interaction_id], persisted_id=123)],
        )
    assert len(snapshots) == 1


@pytest.fixture
def isolated_presentation():
    presentation = MagicMock()
    presentation.visible_spans = [
        SimpleNamespace(interaction_id=999999, request_id="other-chunk")
    ]
    token = current_overflow_presentation.set(presentation)
    try:
        yield presentation
    finally:
        current_overflow_presentation.reset(token)


@pytest.mark.parametrize("expert", [False, True])
def test_dispatch_preserves_pending_metadata_and_each_output_evidence(
    consolidator, storage, expert, isolated_presentation, monkeypatch
):
    monkeypatch.setattr(
        storage, "governance_subject_ref_for_user", lambda _user_id: "owner-subject"
    )
    interaction_id = source(storage, expert=expert)
    ref = {"session_id": "evidence", "turn": interaction_id}
    outputs = [
        {
            "trigger": "before submitting",
            "content": text,
            "rationale": "See cited evidence.",
            "evidence_refs": [ref],
        }
        for text in ["Check authorization.", "Report authorization result."]
    ]
    decisions = {
        "decisions": [
            {
                "kind": "replace",
                "new_ids": ["NEW-0"],
                "existing_ids": ["EXISTING-0"],
                "outputs": outputs,
                "reason": "distinct supported stages",
            }
        ]
    }
    old = row(ids=[999999], persisted_id=123, status=Status.PENDING)
    old.governance_subject_ref = "owner-subject"
    calls = []
    answers = iter([decisions, {"findings": []}])

    def complete(params, _timeout):
        calls.append(params)
        return response(next(answers))

    with (
        patch.object(
            consolidator.client, "_completion_with_hard_timeout", side_effect=complete
        ),
        patch(
            "reflexio.server.services.playbook.components.overflow_evidence.remember_historical_playbooks"
        ) as remember,
    ):
        candidate = row(ids=[interaction_id], status=Status.PENDING)
        candidate.source = "stale-batch-label"
        rows, archives, merges = consolidator.deduplicate(
            [[candidate]],
            generation_request_id="generation",
            agent_version="v1",
            user_id="owner",
            existing_playbooks=[old],
        )
        remember.assert_called_once_with([old])
        assert current_overflow_presentation.get() is isolated_presentation
    assert len(calls) == 2
    assert len(rows) == 2 and archives == [123] and merges == []
    assert consolidator.lineage_sources_by_output_index == {0: ["123"], 1: ["123"]}
    assert all(
        saved.status is Status.PENDING
        and saved.source_interaction_ids == [interaction_id]
        for saved in rows
    )
    assert all(
        saved.reader_angle == "operator" and saved.rationale == "See cited evidence."
        for saved in rows
    )
    assert all(saved.source == "agent" for saved in rows)
    assert all(saved.governance_subject_ref == "owner-subject" for saved in rows)
    audit_payload = json.loads(calls[1]["messages"][1]["content"])
    assert audit_payload["existing"][0]["evidence_refs"] == []
    if expert:
        assert (
            "Report authorization result."
            in audit_payload["sessions"][0]["messages"][0]["content"]
        )
        assert (
            audit_payload["sessions"][0]["messages"][0]["expert_feedback"]
            == "Report authorization result."
        )
        assert (
            audit_payload["sessions"][0]["messages"][0]["agent_response"]
            == "Check authorization before submitting."
        )


@pytest.mark.parametrize(
    "owner,version", [("other-user", "v1"), ("owner", "other-agent")]
)
def test_wrong_owner_or_agent_evidence_never_reaches_provider(
    consolidator, storage, owner, version
):
    interaction_id = source(storage, owner=owner, agent_version=version)
    with patch.object(consolidator.client, "_completion_with_hard_timeout") as provider:
        with pytest.raises(
            (PlaybookReviewWindowError, AuditedConsolidationAbstainedError)
        ):
            consolidator.deduplicate(
                [[row(ids=[interaction_id])]],
                generation_request_id="generation",
                agent_version="v1",
                user_id="owner",
                existing_playbooks=[],
            )
        provider.assert_not_called()


def test_missing_new_evidence_abstains_without_unaudited_independent_shortcut(
    consolidator,
):
    with patch.object(consolidator.client, "_completion_with_hard_timeout") as provider:
        with pytest.raises(
            AuditedConsolidationAbstainedError, match="persisted source"
        ):
            consolidator.deduplicate(
                [[row(ids=[])]],
                generation_request_id="generation",
                agent_version="v1",
                user_id="owner",
                existing_playbooks=[],
            )
        provider.assert_not_called()


def test_supplied_governance_subject_cannot_override_trusted_storage_owner(
    consolidator, storage
):
    interaction_id = source(storage)
    incumbent = row(ids=[interaction_id], persisted_id=123)
    incumbent.governance_subject_ref = "other-subject"
    with patch.object(consolidator.client, "_completion_with_hard_timeout") as provider:
        with pytest.raises(AuditedConsolidationAbstainedError, match="trusted owner"):
            consolidator.deduplicate(
                [[row(ids=[interaction_id])]],
                generation_request_id="generation",
                user_id="owner",
                agent_version="v1",
                existing_playbooks=[incumbent],
            )
        provider.assert_not_called()


@pytest.mark.parametrize("same_request", [False, True])
def test_backfilled_event_chronology_is_independent_of_citation_ids(
    consolidator, storage, same_request
):
    storage.add_request(
        Request(
            request_id="newer",
            user_id="owner",
            session_id="session",
            agent_version="v1",
            source="agent",
            created_at=200,
        )
    )
    storage.add_user_interaction(
        "owner",
        Interaction(
            user_id="owner", request_id="newer", created_at=200, content="Use format B."
        ),
    )
    newer_id = storage.get_interactions_by_request_ids(["newer"])[0].interaction_id
    older_request = "newer" if same_request else "older"
    if not same_request:
        storage.add_request(
            Request(
                request_id=older_request,
                user_id="owner",
                session_id="session",
                agent_version="v1",
                source="agent",
                created_at=100,
            )
        )
    storage.add_user_interaction(
        "owner",
        Interaction(
            user_id="owner",
            request_id=older_request,
            created_at=100,
            content="Use format A.",
        ),
    )
    older_id = max(
        item.interaction_id
        for item in storage.get_interactions_by_request_ids([older_request])
    )
    assert older_id > newer_id
    requests = []
    answers = iter(
        [
            {
                "decisions": [
                    {
                        "kind": "create",
                        "new_ids": ["NEW-0"],
                        "reason": "latest correction",
                        "outputs": [
                            {
                                "trigger": "when formatting",
                                "content": "Use format B.",
                                "rationale": "See cited evidence.",
                                "evidence_refs": [
                                    {"session_id": "newer", "turn": newer_id}
                                ],
                            }
                        ],
                    }
                ]
            },
            {"findings": []},
        ]
    )

    def complete(params, _timeout):
        requests.append(params)
        return response(next(answers))

    with patch.object(
        consolidator.client, "_completion_with_hard_timeout", side_effect=complete
    ):
        consolidator.deduplicate(
            [[row(ids=[newer_id, older_id])]],
            generation_request_id="generation",
            user_id="owner",
            agent_version="v1",
            existing_playbooks=[],
        )
    for request in requests:
        sessions = json.loads(request["messages"][1]["content"])["sessions"]
        messages = [message for session in sessions for message in session["messages"]]
        assert [message["turn"] for message in messages] == [older_id, newer_id]
        assert [message["created_at"] for message in messages] == [100, 200]


def test_existing_only_remove_reaches_audited_engine(consolidator, storage):
    interaction_id = source(storage)
    answers = iter(
        [
            {
                "decisions": [
                    {
                        "kind": "remove",
                        "new_ids": [],
                        "existing_ids": ["EXISTING-0"],
                        "reason": "procedure retired by source",
                        "evidence_refs": [
                            {"session_id": "evidence", "turn": interaction_id}
                        ],
                    }
                ]
            },
            {"findings": []},
        ]
    )
    with patch.object(
        consolidator.client,
        "_completion_with_hard_timeout",
        side_effect=lambda *_args: response(next(answers)),
    ):
        rows, archives, merges = consolidator.deduplicate(
            [],
            generation_request_id="generation",
            user_id="owner",
            agent_version="v1",
            existing_playbooks=[row(ids=[interaction_id], persisted_id=123)],
        )
    assert rows == [] and archives == [123] and merges == []


def test_overflow_uses_verified_quotations_not_full_large_interaction(
    consolidator, storage
):
    quote = "Check authorization before submitting."
    large = "noise " * 20000 + quote + " trailing noise" * 10000
    storage.add_request(
        Request(
            request_id="evidence",
            user_id="owner",
            session_id="session",
            agent_version="v1",
            source="agent",
        )
    )
    storage.add_user_interaction(
        "owner", Interaction(user_id="owner", request_id="evidence", content=large)
    )
    interaction_id = storage.get_interactions_by_request_ids(["evidence"])[
        0
    ].interaction_id
    start = large.index(quote)
    span = SourceSpan(
        interaction_id=interaction_id,
        request_id="evidence",
        role="User",
        field="content",
        start=start,
        end=start + len(quote),
        text=quote,
        source_order=1,
    )
    coordinator = MagicMock()
    coordinator.require_prepared.return_value = SimpleNamespace(
        chunks=[SimpleNamespace(spans=[span], neighbors=[])]
    )
    coordinator.playbook_key.return_value = "new"
    coordinator.candidate_records = {
        "new": ProtectedRecord.model_validate(
            {
                "evidence": [
                    {
                        "kind": "instruction",
                        "quote": quote,
                        "evidence_refs": [span.evidence_ref],
                    }
                ]
            }
        )
    }
    coordinator.checkpoint = {"completed_steps": {}}
    coordinator.step_key.side_effect = lambda stage, data: hashlib.sha256(
        json.dumps([stage, data], sort_keys=True).encode()
    ).hexdigest()
    coordinator.completed_step.return_value = None
    ref = {"session_id": "evidence", "turn": interaction_id}
    answers = iter(
        [
            {
                "decisions": [
                    {
                        "kind": "create",
                        "new_ids": ["NEW-0"],
                        "reason": "supported",
                        "outputs": [
                            {
                                "content": quote,
                                "trigger": "before submitting",
                                "rationale": "See cited evidence.",
                                "evidence_refs": [ref],
                            }
                        ],
                    }
                ]
            },
            {"findings": []},
        ]
    )
    requests = []

    def complete(params, _timeout):
        requests.append(params)
        return response(next(answers))

    token = current_overflow_coordinator.set(coordinator)
    try:
        with patch.object(
            consolidator.client, "_completion_with_hard_timeout", side_effect=complete
        ):
            rows, _archives, _merges = consolidator.deduplicate(
                [[row(ids=[interaction_id])]],
                generation_request_id="generation",
                user_id="owner",
                agent_version="v1",
                existing_playbooks=[],
            )
    finally:
        current_overflow_coordinator.reset(token)
    assert rows[0].source_span == quote
    for request in requests:
        shown = json.loads(request["messages"][1]["content"])
        assert shown["sessions"][0]["messages"][0]["content"] == quote
        assert len(json.dumps(shown)) < 4000
    coordinator.require_full_source_coverage.assert_called_once()


@pytest.mark.parametrize("tool_and_repair", [False, True])
def test_uncapped_client_durable_admission_suspends_and_resumes_once_per_call(
    durable_pipeline, monkeypatch, tool_and_repair
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-provider")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    storage, window, token = _overflow_coordinator(durable_pipeline)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=None, fallback_models=[])
    )
    answers = iter(
        [
            {"decisions": [reconciliation_replacement()]},
            {"findings": []},
        ]
    )
    requests = []
    active = []
    allowance = 4 if tool_and_repair else 2

    def dispatch(params, _hard_timeout):
        requests.append(params)
        assert params.get("max_tokens", params.get("max_completion_tokens", 0)) > 0
        if tool_and_repair and len(requests) <= 2:
            # End each worker turn after one successful provider response.
            active[0]._deadline = 0
            if len(requests) == 1:
                answer = response({})
                answer.choices[0].message.content = None
                answer.choices[0].message.tool_calls = [
                    litellm.ChatCompletionMessageToolCall(**call)
                    for call in [
                        {
                            "id": "search-1",
                            "type": "function",
                            "function": {
                                "name": "search_playbooks",
                                "arguments": '{"query":"authorization"}',
                            },
                        }
                    ]
                ]
                answer.choices[0].finish_reason = "tool_calls"
                return answer
            return response({"decisions": []})
        return response(next(answers))

    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)

    def coordinator(seconds):
        parent = storage.get_agent_run(f"window:{window.window_id}")
        assert parent is not None
        return OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={"gpt-4o": 128000},
            turn_seconds=seconds,
        )

    first = coordinator(1)
    with first.provider_scope():
        engine = Reconciler(client, "gpt-4o", max_calls=allowance)
        with pytest.raises(OverflowSuspendedError):
            engine.run(reconciliation_payload(), {("request", 17): object()})
        assert engine.calls == 0
    assert requests == []
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    assert parent.generation_request_snapshot["overflow"]["attempts"] == []

    if tool_and_repair:
        for spent in (1, 2):
            partial = coordinator(300)
            active[:] = [partial]
            with partial.provider_scope():
                engine = Reconciler(client, "gpt-4o", max_calls=allowance)
                with pytest.raises(OverflowSuspendedError):
                    engine.run(reconciliation_payload(), {("request", 17): object()})
                assert engine.calls == spent
        state = partial.checkpoint["audited_conversation"]
        assert state["turn"] == 2 and state["repaired"] is True
        assert any(message["role"] == "tool" for message in state["messages"])

    resumed = coordinator(300)
    active[:] = [resumed]
    with resumed.provider_scope():
        engine = Reconciler(client, "gpt-4o", max_calls=allowance)
        engine.run(reconciliation_payload(), {("request", 17): object()})
        assert engine.calls == allowance
    assert len(requests) == allowance
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    attempts = parent.generation_request_snapshot["overflow"]["attempts"]
    assert len(attempts) == allowance
    assert resumed.checkpoint["audited_consolidation_budget"]["calls"] == allowance
    if tool_and_repair:
        assert requests[2].get("tools") is None
        assert "deterministic validation" in requests[2]["messages"][-1]["content"]
    assert all(attempt["usage"] is not None for attempt in attempts)


def _publication_case(
    consolidator, storage, monkeypatch, *, expert=False, pending=False
):
    """Keep real method/service/storage; replace only external inference and search."""
    monkeypatch.setattr(storage, "_get_embedding", lambda _text: [0.1] * 512)
    interaction_id = source(storage, expert=expert)
    incumbent = row(ids=[interaction_id])
    incumbent.request_id = "incumbent-generation"
    storage.save_user_playbooks([incumbent], skip_embedding=True)
    assert incumbent.user_playbook_id > 0
    ref = {"session_id": "evidence", "turn": interaction_id}
    outputs = [
        {
            "trigger": "before submitting",
            "content": text,
            "rationale": "See cited evidence.",
            "evidence_refs": [ref],
        }
        for text in ["Check authorization.", "Report authorization result."]
    ]
    answers = iter(
        [
            {
                "decisions": [
                    {
                        "kind": "replace",
                        "new_ids": ["NEW-0"],
                        "existing_ids": ["EXISTING-0"],
                        "outputs": outputs,
                        "reason": "separate supported steps",
                    }
                ]
            },
            {"findings": []},
        ]
    )
    calls = []

    def dispatch(params, _timeout):
        calls.append(params)
        return response(next(answers))

    monkeypatch.setattr(consolidator.client, "_completion_with_hard_timeout", dispatch)
    context = consolidator.request_context
    context.configurator.set_config_by_name(
        "user_playbook_extractor_config",
        UserPlaybookExtractorConfig(
            extraction_definition_prompt="Extract source-supported instructions",
            deduplication_config=DeduplicationConfig(strategy="audited"),
        ),
    )
    service = PlaybookGenerationService(
        consolidator.client,
        context,
        skip_aggregation=True,
        output_pending_status=pending,
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="generation",
        agent_version="v1",
        user_id="owner",
        source="agent",
        auto_run=False,
    )
    # Candidate extraction/review and nearest-neighbor quality are separate
    # contracts; exercise consolidation through the configured service here.
    monkeypatch.setattr(PlaybookCandidateReviewer, "is_enabled", lambda _self: False)
    monkeypatch.setattr(
        PlaybookConsolidator,
        "retrieve_existing_playbooks",
        lambda *_args, **_kwargs: [incumbent],
    )
    return SimpleNamespace(
        service=service,
        candidate=row(ids=[interaction_id]),
        incumbent=incumbent,
        calls=calls,
        interaction_id=interaction_id,
        contents={output["content"] for output in outputs},
    )


@pytest.mark.parametrize("expert", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_audited_multioutput_service_publication_preserves_each_source_link(
    consolidator, storage, monkeypatch, expert, pending
):
    case = _publication_case(
        consolidator, storage, monkeypatch, expert=expert, pending=pending
    )
    original_events = storage.get_lineage_events()
    plan = case.service._resolve_write_plan([[case.candidate]])
    assert plan is not None and len(plan.new_playbooks) == 2
    assert len(case.calls) == 2  # One decision and its independent audit.
    assert all(playbook.embedding for playbook in plan.new_playbooks)
    monkeypatch.setattr(
        storage,
        "_get_embedding",
        MagicMock(side_effect=AssertionError("Persistence must reuse embeddings")),
    )
    # Computing a valid plan must not publish outputs or retire its incumbent.
    assert storage.get_user_playbooks() == [case.incumbent]
    assert storage.get_lineage_events() == original_events
    with storage.commit_scope():
        case.service._persist_write_plan(plan)
    current = storage.get_user_playbooks(user_id="owner", agent_version="v1")
    assert {playbook.content for playbook in current} == case.contents
    assert all(
        playbook.status == (Status.PENDING if pending else None)
        and playbook.source_interaction_ids == [case.interaction_id]
        for playbook in current
    )
    original = storage.get_user_playbook_by_id(
        case.incumbent.user_playbook_id, include_tombstones=True
    )
    assert original is not None and original.status == Status.SUPERSEDED
    assert original.merged_into is None  # A split has no arbitrary single survivor.
    created = [
        event
        for event in storage.get_lineage_events(request_id="generation")
        if event.op == "create"
    ]
    assert len(created) == 2
    assert {event.entity_id for event in created} == {
        str(playbook.user_playbook_id) for playbook in current
    }
    assert all(
        str(case.incumbent.user_playbook_id) in event.source_ids for event in created
    )


@pytest.mark.parametrize("failure_point", ["partial_insert", "after_retirement"])
def test_audited_service_publication_rolls_back_rows_and_lineage_together(
    consolidator, storage, monkeypatch, failure_point
):
    case = _publication_case(consolidator, storage, monkeypatch)
    original_events = storage.get_lineage_events()
    plan = case.service._resolve_write_plan([[case.candidate]])
    assert plan is not None and len(plan.new_playbooks) == 2
    assert len(case.calls) == 2
    monkeypatch.setattr(
        storage,
        "_get_embedding",
        MagicMock(side_effect=AssertionError("Persistence must reuse embeddings")),
    )
    observed = []
    if failure_point == "partial_insert":
        save = storage.save_user_playbooks

        def fail_after_first_insert(playbooks, **kwargs):
            kwargs["lineage_contexts"] = kwargs["lineage_contexts"][:1]
            save(playbooks[:1], **kwargs)
            assert len(storage.get_user_playbooks()) == 2
            observed.append("first output inserted")
            raise RuntimeError("injected publication failure")

        monkeypatch.setattr(storage, "save_user_playbooks", fail_after_first_insert)
    else:
        retire = storage.supersede_user_playbooks_by_ids

        def fail_after_retirement(*args, **kwargs):
            retire(*args, **kwargs)
            original = storage.get_user_playbook_by_id(
                case.incumbent.user_playbook_id, include_tombstones=True
            )
            assert original is not None and original.status == Status.SUPERSEDED
            observed.append("incumbent retired")
            raise RuntimeError("injected publication failure")

        monkeypatch.setattr(
            storage, "supersede_user_playbooks_by_ids", fail_after_retirement
        )
    with (
        pytest.raises(RuntimeError, match="injected publication failure"),
        storage.commit_scope(),
    ):
        case.service._persist_write_plan(plan)
    assert len(observed) == 1  # The fault must happen after a real mutation.
    assert storage.get_user_playbooks() == [case.incumbent]
    assert storage.get_lineage_events() == original_events
    assert len(case.calls) == 2  # Persistence never dispatches another model call.
