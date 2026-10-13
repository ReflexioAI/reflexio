"""Real stored provenance, consolidator dispatch and LiteLLM request seam."""

import hashlib
import json
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
from reflexio.models.config_schema import DeduplicationConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.input_preparation import SourceSpan
from reflexio.server.services.extraction.overflow_coordinator import (
    current_overflow_coordinator,
)
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.components.reconciliation import (
    AuditedConsolidationAbstainedError,
)
from reflexio.server.services.playbook.review_window import PlaybookReviewWindowError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


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
    consolidator, storage, expert, isolated_presentation
):
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
