"""Earlier profile prompt history cannot escape ordinary publication proof."""

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import (
    test_publication_history_integration as history,
)
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
@pytest.mark.parametrize("change", [None, "content", "delete", "status", "owner"])
def test_candidate_history_is_checked_even_when_no_consolidation_is_needed(
    shared_worker: DurableLearningWorker, publication, monkeypatch, change
):
    storage, worker, kind, dispatches, events = publication
    attempts = base._capture_output_inputs(storage, monkeypatch)
    original = history._seed_history(storage, kind)
    from reflexio.server.services.extraction.profile_package_state import (
        profile_witness,
    )

    expected = {original.profile_id: profile_witness(original)}
    calls = []

    def model(**kwargs):
        assert "Historical approval policy." in str(kwargs["messages"])
        calls.append(1)
        if change == "content":
            storage._execute(
                "UPDATE profiles SET content='Changed during candidate extraction' WHERE profile_id=?",
                (original.profile_id,),
            )
        elif change == "delete":
            storage._execute(
                "DELETE FROM profiles WHERE profile_id=?", (original.profile_id,)
            )
        elif change == "status":
            storage._execute(
                "UPDATE profiles SET status='pending' WHERE profile_id=?",
                (original.profile_id,),
            )
        elif change == "owner":
            storage._execute(
                "UPDATE profiles SET user_id='other-user' WHERE profile_id=?",
                (original.profile_id,),
            )
        # The first-stage prompt suppresses known facts. An empty candidate set
        # never invokes consolidation, so its own history cannot be a fallback.
        return base._loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert calls == [1] and result is not None
    assert len(attempts) == 1 and attempts[0][0] == result.id
    candidate = attempts[0][1]
    assert candidate["history_inputs"] == {
        "profiles": expected,
        "playbooks": {},
    }
    assert not [event for event in events if event.event_name == "learnings_generated"]
    if change is None:
        assert result.status == AgentRunStatus.FINALIZED
        assert result.committed_output is not None
        assert dispatches == ["tagging"]
        assert all(
            d.consumed_at is not None
            for d in storage.list_run_tool_dependencies(result.id)
        )
    else:
        assert result.status == AgentRunStatus.RESUME_READY
        assert result.committed_output is None
        assert not dispatches
        assert all(
            d.consumed_at is None for d in storage.list_run_tool_dependencies(result.id)
        )
        with storage._stream_sql(read_only=True) as db:
            assert (
                storage._get_output_publication_receipt(db, result.id, candidate["id"])
                is None
            )


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
@pytest.mark.parametrize("change", [None, "content"])
def test_fresh_extractor_retains_original_prompt_history(
    publication, monkeypatch, change
):
    from reflexio.server.services.extraction.profile_package_state import (
        profile_witness,
    )
    from reflexio.server.services.extraction.resume_worker import (
        _request_interaction_models_from_ids,
    )
    from reflexio.server.services.profile.components.extractor import ProfileExtractor
    from reflexio.server.services.profile.service import ProfileGenerationServiceConfig

    storage, worker, kind, _, _ = publication
    attempts = base._capture_output_inputs(storage, monkeypatch)
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    original = history._seed_history(storage, kind)
    expected = {original.profile_id: profile_witness(original)}
    config = worker.request_context.configurator.get_config().profile_extractor_config
    service_config = ProfileGenerationServiceConfig(
        user_id="user_1",
        request_id="request_1",
        source="api",
        auto_run=False,
        force_extraction=True,
    )
    service_config.extraction_window_id = "fresh-history-proof"
    extractor = ProfileExtractor(
        worker.request_context,
        worker.client,
        config,
        service_config,
        "Deployment guidance",
    )

    def model(**kwargs):
        assert "Historical approval policy." in str(kwargs["messages"])
        # Neither caller mutation nor a newer DB read may replace original proof.
        original.content = "Caller changed the previously supplied model"
        if change:
            storage._execute(
                "UPDATE profiles SET content='Changed during candidate extraction' WHERE profile_id=?",
                (original.profile_id,),
            )
        return base._loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    groups = _request_interaction_models_from_ids(
        storage, [1, 2, 3], fallback_source="api", fallback_agent_version="v1"
    )
    if change:
        with pytest.raises(
            InputBudgetConfigurationError, match="Historical profile changed"
        ):
            extractor._generate_raw_updates_from_sessions(groups, [original])
    else:
        assert extractor._generate_raw_updates_from_sessions(groups, [original]) == []
    assert len(attempts) == 1
    assert attempts[0][1]["history_inputs"] == {"profiles": expected, "playbooks": {}}
    saved = storage.get_agent_run(attempts[0][0])
    assert saved is not None
    if change:
        assert saved.committed_output is None
    else:
        assert saved.committed_output is not None
        assert (
            saved.committed_output["publication"]["history_inputs"]
            == attempts[0][1]["history_inputs"]
        )


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
@pytest.mark.parametrize("change", ["missing", "stale"])
def test_invalid_history_refuses_before_provider_or_attempt_charge(
    publication, monkeypatch, change
):
    from unittest.mock import MagicMock

    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.extraction.history_inputs import HistoricalInputs
    from reflexio.server.services.extraction.profile_package_state import (
        profile_witness,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.extraction.resume_worker import (
        _request_interaction_models_from_ids,
    )
    from reflexio.server.services.extraction.source_inputs import capture_source_inputs
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    storage, worker, kind, _, _ = publication
    original = history._seed_history(storage, kind)
    declared = HistoricalInputs(
        profiles={original.profile_id: profile_witness(original)}
    )
    run = storage.get_agent_run("run_1")
    assert run is not None
    sources = capture_source_inputs(
        storage,
        run,
        _request_interaction_models_from_ids(
            storage, [1, 2, 3], fallback_source="api", fallback_agent_version="v1"
        ),
    )
    storage._execute(
        "UPDATE profiles SET content='Changed before model admission' WHERE profile_id=?",
        (original.profile_id,),
    )
    model = MagicMock(side_effect=AssertionError("Invalid history reached provider"))
    before_model = MagicMock(side_effect=lambda run: run)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    with pytest.raises(InputBudgetConfigurationError):
        ResumableExtractionAgent(client=worker.client, storage=storage).resume(
            run=run,
            messages=[],
            history_inputs=None if change == "missing" else declared,
            source_inputs=sources,
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[],
            before_model=before_model,
        )
    model.assert_not_called()
    before_model.assert_not_called()
