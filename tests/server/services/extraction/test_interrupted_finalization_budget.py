"""An interrupted secondary computation must consume the existing retry budget."""

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
pytestmark = pytest.mark.integration
_REAL_EMBEDDING = SQLiteStorage._get_embedding
_REAL_EMBED_TEXTS = LiteLLMClient._embed_texts


@pytest.fixture(autouse=True)
def actual_embedding_dispatch(storage, monkeypatch):
    from types import MethodType, SimpleNamespace

    import litellm

    # Restore the actual SDK dispatcher. Script only the external API response,
    # so admission must pass through the real provider slot and dispatch hook.
    monkeypatch.setattr(storage, "_get_embedding", MethodType(_REAL_EMBEDDING, storage))
    monkeypatch.setattr(storage, "embedding_model_name", "text-embedding-3-small")
    monkeypatch.setattr(storage, "embedding_dimensions", 512)
    monkeypatch.setattr(storage, "_should_expand_documents", lambda: False)
    monkeypatch.setattr(
        storage, "llm_client", LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    )
    monkeypatch.setattr(
        storage.llm_client,
        "_embed_texts",
        MethodType(_REAL_EMBED_TEXTS, storage.llm_client),
    )
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda **_kwargs: SimpleNamespace(
            data=[{"index": 0, "embedding": [0.0] * 512}]
        ),
    )


class SimulatedProcessExit(BaseException):
    """Bypass ordinary Exception cleanup as an exited process would."""


@pytest.mark.parametrize("allowance", [1, 2])
def test_unknown_completed_compute_is_charged_before_restart(
    shared_worker: DurableLearningWorker, publication, monkeypatch, allowance
):
    storage, worker, kind, _, events = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = allowance
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan
    computes = []

    def interrupted(self, groups):
        plan = resolve(self, groups)
        computes.append(1)
        if len(computes) == 1:
            raise SimulatedProcessExit(
                "Exit after compute but before durable preparation"
            )
        return plan

    monkeypatch.setattr(service_type, "_resolve_write_plan", interrupted)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    with pytest.raises(SimulatedProcessExit):
        shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    abandoned = storage.get_agent_run("run_1")
    assert abandoned is not None and abandoned.status == AgentRunStatus.FINALIZING
    assert abandoned.committed_output is not None
    assert computes == [1] and not base._learnings(storage, kind)
    assert not [event for event in events if event.event_name == "learnings_generated"]
    # The exited turn released its actual user owner. Restart immediately;
    # no logical timestamp or private parent token is rewritten.
    restarted = DurableLearningWorker(lambda _org: worker.request_context)
    for _ in range(6):
        if restarted.drain_org(storage.org_id, batch_size=1, lease_seconds=300):
            break
    else:
        pytest.fail("Restart did not reach the retained output in two class rotations")
    result = storage.get_agent_run("run_1")
    assert result is not None
    assert result.finalization_attempts == 1
    assert result.committed_output == abandoned.committed_output
    if allowance == 1:
        assert result.status == AgentRunStatus.FAILED
        assert computes == [1], (
            "A restart must not refund the interrupted compute attempt"
        )
        assert not base._learnings(storage, kind)
        assert all(
            dependency.consumed_at is None
            for dependency in storage.list_run_tool_dependencies(abandoned.id)
        )
    else:
        assert result.status == AgentRunStatus.FINALIZED
        assert computes == [1, 1]
        assert len(base._learnings(storage, kind)) == 1
        assert all(
            dependency.consumed_at is not None
            for dependency in storage.list_run_tool_dependencies(abandoned.id)
        )
        assert restarted.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 0
        assert computes == [1, 1]


@pytest.mark.parametrize("phase", ["before_entry", "after_entry"])
def test_readiness_is_free_only_before_secondary_compute_admission(
    shared_worker: DurableLearningWorker, publication, monkeypatch, phase
):
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    storage, worker, kind, dispatches, _events = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan
    computes = []

    def unavailable(self, groups):
        if phase == "after_entry":
            resolve(self, groups)
            computes.append(1)
        raise ReadinessUnavailableError("Controlled readiness failure")

    monkeypatch.setattr(service_type, "_resolve_write_plan", unavailable)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.committed_output is not None
    charged = int(phase == "after_entry")
    assert result.finalization_attempts == charged
    assert result.status == (
        AgentRunStatus.FAILED if charged else AgentRunStatus.FINALIZATION_FAILED
    )
    assert len(computes) == charged
    assert not base._learnings(storage, kind) and not dispatches
    assert (
        storage.conn.execute(
            "SELECT run_id FROM _agent_output_preparations WHERE run_id=?", (result.id,)
        ).fetchone()
        is None
    )


def test_successful_distinct_outputs_preserve_failure_allowance(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from tests.server.services.extraction import (
        test_publication_preparation_integration as prepared,
    )

    storage, worker, _kind, _dispatches, _events = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    prepared.test_new_output_replaces_preparation_only_after_previous_receipt(
        shared_worker, publication, monkeypatch
    )
    completed = storage.get_agent_run("run_1")
    assert completed is not None and completed.status == AgentRunStatus.FINALIZED
    assert completed.finalization_attempts == 0
    assert (
        storage.conn.execute(
            "SELECT active_attempt FROM _agent_output_preparations WHERE run_id=?",
            (completed.id,),
        ).fetchone()[0]
        is None
    )


def test_absent_entry_is_not_inferred_from_finalizing_status(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    # Pure scripted local compute performs no external request. A status alone
    # cannot prove paid work or manufacture an entry for unknown old history.
    monkeypatch.setattr(
        publication[0], "_get_embedding", lambda *_a, **_kw: [0.0] * 512
    )
    storage, worker, kind, _dispatches, _events = publication
    worker.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan
    calls = []

    def pure(self, groups):
        plan = resolve(self, groups)
        calls.append(1)
        if len(calls) == 1:
            raise SimulatedProcessExit("Exit without external compute")
        return plan

    monkeypatch.setattr(service_type, "_resolve_write_plan", pure)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    with pytest.raises(SimulatedProcessExit):
        shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert (
        storage.conn.execute(
            "SELECT run_id FROM _agent_output_preparations WHERE run_id='run_1'"
        ).fetchone()
        is None
    )
    # Release of the exited actual owner makes saved work immediately eligible.
    restarted = DurableLearningWorker(lambda _org: worker.request_context)
    for _ in range(6):
        if restarted.drain_org(storage.org_id, batch_size=1, lease_seconds=300):
            break
    else:
        pytest.fail("Restart did not reach the retained output in two class rotations")
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.finalization_attempts == 0 and calls == [1, 1]


def test_live_same_claim_is_not_settled_as_an_interrupted_operation(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    import litellm

    from reflexio.server.services.storage.storage_base import LeaseLostError

    storage, worker, kind, _dispatches, _events = publication
    provider = litellm.embedding
    refusals = []
    executing = []
    publish = ExtractionResumeWorker._publish_retained_output

    def observe(active_worker, run):
        executing.append(active_worker)
        try:
            return publish(active_worker, run)
        finally:
            executing.pop()

    monkeypatch.setattr(ExtractionResumeWorker, "_publish_retained_output", observe)

    def response(**kwargs):
        active_worker = executing[-1]
        active = active_worker._active_claim
        assert active is not None and active.status == AgentRunStatus.FINALIZING
        pending = storage.pending_agent_preparation_attempt(active)
        assert pending is not None
        with pytest.raises(LeaseLostError, match="already has secondary compute"):
            active_worker._publish_retained_output(active)
        assert storage.get_agent_run(active.id).finalization_attempts == 0
        assert storage.pending_agent_preparation_attempt(active) == pending
        refusals.append(pending.identity)
        return provider(**kwargs)

    monkeypatch.setattr(litellm, "embedding", response)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.finalization_attempts == 0 and len(refusals) == 1
    assert len(base._learnings(storage, kind)) == 1


def test_external_compute_refuses_a_borrowed_writer_before_dispatch(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from unittest.mock import MagicMock

    import litellm

    storage, worker, kind, dispatches, _events = publication
    embed = storage._get_embedding
    provider = MagicMock(
        side_effect=AssertionError("Provider entered a borrowed writer")
    )
    monkeypatch.setattr(litellm, "embedding", provider)

    def borrowed(*args, **kwargs):
        with storage.commit_scope():
            return embed(*args, **kwargs)

    monkeypatch.setattr(storage, "_get_embedding", borrowed)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZATION_FAILED
    assert result.committed_output is not None
    provider.assert_not_called()
    assert not base._learnings(storage, kind) and not dispatches
    assert (
        storage.conn.execute(
            "SELECT run_id FROM _agent_output_preparations WHERE run_id=?", (result.id,)
        ).fetchone()
        is None
    )


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
def test_parallel_embedding_and_expansion_share_one_pending_entry(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from threading import Barrier, Lock
    from types import SimpleNamespace

    import litellm

    storage, worker, kind, _dispatches, _events = publication
    monkeypatch.setattr(storage, "_should_expand_documents", lambda: True)
    monkeypatch.setattr(storage, "_expansion_model_name", "gpt-4o")
    barrier = Barrier(2)
    observed = []
    observations = Lock()

    def rendezvous(channel):
        active = storage.get_agent_run("run_1")
        assert active is not None and active.status == AgentRunStatus.FINALIZING
        pending = storage.pending_agent_preparation_attempt(active)
        assert pending is not None
        with observations:
            observed.append((channel, pending.identity))
        # This would time out if a scope mutex or writer covered provider I/O.
        barrier.wait(timeout=10)

    def embedding(**_kwargs):
        rendezvous("embedding")
        return SimpleNamespace(data=[{"index": 0, "embedding": [0.0] * 512}])

    def text(_params, _timeout):
        rendezvous("text")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"deploy":["release"]}', tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="gpt-4o",
            _hidden_params={"custom_llm_provider": "openai"},
        )

    monkeypatch.setattr(litellm, "embedding", embedding)
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(text)
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.finalization_attempts == 0
    assert {channel for channel, _identity in observed} == {"embedding", "text"}
    assert (
        len(observed) == 2 and len({identity for _channel, identity in observed}) == 1
    )
    rows = base._learnings(storage, kind)
    assert len(rows) == 1 and rows[0].expanded_terms == "deploy, release"
    assert (
        storage.conn.execute(
            "SELECT active_attempt FROM _agent_output_preparations WHERE run_id=?",
            (result.id,),
        ).fetchone()[0]
        is None
    )
