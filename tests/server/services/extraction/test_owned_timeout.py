"""Timeout cancellation belongs to one actual turn, not every matching request."""

from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from reflexio.server.llm import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.base_generation_service import ExtractorExecutionError
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    create_owned_agent_run,
    owned_agent_run_write,
    owned_user_work_context,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus, LeaseLostError
from tests.server.services import test_durable_window_pipeline as durable
from tests.server.services.test_base_generation_service import (
    ConcreteGenerationService,
    MockExtractorConfig,
)

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("invariant", ["unrelated_run", "late_output"])
def test_timeout_cancels_its_turn_without_request_wide_status_writes(
    pipeline, monkeypatch, invariant
):
    storage, window, token = durable._overflow_coordinator(pipeline)
    engine, _, _ = pipeline
    started, resume = Event(), Event()
    outcomes = []
    actual_futures = []
    pool = RealThreadPoolExecutor(max_workers=1)
    try:
        with owned_user_work_context(storage, window.user_id, token, 300):
            template = storage.get_agent_run(f"window:{window.window_id}")
            assert template is not None
            active, unrelated = (
                create_owned_agent_run(
                    storage,
                    replace(template, id=identity, claimed_by=None, claimed_at=None),
                )
                for identity in ("timed_out_execution", "same_request_other_execution")
            )

            class PausedExtractor:
                def run(self):
                    started.set()
                    assert resume.wait(5), "Test must release the owned child thread"
                    try:
                        with owned_agent_run_write(storage, active):
                            storage.update_agent_run_status(
                                active.id,
                                AgentRunStatus.AGENT_COMPLETED,
                                expected_statuses=(AgentRunStatus.RUNNING,),
                                expected_claimed_by=active.claimed_by,
                                expected_claimed_at=active.claimed_at,
                            )
                    except LeaseLostError:
                        outcomes.append("rejected")
                    else:
                        outcomes.append("published")

            def submit(fn, *args):
                actual_futures.append(pool.submit(fn, *args))

                def expire(*, timeout):
                    assert started.wait(5), "Real extractor did not start"
                    raise FutureTimeoutError("Controlled enclosing timeout")

                return SimpleNamespace(result=expire)

            # Use a real child thread and copied production context; only the
            # enclosing Future wait is controlled to avoid scheduler races.
            executor = SimpleNamespace(submit=submit, shutdown=lambda **_: None)
            monkeypatch.setattr(
                "reflexio.server.services.base_generation._extraction_lifecycle.ThreadPoolExecutor",
                lambda **_: executor,
            )
            config = MockExtractorConfig(extractor_name=window.kind)
            service = ConcreteGenerationService(
                LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
                engine.request_context,
                extractor_config=config,
            )
            service.service_config = SimpleNamespace(
                user_id=window.user_id, request_id=template.binding.request_id
            )
            monkeypatch.setattr(
                service, "_create_extractor", lambda *_: PausedExtractor()
            )
            try:
                with pytest.raises(ExtractorExecutionError, match="timed out"):
                    service._execute_extractor(config, window.user_id)
                after_timeout = storage.get_agent_run(unrelated.id)
            finally:
                resume.set()
                for future in actual_futures:
                    future.result(timeout=5)
            if invariant == "unrelated_run":
                assert after_timeout == unrelated
            else:
                assert outcomes == ["rejected"]
                assert storage.get_agent_run(active.id) == active
            # Local cancellation does not clear the actual lease behind its
            # caller's back: the existing failure/retry handler still owns it.
            storage.fence_user_extraction(window.user_id, token)
            assert (
                durable.DurableLearningWorker._handle_window_failure(
                    storage,
                    window,
                    token,
                    ExtractorExecutionError("controlled timeout"),
                )
                == "failure"
            )
            with pytest.raises(LeaseLostError):
                storage.fence_user_extraction(window.user_id, token)
            assert storage.get_agent_run(unrelated.id) == unrelated
        # Existing durable state remains recoverable under a NEW actual owner;
        # cancellation neither deletes output/attempt history nor poisons it.
        replacement = storage.claim_user_extraction(window.user_id, "replacement", 300)
        assert replacement is not None and replacement != token
        with owned_user_work_context(storage, window.user_id, replacement, 300):
            recovered = adopt_agent_run_for_execution(storage, active)
            with owned_agent_run_write(storage, recovered):
                finished = storage.update_agent_run_status(
                    recovered.id,
                    AgentRunStatus.AGENT_COMPLETED,
                    expected_statuses=(AgentRunStatus.RUNNING,),
                    expected_claimed_by=recovered.claimed_by,
                    expected_claimed_at=recovered.claimed_at,
                )
            assert finished is not None
            assert finished.status == AgentRunStatus.AGENT_COMPLETED
            assert (
                finished.generation_request_snapshot
                == active.generation_request_snapshot
            )
        storage.release_user_extraction(window.user_id, replacement)
    finally:
        resume.set()
        pool.shutdown(wait=True, cancel_futures=True)
