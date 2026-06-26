"""Verify the producer retries save_agent_success_evaluation_results with backoff
and records a failure into EvalHealth when retries are exhausted."""

from unittest.mock import MagicMock

from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation.service import (
    AgentSuccessEvaluationService,
)


def _make_service_with_storage(storage_mock):
    """Build the service with internals stubbed enough to exercise _process_results."""
    svc = AgentSuccessEvaluationService.__new__(AgentSuccessEvaluationService)
    svc.last_run_result_count = 0
    svc.last_run_saved_result_count = 0
    svc.last_run_save_failed = False
    svc._last_extractor_run_stats = {"failed": 0}
    svc.storage = storage_mock
    svc.service_config = MagicMock(session_id="sess_test")
    return svc


def test_save_retried_three_times_with_backoff(monkeypatch) -> None:
    _eval_health._HEALTH.__init__()
    storage = MagicMock()
    storage.save_agent_success_evaluation_results.side_effect = RuntimeError("nope")

    slept: list[float] = []
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation.service.time.sleep",
        lambda s: slept.append(s),
    )

    svc = _make_service_with_storage(storage)
    fake_result = MagicMock()
    svc._process_results([[fake_result]])

    assert storage.save_agent_success_evaluation_results.call_count == 3
    assert slept == [1, 4]  # backoffs *between* attempts (no sleep after final)
    assert svc.last_run_save_failed is True
    assert _eval_health.get_status()["producer_failures_24h"] == 1


def test_save_succeeds_on_second_attempt(monkeypatch) -> None:
    _eval_health._HEALTH.__init__()
    calls = {"n": 0}

    def flaky(_results):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flake")

    storage = MagicMock()
    storage.save_agent_success_evaluation_results.side_effect = flaky
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation.service.time.sleep",
        lambda _: None,
    )

    svc = _make_service_with_storage(storage)
    fake_result = MagicMock()
    svc._process_results([[fake_result]])

    assert calls["n"] == 2
    assert svc.last_run_save_failed is False
    assert svc.last_run_saved_result_count == 1
    assert _eval_health.get_status()["producer_failures_24h"] == 0
