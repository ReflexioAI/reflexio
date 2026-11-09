"""An outage is one WARNING (the breaker opening), not one per waiting window.

Production 2026-11-02..09: 12,821 "waiting for provider" WARNINGs, 10.2k of
them on one day of a provider quota outage, while the breaker's own
``event=provider_circuit_opened`` already reported that outage.
"""

from __future__ import annotations

import logging
from typing import cast
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.provider_circuit import (
    OPEN_AFTER_FAILURES,
    ProviderCircuitOpenError,
    circuit_route,
    provider_circuit,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import Window

MODEL = "test/provider-wait-model"
ROUTE = circuit_route({"model": MODEL})


def _window(window_id: str) -> Window:
    return Window(
        window_id=window_id,
        user_id="u",
        kind="profile",
        project_id="",
        predecessor=0,
        end_seq=1,
        manifest=[],
        policy={},
        force=False,
        skip_aggregation=False,
    )


@pytest.fixture
def clean_circuit():
    provider_circuit.reset()
    yield
    provider_circuit.reset()


def test_outage_warns_once_and_logs_each_window_wait_at_info(clean_circuit, caplog):
    storage = MagicMock(spec=BaseStorage)
    windows = [_window(f"w{i}") for i in range(3)]

    with caplog.at_level(logging.INFO):
        for _ in range(OPEN_AFTER_FAILURES):
            provider_circuit.admit(ROUTE, model=MODEL).failed()
        outcomes = []
        for window in windows:
            refused = ProviderCircuitOpenError("refused")
            refused.circuit_route = ROUTE
            outcomes.append(
                DurableLearningWorker._handle_window_failure(
                    cast(BaseStorage, storage), window, "token", refused
                )
            )

    # Retry/wait behaviour is unchanged: every window waits, uncharged.
    assert outcomes == ["provider_wait"] * len(windows)
    assert storage.retry_extraction.call_count == len(windows)
    assert all(
        call.kwargs["wait_seconds"] > 0
        for call in storage.retry_extraction.call_args_list
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "event=provider_circuit_opened" in warnings[0]
    waits = [
        r
        for r in caplog.records
        if "Extraction window waiting for provider" in r.getMessage()
    ]
    assert len(waits) == len(windows)
    assert {r.levelno for r in waits} == {logging.INFO}
