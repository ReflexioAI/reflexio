"""The publish-timing line must survive the paths `GenerationService.run` takes.

`reflexio/server/publish_timing.py` carries its own module-level guards in
`reflexio_ext/tests/test_publish_timing.py`. What those cannot reach is where
the request path calls it from. A publish has three ways out and each is a
separate placement:

* it SUCCEEDS -- and must produce exactly one line, not two, even though a
  `finally` backstop also runs; and if it was FAST it must produce none at all,
  however long the caller then waits for coverage afterwards;
* it RAISES -- the case most worth a breakdown, and before this it produced
  nothing at all, because the only `emit()` sat on the success path; and
* it is CANCELLED while still queued for admission -- the dominant production
  outcome (97.4% of publishes end as an ELB 460), which never reaches
  `GenerationService.run` and so needs the route's own emit.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.requests import Request

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server import publish_timing
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.rate_limit import limiter
from reflexio.server.services.generation_service import GenerationService

_ORG_ID = "test_org_publish_timing"
_USER_ID = "test_user_publish_timing"


@pytest.fixture(autouse=True)
def _timing_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Report every publish, with no throttle, so nothing is time-dependent."""
    publish_timing.reset_for_tests()
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setenv(publish_timing.ENV_ENABLED, "true")
    monkeypatch.setenv(publish_timing.ENV_THRESHOLD_MS, "0")
    monkeypatch.setenv(publish_timing.ENV_INTERVAL_SECONDS, "0")
    yield
    publish_timing.reset_for_tests()


def _publish_request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        user_id=_USER_ID,
        interaction_data_list=[
            InteractionData(
                content="test interaction",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
        ],
        session_id="test_session_id",
    )


def _timing_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == publish_timing.__name__
        and record.levelno == publish_timing.PUBLISH_TIMING_LOG_LEVEL
    ]


def _starlette_request() -> Request:
    """A real `Request`: slowapi's decorator rejects anything else."""
    app = SimpleNamespace(state=SimpleNamespace(limiter=limiter))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/publish_interaction",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
            "app": app,
        }
    )


def _service(temp_dir: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(org_id=_ORG_ID, storage_base_dir=temp_dir),
    )


def test_a_failed_publish_reports_its_phase_breakdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The phases measured before the raise are the reason the request failed.

    Without this, a storage timeout produced only `publish_request_failed` with
    an aggregate duration -- the breakdown this change adds was available for
    healthy slow requests and missing for broken ones.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        service = _service(temp_dir)
        assert service.storage is not None
        with (
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            patch.object(
                type(service.storage),
                "add_user_interactions_bulk",
                side_effect=RuntimeError("storage timed out"),
            ),
            publish_timing.collect(),
            pytest.raises(RuntimeError, match="storage timed out"),
        ):
            service.run(_publish_request(), defer_learning=True)

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    line = lines[0]
    assert "event=publish_timing" in line
    # The phases that closed before the raise, plus the failure metering event.
    for field in ("add_request_ms=", "embeddings_ms=", "metering_ms=", "total_ms="):
        assert field in line, f"{field!r} missing from {line!r}"


def test_a_successful_publish_reports_exactly_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The `finally` backstop must not double-log a request that succeeded.

    `run()` emits on the success path and again from a `finally` that catches
    every other exit. Both are reached on a successful publish, and the second
    must be suppressed by the at-most-once latch rather than by the throttle --
    which is set to 0 here for exactly that reason.
    """
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
        publish_timing.collect(),
    ):
        result = _service(temp_dir).run(_publish_request(), defer_learning=True)
        assert result.request_id is not None

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    assert "n_interactions=1" in lines[0]


def test_the_backstop_never_folds_the_coverage_wait_into_the_duration(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast publish must stay silent even if the caller then waits.

    `total_ms` is the time spent SERVING the publish. The in-process coverage
    wait that `defer_learning=False` performs afterwards is a different
    quantity -- blocking on a background worker -- and folding it in would make
    every waited request look slow.

    The at-most-once latch does not cover this on its own, which is the whole
    bug: a publish below the threshold leaves the latch unset, so the `finally`
    backstop got a second look at a clock that had meanwhile absorbed the wait.
    Here the publish is far under 300ms and the wait is 500ms over it.
    """
    monkeypatch.setenv(publish_timing.ENV_THRESHOLD_MS, "300")

    def slow_poll(*_args: object, **_kwargs: object) -> dict[str, str]:
        time.sleep(0.5)
        return {"status": "done", "reason": "complete"}

    with tempfile.TemporaryDirectory() as temp_dir:
        service = _service(temp_dir)
        assert service.storage is not None
        with (
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            patch.object(type(service.storage), "extraction_status", slow_poll),
            publish_timing.collect(),
        ):
            service.run(_publish_request(), defer_learning=False)
            elapsed_ms = publish_timing.snapshot() is not None

    assert elapsed_ms  # the scope really was open for the whole call
    assert _timing_lines(caplog) == [], (
        "a fast publish was reported as slow because the coverage wait was "
        f"counted against it: {_timing_lines(caplog)}"
    )


def test_a_publish_cancelled_while_queued_still_reports(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dominant production shape: the client gives up before admission.

    97.4% of production publishes end as an ELB 460 -- the client's own ~8s
    timeout fires before the load balancer can respond. A request cancelled
    while still queued in `acquire_ingestion` never reaches
    `GenerationService.run`, so neither the success `emit()` nor the `finally`
    backstop there can report it. Without the route's own admission-side emit
    the instrument would be silent on the majority of slow publishes.

    A request that HAS reached the worker needs no such help: the work is
    shielded and runs to completion in a thread holding its own copy of this
    context, so it emits the full duration itself. That is why the route's
    emit is scoped to the admission exits rather than to the whole handler.
    """
    from reflexio.server.routes import interactions
    from reflexio.server.services.durable_learning import waiting

    async def never_admits(org_id: str, deadline: float) -> bool:
        await asyncio.sleep(30)  # queued behind other publishes for this org
        return True

    monkeypatch.setattr(waiting, "acquire_ingestion", never_admits)

    async def scenario() -> None:
        task = asyncio.create_task(
            interactions.publish_user_interaction(
                request=_starlette_request(),
                payload=_publish_request(),
                org_id=_ORG_ID,
            )
        )
        await asyncio.sleep(0.15)
        task.cancel()  # the client closed the connection
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.WARNING, logger=publish_timing.__name__):
        asyncio.run(scenario())

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    assert "admission_ms=" in lines[0], lines[0]
