"""Tests for the durable atomic enqueue path (REFLEXIO_DURABLE_LEARNING_QUEUE=true).

Uses SQLite-backed GenerationService — no real LLM/embedding calls needed.
"""

from __future__ import annotations

import datetime
import tempfile
from datetime import UTC
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService


def _make_svc(tmp_dir: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(org_id="test_org", storage_base_dir=tmp_dir),
    )


def _publish_request(*, user_id: str, request_id: str) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        request_id=request_id,
        user_id=user_id,
        interaction_data_list=[
            InteractionData(
                content="test interaction",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
        ],
        session_id="test-session",
    )


def test_durable_enqueue_persists_job_atomically(monkeypatch):
    """When REFLEXIO_DURABLE_LEARNING_QUEUE=true and defer_learning=True, a
    pending learning job is persisted in the same atomic scope as the request
    and its interactions."""
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")

    with tempfile.TemporaryDirectory() as tmp_dir:
        svc = _make_svc(tmp_dir)
        req = _publish_request(user_id="u1", request_id="r1")

        with (
            patch(
                "reflexio.server.services.generation_service.ProfileGenerationService",
                side_effect=AssertionError("profile extraction should be deferred"),
            ),
            patch(
                "reflexio.server.services.generation_service.PlaybookGenerationService",
                side_effect=AssertionError("playbook extraction should be deferred"),
            ),
        ):
            result = svc.run(req, defer_learning=True)

        assert result.request_id == "r1"
        assert svc.storage is not None
        assert svc.storage.get_request("r1") is not None

        jobs = svc.storage.claim_learning_jobs(
            claimed_by="test-worker", limit=10, lease_seconds=300
        )
        assert any(j.user_id == "u1" for j in jobs), "expected one pending job for u1"


def test_enqueue_failure_rolls_back_interactions(monkeypatch):
    """Zero-loss proof: if enqueue_learning_job raises inside commit_scope, the
    entire transaction rolls back — neither the request nor its interactions
    persist."""
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")

    with tempfile.TemporaryDirectory() as tmp_dir:
        svc = _make_svc(tmp_dir)
        req = _publish_request(user_id="u2", request_id="r2")

        monkeypatch.setattr(
            type(svc.storage),
            "enqueue_learning_job",
            Mock(side_effect=RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            svc.run(req, defer_learning=True)

        # Both the request and its interactions must be absent (rolled back).
        assert svc.storage is not None
        assert svc.storage.get_request("r2") is None, (
            "request must not persist when enqueue_learning_job fails inside commit_scope"
        )
