"""A participant that lost its own fence refuses admission without an ERROR.

Real SQLite storage, mocked LLM. The participant raises
``ExternalAdmissionFencedError`` inside the admission transaction; nothing may
land, the error reaches the caller unchanged, and the publish is logged as a
WARNING rather than as a profile-refresh fault.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.external_admission import (
    ExternalAdmissionFencedError,
)
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_GENERATION_LOGGER = "reflexio.server.services.generation_service"


class _LeaseLostError(ExternalAdmissionFencedError):
    """A participant-owned subclass, as a real participant would raise."""


@pytest.fixture(autouse=True)
def _no_embeddings_or_scheduler(monkeypatch):
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )


def _service(tmp_path, worker_id: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(
            org_id=f"fenced_{worker_id}", storage_base_dir=str(tmp_path)
        ),
    )


def _request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        request_id="fenced-req",
        user_id="fenced-user",
        session_id="s1",
        interaction_data_list=[
            InteractionData(role="User", content="hello", created_at=1_700_000_000),
            InteractionData(role="Agent", content="hi", created_at=1_700_000_000),
        ],
    )


def _participant(stage: str) -> Mock:
    """Fence lost in the admission transaction's claim, or in complete."""
    participant = Mock()
    secret = "lease token tok-SECRET changed"
    if stage == "claim":
        participant.claim.side_effect = _LeaseLostError(secret)
    else:
        participant.claim.return_value = None
        participant.complete.side_effect = _LeaseLostError(secret)
    return participant


@pytest.mark.parametrize("stage", ["claim", "complete"])
def test_lost_fence_is_a_warning_and_admits_nothing(tmp_path, worker_id, caplog, stage):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    caplog.set_level(logging.DEBUG, logger=_GENERATION_LOGGER)

    with pytest.raises(_LeaseLostError):
        svc.run(
            _request(),
            defer_learning=True,
            admission_participant=_participant(stage),
        )

    assert store.get_request("fenced-req") is None
    assert store.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] == 0
    records = [r for r in caplog.records if r.name == _GENERATION_LOGGER]
    fenced = [r for r in records if "event=publish_admission_fenced" in r.getMessage()]
    assert len(fenced) == 1, caplog.text
    assert fenced[0].levelno == logging.WARNING
    assert fenced[0].exc_info is None
    assert "error_type=_LeaseLostError" in fenced[0].getMessage()
    assert "fenced-user" in fenced[0].getMessage()
    assert not [r for r in records if r.levelno >= logging.ERROR], caplog.text
    assert "tok-SECRET" not in caplog.text
