"""Tests for `_schedule_group_evaluation_if_needed` in GenerationService.

Guards against regression of the bug where the agentic extraction backend
silently bypassed the scheduler call, leaving /evaluations permanently empty.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.generation_service import GenerationService


@pytest.fixture
def service() -> GenerationService:
    """Build a bare GenerationService with the attributes the helper reads.

    Bypasses __init__ entirely because the helper only reads three instance
    attributes: org_id, request_context, and client. The full constructor would
    require a configurator + storage + LLM client that aren't needed here.
    """
    svc = GenerationService.__new__(GenerationService)
    svc.org_id = "org_test"
    svc.request_context = MagicMock(name="request_context")
    svc.client = MagicMock(name="llm_client")
    return svc


def test_no_schedule_when_session_id_is_none(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When new_request.session_id is None, the helper short-circuits."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    request_with_no_session = MagicMock(session_id=None)

    service._schedule_group_evaluation_if_needed(
        new_request=request_with_no_session,
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )

    scheduler.schedule.assert_not_called()


def test_schedules_with_correct_key_when_session_id_present(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper schedules with key=(org_id, user_id, session_id)."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    request_with_session = MagicMock(session_id="sess_42")

    service._schedule_group_evaluation_if_needed(
        new_request=request_with_session,
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    scheduler.schedule.assert_called_once()
    call_args = scheduler.schedule.call_args
    key = call_args[0][0] if call_args[0] else call_args.kwargs.get("key")
    callback = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("callback")
    assert key == ("org_test", "user_test", "sess_42")
    assert callable(callback)
