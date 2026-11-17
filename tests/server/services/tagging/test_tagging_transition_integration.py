"""Actual SQLite user/root ownership for direct and scheduled profile tagging."""

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)
from reflexio.server.services.tagging.tagging_scheduler import TaggingScheduler
from reflexio.test_support.tagging_transition import exercise_open, exercise_sealed
from tests.server.services.extraction.test_selected_recovery_integration import (
    context as context,
)


def test_open_direct_tagging_has_actual_user_root_ownership(context, monkeypatch):
    exercise_open(context, monkeypatch)


@pytest.mark.parametrize("seal_after_response", [False, True])
def test_seal_refuses_tag_compute_or_returned_response_write(
    context, monkeypatch, seal_after_response
):
    exercise_sealed(context, monkeypatch, seal_after_response=seal_after_response)


def test_scheduler_retains_typed_transition_hold(monkeypatch):
    scheduled = []
    scheduler = MagicMock()
    scheduler.schedule.side_effect = lambda *args, **kwargs: scheduled.append(
        (args, kwargs)
    )
    monkeypatch.setattr(
        TaggingScheduler, "get_instance", classmethod(lambda _cls: scheduler)
    )

    def blocked():
        raise LearningTransitionBlockedError("sealed")

    key = ("org", None, "user", "v1")
    TaggingScheduler._run_callback(key, blocked)
    assert scheduled == [((key, blocked), {"delay_seconds": 15})]
