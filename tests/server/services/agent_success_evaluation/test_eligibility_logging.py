"""Verify each skip path in run_group_evaluation bumps the matching counter."""

from unittest.mock import MagicMock

from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)


def _make_mock_context(*, get_op_state=None, get_requests=None, get_interactions=None):
    ctx = MagicMock()
    ctx.storage.get_operation_state.return_value = get_op_state
    ctx.storage.get_requests_by_session.return_value = get_requests or []
    ctx.storage.get_interactions_by_request_ids.return_value = get_interactions or []
    return ctx


def test_already_evaluated_path_bumps_counter() -> None:
    _eval_health._HEALTH.__init__()
    ctx = _make_mock_context(
        get_op_state={"operation_state": {"evaluated": True}},
    )

    run_group_evaluation("org", "user", "sess", "v0", None, ctx, MagicMock())

    assert _eval_health.get_status()["skip_counts"]["already_evaluated"] == 1


def test_no_requests_path_bumps_counter() -> None:
    _eval_health._HEALTH.__init__()
    ctx = _make_mock_context(get_requests=[])

    run_group_evaluation("org", "user", "sess", "v0", None, ctx, MagicMock())

    assert _eval_health.get_status()["skip_counts"]["no_requests"] == 1
