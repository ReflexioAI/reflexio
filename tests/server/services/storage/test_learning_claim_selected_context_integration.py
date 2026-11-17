"""Only current program-owned selected context can schedule under a seal."""

import pytest

from reflexio.server.services.durable_learning.learning_admission import (
    preview_learning_work,
    run_selected_learning_work,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    _change_learning_policy_authority_db,
)
from tests.server.services.durable_learning.test_selected_learning_integration import (
    grant,
)
from tests.server.services.extraction.test_selected_recovery_integration import (
    context as context,
)
from tests.server.services.extraction.test_selected_recovery_integration import seed

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mismatch", ["nonce", "generation", "armed"])
def test_stale_or_armed_selected_context_never_claims_or_reserves(context, mismatch):
    storage = context.storage
    original = seed(context)
    selection = preview_learning_work(context, run_id=original.id)
    state = grant(context, selection)
    expected_nonce = state.authority.authority_nonce
    expected_generation = state.authority.generation
    if mismatch == "nonce":
        expected_nonce = "b" * 32 if expected_nonce != "b" * 32 else "c" * 32
    elif mismatch == "generation":
        expected_generation += 1
    else:
        storage.grant_learning_transition(
            "release",
            expected_authority_nonce=expected_nonce,
            expected_authority_generation=expected_generation,
            expected_nonce=state.project.nonce,
            expected_revision=state.project.revision,
            selection_sha256=None,
        )
        # Deliberately place this fixture into ARMED to falsify scheduling;
        # this negative test does not qualify its unfinished original run.
        with storage._learning_policy_authority_scope() as (db, current):
            _change_learning_policy_authority_db(
                db,
                action="arm",
                operation_id="release",
                expected_nonce=current.authority_nonce,
                expected_generation=current.generation,
            )
    before = tuple(
        storage.conn.execute(
            "SELECT highwater,revision,last_turn,due_at,lease_token,lease_until FROM learning_work WHERE user_id='user'"
        ).fetchone()
    )
    result = run_selected_learning_work(
        context,
        selection,
        operation_id="release",
        expected_nonce=state.project.nonce,
        expected_revision=state.project.revision,
        expected_authority_nonce=expected_nonce,
        expected_authority_generation=expected_generation,
    )
    assert result.status == "held"
    assert (
        tuple(
            storage.conn.execute(
                "SELECT highwater,revision,last_turn,due_at,lease_token,lease_until FROM learning_work WHERE user_id='user'"
            ).fetchone()
        )
        == before
    )
    assert storage.get_agent_run(original.id) == original


def test_ordinary_resume_class_reports_hold_without_token_or_failure(
    context, monkeypatch
):
    from unittest.mock import MagicMock

    from reflexio.server.services.durable_learning.worker import (
        DurableLearningWorker,
        _TurnRecord,
    )

    storage = context.storage
    original = seed(context)
    state = grant(context, preview_learning_work(context, run_id=original.id))
    worker = DurableLearningWorker(lambda _org: context)
    worker._financial_delivery.deliver_one = MagicMock(return_value=False)
    failures = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.resume_selection.report_failure_at_interval",
        failures,
    )
    # One normal learning opportunity, then its existing finalization class.
    for _ in range(2):
        record = _TurnRecord()
        assert worker._turn(storage.org_id, 60, record) == 0
        assert record.outcome == "transition_held" and not record.completed
    assert storage.get_agent_run(original.id) == original
    assert storage.read_learning_transition() == state
    assert (
        storage.conn.execute(
            "SELECT lease_token FROM learning_work WHERE user_id='user'"
        ).fetchone()[0]
        is None
    )
    failures.assert_not_called()
