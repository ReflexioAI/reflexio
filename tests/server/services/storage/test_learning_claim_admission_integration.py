"""Queue preservation and scheduling-only selected claims under durable roots."""

from unittest.mock import MagicMock

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.worker import (
    DurableLearningWorker,
    _TurnRecord,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)
from reflexio.test_support.transition_closure import seed
from tests.server.services.storage.test_learning_transition_integration import (
    _deny_provider_dispatch as _deny_provider_dispatch,
)
from tests.server.services.storage.test_learning_transition_integration import (
    storage as storage,
)
from tests.server.services.storage.test_learning_transition_integration import (
    transition,
)

pytestmark = pytest.mark.integration


def _queue(storage):
    with storage.commit_scope(), storage._stream_sql() as db:
        seed(db, storage.org_id, "")


def _snapshot(storage):
    return {
        "work": tuple(
            storage.conn.execute(
                "SELECT org_id,user_id,highwater,pending_since,revision,last_turn,due_at,lease_token,lease_owner,lease_until FROM learning_work"
            ).fetchone()
        ),
        "cursors": [
            tuple(row)
            for row in storage.conn.execute(
                "SELECT user_id,kind,completed_seq,started,window_id,attempts,retry_at,last_error,last_turn FROM extraction_cursors ORDER BY kind"
            )
        ],
        "windows": storage.conn.execute(
            "SELECT count(*) FROM extraction_windows"
        ).fetchone()[0],
        "runs": storage.conn.execute("SELECT count(*) FROM _agent_runs").fetchone()[0],
        "preparations": storage.conn.execute(
            "SELECT count(*) FROM _agent_output_preparations"
        ).fetchone()[0],
    }


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("phase", ["sealed", "armed"])
def test_held_claims_preserve_nonempty_queue_and_no_model_attempt(
    storage, manual, phase
):
    _queue(storage)
    transition(
        storage, "seal", target=storage.read_learning_policy_authority().active_policy
    )
    if phase == "armed":
        transition(storage, "arm")
    before = _snapshot(storage)
    assert before["work"][2] == 2 and before["work"][7] is None
    with pytest.raises(LearningTransitionBlockedError):
        if manual:
            storage.claim_user_extraction("user", "manual", 60)
        else:
            storage.claim_extraction("automatic", 60)
    assert _snapshot(storage) == before


@pytest.mark.parametrize("manual", [False, True])
def test_actual_wrong_source_root_refuses_before_token(storage, monkeypatch, manual):
    from reflexio import profile_authoring_protocol

    _queue(storage)
    before = _snapshot(storage)
    different_policy = next(
        policy
        for policy in profile_authoring_protocol.AUTHORING_POLICIES
        if policy != profile_authoring_protocol.PROFILE_AUTHORING_POLICY
    )
    monkeypatch.setattr(
        profile_authoring_protocol, "PROFILE_AUTHORING_POLICY", different_policy
    )
    with pytest.raises(LearningTransitionBlockedError):
        if manual:
            storage.claim_user_extraction("user", "manual", 60)
        else:
            storage.claim_extraction("automatic", 60)
    assert _snapshot(storage) == before


@pytest.mark.parametrize("manual", [False, True])
def test_missing_authority_refuses_before_token(storage, manual):
    _queue(storage)
    # Explicit fixture corruption of its own database, never a product repair.
    storage.conn.execute("DELETE FROM _learning_policy_authority")
    storage.conn.commit()
    before = _snapshot(storage)
    with pytest.raises(LearningTransitionBlockedError):
        if manual:
            storage.claim_user_extraction("user", "manual", 60)
        else:
            storage.claim_extraction("automatic", 60)
    assert _snapshot(storage) == before


@pytest.mark.parametrize("acknowledged", [False, True])
def test_normal_worker_classifies_claim_hold_without_empty_success_or_reservation(
    storage, monkeypatch, acknowledged
):
    _queue(storage)
    transition(
        storage, "seal", target=storage.read_learning_policy_authority().active_policy
    )
    before = _snapshot(storage)
    context = RequestContext.__new__(RequestContext)
    context.storage, context.org_id = storage, storage.org_id
    worker = DurableLearningWorker(lambda _org: context)
    # Unit seam: independent delivery reports real progress separately from a
    # held claimant. No fake receipt is installed in the actual fixture store.
    worker._financial_delivery.deliver_one = MagicMock(return_value=acknowledged)
    prepare = MagicMock(
        side_effect=AssertionError("Held work cannot reserve generation")
    )
    monkeypatch.setattr(storage, "prepare_extraction", prepare)
    success = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.worker.report_background_success",
        success,
    )
    record = _TurnRecord()
    assert worker._turn(storage.org_id, 60, record) == int(acknowledged)
    assert (
        record.outcome == "transition_held"
        and not record.completed
        and record.user_id is None
    )
    prepare.assert_not_called()
    success.assert_not_called()
    assert _snapshot(storage) == before


def test_live_renewal_release_and_intake_remain_available_after_seal(storage):
    _queue(storage)
    token = storage.claim_user_extraction("user", "already-owned", 60)
    assert token is not None
    transition(
        storage, "seal", target=storage.read_learning_policy_authority().active_policy
    )
    assert storage.renew_extraction("user", token, 60)
    storage.conn.execute("UPDATE learning_work SET highwater=3 WHERE user_id='user'")
    storage.conn.commit()
    storage.release_user_extraction("user", token)
    assert tuple(
        storage.conn.execute(
            "SELECT highwater,lease_token,lease_until FROM learning_work"
        ).fetchone()
    ) == (3, None, 0)
