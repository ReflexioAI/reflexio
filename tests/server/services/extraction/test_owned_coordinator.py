"""A live user token cannot let a superseded logical parent mutate its journal."""

import copy

import pytest

from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    owned_user_work_context,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.storage.storage_base import LeaseLostError
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "operation", ["checkpoint", "reservation", "observation", "cache_validation"]
)
def test_superseded_parent_cannot_mutate_or_validate_execution(pipeline, operation):
    storage, window, token = durable._overflow_coordinator(pipeline)
    with owned_user_work_context(storage, window.user_id, token, 300):
        parent = storage.get_agent_run(f"window:{window.window_id}")
        assert parent is not None
        adopted = adopt_agent_run_for_execution(storage, parent)
        manifest = adopted.generation_request_snapshot["overflow"]
        coordinator = OverflowCoordinator(
            storage,
            window,
            token,
            manifest,
            source_digest=manifest["source_digest"],
            context_limits={"gpt-4o-mini": 128000},
        )
        coordinator.saved_run = adopted
        if operation == "observation":
            coordinator._advance(reserve_attempt_id="paid-before-replacement")
            coordinator._active_attempt.set("paid-before-replacement")
        before = copy.deepcopy(
            storage.get_agent_run(parent.id).generation_request_snapshot
        )
        storage._execute(
            "UPDATE _agent_runs SET claimed_by=? WHERE id=?",
            ("superseding-logical-owner", parent.id),
        )
        storage.fence_user_extraction(window.user_id, token)
        expected_error = (
            ProviderRequestGuardError if operation == "reservation" else LeaseLostError
        )
        with pytest.raises(expected_error) as rejected:
            if operation == "checkpoint":
                coordinator.save_checkpoint(
                    {**coordinator.checkpoint, "stage": "extracted"}
                )
            elif operation == "reservation":
                coordinator._guard(
                    {
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "synthetic input"}],
                        "max_tokens": 64,
                    },
                    1,
                    ("gpt-4o-mini",),
                    False,
                )
            elif operation == "observation":
                coordinator._observe(
                    ModelProvenance("fixture", "provider"),
                    RunTokenTotals(prompt_tokens=13, completion_tokens=4),
                )
            else:
                coordinator.validate_current_parent()
        if operation == "reservation":
            assert isinstance(rejected.value.__cause__, LeaseLostError)
        after = storage.get_agent_run(parent.id)
        assert after.generation_request_snapshot == before
        assert after.claimed_by == "superseding-logical-owner"
        if operation == "observation":
            attempts = after.generation_request_snapshot["overflow"]["attempts"]
            assert len(attempts) == 1
            assert attempts[0]["id"] == "paid-before-replacement"
            assert attempts[0]["usage"] is None


def test_another_valid_run_cannot_authorize_this_window_checkpoint(pipeline):
    from dataclasses import replace

    from reflexio.server.services.durable_learning.user_lease import (
        create_owned_agent_run,
    )

    storage, window, token = durable._overflow_coordinator(pipeline)
    with owned_user_work_context(storage, window.user_id, token, 300):
        original = storage.get_agent_run(f"window:{window.window_id}")
        assert original is not None
        parent = adopt_agent_run_for_execution(storage, original)
        other = create_owned_agent_run(
            storage,
            replace(parent, id="another_owned_run", claimed_by=None, claimed_at=None),
        )
        manifest = parent.generation_request_snapshot["overflow"]
        current = OverflowCoordinator(
            storage,
            window,
            token,
            manifest,
            source_digest=manifest["source_digest"],
            context_limits={},
        )
        # Both run and owner are real and valid, but this is the WRONG run.
        current.saved_run = other
        storage._execute(
            "UPDATE _agent_runs SET claimed_by='replacement' WHERE id=?", (parent.id,)
        )
        with pytest.raises(LeaseLostError):
            current.save_checkpoint({**current.checkpoint, "stage": "extracted"})
        assert (
            storage.get_agent_run(parent.id).generation_request_snapshot
            == parent.generation_request_snapshot
        )
        assert storage.get_agent_run(other.id) == other
