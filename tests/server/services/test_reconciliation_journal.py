"""A crash cannot refund a reconciliation request reserved by the parent CAS."""

from contextlib import contextmanager
from copy import deepcopy

import pytest

from reflexio.server.llm._litellm_types import ProviderRequestGuardError
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
)
from tests.server.services.test_durable_window_pipeline import _overflow_coordinator
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
journal_pipeline = imported_pipeline
OPERATION = "c" * 64
PARAMS = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "fixture"}],
    "max_tokens": 64,
}


@pytest.fixture(autouse=True)
def scripted_pricing(monkeypatch):
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )


@contextmanager
def coordinator(storage, window, token):
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        owned_user_work_context,
    )

    with owned_user_work_context(storage, window.user_id, token, 300):
        parent = storage.get_agent_run(f"window:{window.window_id}")
        assert parent is not None
        adopted = adopt_agent_run_for_execution(storage, parent)
        current = OverflowCoordinator(
            storage,
            window,
            token,
            adopted.generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={},
        )
        current.saved_run = adopted
        yield current


def reserve(owner):
    with owner.reconciliation_request(OPERATION, 2):
        owner._guard(PARAMS, 1.0, ("gpt-4o",), False)


def test_crash_after_parent_commit_preserves_both_allowances(
    journal_pipeline, monkeypatch
):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    with coordinator(storage, window, token) as first:
        first.begin_reconciliation(OPERATION, 2)
        checkpoint = OverflowCoordinator._advance

        def committed_then_crashed(current, **kwargs):
            checkpoint(current, **kwargs)
            raise RuntimeError("crash after parent reservation committed")

        monkeypatch.setattr(OverflowCoordinator, "_advance", committed_then_crashed)
        with pytest.raises(ProviderRequestGuardError, match="reservation failed"):
            reserve(first)
        monkeypatch.setattr(OverflowCoordinator, "_advance", checkpoint)
    with coordinator(storage, window, token) as resumed:
        context = resumed.checkpoint["reconciliation_context"]
        assert (
            len(context["attempt_ids"])
            == resumed.checkpoint["audited_consolidation_budget"]["calls"]
            == 1
        )
        assert resumed._manifest["attempts"][-1]["usage"] is None
        reserve(resumed)
        with pytest.raises(ProviderRequestGuardError, match="reservation failed"):
            reserve(resumed)
    with coordinator(storage, window, token) as final:
        assert final.checkpoint["audited_consolidation_budget"]["calls"] == 2
        assert len(final._manifest["attempts"]) == 2


def test_reconciliation_scope_cannot_be_erased_or_refunded(journal_pipeline):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    with coordinator(storage, window, token) as owner:
        owner.begin_reconciliation(OPERATION, 2)
        reserve(owner)
        before = owner.checkpoint
        for change in ("erase", "refund", "change_limit"):
            altered = deepcopy(before)
            if change == "erase":
                del altered["reconciliation_context"]
            elif change == "refund":
                altered["reconciliation_context"]["attempt_ids"] = []
                altered["audited_consolidation_budget"]["calls"] = 0
            else:
                altered["reconciliation_context"]["limit"] = 3
            with pytest.raises(OverflowCheckpointConflictError, match="immutable"):
                owner.save_checkpoint(altered)
        with coordinator(storage, window, token) as replay:
            assert replay.checkpoint == before


def test_existing_operation_cannot_change_its_scope(journal_pipeline):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    with coordinator(storage, window, token) as owner:
        owner.begin_reconciliation(OPERATION, 2)
        with pytest.raises(ProviderRequestGuardError, match="contract changed"):
            owner.begin_reconciliation("d" * 64, 2)
        assert owner.checkpoint["audited_consolidation_budget"]["calls"] == 0
