"""A crash cannot refund a reconciliation request reserved by the parent CAS."""

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


def coordinator(storage, window, token):
    parent = storage.get_agent_run(f"window:{window.window_id}")
    return OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )


def reserve(owner):
    with owner.reconciliation_request(OPERATION, 2):
        owner._guard(PARAMS, 1.0, ("gpt-4o",), False)


def test_crash_after_parent_commit_preserves_both_allowances(
    journal_pipeline, monkeypatch
):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    first = coordinator(storage, window, token)
    first.begin_reconciliation(OPERATION, 2)
    checkpoint = storage.checkpoint_extraction_overflow

    def committed_then_crashed(*args, **kwargs):
        checkpoint(*args, **kwargs)
        raise RuntimeError("crash after parent reservation committed")

    monkeypatch.setattr(
        storage, "checkpoint_extraction_overflow", committed_then_crashed
    )
    with pytest.raises(ProviderRequestGuardError, match="reservation failed"):
        reserve(first)
    monkeypatch.setattr(storage, "checkpoint_extraction_overflow", checkpoint)
    resumed = coordinator(storage, window, token)
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
    final = coordinator(storage, window, token)
    assert final.checkpoint["audited_consolidation_budget"]["calls"] == 2
    assert len(final._manifest["attempts"]) == 2


def test_reconciliation_scope_cannot_be_erased_or_refunded(journal_pipeline):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    owner = coordinator(storage, window, token)
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
    assert coordinator(storage, window, token).checkpoint == before


def test_existing_operation_cannot_change_its_scope(journal_pipeline):
    storage, window, token = _overflow_coordinator(journal_pipeline)
    owner = coordinator(storage, window, token)
    owner.begin_reconciliation(OPERATION, 2)
    with pytest.raises(ProviderRequestGuardError, match="contract changed"):
        owner.begin_reconciliation("d" * 64, 2)
    assert owner.checkpoint["audited_consolidation_budget"]["calls"] == 0
