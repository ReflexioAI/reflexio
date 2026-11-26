"""Exact persisted package generation through normal selection, never a provider."""

from unittest.mock import Mock

import litellm
import pytest

from reflexio.server.services.durable_learning.learning_admission import (
    preview_learning_work,
)
from reflexio.server.services.durable_learning.profile_package_executor import (
    ProfilePackageWindowExecutor,
)
from reflexio.server.services.durable_learning.user_lease import owned_user_work_context
from tests.server.services.durable_learning.test_selected_learning_integration import (
    grant,
    invoke,
)
from tests.server.services.test_profile_package_executor import pipeline as pipeline
from tests.server.services.test_profile_package_executor import setup

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("checkpoint", ["before_inference", "gate_saved", "all_saved"])
def test_exact_persisted_package_continues_generation_in_normal_worker(
    pipeline, monkeypatch, checkpoint
):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    # Freeze the REAL recipe/parent using its normal producer before any model call.
    service, request, config = executor.service(window)
    with owned_user_work_context(storage, window.user_id, token, 600):
        executor._coordinator(window, token, service, request, config)
    assert calls == []
    if checkpoint == "gate_saved":
        from reflexio.server.services.extraction.overflow_coordinator import (
            OverflowSuspendedError,
        )
        from reflexio.server.services.extraction.resumable_agent import (
            ResumableExtractionAgent,
        )

        with monkeypatch.context() as interrupted:
            interrupted.setattr(
                ResumableExtractionAgent,
                "start_profile_package",
                Mock(side_effect=OverflowSuspendedError("synthetic next turn")),
            )
            with pytest.raises(OverflowSuspendedError):
                executor.execute(window, token)
        assert len(calls) == 1
    if checkpoint == "all_saved":
        original_save = storage.save_extraction_outcome
        monkeypatch.setattr(
            storage,
            "save_extraction_outcome",
            Mock(side_effect=RuntimeError("synthetic interrupted publication")),
        )
        with pytest.raises(RuntimeError, match="synthetic interrupted publication"):
            executor.execute(window, token)
        monkeypatch.setattr(storage, "save_extraction_outcome", original_save)
        assert len(calls) == 3
    parent_before = storage.get_agent_run(f"window:{window.window_id}")
    storage.release_user_extraction(window.user_id, token)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _: client
    )
    provider = Mock(side_effect=AssertionError("External provider forbidden"))
    monkeypatch.setattr(litellm, "completion", provider)
    selection = preview_learning_work(
        engine.request_context, window_id=window.window_id
    )
    assert selection.eligible, selection.blockers
    assert storage.get_agent_run(parent_before.id) == parent_before
    state = grant(engine.request_context, selection)
    result = invoke(engine.request_context, selection, state)
    assert result.status == "completed", result.to_dict()
    assert len(calls) == 3
    assert len(storage.get_user_profile("u")) == 1
    parent = storage.get_agent_run(parent_before.id)
    assert parent.status.value == "finalized"
    manifest = parent.generation_request_snapshot["overflow"]
    assert len(manifest["attempts"]) == 3
    assert storage.extraction_status("u", "package-parent")["status"] == "done"
    assert storage.read_learning_transition() == state
    provider.assert_not_called()


def test_changed_native_recipe_holds_before_claim_or_model(pipeline, monkeypatch):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    service, request, config = executor.service(window)
    with owned_user_work_context(storage, window.user_id, token, 600):
        executor._coordinator(window, token, service, request, config)
    storage.release_user_extraction(window.user_id, token)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _: client
    )
    selection = preview_learning_work(
        engine.request_context, window_id=window.window_id
    )
    assert selection.eligible
    state = grant(engine.request_context, selection)
    original = client._completion_request_contract_from_params
    monkeypatch.setattr(
        client,
        "_completion_request_contract_from_params",
        lambda params: {**original(params), "changed_native_instruction": True},
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert invoke(engine.request_context, selection, state).status == "held"
    changed = preview_learning_work(engine.request_context, window_id=window.window_id)
    assert not changed.eligible
    assert "original_generation_contract_incompatible" in changed.blockers
    assert storage.get_agent_run(parent.id) == parent
    assert calls == []


def test_ordinary_window_cannot_borrow_a_package_generation_grant(
    pipeline, monkeypatch
):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    storage.release_user_extraction(window.user_id, token)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _: client
    )
    selection = preview_learning_work(
        engine.request_context, window_id=window.window_id
    )
    assert not selection.eligible
    assert selection.blockers == ("original_generation_instructions_unqualified",)
    state = grant(engine.request_context, selection)
    assert invoke(engine.request_context, selection, state).status == "held"
    assert storage.get_agent_run(f"window:{window.window_id}") is None
    assert calls == []
