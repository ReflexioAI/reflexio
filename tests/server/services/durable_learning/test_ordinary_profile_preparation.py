"""Ordinary future roots retain their original native instruction authority."""

import copy
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning import (
    ordinary_profile_preparation as preparation,
)
from reflexio.server.services.durable_learning.selected_learning import (
    preview_learning_work,
)
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.test_support.owned_window_recovery import (
    SimulatedProcessDeath,
    install_crashing_model,
    recovery_publish_request,
)
from tests.server.services import test_durable_window_pipeline as durable
from tests.server.services.durable_learning.test_selected_learning_integration import (
    grant,
    invoke,
)

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def interrupted(pipeline, monkeypatch, *, force=True):
    engine, worker, client = pipeline
    calls = install_crashing_model(monkeypatch)
    payload = recovery_publish_request()
    payload["force_extraction"] = force
    assert client.post("/api/publish_interaction", json=payload).json()["success"]
    with pytest.raises(SimulatedProcessDeath):
        worker.drain_org(engine.request_context.org_id, 1, 300)
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        rows = db.query("SELECT id FROM _agent_runs WHERE user_id=?", ("u",))
    assert len(rows) == 1
    parent = storage.get_agent_run(rows[0]["id"])
    assert parent.generation_request_snapshot["overflow"]["version"] == 1
    assert parent.generation_request_snapshot[preparation.KEY]["version"] == 1
    assert calls == [1]
    return engine.request_context, parent, calls


def test_before_first_gate_preview_is_provider_free_and_does_not_adopt(
    pipeline, monkeypatch
):
    context, parent, calls = interrupted(pipeline, monkeypatch, force=False)
    before = copy.deepcopy(parent)
    forbidden = Mock(side_effect=AssertionError("Preview must not adopt or dispatch"))
    monkeypatch.setattr(WindowExecutor, "_coordinator", forbidden)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", forbidden)
    selection = preview_learning_work(
        context, window_id=parent.id.removeprefix("window:")
    )
    assert selection.eligible, selection.blockers
    assert context.storage.get_agent_run(parent.id) == before
    assert calls == [1]
    forbidden.assert_not_called()


def test_selected_ordinary_continuation_uses_normal_executor(pipeline, monkeypatch):
    context, parent, calls = interrupted(pipeline, monkeypatch)
    selection = preview_learning_work(
        context, window_id=parent.id.removeprefix("window:")
    )
    assert selection.eligible, selection.blockers
    state = grant(context, selection)
    from reflexio.server.services.durable_learning.profile_package_executor import (
        ProfilePackageWindowExecutor,
    )

    monkeypatch.setattr(
        ProfilePackageWindowExecutor,
        "execute",
        Mock(side_effect=AssertionError("Ordinary work must not switch to v2")),
    )
    result = invoke(context, selection, state)
    assert result.status == "completed", result.to_dict()
    assert len(calls) > 1
    assert context.storage.get_user_profile("u")


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "version",
        "source",
        "templates",
        "schemas",
        "requests",
        "instructions",
    ],
)
def test_unknown_or_changed_original_contract_is_held_without_provider(
    pipeline, monkeypatch, change
):
    context, parent, calls = interrupted(pipeline, monkeypatch)
    run = context.storage.get_agent_run(parent.id)
    if change == "missing":
        run.generation_request_snapshot.pop(preparation.KEY)
    elif change == "version":
        run.generation_request_snapshot[preparation.KEY]["version"] = True
    else:
        run.generation_request_snapshot[preparation.KEY][change + "_sha256"] = "0" * 64
    # Test-owned historical corruption; preserve the actual normal root identity.
    with context.storage._stream_sql() as db:
        import json

        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(run.generation_request_snapshot), run.id),
        )
    forbidden = Mock(side_effect=AssertionError("Held preview must not dispatch"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", forbidden)
    selection = preview_learning_work(
        context, window_id=parent.id.removeprefix("window:")
    )
    assert not selection.eligible
    forbidden.assert_not_called()
    assert calls == [1]


def test_source_cache_is_only_for_immutable_source_not_live_resources(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("first\n")
    module = source / "server/services/durable_learning/module.py"
    module.parent.mkdir(parents=True)
    module.write_text("source\n")
    monkeypatch.setattr(preparation, "__file__", str(module))
    preparation._installed_source_digest.cache_clear()
    try:
        first = preparation._installed_source_digest()
        (source / "__init__.py").write_text("changed\n")
        assert preparation._installed_source_digest() == first
        preparation._installed_source_digest.cache_clear()
        assert preparation._installed_source_digest() != first
        bank = tmp_path / "templates"
        bank.mkdir()
        resource = bank / "template.txt"
        resource.write_text("one")
        before = preparation._files_digest(bank, python_only=False)
        resource.write_text("two")
        assert preparation._files_digest(bank, python_only=False) != before
    finally:
        preparation._installed_source_digest.cache_clear()


@pytest.mark.parametrize("kind", ["missing", "empty", "symlink", "fifo"])
def test_unknown_resource_layout_cannot_prove_consumer(tmp_path, kind):
    root = tmp_path / "resources"
    if kind != "missing":
        root.mkdir()
    if kind == "symlink":
        actual = tmp_path / "actual.txt"
        actual.write_text("native")
        (root / "linked.txt").symlink_to(actual)
    if kind == "fifo":
        import os

        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO requires POSIX")
        os.mkfifo(root / "stream")
    with pytest.raises(InputBudgetConfigurationError):
        preparation._files_digest(root, python_only=False)


def test_dynamic_incumbents_are_not_frozen_into_instruction_witness(
    pipeline, monkeypatch
):
    context, parent, _ = interrupted(pipeline, monkeypatch)
    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.services.durable_learning.selected_learning import (
        _existing_window_rows,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import Window

    row, _ = _existing_window_rows(context, parent.id.removeprefix("window:"))
    window = Window.from_row(row)
    executor = WindowExecutor(context, create_generation_litellm_client(context))
    service, _, config = executor.service(window)
    before = preparation.validate_ordinary_profile_preparation(
        executor, window, service, config, parent
    )
    service.service_config.existing_data = ["new native lookup state"]
    assert (
        preparation.validate_ordinary_profile_preparation(
            executor, window, service, config, parent
        )
        == before
    )
    from dataclasses import replace

    with pytest.raises(InputBudgetConfigurationError):
        preparation.validate_ordinary_profile_preparation(
            executor, replace(window, force=not window.force), service, config, parent
        )


def test_midturn_instruction_drift_holds_and_preserves_actual_usage(
    pipeline, monkeypatch
):
    context, parent, calls = interrupted(pipeline, monkeypatch)
    selection = preview_learning_work(
        context, window_id=parent.id.removeprefix("window:")
    )
    state = grant(context, selection)
    original: Any = LiteLLMClient._completion_with_hard_timeout
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    failure_owner = DurableLearningWorker._handle_window_failure
    outcomes = []

    def observe_failure(*args):
        outcome = failure_owner(*args)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(
        DurableLearningWorker, "_handle_window_failure", staticmethod(observe_failure)
    )

    def respond_then_change(params, hard_timeout):
        response = original(params, hard_timeout)
        settings = context.configurator.get_config().model_copy(deep=True)
        settings.agent_context_prompt = "Changed while provider returned"
        context.configurator.set_config(settings)
        return response

    monkeypatch.setattr(
        LiteLLMClient,
        "_completion_with_hard_timeout",
        staticmethod(respond_then_change),
    )
    result = invoke(context, selection, state)
    assert result.status == "held"
    assert outcomes == ["transition_held"]
    # Exactly one successful physical response followed the original crash;
    # no further changed-instruction call or business publication was permitted.
    assert len(calls) == 2
    assert not context.storage.get_user_profile("u")
    after = context.storage.get_agent_run(parent.id)
    assert after.status.value != "failed"
    attempts = after.generation_request_snapshot["overflow"]["attempts"]
    assert attempts[0] == parent.generation_request_snapshot["overflow"]["attempts"][0]
    assert any(attempt["usage"] is not None for attempt in attempts[1:])


def test_actual_storage_expansion_settings_are_part_of_original_contract(
    pipeline, monkeypatch
):
    pipeline[0].get_storage()._enable_document_expansion = True
    context, parent, calls = interrupted(pipeline, monkeypatch)
    before = preview_learning_work(context, window_id=parent.id.removeprefix("window:"))
    assert before.eligible
    context.storage._expansion_model_name = "gpt-4o"
    after = preview_learning_work(context, window_id=parent.id.removeprefix("window:"))
    assert not after.eligible
    assert calls == [1]


@pytest.mark.parametrize("operation", ["embedding", "expansion"])
def test_common_guard_blocks_actual_storage_dispatch_after_instruction_drift(
    pipeline, monkeypatch, operation
):
    from types import SimpleNamespace
    from typing import cast

    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.llm import _litellm_embedding as embeddings
    from reflexio.server.llm.compute_admission import ComputeAdmissionError
    from reflexio.server.services.durable_learning import selected_learning
    from reflexio.server.services.pre_retrieval import DocumentExpander
    from reflexio.server.services.storage.storage_base._extraction_stream import Window

    storage = pipeline[0].get_storage()
    storage._enable_document_expansion = True
    storage.embedding_model_name = "openai/text-embedding-3-small"
    context, parent, _ = interrupted(pipeline, monkeypatch)
    row, _ = selected_learning._existing_window_rows(
        context, parent.id.removeprefix("window:")
    )
    window = Window.from_row(row)
    executor = WindowExecutor(context, create_generation_litellm_client(context))
    service, _, config = executor.service(window)
    monkeypatch.setattr(
        selected_learning,
        "current_selected_learning",
        lambda: SimpleNamespace(effects_only=False),
    )
    physical = Mock(
        side_effect=AssertionError(
            "Instruction drift must refuse before storage transport"
        )
    )
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", physical)
    monkeypatch.setattr(embeddings.litellm, "embedding", physical)
    monkeypatch.setattr(embeddings, "embedding_provider_mode", lambda _: "required")
    monkeypatch.setattr(
        embeddings, "resolve_service_configured_model", lambda model: model
    )
    monkeypatch.setattr(embeddings, "should_use_embedding_service", lambda _: False)
    monkeypatch.setattr(embeddings, "_truncate_for_embedding", lambda text, _: text)
    monkeypatch.setattr(DocumentExpander, "_provider_down", lambda _: False)
    with preparation.original_ordinary_instruction_scope(
        executor,
        cast(Any, SimpleNamespace(saved_run=parent, window=window)),
        service,
        config,
    ):
        settings = context.configurator.get_config().model_copy(deep=True)
        settings.agent_context_prompt = "New instructions after selected admission"
        context.configurator.set_config(settings)
        if operation == "embedding":
            with pytest.raises(ComputeAdmissionError):
                embeddings.EmbeddingMixin._embed_texts(
                    storage.llm_client,
                    ["test"],
                    storage.embedding_model_name,
                    3,
                    batch=False,
                )
        else:
            result = DocumentExpander(
                storage.llm_client,
                context.prompt_manager,
                model_name=storage._expansion_model_name,
            ).expand("test")
            assert result.expanded_text == ""
    physical.assert_not_called()


def _distribution(name, version):
    from email.message import Message
    from types import SimpleNamespace

    metadata = Message()
    if name is not None:
        metadata["Name"] = name
    if version is not None:
        metadata["Version"] = version
    return SimpleNamespace(metadata=metadata)


@pytest.fixture
def runtime_metadata(monkeypatch):
    records = [_distribution("Pydantic", "2.12.0"), _distribution("LiteLLM", "1.90.0")]
    monkeypatch.setattr(preparation, "distributions", lambda: iter(records))
    preparation._installed_runtime_digest.cache_clear()
    try:
        yield records
    finally:
        preparation._installed_runtime_digest.cache_clear()


def test_runtime_versions_are_canonical_order_independent_and_process_cached(
    runtime_metadata,
):
    first = preparation._installed_runtime_digest()
    runtime_metadata.reverse()
    preparation._installed_runtime_digest.cache_clear()
    assert preparation._installed_runtime_digest() == first
    runtime_metadata[0].metadata.replace_header("Version", "1.91.0")
    assert preparation._installed_runtime_digest() == first
    preparation._installed_runtime_digest.cache_clear()
    assert preparation._installed_runtime_digest() != first


@pytest.mark.parametrize(
    "bad",
    [
        "missing-name",
        "missing-version",
        "duplicate-header",
        "duplicate-name",
        "invalid-version",
        "empty",
    ],
)
def test_ambiguous_or_missing_installed_metadata_cannot_prove_runtime(
    runtime_metadata, bad
):
    if bad == "missing-name":
        runtime_metadata.append(_distribution(None, "1.0"))
    elif bad == "missing-version":
        runtime_metadata.append(_distribution("parser", None))
    elif bad == "duplicate-header":
        runtime_metadata[0].metadata["Name"] = "Other"
    elif bad == "invalid-version":
        runtime_metadata.append(_distribution("parser", "/not/a/version"))
    elif bad == "duplicate-name":
        runtime_metadata.extend(
            [_distribution("my_parser", "1"), _distribution("MY-parser", "1")]
        )
    else:
        runtime_metadata.clear()
    with pytest.raises(InputBudgetConfigurationError):
        preparation._installed_runtime_digest()


def test_python_identity_change_moves_runtime_fingerprint(
    runtime_metadata, monkeypatch
):
    from types import SimpleNamespace

    first = preparation._installed_runtime_digest()
    actual = preparation.sys
    monkeypatch.setattr(
        preparation,
        "sys",
        SimpleNamespace(
            implementation=actual.implementation,
            version_info=(99, 0, 0, "final", 0),
            platform=actual.platform,
            byteorder=actual.byteorder,
            abiflags=getattr(actual, "abiflags", ""),
        ),
    )
    preparation._installed_runtime_digest.cache_clear()
    assert preparation._installed_runtime_digest() != first


def test_changed_runtime_versions_hold_original_window_without_dispatch(
    pipeline, monkeypatch
):
    context, parent, calls = interrupted(pipeline, monkeypatch)
    monkeypatch.setattr(
        preparation,
        "distributions",
        lambda: iter([_distribution("DifferentRuntime", "1")]),
    )
    preparation._installed_runtime_digest.cache_clear()
    try:
        selection = preview_learning_work(
            context, window_id=parent.id.removeprefix("window:")
        )
        assert not selection.eligible
        assert calls == [1]
    finally:
        preparation._installed_runtime_digest.cache_clear()
