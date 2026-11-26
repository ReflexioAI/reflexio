"""Terminal deliveries and effective provider policies remain bound on retry."""

import json
from copy import deepcopy

import litellm
import pytest

from reflexio.server.llm._litellm_types import ProviderRequestGuardError
from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.playbook.procedural import native
from reflexio.server.services.playbook.procedural.models import ContractError
from reflexio.server.services.playbook.procedural.transport import (
    BATCH_TOOL,
    StageClient,
)
from tests.server.services.playbook.test_procedural_window_integration import (
    cli_setup,
    setup,
)
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline


def test_resolved_request_hashes_bind_inherited_policy_and_fallback_routes(monkeypatch):
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", top_p=0.1, fallback_models=["gpt-4o"])
    )
    first = StageClient(client, "gpt-4o-mini")
    client.config.top_p = 0.9
    second = StageClient(client, "gpt-4o-mini")
    assert first.profiles == second.profiles
    assert first.model_ladders == second.model_ladders
    assert all(
        first.request_contracts[s] != second.request_contracts[s]
        for s in first.profiles
    )
    resolve = client.get_completion_request_contract
    secret_url = "https://fixture.invalid/v1?api_key=fixture-route-secret"

    def changed_fallback(**kwargs):
        value = resolve(**kwargs)
        if kwargs.get("model") == "gpt-4o":
            value["api_base"] = secret_url
        return value

    monkeypatch.setattr(client, "get_completion_request_contract", changed_fallback)
    third = StageClient(client, "gpt-4o-mini")
    for stage in second.profiles:
        assert (
            third.request_contracts[stage]["primary"]
            == second.request_contracts[stage]["primary"]
        )
        assert (
            third.request_contracts[stage]["rungs"][0]
            == second.request_contracts[stage]["rungs"][0]
        )
        assert (
            third.request_contracts[stage]["rungs"][1]
            != second.request_contracts[stage]["rungs"][1]
        )
    assert "fixture-route-secret" not in json.dumps(third.request_contracts)
    assert "fixture.invalid" not in json.dumps(third.request_contracts)


@pytest.mark.integration
@pytest.mark.parametrize("adapter", ["native", "inline"])
def test_changed_effective_policy_holds_partial_window_before_cached_work_or_dispatch(
    pipeline, monkeypatch, tmp_path, adapter
):
    if adapter == "inline":
        engine, storage, window, token, client, calls, _, wire = cli_setup(
            pipeline, monkeypatch, tmp_path
        )
    else:
        engine, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
        wire = client._completion_with_hard_timeout
    library_before = storage.get_user_playbooks(user_id="u", agent_version="v")
    client.config.top_p = 0.1
    resolve = client.get_completion_request_contract
    secret_url = "https://fixture.invalid/v1?api_key=fixture-route-secret"

    def private_descriptor(**kwargs):
        value = resolve(**kwargs)
        value["api_base"] = secret_url
        return value

    monkeypatch.setattr(client, "get_completion_request_contract", private_descriptor)

    def outage(params, timeout):
        if "source_observations" in json.loads(params["messages"][1]["content"]):
            calls.append(params)
            raise ProviderRequestGuardError("synthetic interruption after reader")
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", outage)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises(ProviderRequestGuardError, match="synthetic"):
        executor.execute(window, token)
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run is not None and run.committed_output is None
    if adapter == "inline":
        retained = run.generation_request_snapshot["procedural_execution"]
        contract = retained["contract"]
        assert len(retained["completed"]) == 1
    else:
        retained = run.generation_request_snapshot["overflow"]["checkpoint"][native.KEY]
        contract = retained
        assert len(retained["reader_steps"]) == 1
    assert set(contract["request_contracts"]) == {"source", "maintain", "review"}
    assert "fixture-route-secret" not in json.dumps(retained)
    assert "fixture.invalid" not in json.dumps(retained)
    before = deepcopy(retained)
    count = len(calls)
    assert count == 2
    client.config.top_p = 0.9
    window = storage.prepare_extraction(window.user_id, token)
    with pytest.raises((ExtractionInputIntegrityError, RuntimeError)):
        executor.execute(window, token)
    assert len(calls) == count
    current = storage.get_agent_run(run.id)
    assert current is not None and current.committed_output is None
    if adapter == "inline":
        assert current.generation_request_snapshot["procedural_execution"] == before
    else:
        assert (
            current.generation_request_snapshot["overflow"]["checkpoint"][native.KEY]
            == before
        )
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == library_before


@pytest.mark.integration
@pytest.mark.parametrize("mixed", [False, True])
def test_exhausted_outer_json_is_terminal_across_actual_window_retry(
    pipeline, monkeypatch, tmp_path, mixed
):
    engine, storage, window, token, client, calls, _, _ = cli_setup(
        pipeline, monkeypatch, tmp_path
    )
    if mixed:
        configured = engine.request_context.configurator.get_config().model_copy(
            deep=True
        )
        configured.llm_config.generation_model_name = "gpt-4o-mini"
        engine.request_context.configurator.set_config(configured)
        client.config.model = "gpt-4o-mini"
        client.config.fallback_models = ["local-cli/codex/fixture-model"]

    def malformed(params, _timeout):
        calls.append(params)
        assert params["tools"][0]["function"]["name"] == BATCH_TOOL
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "malformed-batch",
                                "type": "function",
                                "function": {
                                    "name": BATCH_TOOL,
                                    "arguments": "{broken-json",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", malformed)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises((ContractError, RuntimeError), match="JSON|did not complete"):
        executor.execute(window, token)
    assert len(calls) == 2
    if mixed:
        assert all(call["model"] == "gpt-4o-mini" for call in calls)
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run is not None and run.committed_output is None
    snapshot = run.generation_request_snapshot
    if "procedural_execution" in snapshot:
        failed = snapshot["procedural_execution"]["failed"]
    else:
        failed = snapshot["overflow"]["checkpoint"][native.KEY]["failed_steps"]
    assert len(failed) == 1
    assert (
        next(iter(failed.values())).get(
            "kind", next(iter(failed.values())).get("error_type")
        )
        == "ContractError"
    )
    assert "broken-json" not in json.dumps(failed)
    window = storage.prepare_extraction(window.user_id, token)
    with pytest.raises((ContractError, RuntimeError), match="JSON|did not complete"):
        executor.execute(window, token)
    assert len(calls) == 2


def test_native_journal_retains_exhausted_batch_json_without_redraw():
    from reflexio.server.llm._litellm_types import (
        CompletionResult,
        ModelProvenance,
        ToolCallingChatResponse,
    )
    from tests.server.services.playbook.test_procedural_native import (
        journal_fixture,
        observation_models,
    )
    from tests.server.services.playbook.test_procedural_transports import _cli_client

    coordinator, _, data, _ = journal_fixture()
    delegate, client = _cli_client("local-cli/codex/test-model", [])

    def malformed(**kwargs):
        kwargs["provider_request_guard"]({}, 240, (kwargs["model"],), False)
        return CompletionResult(
            value=ToolCallingChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "batch",
                        "type": "function",
                        "function": {"name": BATCH_TOOL, "arguments": "{broken-json"},
                    }
                ],
                finish_reason="tool_calls",
            ),
            provenance=ModelProvenance(model_name="test-model", provider="test"),
        )

    client.generate_chat_response_with_provenance.side_effect = malformed
    journal = native.JournalClient(
        coordinator, delegate, data, {1: ["Require approval."]}
    )
    options = (
        "source",
        "Read original source",
        {},
        observation_models(),
        lambda rows: rows,
        lambda _value: None,
    )
    with pytest.raises(ContractError, match="Malformed procedural batch JSON"):
        journal.ask(*options)
    assert client.generate_chat_response_with_provenance.call_count == 2
    failed = coordinator.checkpoint[native.KEY]["failed_steps"]
    assert next(iter(failed.values()))["error_type"] == "ContractError"
    assert "broken-json" not in json.dumps(failed)
    with pytest.raises(ContractError, match="Retained procedural method failure"):
        journal.ask(*options)
    assert client.generate_chat_response_with_provenance.call_count == 2
    assert delegate.physical_calls == 2
