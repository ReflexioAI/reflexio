from __future__ import annotations

import inspect
import json
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.domain import AgentPlaybook, Interaction
from reflexio.models.config_schema import (
    APIKeyConfig,
    AzureOpenAIConfig,
    OpenAIConfig,
)
from reflexio.server.llm import _litellm_text_generation as generation
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook_optimizer import judge as judge_module
from reflexio.server.services.playbook_optimizer.judge import (
    PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID,
    PairwiseJudge,
)
from reflexio.server.services.playbook_optimizer.models import (
    RolloutTrace,
    ScenarioWindow,
)


def _inputs() -> dict[str, Any]:
    return {
        "window": ScenarioWindow(
            user_playbook_id=1,
            source_interaction_ids=[101],
            interactions=[
                Interaction(
                    interaction_id=101,
                    user_id="u1",
                    request_id="request-1",
                    role="User",
                    content="Help with a refund",
                )
            ],
        ),
        "incumbent": AgentPlaybook(
            agent_version="v1",
            content="old guidance",
            trigger="refund request",
        ),
        "candidate": AgentPlaybook(
            agent_version="v1",
            content="new guidance",
            trigger="refund request",
        ),
        "incumbent_rollout": RolloutTrace(),
        "candidate_rollout": RolloutTrace(),
    }


def _freeze(
    prompt_manager: PromptManager,
    client: LiteLLMClient,
) -> dict[str, Any]:
    return judge_module.build_pairwise_judge_request_plan(
        prompt_manager=prompt_manager,
        llm_client=client,
        model_name="azure/judge-deployment",
    )


@pytest.mark.parametrize(
    "drift",
    [
        "instance_method",
        "helper",
        "config",
        "temperature",
        "completion_transport",
        "prompt_version",
        "native_launch",
        "native_retirement",
        "native_worker_alias",
        "native_directory_alias",
        "native_remaining",
        "native_remaining_alias",
        "native_error_transport",
        "native_launch_owner_alias",
        "native_launch_execution",
        "native_executable_source",
    ],
)
def test_pairwise_judge_rejects_frozen_plan_drift_before_provider_execution(
    monkeypatch, tmp_path, drift
):
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="azure/judge-deployment"))
    provider = Mock()
    cast(Any, client)._completion_with_hard_timeout = provider
    if drift == "native_executable_source":
        source = judge_module.native_cli_module._NATIVE_OWNER_SCRIPT
        copied = tmp_path / source.name
        copied.write_bytes(source.read_bytes())
        monkeypatch.setattr(
            judge_module.native_cli_module, "_NATIVE_OWNER_SCRIPT", copied
        )
    frozen_plan = _freeze(prompt_manager, client)
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "azure/judge-deployment",
        frozen_request_plan=frozen_plan,
    )

    if drift == "instance_method":
        cast(Any, client)._make_request = Mock()
    elif drift == "helper":
        monkeypatch.setattr(
            judge_module,
            "build_pairwise_judge_request_plan",
            Mock(return_value=frozen_plan),
        )
    elif drift == "native_launch":
        monkeypatch.setattr(judge_module.native_cli_provider, "run_native_cli", Mock())
    elif drift == "native_worker_alias":
        monkeypatch.setattr(judge_module.worker_module, "native_cli_worker", Mock())
    elif drift == "native_directory_alias":
        monkeypatch.setattr(
            judge_module.native_cli_provider, "native_cli_directory", Mock()
        )
    elif drift == "native_remaining":
        monkeypatch.setattr(
            judge_module.native_cli_module, "remaining_native_time", Mock()
        )
    elif drift == "native_remaining_alias":
        monkeypatch.setattr(
            judge_module.native_cli_provider, "remaining_native_time", Mock()
        )
    elif drift == "native_error_transport":
        monkeypatch.setattr(generation, "_raise_completion_error", Mock())
    elif drift == "native_launch_owner_alias":
        monkeypatch.setattr(
            judge_module.native_cli_module,
            "native_owner_identity",
            Mock(return_value={"sha256": "foreign"}),
        )
    elif drift == "native_executable_source":
        with judge_module.native_cli_module._NATIVE_OWNER_SCRIPT.open("a") as stream:
            stream.write("\n# changed executable source\n")
    elif drift == "native_launch_execution":
        monkeypatch.setattr(judge_module.native_process_module, "_run", Mock())
    elif drift == "native_retirement":
        monkeypatch.setattr(generation.NativeCLIOwner, "_run", Mock())
    elif drift == "config":
        client.config.top_p = 0.25
    elif drift == "temperature":
        client.config.temperature = 0.25
    elif drift == "completion_transport":
        monkeypatch.setattr(client, "_run_completion", Mock())
    else:
        prompt_manager.version_override = {PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID: "1.1.0"}

    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge.judge(**_inputs())
    provider.assert_not_called()


def test_pairwise_judge_provider_params_match_frozen_sanitized_plan(monkeypatch):
    # This checks the configured frozen plan, with no elapsed admission time.
    # Patch only the generation module; deadline tests exercise advancing clocks.
    monkeypatch.setattr(
        generation,
        "time",
        SimpleNamespace(
            monotonic=lambda: 1000.0,
            perf_counter=time.perf_counter,
            sleep=time.sleep,
            time=time.time,
        ),
    )
    endpoint = "https://example.com:4443/"
    secret = "azure-secret"
    client = LiteLLMClient(
        LiteLLMConfig(
            model="azure/judge-deployment",
            temperature=0.2,
            top_p=0.7,
            max_tokens=300,
            api_key_config=APIKeyConfig(
                openai=OpenAIConfig(
                    azure_config=AzureOpenAIConfig(
                        api_key=secret,
                        endpoint=cast(Any, endpoint),
                        api_version="2024-02-15-preview",
                    )
                )
            ),
        )
    )
    captured: dict[str, Any] = {}
    compared: dict[str, Any] = {}
    sanitize = judge_module.sanitize_pairwise_judge_provider_params

    def recording_sanitize(params: Any) -> dict[str, Any]:
        compared["params"] = params
        return sanitize(params)

    monkeypatch.setattr(
        judge_module,
        "sanitize_pairwise_judge_provider_params",
        recording_sanitize,
    )

    def completion(params: dict[str, Any], hard_timeout: float) -> Any:
        captured["params"] = params
        captured["hard_timeout"] = hard_timeout
        message = SimpleNamespace(
            content=json.dumps(
                {
                    "verdict": "candidate",
                    "score": 0.9,
                    "likert": 5,
                    "rationale": "candidate is clearer",
                }
            ),
            refusal=None,
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=None,
            stop_reason=None,
        )

    cast(Any, client)._completion_with_hard_timeout = completion
    prompt_manager = PromptManager()
    frozen_plan = _freeze(prompt_manager, client)
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "azure/judge-deployment",
        frozen_request_plan=frozen_plan,
    )

    result = judge.judge(**_inputs())

    rung = frozen_plan["judge_generation_settings"]["rungs"][0]
    assert result.verdict == "candidate"
    assert compared["params"] is captured["params"]
    assert sanitize(captured["params"]) == rung["provider_params"]
    assert captured["hard_timeout"] == float(rung["hard_timeout_seconds"])
    serialized = json.dumps(frozen_plan, sort_keys=True)
    assert "seed" not in rung["provider_params"]
    assert secret not in serialized
    assert endpoint not in serialized


def test_pairwise_judge_rejects_drift_during_prompt_render_before_provider(
    monkeypatch,
):
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="azure/judge-deployment"))
    provider = Mock()
    cast(Any, client)._completion_with_hard_timeout = provider
    render = prompt_manager.render_prompt_from_identity

    def mutating_render(*args: Any, **kwargs: Any) -> str:
        rendered = render(*args, **kwargs)
        client.config.top_p = 0.25
        return rendered

    cast(Any, prompt_manager).render_prompt_from_identity = mutating_render
    frozen_plan = _freeze(prompt_manager, client)
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "azure/judge-deployment",
        frozen_request_plan=frozen_plan,
    )

    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge.judge(**_inputs())
    provider.assert_not_called()


@pytest.mark.parametrize(
    "module_name,attribute",
    [
        ("native_cli_provider", "NativeCLILLM"),
        ("native_cli_provider", "NativeCLILLM.completion"),
        ("native_cli_provider", "NativeCLILLM._cli_path"),
        ("native_cli_provider", "NativeCLILLM._timeout"),
        ("native_cli_provider", "resolve_cli_request_timeout"),
        ("native_cli_provider", "_resolve_cli_path"),
        ("native_cli_provider", "_resolve_cli_override_path"),
        ("native_cli_provider", "_windows_cli_suffixes"),
        ("native_cli_provider", "_is_windows"),
        ("native_cli_provider", "validate_native_selection"),
        ("native_cli_provider", "request_timeout_seconds"),
        ("native_cli_provider", "_codex_prompt"),
        ("native_cli_provider", "_temporary_output_path"),
        ("native_cli_provider", "_split_system_and_dialogue"),
        ("native_cli_provider", "_maybe_append_schema"),
        ("native_cli_provider", "_maybe_append_tools_instruction"),
        ("native_cli_provider", "_parse_tool_use"),
        ("native_stream_parser", "parse_stream_json"),
        ("native_tool_protocol", "append_schema_instruction"),
        ("native_cli_provider", "_run_cli_stream"),
        ("native_model_ref", "parse_native_model_ref"),
        ("native_model_ref", "configured_native_model"),
        ("native_model_ref", "env_truthy"),
        ("native_cli_provider", "env_str"),
        ("native_model_ref", "authorize_native_model"),
        ("native_model_ref", "native_policy_digest"),
        ("native_model_ref", "reject_native_selection"),
        ("native_cli_provider", "authorize_native_model"),
        ("native_cli_provider", "reject_native_selection"),
        ("worker_module", "authorize_native_model"),
        ("worker_module", "native_request_authority"),
        ("text_generation_module", "authorize_native_model"),
        ("native_cli_provider", "_compat_host"),
        ("opencode_module", "run_opencode"),
        ("opencode_module", "resolve_executable"),
        ("opencode_executable_module", "_npm_opencode_executable"),
        ("opencode_executable_module", "resolve_executable"),
        ("opencode_module", "_run_options"),
        ("opencode_module", "parse_output"),
        ("opencode_module", "_error_message"),
        ("opencode_module", "run_native_cli"),
        ("opencode_module", "make_private_cli_directory"),
        ("opencode_module", "release_private_cli_directory"),
        ("native_cli_provider", "private_cli_directory"),
        ("opencode_module", "remaining_native_time"),
        ("opencode_module", "classify_stall"),
        ("opencode_module", "parse_reset_estimate"),
    ],
)
def test_frozen_judge_rejects_each_opencode_binding_before_dispatch(
    monkeypatch, module_name, attribute
):
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="azure/judge-deployment"))
    dispatch = Mock()
    cast(Any, client)._completion_with_hard_timeout = dispatch
    frozen = _freeze(prompt_manager, client)
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "azure/judge-deployment",
        frozen_request_plan=frozen,
    )
    owner = getattr(judge_module, module_name)
    for component in attribute.split(".")[:-1]:
        owner = getattr(owner, component)
    monkeypatch.setattr(owner, attribute.split(".")[-1], Mock())
    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge.judge(**_inputs())
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/codex/other"),
        ("CLAUDE_SMART_CLI_PATH", "/owned/different-executable"),
        ("CLAUDE_SMART_CLI_TIMEOUT", "17"),
        ("CODEX_HOME", "/owned/different-codex-account"),
        ("CLAUDE_CONFIG_DIR", "/owned/different-claude-account"),
    ],
)
def test_native_frozen_policy_cannot_change_after_job_creation(
    monkeypatch, setting, value
):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/codex/owned")
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="local-cli/codex/owned", max_tokens=32))
    frozen = judge_module.build_pairwise_judge_request_plan(
        prompt_manager=prompt_manager,
        llm_client=client,
        model_name=None,
    )
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "local-cli/codex/owned",
        frozen_request_plan=frozen,
    )
    judge._assert_frozen_implementation_identities(frozen)
    monkeypatch.setenv(setting, value)
    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge._assert_frozen_implementation_identities(frozen)


def test_native_frozen_policy_binds_effective_home(monkeypatch, tmp_path):
    # Path.home also covers Windows USERPROFILE and platform home resolution;
    # binding only the raw HOME variable would miss those accounts.
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/codex/owned")
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(judge_module.native_model_ref.Path, "home", lambda: first)
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="local-cli/codex/owned", max_tokens=32))
    frozen = judge_module.build_pairwise_judge_request_plan(
        prompt_manager=prompt_manager, llm_client=client, model_name=None
    )
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "local-cli/codex/owned",
        frozen_request_plan=frozen,
    )
    judge._assert_frozen_implementation_identities(frozen)
    monkeypatch.setattr(judge_module.native_model_ref.Path, "home", lambda: second)
    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge._assert_frozen_implementation_identities(frozen)


@pytest.mark.parametrize(
    "method",
    [
        name
        for name, function in vars(
            judge_module.native_cli_provider.NativeCLILLM
        ).items()
        if inspect.isfunction(function) and name != "__init__"
    ],
)
def test_registered_native_handler_override_invalidates_frozen_plan(
    monkeypatch, method
):
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="azure/judge-deployment"))
    handler = judge_module.native_cli_provider.NativeCLILLM()
    monkeypatch.setattr(
        judge_module.litellm,
        "custom_provider_map",
        [
            {"provider": "local-cli", "custom_handler": handler},
        ],
    )
    frozen = _freeze(prompt_manager, client)
    judge = PairwiseJudge(
        cast(Any, SimpleNamespace(prompt_manager=prompt_manager)),
        client,
        "azure/judge-deployment",
        frozen_request_plan=frozen,
    )
    monkeypatch.setattr(handler, method, Mock())
    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge._assert_frozen_implementation_identities(frozen)


def test_native_helper_census_covers_actual_owned_functions_and_aliases():
    modules = {
        "native_cli_provider": judge_module.native_cli_provider,
        "native_model_ref": judge_module.native_model_ref,
        "opencode": judge_module.opencode_module,
        "opencode_executable": judge_module.opencode_executable_module,
        "native_stream_parser": judge_module.native_stream_parser,
        "native_tool_protocol": judge_module.native_tool_protocol,
    }
    bindings = judge_module._native_helper_identities()
    expected = {
        f"{label}.{name}"
        for label, module in modules.items()
        for name, value in vars(module).items()
        if (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", "").startswith("reflexio.")
    }
    assert "native_cli_provider._codex_prompt" in expected
    assert "native_cli_provider.request_timeout_seconds" in expected
    assert expected <= set(bindings)
