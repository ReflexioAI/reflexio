from __future__ import annotations

import json
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
    ["instance_method", "helper", "config", "seed", "grace", "prompt_version"],
)
def test_pairwise_judge_rejects_frozen_plan_drift_before_provider_execution(
    monkeypatch, drift
):
    monkeypatch.setenv("REFLEXIO_LLM_SEED", "42")
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    prompt_manager = PromptManager()
    client = LiteLLMClient(LiteLLMConfig(model="azure/judge-deployment"))
    provider = Mock()
    cast(Any, client)._completion_with_hard_timeout = provider
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
    elif drift == "config":
        client.config.top_p = 0.25
    elif drift == "seed":
        monkeypatch.setenv("REFLEXIO_LLM_SEED", "43")
    elif drift == "grace":
        monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "6")
    else:
        prompt_manager.version_override = {PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID: "1.1.0"}

    with pytest.raises(judge_module.FrozenEvaluatorPlanDriftError):
        judge.judge(**_inputs())
    provider.assert_not_called()


def test_pairwise_judge_provider_params_match_frozen_sanitized_plan(monkeypatch):
    monkeypatch.setenv("REFLEXIO_LLM_SEED", "47")
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "7")
    endpoint = "https://azure-one.example.test/"
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
    assert rung["provider_params"]["seed"] == 47
    assert secret not in serialized
    assert endpoint not in serialized


def test_pairwise_judge_rejects_drift_during_prompt_render_before_provider(
    monkeypatch,
):
    monkeypatch.setenv("REFLEXIO_LLM_SEED", "42")
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
