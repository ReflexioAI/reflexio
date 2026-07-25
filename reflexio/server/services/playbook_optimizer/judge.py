from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, cast

import litellm

from reflexio.models.api_schema.domain import AgentPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm import _litellm_structured_output as structured_output_module
from reflexio.server.llm import _litellm_text_generation as text_generation_module
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.service_utils import log_model_response

from .models import JudgeOutput, RolloutTrace, ScenarioWindow

logger = logging.getLogger(__name__)

PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID = "playbook_optimizer_judge"
PAIRWISE_JUDGE_TIMEOUT_SECONDS = 120
PAIRWISE_JUDGE_MAX_RETRIES = 1

_LLM_IMPLEMENTATION_METHODS = (
    "generate_chat_response",
    "_make_request",
    "_resolve_ladder",
    "_resolve_primary_model",
    "_build_completion_params",
    "_is_temperature_restricted_model",
    "_resolve_api_key",
    "_provider_for_model",
    "_structured_output_strategy",
    "_apply_structured_output_transport",
    "_provider_response_format",
    "_completion_with_hard_timeout",
    "_coerce_timeout_seconds",
    "_hard_timeout_grace_seconds",
    "_should_process_isolate_completion",
    "_maybe_parse_structured_output",
    "_apply_prompt_caching",
)

_PROMPT_IMPLEMENTATION_METHODS = (
    "get_prompt_template_identity",
    "render_prompt_from_identity",
    "get_active_version",
    "_find_active_version",
    "_get_prompt",
    "_load_prompt",
    "_render_prompt",
)

_PROVIDER_PARAM_KEYS = frozenset(
    {
        "allowed_openai_params",
        "api_base",
        "api_key",
        "api_version",
        "drop_params",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "num_retries",
        "response_format",
        "seed",
        "temperature",
        "timeout",
        "top_p",
    }
)


class FrozenEvaluatorPlanDriftError(RuntimeError):
    """Raised before evaluation when live judge inputs differ from the frozen plan."""


def canonicalize_pairwise_judge_request_plan(plan: Mapping[str, Any]) -> str:
    """Return immutable canonical JSON bytes-as-text for a sanitized plan."""
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sanitize_pairwise_judge_provider_params(
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Select every non-secret, non-content parameter sent to the provider."""
    unexpected = set(params) - _PROVIDER_PARAM_KEYS
    if unexpected:
        raise ValueError(
            f"Unsupported PairwiseJudge provider params: {sorted(unexpected)}"
        )

    sanitized: dict[str, Any] = {}
    for key, value in sorted(params.items()):
        if key in {"api_key", "messages", "metadata"}:
            continue
        if key == "api_base":
            sanitized["api_base_digest"] = _text_digest(str(value)) if value else None
        elif key == "response_format":
            sanitized[key] = _response_format_identity(value)
        else:
            sanitized[key] = _sanitize_provider_value(value)
    return sanitized


def build_pairwise_judge_request_plan(
    *,
    prompt_manager: PromptManager,
    llm_client: LiteLLMClient,
    model_name: str | None,
) -> dict[str, Any]:
    """Build the sanitized plan persisted and verified by ``PairwiseJudge``."""
    client = cast(Any, llm_client)
    requested_model = model_name or llm_client.config.model
    ladder = client._resolve_ladder(model=requested_model)
    grace_seconds = client._hard_timeout_grace_seconds()
    callables = {
        f"llm_client.{name}": _bound_callable_identity(client, name)
        for name in _LLM_IMPLEMENTATION_METHODS
    }
    callables.update(
        {
            f"prompt_manager.{name}": _bound_callable_identity(prompt_manager, name)
            for name in _PROMPT_IMPLEMENTATION_METHODS
        }
    )
    callables.update(
        {
            "judge.PairwiseJudge.judge": _callable_identity(PairwiseJudge.judge),
            **_judge_helper_identities(),
            **_implementation_helper_identities(),
        }
    )
    return {
        "judge_class": _type_identity(PairwiseJudge),
        "judge_code_digest": _code_digest(PairwiseJudge),
        "judge_model_id": requested_model,
        "judge_prompt_id": PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID,
        "judge_prompt_identity": prompt_manager.get_prompt_template_identity(
            PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID
        ),
        "llm_client_class": _type_identity(type(llm_client)),
        "llm_client_code_digest": _code_digest(type(llm_client)),
        "judge_output_schema_class": _type_identity(JudgeOutput),
        "judge_output_schema_code_digest": _code_digest(JudgeOutput),
        "judge_output_schema_digest": _json_digest(JudgeOutput.model_json_schema()),
        "implementation_callables": callables,
        "judge_generation_settings": {
            "requested_model": requested_model,
            "resolved_primary_model": ladder[0],
            "fallback_model_order": ladder[1:],
            "resolved_model_ladder": ladder,
            "pairwise_judge_max_retries": PAIRWISE_JUDGE_MAX_RETRIES,
            "rungs": [
                _pairwise_judge_rung(llm_client, model, grace_seconds)
                for model in ladder
            ],
        },
    }


def _pairwise_judge_rung(
    llm_client: LiteLLMClient,
    model: str,
    grace_seconds: float,
) -> dict[str, Any]:
    client = cast(Any, llm_client)
    params, _response_format, parse_structured_output, _max_retries, _fallbacks = (
        client._build_completion_params(
            [{"role": "user", "content": ""}],
            model=model,
            response_format=JudgeOutput,
            timeout=PAIRWISE_JUDGE_TIMEOUT_SECONDS,
            max_retries=PAIRWISE_JUDGE_MAX_RETRIES,
            fallback_models=[],
        )
    )
    params["num_retries"] = 0
    params.pop("fallbacks", None)
    resolved_model = str(params["model"])
    timeout_seconds = client._coerce_timeout_seconds(params)
    provider_params = sanitize_pairwise_judge_provider_params(params)
    return {
        "model": resolved_model,
        "temperature": _decimal_string(provider_params["temperature"]),
        "top_p": _decimal_string(provider_params.get("top_p", 1.0)),
        "max_tokens": provider_params.get("max_tokens"),
        "seed": provider_params["seed"],
        "timeout_seconds": _decimal_string(timeout_seconds),
        "hard_timeout_grace_seconds": _decimal_string(grace_seconds),
        "hard_timeout_seconds": _decimal_string(timeout_seconds + grace_seconds),
        "process_isolation": client._should_process_isolate_completion(
            timeout_seconds, grace_seconds
        ),
        "provider_kind": client._provider_for_model(resolved_model) or "unconfigured",
        "api_base_digest": provider_params.get("api_base_digest"),
        "api_version": provider_params.get("api_version"),
        "structured_output_strategy": client._structured_output_strategy(
            model=resolved_model,
            strict_response_format=True,
        ),
        "parse_structured_output": parse_structured_output,
        "provider_params": provider_params,
    }


class PairwiseJudge:
    """LLM-based comparator for paired playbook rollouts.

    Given two rollouts that share the same user turns and differ only in
    the playbook injected into the assistant, ``judge`` asks an LLM (using
    the ``playbook_optimizer_judge`` prompt) to pick a winner and assign a
    score, Likert rating, and structured rationale.

    If the two playbooks have identical content, ``judge`` short-circuits
    to a tie without spending an LLM call — the rollouts will be identical
    by construction.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        model_name: str | None,
        *,
        frozen_request_plan: Mapping[str, Any] | None = None,
    ) -> None:
        self.request_context = request_context
        self.llm_client = llm_client
        self.model_name = model_name or llm_client.config.model
        self._frozen_request_plan_json = (
            canonicalize_pairwise_judge_request_plan(frozen_request_plan)
            if frozen_request_plan is not None
            else None
        )

    def judge(
        self,
        *,
        window: ScenarioWindow,
        incumbent: AgentPlaybook,
        candidate: AgentPlaybook,
        incumbent_rollout: RolloutTrace,
        candidate_rollout: RolloutTrace,
    ) -> JudgeOutput:
        if incumbent.content == candidate.content:
            return JudgeOutput(
                verdict="tie",
                score=0.5,
                likert=3,
                rationale="Candidate content is identical to incumbent content.",
            )
        frozen_plan = self._verify_frozen_request_plan()
        variables = {
            "source_window_json": _json(
                [interaction.model_dump() for interaction in window.interactions]
            ),
            "incumbent_playbook_json": _json(_playbook_payload(incumbent)),
            "candidate_playbook_json": _json(_playbook_payload(candidate)),
            "incumbent_rollout_json": incumbent_rollout.model_dump_json(),
            "candidate_rollout_json": candidate_rollout.model_dump_json(),
        }
        if frozen_plan is None:
            prompt = self.request_context.prompt_manager.render_prompt(
                PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID, variables
            )
        else:
            prompt = self.request_context.prompt_manager.render_prompt_from_identity(
                PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID,
                variables,
                frozen_plan["judge_prompt_identity"],
            )
        response = self.llm_client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format=JudgeOutput,
            timeout=PAIRWISE_JUDGE_TIMEOUT_SECONDS,
            max_retries=PAIRWISE_JUDGE_MAX_RETRIES,
        )
        log_model_response(logger, "Playbook optimizer judge response", response)
        if isinstance(response, JudgeOutput):
            return response
        return JudgeOutput(
            verdict="tie",
            score=0.5,
            likert=3,
            rationale=f"Judge response was not parsed: {type(response).__name__}",
        )

    def _verify_frozen_request_plan(self) -> dict[str, Any] | None:
        if self._frozen_request_plan_json is None:
            return None
        frozen = json.loads(self._frozen_request_plan_json)
        frozen_callables = frozen["implementation_callables"]
        current_helpers = _judge_helper_identities()
        if any(
            frozen_callables.get(name) != identity
            for name, identity in current_helpers.items()
        ):
            raise FrozenEvaluatorPlanDriftError(
                "PairwiseJudge evaluator helper drifted after job creation"
            )
        current = build_pairwise_judge_request_plan(
            prompt_manager=self.request_context.prompt_manager,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        if (
            canonicalize_pairwise_judge_request_plan(current)
            != self._frozen_request_plan_json
        ):
            raise FrozenEvaluatorPlanDriftError(
                "PairwiseJudge evaluator plan drifted after job creation"
            )
        return frozen


def _bound_callable_identity(owner: Any, name: str) -> dict[str, Any]:
    return {
        **_callable_identity(getattr(owner, name)),
        "instance_override": name in vars(owner),
    }


def _implementation_helper_identities() -> dict[str, dict[str, str]]:
    helpers = {
        "litellm.completion": litellm.completion,
        "litellm.get_llm_provider": litellm.get_llm_provider,
        "litellm.supports_response_schema": litellm.supports_response_schema,
        "structured_output._extract_json_from_string": (
            structured_output_module._extract_json_from_string
        ),
        "structured_output._looks_truncated_json": (
            structured_output_module._looks_truncated_json
        ),
        "structured_output._sanitize_json_string": (
            structured_output_module._sanitize_json_string
        ),
        "structured_output._validate_structured_payload": (
            structured_output_module._validate_structured_payload
        ),
        "structured_output.assert_provider_safe_schema": (
            structured_output_module.assert_provider_safe_schema
        ),
        "structured_output.prompt_schema_instruction": (
            structured_output_module.prompt_schema_instruction
        ),
        "structured_output.strict_response_format_for_model": (
            structured_output_module.strict_response_format_for_model
        ),
        "text_generation._litellm_completion_worker": (
            text_generation_module._litellm_completion_worker
        ),
        "text_generation.default_max_tokens_for_model": (
            text_generation_module.default_max_tokens_for_model
        ),
        "text_generation.resolve_model_name": text_generation_module.resolve_model_name,
    }
    return {name: _callable_identity(value) for name, value in helpers.items()}


def _judge_helper_identities() -> dict[str, dict[str, str]]:
    helpers = {
        "judge.build_pairwise_judge_request_plan": build_pairwise_judge_request_plan,
        "judge.canonicalize_pairwise_judge_request_plan": (
            canonicalize_pairwise_judge_request_plan
        ),
        "judge.sanitize_pairwise_judge_provider_params": (
            sanitize_pairwise_judge_provider_params
        ),
        "judge._pairwise_judge_rung": _pairwise_judge_rung,
        "judge._response_format_identity": _response_format_identity,
        "judge._sanitize_provider_value": _sanitize_provider_value,
    }
    return {name: _callable_identity(value) for name, value in helpers.items()}


def _callable_identity(value: Any) -> dict[str, str]:
    target = getattr(value, "__func__", value)
    target_type = type(target)
    identity = (
        f"{getattr(target, '__module__', target_type.__module__)}."
        f"{getattr(target, '__qualname__', target_type.__qualname__)}"
    )
    return {"identity": identity, "code_digest": _code_digest(target)}


def _type_identity(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _code_digest(value: Any) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = _callable_fallback_identity(value)
    return _text_digest(source)


def _callable_fallback_identity(value: Any) -> str:
    value_type = type(value)
    return (
        f"{getattr(value, '__module__', value_type.__module__)}."
        f"{getattr(value, '__qualname__', value_type.__qualname__)}"
    )


def _response_format_identity(value: Any) -> dict[str, str]:
    if inspect.isclass(value):
        return {"kind": "class", "identity": _type_identity(value)}
    try:
        digest = _json_digest(value)
    except (TypeError, ValueError):
        return {"kind": "type", "identity": _type_identity(type(value))}
    return {"kind": "json", "digest": digest}


def _sanitize_provider_value(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_provider_value(item) for key, item in value.items()}
    return value


def _json_digest(value: Any) -> str:
    return _text_digest(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _decimal_string(value: Any) -> str:
    return repr(float(value))


def _playbook_payload(playbook: AgentPlaybook) -> dict[str, object]:
    return {
        "id": playbook.agent_playbook_id,
        "content": playbook.content,
        "trigger": playbook.trigger,
        "rationale": playbook.rationale,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
