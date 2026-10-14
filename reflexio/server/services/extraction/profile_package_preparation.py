"""Provider-free preparation of complete profile package consumer requests.

The caller captures prior knowledge and authorizes the original window first.
This module plans compression and freezes native request contracts; it does not
admit a parent, dispatch a consumer, mark coverage, publish, or enable a worker.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.config_schema import PendingToolCallConfig, ProfileExtractorConfig
from reflexio.server.llm._litellm_text_generation import (
    _is_structured_output_validation_repair_turn,
)
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    check_request_input_budget,
    extraction_provider_guard,
    measure_request_input_tokens,
    request_input_capacity,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    is_structured_output_correction_turn,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import ToolRegistry
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.compression import (
    ExtractionCompressionPlan,
    plan_extraction_window,
)
from reflexio.server.services.extraction.overflow_protocol import package_stage_caps
from reflexio.server.services.extraction.prior_answer_search import (
    CapturedPriorKnowledge,
    build_knowledge_need_query,
)
from reflexio.server.services.extraction.resumable_agent import (
    create_pending_info_tools_for_extractor_kind,
)
from reflexio.server.services.extraction.source_package import render_source_package
from reflexio.server.services.input_compression import CompressionBudget, EvidencePacket
from reflexio.server.services.profile.components.consolidator import (
    ProfileDeduplicationOutput,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    PROFILE_SOURCE_EVIDENCE_INSTRUCTION,
    StructuredProfilesOutput,
    build_profile_gate_prompt,
    construct_profile_extraction_messages_from_sessions,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationServiceConstants as Prompts,
)
from reflexio.server.services.service_utils import format_sessions_to_history_string
from reflexio.server.services.storage.storage_base import canonical_json

PROFILE_PACKAGE_INPUT_TARGET = 32_768


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def build_profile_package_request(
    client: LiteLLMClient, messages: list[dict[str, Any]], **options: Any
) -> dict[str, Any]:
    """Use the exact scoped native builder without allowing any dispatch.

    Reservation must occur BEFORE native model policy, as in actual extraction.
    Supplying it after an unscoped build is not equivalent for every provider.
    Preserve any enclosing guard, and keep raw messages detached from transforms.
    """

    def refuse_dispatch(*_args: Any) -> None:
        raise InputBudgetConfigurationError("Profile preparation forbids dispatch")

    token = extraction_provider_guard.set(refuse_dispatch)
    try:
        return client._build_completion_params(
            messages=copy.deepcopy(messages), **copy.deepcopy(options)
        )[0]
    finally:
        extraction_provider_guard.reset(token)


def profile_package_original_seed(
    *,
    prompt_manager: PromptManager,
    groups: list[RequestInteractionDataModel],
    existing_profiles: list[UserProfile],
    agent_context: str,
    extractor_config: ProfileExtractorConfig,
    template_identities: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build the native pre-compression seed for the existing bounded lookup.

    History formatting and its existing display limits are unchanged; the prior
    lookup still consumes only its existing4,000-character query. Compression
    itself separately receives all authoritative original fields.
    """
    return [
        *construct_profile_extraction_messages_from_sessions(
            prompt_manager=prompt_manager,
            request_interaction_data_models=groups,
            existing_profiles=existing_profiles,
            agent_context_prompt=agent_context,
            context_prompt=extractor_config.context_prompt.strip()
            if extractor_config.context_prompt
            else "",
            extraction_definition_prompt=extractor_config.extraction_definition_prompt.strip(),
            source_text=format_sessions_to_history_string(groups),
            template_identities=template_identities,
        ),
        {"role": "system", "content": PROFILE_SOURCE_EVIDENCE_INSTRUCTION},
    ]


@dataclass(frozen=True)
class ProfilePackagePreparation:
    compression: ExtractionCompressionPlan
    contract_digest: str
    _snapshot_json: str = field(repr=False)
    _render_agent: Callable[[str], list[dict[str, Any]]] = field(
        repr=False, compare=False
    )
    _render_gate: Callable[[str], list[dict[str, Any]]] = field(
        repr=False, compare=False
    )
    _check_request: Callable[
        [str, list[dict[str, Any]], dict[str, Any], tuple[str, ...], bool], None
    ] = field(repr=False, compare=False)

    @property
    def snapshot(self) -> dict[str, Any]:
        """Private frozen context; caller validates answers before admission."""
        return json.loads(self._snapshot_json)

    def agent_messages(self, source_text: str) -> list[dict[str, Any]]:
        return self._render_agent(source_text)

    def gate_messages(self, source_text: str) -> list[dict[str, Any]]:
        return self._render_gate(source_text)

    def check_request(
        self,
        stage: str,
        raw_messages: list[dict[str, Any]],
        params: dict[str, Any],
        ladder: tuple[str, ...],
        parse_structured_output: bool,
    ) -> None:
        """Check a real dispatch before the caller reserves its durable attempt.

        The runtime caller owns the authorized seed/continuation and binds this
        contract digest to its admitted parent. This check neither installs a
        provider guard nor authorizes a caller-supplied message history.
        """
        self._check_request(
            stage, raw_messages, params, ladder, parse_structured_output
        )


def prepare_profile_package(
    groups: list[RequestInteractionDataModel],
    *,
    org_id: str,
    project_id: str,
    user_id: str,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    prompt_manager: PromptManager,
    extractor_config: ProfileExtractorConfig,
    existing_profiles: list[UserProfile],
    agent_context: str,
    gate_enabled: bool,
    gate_model: str,
    dedup_model: str,
    prior_context: CapturedPriorKnowledge | None = None,
    pending_tools_active: bool = False,
    pending_tool_config: PendingToolCallConfig | None = None,
) -> ProfilePackagePreparation:
    """Price full gate/agent input across native ladders before compression.

    ``gate_enabled`` and ``pending_tools_active`` are the caller's already
    resolved feature/configuration decisions. Existing-profile selection belongs
    to the caller; pass the same selected context as the actual extractor.
    Dedup's dynamic NEW/OLD prompt must still be frozen and checked at dispatch.
    This is preparation, not a complete consumer/runtime qualification.
    """
    context_limits = dict(context_limits)
    config = extractor_config.model_copy(deep=True)
    profiles = [profile.model_copy(deep=True) for profile in existing_profiles]
    if any(profile.user_id != user_id for profile in profiles):
        raise InputBudgetConfigurationError("Profile context owner changed")
    prior = prior_context.snapshot if prior_context is not None else None
    if prior is not None and (prior["org_id"] != org_id or not pending_tools_active):
        raise InputBudgetConfigurationError("Profile prior context scope changed")
    if pending_tools_active and pending_tool_config is None:
        raise InputBudgetConfigurationError("Profile tool configuration is required")
    suffix = prior["suffix"] if prior is not None else []
    tools = ToolRegistry(
        create_pending_info_tools_for_extractor_kind("profile")
        if pending_tools_active
        else []
    ).openai_specs()
    templates = {
        prompt_id: prompt_manager.get_prompt_template_identity(prompt_id)
        for prompt_id in (
            Prompts.PROFILE_UPDATE_INSTRUCTION_START_PROMPT_ID,
            Prompts.PROFILE_UPDATE_MAIN_PROMPT_ID,
            *([Prompts.PROFILE_SHOULD_GENERATE_PROMPT_ID] if gate_enabled else []),
        )
    }
    if prior_context is not None and pending_tool_config is not None:
        query = build_knowledge_need_query(
            extractor_kind="profile",
            extractor_config=config,
            source=groups[-1].request.source if groups else None,
            agent_version=None,
            messages=profile_package_original_seed(
                prompt_manager=prompt_manager,
                groups=groups,
                existing_profiles=profiles,
                agent_context=agent_context,
                extractor_config=config,
                template_identities=templates,
            ),
        )
        prior_context.require_lookup(
            org_id=org_id,
            knowledge_need_query=query,
            similarity_threshold=pending_tool_config.for_tool(
                "attach_pending_info_request"
            ).similarity_threshold,
        )

    def render_agent(source_text: str) -> list[dict[str, Any]]:
        return [
            *construct_profile_extraction_messages_from_sessions(
                prompt_manager=prompt_manager,
                request_interaction_data_models=[],
                existing_profiles=profiles,
                agent_context_prompt=agent_context,
                context_prompt=config.context_prompt.strip()
                if config.context_prompt
                else "",
                extraction_definition_prompt=config.extraction_definition_prompt.strip(),
                source_text=source_text,
                template_identities=templates,
            ),
            {"role": "system", "content": PROFILE_SOURCE_EVIDENCE_INSTRUCTION},
            *copy.deepcopy(suffix),
        ]

    def render_gate(source_text: str) -> list[dict[str, Any]]:
        if not gate_enabled:
            return []
        prompt = build_profile_gate_prompt(
            prompt_manager=prompt_manager,
            agent_context_prompt=agent_context,
            extraction_definition_prompt=config.extraction_definition_prompt,
            condition_prompt=config.should_extract_profile_prompt_override,
            source_text=source_text,
            template_identity=templates[Prompts.PROFILE_SHOULD_GENERATE_PROMPT_ID],
        )
        if prompt is None:
            raise InputBudgetConfigurationError("Enabled profile gate lacks criteria")
        return [{"role": "user", "content": prompt}]

    stage_options: dict[str, dict[str, Any]] = {
        "gate": {"model": gate_model},
        "agent": {
            "model_role": ModelRole.EXTRACTION_AGENT,
            "response_format": StructuredProfilesOutput,
            "tools": tools or None,
            "tool_choice": "auto" if tools else None,
        },
        "dedup": {"model": dedup_model, "response_format": ProfileDeduplicationOutput},
    }
    ladders = {
        stage: list(client.get_completion_model_ladder(**options))
        if stage != "gate" or gate_enabled
        else []
        for stage, options in stage_options.items()
    }
    consumer = {
        "kind": "profile",
        "gate_enabled": gate_enabled,
        "ladders": {**ladders, "intent": []},
    }
    caps = package_stage_caps(consumer)

    def requests(stage: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The native ladder rebuilds each explicit rung without model_role.
        options = {**stage_options[stage], "fallback_models": []}
        options.pop("model_role", None)
        return [
            build_profile_package_request(
                client, messages, **{**options, "model": rung}
            )
            for rung in ladders[stage]
        ]

    contracts = {
        stage: [
            _digest(client._completion_request_contract_from_params(params))
            for params in requests(stage, [])
        ]
        for stage in stage_options
    }
    # A consumer's output reservation may leave less than the nominal32K. All
    # measured complete requests use the same budgeted INPUT-token unit.
    ceiling = min(
        request_input_capacity(
            params,
            context_limits=context_limits,
            prompt_target=PROFILE_PACKAGE_INPUT_TARGET,
        )
        for stage in ("gate", "agent")
        for params in requests(stage, [])
    )
    if ceiling <= 0:
        raise InputBudgetConfigurationError("Profile fixed request lacks capacity")
    snapshot = {
        "version": 1,
        "org_id": org_id,
        "project_id": project_id,
        "user_id": user_id,
        "extractor_config": config.model_dump(mode="json"),
        "context_profiles": [profile.model_dump(mode="json") for profile in profiles],
        "agent_context": agent_context,
        "templates": templates,
        "prior_context": prior,
        "tools": tools,
        "pending_tool_config": pending_tool_config.model_dump(mode="json")
        if pending_tools_active and pending_tool_config is not None
        else None,
        "consumer": consumer,
        "stage_caps": caps,
        "native_contracts": contracts,
        "schemas": {
            "agent": StructuredProfilesOutput.model_json_schema(),
            "dedup": ProfileDeduplicationOutput.model_json_schema(),
        },
        "validator_policies": {
            "agent": "exact-visible-original-quote-v1",
            "dedup": "native-profile-dedup-indices-v1",
        },
        "context_limits_digest": _digest(dict(context_limits)),
        "input_target": PROFILE_PACKAGE_INPUT_TARGET,
        "input_ceiling": ceiling,
        "agent_max_steps": 8,
        "source_policy": PROFILE_SOURCE_EVIDENCE_INSTRUCTION,
    }
    contract_digest = _digest(snapshot)

    def measure(packet: EvidencePacket) -> int:
        # Original witness is authoritative, including empty interactions.
        # Assigned once below before the planner invokes this callback.
        source_text = render_source_package(packet, source_digest=source_digest)
        costs = []
        for stage, messages in (
            ("gate", render_gate(source_text)),
            ("agent", render_agent(source_text)),
        ):
            params = requests(stage, messages)
            if [
                _digest(client._completion_request_contract_from_params(request))
                for request in params
            ] != contracts[stage]:
                raise InputBudgetConfigurationError("Profile native contract changed")
            costs.extend(measure_request_input_tokens(request) for request in params)
        return max(costs)

    from reflexio.server.services.extraction.input_preparation import (
        collect_extraction_sources,
    )

    source_digest = collect_extraction_sources(
        groups, user_id=user_id, include_timestamps=True, include_request_source=True
    ).source_digest
    plan = plan_extraction_window(
        groups,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        budget=CompressionBudget(output_tokens=ceiling, max_calls=caps["compress"]),
        client=client,
        context_limits=context_limits,
        measure_output=measure,
        output_contract=f"profile-package-complete-native-v1:{contract_digest}",
    )
    if plan.source_digest != source_digest:
        raise InputBudgetConfigurationError("Profile original source changed")

    def check_request(
        stage: str,
        raw_messages: list[dict[str, Any]],
        params: dict[str, Any],
        ladder: tuple[str, ...],
        parse_structured_output: bool,
    ) -> None:
        if (
            stage not in stage_options
            or ladder != tuple(ladders[stage])
            or not ladder
            or client.get_completion_model_ladder(**stage_options[stage]) != ladder
            or params.get("model") not in ladder
            or parse_structured_output is not True
            or params.get("num_retries") != 0
            or "fallbacks" in params
        ):
            raise InputBudgetConfigurationError("Profile dispatch policy changed")
        if snapshot["schemas"] != {
            "agent": StructuredProfilesOutput.model_json_schema(),
            "dedup": ProfileDeduplicationOutput.model_json_schema(),
        }:
            raise InputBudgetConfigurationError("Profile output schema changed")
        index = ladder.index(params["model"])
        actual_contract = _digest(
            client._completion_request_contract_from_params(params)
        )
        expected = requests(stage, raw_messages)[index]
        if (
            actual_contract != contracts[stage][index]
            or _digest(client._completion_request_contract_from_params(expected))
            != contracts[stage][index]
        ):
            raise InputBudgetConfigurationError("Profile native contract changed")
        actual_messages = params.get("messages")
        if not isinstance(actual_messages, list) or (
            actual_messages != expected["messages"]
            and not (
                stage != "gate"
                and (
                    is_structured_output_correction_turn(
                        expected["messages"], actual_messages
                    )
                    or _is_structured_output_validation_repair_turn(
                        expected["messages"],
                        actual_messages,
                        schema_name=stage_options[stage]["response_format"].__name__,
                    )
                )
            )
        ):
            raise InputBudgetConfigurationError("Profile dispatch messages changed")
        check_request_input_budget(
            params,
            context_limits=context_limits,
            prompt_target=PROFILE_PACKAGE_INPUT_TARGET,
        )

    return ProfilePackagePreparation(
        plan,
        contract_digest,
        canonical_json(snapshot),
        render_agent,
        render_gate,
        check_request,
    )
