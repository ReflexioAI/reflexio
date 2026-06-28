"""Live provider for the playbook ask_human invocation eval."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    PlaybookConfig,
    StorageConfigSQLite,
    ToolUseConfig,
)
from reflexio.server.services.extraction.resumable_agent import (
    run_resumable_extraction_agent,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
    construct_playbook_extraction_messages_from_sessions,
)
from reflexio.server.services.playbook.service import PlaybookGenerationServiceConfig
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)
from tests.eval.playbook_ask_human.case import PlaybookAskHumanCase
from tests.eval.playbook_ask_human.runner import AskHumanPrediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient


_EVAL_USER_ID = "ask-human-eval-user"
_EVAL_AGENT_VERSION = "ask-human-eval-agent"
_EVAL_SOURCE = "ask-human-eval"


def _sessions_to_ridm(case: PlaybookAskHumanCase) -> RequestInteractionDataModel:
    request_id = f"ask-human-eval-{case.id}"
    interactions = [
        Interaction(
            interaction_id=idx,
            user_id=_EVAL_USER_ID,
            request_id=request_id,
            content=turn.content,
            role=turn.role,
            created_at=idx,
        )
        for idx, turn in enumerate(case.sessions, start=1)
    ]
    request = Request(
        request_id=request_id,
        user_id=_EVAL_USER_ID,
        session_id=f"session-{case.id}",
        created_at=0,
        source=_EVAL_SOURCE,
    )
    return RequestInteractionDataModel(
        session_id=request.session_id,
        request=request,
        interactions=interactions,
    )


def _tool_can_use(case: PlaybookAskHumanCase) -> list[ToolUseConfig]:
    tools: list[ToolUseConfig] = []
    for raw in case.tool_can_use:
        name, _, description = raw.partition(":")
        tools.append(
            ToolUseConfig(
                tool_name=name.strip(),
                tool_description=(description or raw).strip(),
            )
        )
    return tools


def _configure_context(
    request_context: RequestContext, case: PlaybookAskHumanCase
) -> None:
    config = Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="ask_human_eval_playbook",
            extraction_definition_prompt=case.extraction_definition_prompt
            or "Extract durable playbooks from natural agent-user trajectories.",
        ),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
        tool_can_use=_tool_can_use(case),
        window_size=max(len(case.sessions), 1),
        stride_size=1,
    )
    request_context.configurator.get_config = lambda: config  # type: ignore[method-assign]


def _seed_prior_pending_tool_calls(
    request_context: RequestContext,
    case: PlaybookAskHumanCase,
) -> None:
    if not case.prior_pending_tool_calls:
        return
    if request_context.storage is None:
        raise RuntimeError("ask_human eval provider requires storage")

    now = datetime.now(UTC)
    scope = human_feedback_scope(request_context.org_id)
    scope_hash = build_scope_hash(scope)
    for prior in case.prior_pending_tool_calls:
        request_context.storage.create_pending_tool_call(
            PendingToolCallRecord(
                id=prior.pending_tool_call_id,
                org_id=request_context.org_id,
                user_id=_EVAL_USER_ID,
                scope=scope,
                scope_hash=scope_hash,
                tool_name="ask_human",
                dedup_key=build_pending_tool_call_dedup_key(
                    tool_name="ask_human",
                    question_text=prior.question_text,
                    answer_format=prior.answer_format,
                ),
                status=PendingToolCallStatus.PENDING,
                question_text=prior.question_text,
                args={
                    "question": prior.question_text,
                    "answer_format": prior.answer_format,
                },
                tags=prior.tags,
                answer_format=prior.answer_format,
                expires_at=now + timedelta(hours=1),
                cache_until=now + timedelta(minutes=5),
            )
        )


def make_ask_human_prediction_provider(
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> Callable[[PlaybookAskHumanCase], AskHumanPrediction]:
    """Build a provider that runs the real playbook extraction prompt/tool loop."""

    def provider(case: PlaybookAskHumanCase) -> AskHumanPrediction:
        _configure_context(request_context, case)
        _seed_prior_pending_tool_calls(request_context, case)
        ridm = _sessions_to_ridm(case)
        tool_can_use = "\n".join(case.tool_can_use)
        messages = construct_playbook_extraction_messages_from_sessions(
            prompt_manager=request_context.prompt_manager,
            request_interaction_data_models=[ridm],
            agent_context_prompt=case.agent_context_prompt,
            extraction_definition_prompt=case.extraction_definition_prompt,
            tool_can_use=tool_can_use,
        )
        extractor_config = (
            request_context.configurator.get_config().user_playbook_extractor_config
        )
        if extractor_config is None:
            raise RuntimeError("ask_human eval provider requires a playbook extractor")
        result = run_resumable_extraction_agent(
            request_context=request_context,
            client=llm_client,
            extractor_kind="playbook",
            user_id=_EVAL_USER_ID,
            request_id=ridm.request.request_id,
            agent_version=_EVAL_AGENT_VERSION,
            source=_EVAL_SOURCE,
            request_interaction_data_models=[ridm],
            extractor_config=extractor_config,
            service_config=PlaybookGenerationServiceConfig(
                request_id=ridm.request.request_id,
                agent_version=_EVAL_AGENT_VERSION,
                user_id=_EVAL_USER_ID,
                source=_EVAL_SOURCE,
            ),
            agent_context=case.agent_context_prompt,
            messages=messages,
            output_schema=StructuredPlaybookList,
            log_label="Playbook ask_human eval",
        )
        tool_names = [turn.tool_name for turn in result.trace.turns]
        question_texts = [
            str(turn.args.get("question", ""))
            for turn in result.trace.turns
            if turn.tool_name == "ask_human"
        ]
        playbook_count = (
            len(result.output.playbooks)
            if isinstance(result.output, StructuredPlaybookList)
            else 0
        )
        return AskHumanPrediction(
            case_id=case.id,
            tool_names=tool_names,
            question_texts=question_texts,
            playbook_count=playbook_count,
        )

    return provider
