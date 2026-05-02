"""LLM call that proposes per-citation reflection decisions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionOutput,
)
from reflexio.server.services.service_utils import (
    log_llm_messages,
    log_model_response,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import (
        Interaction,
        UserPlaybook,
        UserProfile,
    )
    from reflexio.server.api_endpoints.request_context import RequestContext

logger = logging.getLogger(__name__)

REFLECTION_TIMEOUT_SECONDS = 300
REFLECTION_MAX_RETRIES = 1
REFLECTION_PROMPT_ID = "memory_reflection"


class ReflectionExtractor:
    """Build the reflection prompt and ask the LLM for decisions.

    Thin wrapper around one ``LiteLLMClient`` call using
    ``ReflectionOutput`` as the structured-output schema. Does not touch
    storage.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        agent_context: str,
        model_override: str | None,
    ):
        """Build a new extractor.

        Args:
            request_context (RequestContext): For prompt manager and
                config access.
            llm_client (LiteLLMClient): Shared LLM client.
            agent_context (str): Agent-context prompt fragment from
                config (free-form text describing the agent).
            model_override (str | None): If set, used directly. Otherwise
                falls back to ``LLMConfig.generation_model_name`` and the
                site default for ``ModelRole.GENERATION``.
        """
        self.request_context = request_context
        self.client = llm_client
        self.agent_context = agent_context
        self.model_name = self._resolve_model(model_override)

    def _resolve_model(self, model_override: str | None) -> str:
        config = self.request_context.configurator.get_config()
        llm_config = config.llm_config if config else None
        api_key_config = config.api_key_config if config else None
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        return resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            config_override=(
                model_override
                or (llm_config.generation_model_name if llm_config else None)
            ),
            api_key_config=api_key_config,
        )

    def run(
        self,
        *,
        window_interactions: list[Interaction],
        cited_profiles: list[UserProfile],
        cited_user_playbooks: list[UserPlaybook],
    ) -> ReflectionOutput:
        """Render the prompt, call the LLM, return parsed output.

        Returns an empty ``ReflectionOutput`` when nothing to consider
        or when the LLM response cannot be parsed. LLM exceptions
        propagate — the calling service decides whether to swallow.
        """
        if not cited_profiles and not cited_user_playbooks:
            return ReflectionOutput()

        rendered = self.request_context.prompt_manager.render_prompt(
            REFLECTION_PROMPT_ID,
            {
                "agent_context": self.agent_context or "",
                "window_interactions_json": _interactions_to_json(window_interactions),
                "cited_profiles_json": _profiles_to_json(cited_profiles),
                "cited_user_playbooks_json": _playbooks_to_json(cited_user_playbooks),
            },
        )
        messages = [{"role": "user", "content": rendered}]

        logger.info(
            "event=reflection_llm_start cited_profiles=%d cited_playbooks=%d "
            "window_interactions=%d model=%s",
            len(cited_profiles),
            len(cited_user_playbooks),
            len(window_interactions),
            self.model_name,
        )
        log_llm_messages(logger, "Memory reflection", messages)

        response = self.client.generate_chat_response(
            messages=messages,
            model=self.model_name,
            response_format=ReflectionOutput,
            timeout=REFLECTION_TIMEOUT_SECONDS,
            max_retries=REFLECTION_MAX_RETRIES,
        )
        log_model_response(logger, "Memory reflection model response", response)

        if isinstance(response, ReflectionOutput):
            return response
        logger.warning(
            "event=reflection_llm_unparsed response_type=%s",
            type(response).__name__,
        )
        return ReflectionOutput()


def _interactions_to_json(interactions: list[Interaction]) -> str:
    payload: list[dict[str, Any]] = [
        {
            "interaction_id": i.interaction_id,
            "request_id": i.request_id,
            "role": i.role,
            "content": i.content,
            "tools_used": (
                [t.model_dump() for t in i.tools_used] if i.tools_used else []
            ),
            "citations": [c.model_dump() for c in i.citations] if i.citations else [],
            "created_at": i.created_at,
        }
        for i in interactions
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _profiles_to_json(profiles: list[UserProfile]) -> str:
    payload = [
        {
            "profile_id": p.profile_id,
            "content": p.content,
            "profile_time_to_live": p.profile_time_to_live.value,
            "custom_features": p.custom_features,
            "source": p.source,
            "last_modified_timestamp": p.last_modified_timestamp,
        }
        for p in profiles
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _playbooks_to_json(playbooks: list[UserPlaybook]) -> str:
    payload = [
        {
            "user_playbook_id": p.user_playbook_id,
            "playbook_name": p.playbook_name,
            "content": p.content,
            "trigger": p.trigger,
            "rationale": p.rationale,
            "source": p.source,
        }
        for p in playbooks
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
