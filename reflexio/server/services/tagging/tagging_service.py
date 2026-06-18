from __future__ import annotations

import logging
import os

from pydantic import BaseModel, ConfigDict, Field

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.service_utils import log_llm_messages, log_model_response
from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)

TAGGING_PROMPT_ID = "tagging"

# Upper bound on how many current playbooks a single tagging pass will scan for
# untagged entries. Tagging is idempotent (already-tagged entities are skipped),
# so steady-state passes only tag newly generated entities; this bound just keeps
# a one-time backfill from being silently truncated at the storage default of 100.
_TAGGING_FETCH_LIMIT = 1000


class TagsOutput(BaseModel):
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class TaggingService:
    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.configurator = request_context.configurator
        self.storage = request_context.storage

        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        config = self.configurator.get_config()
        llm_config = config.llm_config if config else None
        api_key_config = config.api_key_config if config else None
        self.model_name = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            config_override=llm_config.generation_model_name if llm_config else None,
            api_key_config=api_key_config,
        )

    def run(
        self,
        *,
        user_id: str,
        agent_version: str,
        tag_profiles: bool = True,
        tag_playbooks: bool = True,
    ) -> None:
        if self.storage is None:
            return

        config = self.configurator.get_config()
        if config is None:
            return

        profile_prompt = getattr(
            config.profile_extractor_config, "tagging_definition_prompt", None
        )
        user_playbook_prompt = getattr(
            config.user_playbook_extractor_config, "tagging_definition_prompt", None
        )

        if tag_profiles and profile_prompt:
            self._tag_profiles(
                user_id=user_id, tagging_definition_prompt=profile_prompt
            )
        if tag_playbooks and user_playbook_prompt:
            self._tag_user_playbooks(
                user_id=user_id,
                agent_version=agent_version,
                tagging_definition_prompt=user_playbook_prompt,
            )
            self._tag_agent_playbooks(
                agent_version=agent_version,
                tagging_definition_prompt=user_playbook_prompt,
            )

    def _tag_profiles(self, *, user_id: str, tagging_definition_prompt: str) -> None:
        profiles = self.storage.get_user_profile(user_id)  # type: ignore[union-attr]
        for profile in profiles:
            if profile.tags is not None:
                continue  # already tagged (incl. empty result) — tag each entity once
            tags = self._generate_tags(
                tagging_definition_prompt=tagging_definition_prompt,
                content=self._profile_content(profile),
            )
            self.storage.update_user_profile_tags(  # type: ignore[union-attr]
                user_id, profile.profile_id, tags
            )

    def _tag_user_playbooks(
        self,
        *,
        user_id: str,
        agent_version: str,
        tagging_definition_prompt: str,
    ) -> None:
        playbooks = self.storage.get_user_playbooks(  # type: ignore[union-attr]
            limit=_TAGGING_FETCH_LIMIT,
            user_id=user_id,
            agent_version=agent_version,
            status_filter=[None],
        )
        for playbook in playbooks:
            if playbook.tags is not None:
                continue  # already tagged (incl. empty result) — tag each entity once
            tags = self._generate_tags(
                tagging_definition_prompt=tagging_definition_prompt,
                content=self._playbook_content(playbook),
            )
            self.storage.update_user_playbook(  # type: ignore[union-attr]
                playbook.user_playbook_id,
                tags=tags,
            )

    def _tag_agent_playbooks(
        self,
        *,
        agent_version: str,
        tagging_definition_prompt: str,
    ) -> None:
        playbooks = self.storage.get_agent_playbooks(  # type: ignore[union-attr]
            limit=_TAGGING_FETCH_LIMIT,
            agent_version=agent_version,
            status_filter=[None],
        )
        for playbook in playbooks:
            if playbook.tags is not None:
                continue  # already tagged (incl. empty result) — tag each entity once
            tags = self._generate_tags(
                tagging_definition_prompt=tagging_definition_prompt,
                content=self._playbook_content(playbook),
            )
            self.storage.update_agent_playbook(  # type: ignore[union-attr]
                playbook.agent_playbook_id,
                tags=tags,
            )

    def _generate_tags(
        self, *, tagging_definition_prompt: str, content: str
    ) -> list[str]:
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            return ["example_tag"]

        prompt = self.request_context.prompt_manager.render_prompt(
            TAGGING_PROMPT_ID,
            {
                "tagging_definition_prompt": tagging_definition_prompt.strip(),
                "content": content,
            },
        )
        messages = [{"role": "user", "content": prompt}]
        log_llm_messages(logger, "Tagging", messages)
        response = self.client.generate_chat_response(
            messages=messages,
            model=self.model_name,
            response_format=TagsOutput,
        )
        log_model_response(logger, "Tagging response", response)
        if not isinstance(response, TagsOutput):
            logger.warning(
                "Unexpected response type from tagging LLM: %s", type(response)
            )
            return []
        return [tag.strip() for tag in response.tags if tag.strip()]

    @staticmethod
    def _profile_content(profile: UserProfile) -> str:
        return profile.content

    @staticmethod
    def _playbook_content(playbook: UserPlaybook | AgentPlaybook) -> str:
        parts = [playbook.content]
        if playbook.trigger:
            parts.append(f"Trigger: {playbook.trigger}")
        if playbook.rationale:
            parts.append(f"Rationale: {playbook.rationale}")
        return "\n".join(part for part in parts if part)
