"""Conversation-only proposals; full source remains available to integration."""

from __future__ import annotations

from typing import Any, cast

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.profile.components.extractor import (
    ProfileExtractor,
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.learning.client_policy import ComponentClient
from reflexio.server.services.profile.learning.compiler import (
    BINDING_GUIDANCE,
    TEMPORAL_GUIDANCE,
)
from reflexio.server.services.profile.learning.linked_plan import EXTRACTION_GUIDANCE


def collect_review_source(
    config: Any, sessions: list[RequestInteractionDataModel]
) -> None:
    """Retain native selected inputs without changing later extractors' selection.

    Durable windows remain authoritative. Manual extractors may select different
    windows; integration sees their lossless union, with conflicting source
    identities rejected instead of silently choosing one version.
    """
    if config.window_interactions is not None:
        return
    groups = {
        (row.session_id, row.request.request_id): row
        for row in config.review_interactions
    }
    for session in sessions:
        key = (session.session_id, session.request.request_id)
        if key not in groups:
            row = session.model_copy(deep=True)
            config.review_interactions.append(row)
            groups[key] = row
            continue
        row = groups[key]
        if row.request != session.request or row.arrival_order != session.arrival_order:
            raise ValueError("Conflicting original source request metadata")
        interactions = {item.interaction_id: item for item in row.interactions}
        for item in session.interactions:
            previous = interactions.get(item.interaction_id)
            if previous is not None and previous != item:
                raise ValueError("Conflicting original source interaction identity")
            if previous is None:
                copy = item.model_copy(deep=True)
                row.interactions.append(copy)
                interactions[item.interaction_id] = copy


class SourceOnlyProfileExtractor(ProfileExtractor):
    def _extraction_definition(self) -> str:
        return profile_extraction_definition(
            self.config.extraction_definition_prompt,
            self.config.should_extract_profile_prompt_override,
        ).strip()

    def _generate_raw_updates_from_sessions(
        self, request_interaction_data_models: Any, existing_profiles: Any
    ) -> Any:
        del existing_profiles
        collect_review_source(self.service_config, request_interaction_data_models)
        return super()._generate_raw_updates_from_sessions(
            request_interaction_data_models, existing_profiles=[]
        )


def profile_extraction_definition(definition: str, condition: str | None = None) -> str:
    """The same measured source-only guidance for native and packaged source."""
    if condition and condition.strip():
        definition += (
            "\n\nProfile extraction eligibility condition configured by the project:\n"
            + condition.strip()
            + "\nIf this conversation does not satisfy the condition, return null profiles. "
            "Assess this within extraction; do not invent facts to satisfy it."
        )
    return (
        definition
        + "\n\n"
        + EXTRACTION_GUIDANCE
        + "\n\n"
        + TEMPORAL_GUIDANCE
        + "\n\n"
        + BINDING_GUIDANCE
    )


def create_extractor(
    service: Any, extractor_config: Any, service_config: Any
) -> SourceOnlyProfileExtractor:
    return SourceOnlyProfileExtractor(
        request_context=service.request_context,
        llm_client=cast(
            LiteLLMClient,
            ComponentClient(
                service.client, "native_profile_extraction", StructuredProfilesOutput
            ),
        ),
        extractor_config=extractor_config,
        service_config=service_config,
        agent_context=service.configurator.get_agent_context(),
    )
