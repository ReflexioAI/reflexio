"""Helpers for constructing durable extraction agent run records."""

from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    build_scope_hash,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def extract_source_interaction_ids(
    request_interaction_data_models: list[RequestInteractionDataModel],
) -> list[int]:
    return [
        interaction.interaction_id
        for data_model in request_interaction_data_models
        for interaction in data_model.interactions
        if interaction.interaction_id is not None
    ]


def build_extractor_agent_run_record(
    *,
    org_id: str,
    extractor_kind: str,
    user_id: str | None,
    request_id: str,
    agent_version: str | None,
    source: str | None,
    request_interaction_data_models: list[RequestInteractionDataModel],
    extractor_config: BaseModel,
    service_config: Any,
    agent_context: str,
) -> AgentRunRecord:
    source_interaction_ids = extract_source_interaction_ids(
        request_interaction_data_models
    )
    extractor_config_snapshot = extractor_config.model_dump(mode="json")

    return AgentRunRecord(
        id=f"ar_{uuid.uuid4().hex}",
        binding=AgentBinding(
            org_id=org_id,
            extractor_kind=extractor_kind,
            user_id=user_id,
            request_id=request_id,
            agent_version=agent_version,
            source=source,
            source_interaction_ids=source_interaction_ids,
            window_start_interaction_id=(
                min(source_interaction_ids) if source_interaction_ids else None
            ),
            window_end_interaction_id=(
                max(source_interaction_ids) if source_interaction_ids else None
            ),
            extractor_config_hash=build_scope_hash(extractor_config_snapshot),
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={
            "request_id": request_id,
            "source": source,
            "source_interaction_ids": source_interaction_ids,
            "session_count": len(request_interaction_data_models),
            "extractor_config": extractor_config_snapshot,
        },
        service_config_snapshot=_jsonable(service_config),
        agent_context_snapshot=agent_context,
    )
