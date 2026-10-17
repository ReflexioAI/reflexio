"""Helpers for constructing durable extraction agent run records."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from typing import Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction
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


def source_interaction_digest(interaction: Interaction) -> str:
    """Hash the captured source, excluding derived storage-only values."""
    return build_scope_hash(
        interaction.model_dump(mode="json", exclude={"embedding", "token_count"})
    )


def effective_frozen_extractor_config(frozen: dict[str, Any]) -> dict[str, Any]:
    """Apply only the two behavior-preserving deployed schema transitions.

    Do not round-trip this witness through a model: that would silently accept
    unknown retired fields, changed defaults, or other future schema changes.
    """
    effective = deepcopy(frozen)
    effective.pop("extractor_name", None)
    dedup = effective.get("deduplication_config")
    if isinstance(dedup, dict) and "strategy" not in dedup:
        dedup["strategy"] = "legacy"
    return effective


def frozen_extractor_config_identity(config: BaseModel, frozen: dict[str, Any]) -> str:
    """Reproduce the legacy v1 prior-answer key after frozen binding validation.

    Pydantic's representation was part of that key. The retired root label was
    first; the newly added nested strategy must be absent for older admissions.
    Only the lookup identity uses this representation, never requests or policy.
    """
    fields = []
    if "extractor_name" in frozen:
        fields.append(f"extractor_name={frozen['extractor_name']!r}")
    for name, value in config.__repr_args__():
        if name == "deduplication_config" and isinstance(value, BaseModel):
            dedup = frozen.get("deduplication_config")
            if isinstance(dedup, dict) and "strategy" not in dedup:
                args = ", ".join(
                    f"{key}={item!r}"
                    for key, item in value.__repr_args__()
                    if key != "strategy"
                )
                fields.append(f"{name}={type(value).__name__}({args})")
                continue
        fields.append(f"{name}={value!r}")
    return " ".join(fields)


def matches_frozen_extractor_binding(
    prior: AgentRunRecord, expected: AgentRunRecord, frozen_config: dict[str, Any]
) -> bool:
    """Accept exact bindings or the two known inert schema transitions.

    The compatibility case requires the immutable window policy and saved run
    snapshot to witness the original hash. It never rewrites historical state.
    Ad-hoc resumes without a frozen window policy must keep their strict check.
    """
    if build_scope_hash(asdict(prior.binding)) == build_scope_hash(
        asdict(expected.binding)
    ):
        return True
    if prior.binding.extractor_kind not in {"profile", "playbook"}:
        return False
    saved = prior.generation_request_snapshot.get("extractor_config")
    current = expected.generation_request_snapshot.get("extractor_config")
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    original_hash = prior.binding.extractor_config_hash
    if (
        build_scope_hash(saved) != original_hash
        or build_scope_hash(frozen_config) != original_hash
    ):
        return False
    normalized = effective_frozen_extractor_config(saved)
    current_hash = build_scope_hash(current)
    if (
        build_scope_hash(normalized) != current_hash
        or expected.binding.extractor_config_hash != current_hash
    ):
        return False
    compatible = replace(expected.binding, extractor_config_hash=original_hash)
    return build_scope_hash(asdict(prior.binding)) == build_scope_hash(
        asdict(compatible)
    )


def build_extractor_agent_run_record(
    *,
    org_id: str,
    extractor_kind: str,
    user_id: str,
    agent_version: str | None,
    source: str | None,
    request_interaction_data_models: list[RequestInteractionDataModel],
    extractor_config: BaseModel,
    service_config: Any,
    agent_context: str,
    output_schema_name: str | None = None,
    generation_request_id: str | None = None,
    request_id: str | None = None,
) -> AgentRunRecord:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("Durable extraction runs require a non-empty user_id")
    if any(
        data_model.request.user_id != user_id
        or any(
            interaction.user_id != user_id for interaction in data_model.interactions
        )
        for data_model in request_interaction_data_models
    ):
        raise ValueError(
            "Durable extraction run source evidence must belong to its user_id"
        )
    if generation_request_id is not None:
        if request_id is not None and request_id != generation_request_id:
            raise TypeError(
                "generation_request_id and request_id must match when both are provided"
            )
    elif request_id is not None:
        generation_request_id = request_id
    else:
        raise TypeError("generation_request_id is required")

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
            # The durable storage boundary still uses the legacy ``request_id``
            # field name for run provenance.
            request_id=generation_request_id,
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
            # Snapshot payloads also keep the legacy ``request_id`` key so
            # existing storage readers do not need a migration.
            "request_id": generation_request_id,
            "source": source,
            "source_interaction_ids": source_interaction_ids,
            "source_interaction_digests": {
                str(interaction.interaction_id): source_interaction_digest(interaction)
                for data_model in request_interaction_data_models
                for interaction in data_model.interactions
                if interaction.interaction_id is not None
            },
            "session_count": len(request_interaction_data_models),
            "extractor_config": extractor_config_snapshot,
            # Resume/finalization must parse the committed payload with the
            # schema selected when this run started, even if prompt activation
            # changes before a pending tool call is resolved.
            "output_schema_name": output_schema_name,
        },
        service_config_snapshot=_jsonable(service_config),
        agent_context_snapshot=agent_context,
    )
