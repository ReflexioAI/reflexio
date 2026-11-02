"""Before-provider source evidence for ordinary extraction publication.

Hashes are data, never authority. Request origin is retained so a later real
request cannot silently replace the fallback actually supplied to the model.
"""

from __future__ import annotations

import hashlib
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.input_preparation import source_witness_bytes
from reflexio.server.services.storage.storage_base import AgentRunRecord, BaseStorage
from reflexio.server.work_scope import current_project_id


def fallback_source_request(run: AgentRunRecord, first: Interaction) -> Request:
    """Exactly the historical resumed-input reconstruction, with a frozen anchor."""
    return Request(
        request_id=first.request_id,
        user_id=first.user_id,
        created_at=first.created_at,
        source=run.binding.source or "",
        agent_version=run.binding.agent_version or "",
        session_id=first.request_id,
    )


def capture_source_inputs(
    storage: BaseStorage,
    run: AgentRunRecord,
    models: list[RequestInteractionDataModel],
) -> dict[str, Any]:
    """Freeze the exact complete prompt sources; never reread them after compute."""
    expected = run.binding.source_interaction_ids
    if (
        run.binding.org_id != storage.org_id
        or not run.binding.user_id
        or not expected
        or any(
            type(identifier) is not int or identifier <= 0 for identifier in expected
        )
        or len(set(expected)) != len(expected)
    ):
        raise InputBudgetConfigurationError("Extraction source binding is incomplete")
    sources: dict[str, Any] = {}
    for model in models:
        # The outer session_id is a presentation/grouping label (some loaders
        # use request IDs); Request itself owns persisted session qualifiers.
        if not model.interactions or model.request.user_id != run.binding.user_id:
            raise InputBudgetConfigurationError("Extraction source request changed")
        first = min(
            model.interactions, key=lambda item: (item.created_at, item.interaction_id)
        )
        if model._fallback_request and model.request != fallback_source_request(
            run, first
        ):
            raise InputBudgetConfigurationError("Extraction fallback binding changed")
        for item in model.interactions:
            identifier = str(item.interaction_id)
            if (
                type(item.interaction_id) is not int
                or item.interaction_id <= 0
                or identifier in sources
                or item.user_id != run.binding.user_id
                or item.request_id != model.request.request_id
            ):
                raise InputBudgetConfigurationError(
                    "Extraction source interaction changed"
                )
            sources[identifier] = {
                "digest": hashlib.sha256(
                    source_witness_bytes(model.request, item)
                ).hexdigest(),
                "origin": "fallback" if model._fallback_request else "request",
                **(
                    {"fallback_first_interaction_id": first.interaction_id}
                    if model._fallback_request
                    else {}
                ),
            }
    if set(sources) != {str(identifier) for identifier in expected}:
        raise InputBudgetConfigurationError("Extraction source coverage changed")
    return {"project_id": current_project_id() or "", "sources": sources}


def require_source_inputs(frame: Any, run: AgentRunRecord) -> dict[str, dict[str, Any]]:
    """Reject incomplete/historical proof rather than inventing current evidence."""
    expected = run.binding.source_interaction_ids
    if (
        not expected
        or any(type(i) is not int or i <= 0 for i in expected)
        or len(set(expected)) != len(expected)
        or not isinstance(frame, dict)
        or set(frame) != {"project_id", "sources"}
        or frame["project_id"] != (current_project_id() or "")
        or not isinstance(frame["sources"], dict)
        or set(frame["sources"]) != {str(i) for i in expected}
    ):
        raise InputBudgetConfigurationError("Committed output source proof unavailable")
    sources = frame["sources"]
    for value in sources.values():
        if not isinstance(value, dict):
            raise InputBudgetConfigurationError("Committed output source proof invalid")
        origin = value.get("origin")
        fields = {"digest", "origin"}
        if origin == "fallback":
            fields.add("fallback_first_interaction_id")
        digest = value.get("digest")
        if (
            not isinstance(origin, str)
            or origin not in {"request", "fallback"}
            or set(value) != fields
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise InputBudgetConfigurationError("Committed output source proof invalid")
        if origin == "fallback":
            anchor = value["fallback_first_interaction_id"]
            anchor_value = sources.get(str(anchor))
            if (
                type(anchor) is not int
                or not isinstance(anchor_value, dict)
                or anchor_value.get("origin") != "fallback"
                or anchor_value.get("fallback_first_interaction_id") != anchor
            ):
                raise InputBudgetConfigurationError(
                    "Committed fallback source proof invalid"
                )
    return sources
