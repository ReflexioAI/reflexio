"""Consolidator evidence preparation through the shared hierarchical compressor.

The caller supplies already-authorized source excerpts. Their offsets remain
local to that presentation; original canonical turn coordinates stay unchanged.
This module owns no executor, connection, publication or compression algorithm.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from typing import Any

from reflexio.server.env_utils import env_str
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    check_request_input_budget,
    declared_context_limits,
    extraction_provider_guard,
    measure_request_input_tokens,
    request_input_capacity,
)
from reflexio.server.services import input_compression
from reflexio.server.services.extraction.reconciliation_journal import KEY
from reflexio.server.services.input_compression import CompressionBudget
from reflexio.server.services.input_compression_llm import plan_with_client

CONTRACT = "consolidator-source-excerpts-v1"
# Qualification must price compression plus the consumer, not just the packet.
# The shared module already uses this complete-input target for its requests.
INPUT_TARGET = 32768
TEXT_FIELDS = ("content", "agent_response", "expert_feedback")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _sources(payload: dict, scope: dict[str, str], compression: Any) -> list[Any]:
    if not scope.get("org_id") or not scope.get("user_id") or "project_id" not in scope:
        raise ValueError("Explicit organization, project and user scope required")
    rows = [*payload["new"], *payload["existing"], *payload["sessions"]]
    if any(row.get("user_id") != scope["user_id"] for row in rows):
        raise ValueError("Evidence and playbooks must share one explicit owner")
    ids = [row["id"] for row in [*payload["new"], *payload["existing"]]]
    sessions = [row["id"] for row in payload["sessions"]]
    if len(set(ids)) != len(ids) or len(set(sessions)) != len(sessions):
        raise ValueError("Ambiguous captured identities")
    sources, coordinates = [], set()
    for session in payload["sessions"]:
        for message in session["messages"]:
            coordinate = (session["id"], message["turn"])
            if (
                coordinate in coordinates
                or type(message["turn"]) is not int
                or message["turn"] <= 0
            ):
                raise ValueError("Unique positive persisted turn coordinates required")
            coordinates.add(coordinate)
            for field in TEXT_FIELDS:
                if field not in message:
                    continue
                if not isinstance(message[field], str):
                    raise ValueError("Evidence must be text")
                sources.append(
                    compression.EvidenceSource(
                        source_id=f"evidence:{len(sources)}",
                        request_id=session["id"],
                        interaction_id=message["turn"],
                        field=field,
                        speaker=message["role"],
                        source_order=len(sources),
                        text=message[field],
                        request_source=message.get("request_source", ""),
                        created_at=message.get("created_at"),
                    )
                )
    return sources


def _render(payload: dict, originals: list[Any], packet: Any) -> dict:
    """Validate selected ranges and retain their original canonical turn IDs."""
    by_origin = {(s.request_id, s.interaction_id, s.field): s for s in originals}
    selected: dict[tuple[str, int, str], list[Any]] = defaultdict(list)
    last_order, seen = (-1, -1), set()
    for source in packet.evidence:
        key = (source.request_id, source.interaction_id, source.field)
        original = by_origin.get(key)
        if original is None:
            raise ValueError("Selected evidence has a foreign origin")
        end = source.start + len(source.text)
        allowed_ids = {f"{original.source_id}:{source.start}:{end}"}
        if source.start == 0 and source.text == original.text:
            allowed_ids.add(original.source_id)
        expected = original.model_copy(
            update={
                "source_id": source.source_id,
                "start": source.start,
                "text": original.text[source.start : end],
            }
        )
        order = (source.source_order, source.start)
        if (
            source != expected
            or source.start < 0
            or end > len(original.text)
            or source.source_id not in allowed_ids
            or source.source_id in seen
        ):
            raise ValueError("Selected evidence changed its source text or attribution")
        if order < last_order or (
            selected[key]
            and source.start < selected[key][-1].start + len(selected[key][-1].text)
        ):
            raise ValueError("Selected evidence overlaps or changes chronology")
        selected[key].append(source)
        seen.add(source.source_id)
        last_order = order
    if list(packet.evidence) == originals:
        # A fitting source presentation needs neither new framing nor rewritten
        # quotation carriers. Preserve the exact existing consumer input.
        return deepcopy(payload)
    result = deepcopy(payload)
    result["context_is_reference_excerpt"] = True
    result["evidence_selection"] = {
        "contract": CONTRACT,
        "policy": (
            "Only copied source text is evidence. Gaps and omitted context are unknown, "
            "never contradictions or permission. Range offsets are local to the evidence excerpt."
        ),
        "source_sha256": _digest(payload),
    }
    sessions = []
    for session in result["sessions"]:
        messages = []
        for message in session["messages"]:
            ranges = []
            for field in TEXT_FIELDS:
                spans = selected.get((session["id"], message["turn"], field), [])
                message.pop(field, None)
                if spans:
                    # Explicit separators prevent separated evidence becoming a
                    # fabricated contiguous quotation. They are host framing.
                    message[field] = (
                        "\n\n[gap or boundary between source excerpts]\n\n".join(
                            s.text for s in spans
                        )
                    )
                    ranges.extend(
                        {
                            "field": field,
                            "start_in_evidence_excerpt": s.start,
                            "end": s.start + len(s.text),
                        }
                        for s in spans
                    )
            if ranges:
                message.setdefault("content", "")
                message["selected_excerpt_ranges"] = ranges
                messages.append(message)
        if messages:
            session["messages"] = messages
            sessions.append(session)
    result["sessions"] = sessions
    return result


def _limits(actor: Any) -> dict[str, int]:
    if actor.coordinator is not None:
        return actor.coordinator.context_limits
    return declared_context_limits(
        env_str("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", default="{}")
    )


def admit_request(actor: Any, params: dict[str, Any]) -> None:
    """The actual dispatch, including tool growth/repairs, remains authoritative."""
    check_request_input_budget(
        params, context_limits=_limits(actor), prompt_target=INPUT_TARGET
    )


def _validate_source_owner(actor: Any) -> None:
    coordinator = actor.coordinator
    if actor.source_scope != {
        "org_id": coordinator.storage.org_id,
        "project_id": coordinator.window.project_id,
        "user_id": coordinator.window.user_id,
    }:
        raise InputBudgetConfigurationError("Consolidation source owner changed")
    coordinator.validate_current_parent()


def _budgets(actor: Any) -> CompressionBudget:
    # The client builder owns model-specific output defaults and custom routes.
    consumer = actor.client._build_completion_params(messages=[], model=actor.model)[0]
    compressor = actor.client._build_completion_params(
        messages=[],
        max_tokens=actor.client.config.max_tokens or 8192,
        fallback_models=[],
    )[0]
    output_target = request_input_capacity(
        consumer, context_limits=_limits(actor), prompt_target=INPUT_TARGET
    )
    input_target = request_input_capacity(
        compressor, context_limits=_limits(actor), prompt_target=INPUT_TARGET
    )
    if min(output_target, input_target) <= 0:
        raise InputBudgetConfigurationError("No consolidation input capacity remains")
    return CompressionBudget(
        output_tokens=output_target,
        input_tokens=input_target,
        section_tokens=input_target,
    )


def prepare_context(
    actor: Any, payload: dict[str, Any], references: dict[tuple[str, int], Any]
) -> tuple[dict[str, Any], dict[tuple[str, int], Any]]:
    """Reuse one immutable selected presentation for the entire audit loop."""
    from reflexio.server.services.playbook.components.reconciliation import (
        Audit,
        Decisions,
        project,
    )

    if actor.source_scope is None:
        raise InputBudgetConfigurationError(
            "Authorized consolidation source scope required"
        )
    payload = deepcopy(payload)
    sources = _sources(payload, actor.source_scope, input_compression)
    coordinator = actor.coordinator
    if coordinator is not None:
        _validate_source_owner(actor)
    tools = actor._tools(
        {name: [] for name in ("search_profiles", "search_playbooks", "search_traces")}
    )

    def render_cost(packet: Any) -> int:
        view = _render(payload, sources, packet)
        # Fixed envelopes are priced with the same messages and parameter builder
        # as dispatch. The unknown draft/revision can still grow: admit_request
        # checks each full actual request rather than treating this as a proof.
        prior = {row["id"]: row for row in view["existing"]}
        audit = {**view, "proposed_memory": project(Decisions(decisions=[]), prior)}
        return max(
            measure_request_input_tokens(
                actor.client._build_completion_params(
                    messages=actor._messages(stage, data, schema),
                    model=actor.model,
                    tools=tools,
                )[0]
            )
            for stage, data, schema in (
                ("reconcile-method", view, Decisions),
                ("reconcile-audit", audit, Audit),
            )
        )

    guard_token = extraction_provider_guard.set(actor.provider_guard)
    try:
        plan = plan_with_client(
            sources,
            scope={
                **actor.source_scope,
                "consumer": CONTRACT,
                "payload": _digest(payload),
            },
            policy="extraction",
            budget=_budgets(actor),
            client=actor.client,
            context_limits=_limits(actor),
            measure_output=render_cost,
            output_contract=CONTRACT,
        )
        if coordinator is not None:
            checkpoint = coordinator.checkpoint
            context = checkpoint[KEY]
            witness = {
                "witness": plan.witness,
                "source_digest": plan.compression.source_digest,
                "request_contract_digest": plan.request_contract_digest,
            }
            if context["plan"] is not None and context["plan"] != witness:
                raise InputBudgetConfigurationError(
                    "Consolidation compression plan changed"
                )
            if context["plan"] is None:
                context["plan"] = witness
                coordinator.save_checkpoint(checkpoint)
        active: Any = None
        first_attempt = 0

        def inspect(node: Any) -> None:
            nonlocal active, first_attempt
            active = node
            if coordinator is not None:
                first_attempt = len(coordinator.checkpoint[KEY]["attempt_ids"])

        def load(key: str) -> dict[str, Any] | None:
            if coordinator is None:
                return None
            node = coordinator.checkpoint[KEY]["nodes"].get(key)
            if node is None:
                return None
            if (
                active is None
                or active.key != key
                or node["input_refs_digest"] != active.input_refs_digest
                or node["request_contract_digest"] != plan.request_contract_digest
            ):
                raise InputBudgetConfigurationError(
                    "Consolidation selection binding changed"
                )
            return node["output"]

        def save(key: str, output: dict[str, Any]) -> None:
            if coordinator is None:
                return
            if active is None or active.key != key:
                raise InputBudgetConfigurationError(
                    "Consolidation selection lacks its node"
                )
            coordinator.save_reconciliation_selection(
                key,
                {
                    "input_refs_digest": active.input_refs_digest,
                    "request_contract_digest": plan.request_contract_digest,
                    "attempt_ids": coordinator.checkpoint[KEY]["attempt_ids"][
                        first_attempt:
                    ],
                    "output": output,
                },
            )

        def reserve_consumer() -> None:
            if actor.calls >= actor.max_calls - 2:
                raise InputBudgetConfigurationError(
                    "Compression must reserve draft and audit calls"
                )

        compressed = plan.execute(
            before_call=reserve_consumer, load=load, save=save, inspect_node=inspect
        )
    finally:
        extraction_provider_guard.reset(guard_token)
    if coordinator is not None:
        coordinator.validate_current_parent()
    view = _render(payload, sources, compressed.packet)
    selected: dict[tuple[str, int], list[str]] = defaultdict(list)
    for source in compressed.packet.evidence:
        selected[(source.request_id, source.interaction_id)].append(source.text)
    visible = {
        key: replace(references[key], evidence_texts=tuple(texts))
        for key, texts in selected.items()
    }
    return view, visible
