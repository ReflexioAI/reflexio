"""Authoritative historical snapshots for bounded overflow comparisons."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.service_schemas import UserPlaybook

EVIDENCE_SCOPE_POLICY = (
    " Code and test excerpts may be incomplete upstream. Preserve visible branch "
    "conditions, expected outcomes and uncertainty. Names, array positions and "
    "fixture-specific values establish only their shown local case, not universal "
    "requirements. Keep checks for required permissions distinct from negative "
    "controls, and admission distinct from observed execution. A reusable rule "
    "must state the supported prerequisite and scope; narrow or reject it when "
    "those conditions are unavailable. Revision must not reverse or broaden a "
    "supported condition. Before returning a revision, check that content, trigger "
    "and rationale agree on the exact conditions permitting or prohibiting action. "
    "If the original excerpt does not establish those conditions, reject instead "
    "of inventing a permission rule. Do not infer unseen defaults or workflow steps "
    "from names, searches or file inspection. An exclusion does not establish "
    "that unlisted cases are excluded. Require evidence for a conditional rule's "
    "applicability and exceptions; inspecting an example does not prove its steps "
    "were executed or justify prescribing unstated details."
)

HISTORICAL_PLAYBOOK_FIELDS = (
    "user_playbook_id",
    "user_id",
    "playbook_name",
    "request_id",
    "agent_version",
    "created_at",
    "content",
    "trigger",
    "rationale",
    "blocking_issue",
    "source_interaction_ids",
    "status",
    "source",
    "expanded_terms",
    "tags",
    "source_span",
    "notes",
    "reader_angle",
    "merged_into",
    "superseded_by",
)


def historical_playbook_digest(playbook: UserPlaybook) -> str:
    payload = playbook.model_dump(mode="json", include=set(HISTORICAL_PLAYBOOK_FIELDS))
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def historical_playbook_row_digest(row: dict[str, Any]) -> str:
    """Normalize SQL JSON/time types exactly as the existing read models do."""
    values = dict(row)
    for field in ("blocking_issue", "source_interaction_ids", "tags"):
        if isinstance(values[field], str):
            values[field] = json.loads(values[field])
    values["blocking_issue"] = values["blocking_issue"] or None
    values["source_interaction_ids"] = values["source_interaction_ids"] or []
    created = values["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        created = int(created.timestamp())
    values["created_at"] = created
    return historical_playbook_digest(UserPlaybook.model_validate(values))


def remember_historical_playbooks(playbooks: list[UserPlaybook]) -> None:
    from reflexio.server.services.extraction.overflow_coordinator import (
        current_overflow_coordinator,
    )

    coordinator = current_overflow_coordinator.get()
    if coordinator is not None:
        coordinator.remember_historical_playbooks(playbooks)
