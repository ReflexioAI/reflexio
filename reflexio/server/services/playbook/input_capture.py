"""Optional, lazy observation of exact pre-decision consolidator inputs.

OSS owns only this contract. A provider selects its authorized scope before
calling the factory; neither observation nor its failure changes reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from reflexio.models.api_schema.service_schemas import Interaction, UserPlaybook
from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.extraction.agent_run_records import (
    source_interaction_digest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceWitness:
    interaction_id: int
    request_id: str
    digest: str


@dataclass(frozen=True)
class ExistingPlaybookWitness:
    user_playbook_id: int
    trigger: str | None
    content: str
    rationale: str | None
    source_interaction_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConsolidatorInputSnapshot:
    """Detached immutable bytes and witnesses, without client or transport state."""

    generation_request_id: str
    user_id: str
    payload_json: str
    payload_sha256: str
    source_witnesses: tuple[SourceWitness, ...]
    existing_playbook_witnesses: tuple[ExistingPlaybookWitness, ...]
    agent_version: str
    model_name: str | None
    runtime_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return a detached persistence envelope with whitelisted identities."""
        return {
            "generation_request_id": self.generation_request_id,
            "user_id": self.user_id,
            "payload_json": self.payload_json,
            "payload_sha256": self.payload_sha256,
            "source_witnesses": [asdict(row) for row in self.source_witnesses],
            "existing_playbook_witnesses": [
                {
                    **asdict(row),
                    "source_interaction_ids": list(row.source_interaction_ids),
                }
                for row in self.existing_playbook_witnesses
            ],
            "identity": {
                "agent_version": self.agent_version,
                "model_name": self.model_name,
                "runtime_sha256": self.runtime_sha256,
            },
        }


class ConsolidatorInputObserver(Protocol):
    def observe(
        self,
        *,
        storage: Any,
        user_id: str,
        generation_request_id: str,
        snapshot_factory: Callable[[], ConsolidatorInputSnapshot],
    ) -> None: ...


CONSOLIDATOR_INPUT_OBSERVER = ServiceKey[ConsolidatorInputObserver](
    "consolidator_input_observer"
)


def observe_consolidator_input(
    *,
    storage: Any,
    user_id: str,
    generation_request_id: str,
    snapshot_factory: Callable[[], ConsolidatorInputSnapshot],
) -> None:
    """Observe synchronously before the outcome; failures never escape to actors."""
    observer = get_service(CONSOLIDATOR_INPUT_OBSERVER)
    if observer is None:
        return
    try:
        observer.observe(
            storage=storage,
            user_id=user_id,
            generation_request_id=generation_request_id,
            snapshot_factory=snapshot_factory,
        )
    except Exception:
        # Exception text and traceback can contain source text or credentials.
        logger.warning("event=consolidator_input_capture_failed")


def build_consolidator_input_snapshot(
    *,
    payload: dict[str, Any],
    interactions: Mapping[int, Interaction],
    existing_rows: list[UserPlaybook],
    user_id: str,
    agent_version: str,
    generation_request_id: str,
    model_name: str | None,
    runtime_path: str,
) -> ConsolidatorInputSnapshot:
    """Serialize only when selected, preserving the actor's native JSON order."""
    text = json.dumps(payload, ensure_ascii=False)
    source_ids = {
        message["turn"]
        for session in payload["sessions"]
        for message in session["messages"]
    }
    return ConsolidatorInputSnapshot(
        generation_request_id=generation_request_id,
        user_id=user_id,
        payload_json=text,
        payload_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_witnesses=tuple(
            SourceWitness(
                interaction_id=source_id,
                request_id=interactions[source_id].request_id,
                digest=source_interaction_digest(interactions[source_id]),
            )
            for source_id in sorted(source_ids)
        ),
        existing_playbook_witnesses=tuple(
            ExistingPlaybookWitness(
                user_playbook_id=row.user_playbook_id,
                trigger=row.trigger,
                content=row.content,
                rationale=row.rationale,
                source_interaction_ids=tuple(row.source_interaction_ids),
            )
            for row in existing_rows
        ),
        agent_version=agent_version,
        model_name=model_name,
        runtime_sha256=hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest(),
    )
