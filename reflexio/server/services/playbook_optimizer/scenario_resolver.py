from __future__ import annotations

from reflexio.models.api_schema.domain import Status
from reflexio.server.services.storage.storage_base import BaseStorage

from .models import ScenarioWindow


class ScenarioResolver:
    """Builds the rollout test set from a playbook's lineage.

    Each agent playbook is generated from a cluster of user playbooks,
    which were themselves extracted from real interactions. To evaluate a
    candidate change, the optimizer needs to replay those original
    interactions — this class walks the lineage and materialises one
    ``ScenarioWindow`` per source user playbook.

    Empty windows (no source interactions, archived rows, etc.) are
    skipped silently — the optimizer treats "no windows" as a reason to
    skip the run.
    """

    def __init__(self, storage: BaseStorage) -> None:
        self.storage = storage

    def for_agent_playbook(self, agent_playbook_id: int) -> list[ScenarioWindow]:
        user_playbook_ids = (
            self.storage.get_source_user_playbook_ids_for_agent_playbook(
                agent_playbook_id
            )
        )
        user_playbooks = self.storage.get_user_playbooks_by_ids_any_user(
            user_playbook_ids, status_filter=None
        )
        windows: list[ScenarioWindow] = []
        for playbook in user_playbooks:
            if not playbook.source_interaction_ids:
                continue
            interactions = self.storage.get_interactions_by_ids(
                playbook.source_interaction_ids
            )
            if interactions:
                windows.append(
                    ScenarioWindow(
                        user_playbook_id=playbook.user_playbook_id,
                        source_interaction_ids=playbook.source_interaction_ids,
                        interactions=interactions,
                    )
                )
        return windows

    def for_user_playbook(self, user_playbook_id: int) -> list[ScenarioWindow]:
        playbook = self.storage.get_user_playbook_by_id(user_playbook_id)
        if (
            playbook is None
            or playbook.status is not None
            or not playbook.source_interaction_ids
        ):
            return []
        interactions = self.storage.get_interactions_by_ids(
            playbook.source_interaction_ids
        )
        if not interactions:
            return []
        return [
            ScenarioWindow(
                user_playbook_id=playbook.user_playbook_id,
                source_interaction_ids=playbook.source_interaction_ids,
                interactions=interactions,
            )
        ]


def is_current_status(status: Status | None) -> bool:
    return status is None
