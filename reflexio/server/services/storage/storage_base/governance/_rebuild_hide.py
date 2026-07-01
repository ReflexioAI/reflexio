from __future__ import annotations

from abc import ABC, abstractmethod


class RebuildHideMixin(ABC):
    """Backend-neutral governance rebuild-hide store contract.

    Extracted verbatim from ``_governance.py`` (the RebuildHide bucket): the two
    public methods (``hide_governance_agent_playbooks_for_rebuild``,
    ``apply_governance_agent_playbook_rebuild``). The residual ``GovernanceMixin``
    ABC stays composed alongside this mixin.
    """

    @abstractmethod
    def hide_governance_agent_playbooks_for_rebuild(self, purge_id: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def apply_governance_agent_playbook_rebuild(
        self,
        purge_id: str,
        agent_playbook_id: int,
        remaining_source_windows: list[dict[str, object]],
        content: str | None,
        trigger: str | None,
        rationale: str | None,
        blocking_issue: dict[str, object] | None,
        expanded_terms: str | None,
        tags: list[str] | None,
    ) -> None:
        raise NotImplementedError
