"""Regression tests for managed aggregation effect scopes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import PlaybookAggregatorConfig
from reflexio.server.services.playbook.components import aggregator as aggregator_module
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)


class _OriginalEntryError(RuntimeError):
    pass


class _EntryFailureScope:
    def __init__(self) -> None:
        self.exit_called = False

    def __enter__(self) -> None:
        raise _OriginalEntryError("effect transaction entry failed")

    def __exit__(self, *_args: object) -> None:
        self.exit_called = True
        raise RuntimeError("unentered scope exit masked original error")


def test_effect_scope_entry_failure_preserves_original_error(monkeypatch) -> None:
    storage = MagicMock()
    storage.count_user_playbooks.return_value = 2
    storage.get_agent_playbooks.return_value = []
    user_playbooks = [
        UserPlaybook(
            user_playbook_id=index,
            request_id=f"request-{index}",
            agent_version="v1",
            playbook_name="feedback",
            content=f"content-{index}",
        )
        for index in (1, 2)
    ]
    storage.get_user_playbooks.return_value = user_playbooks
    config = SimpleNamespace(
        user_playbook_extractor_config=SimpleNamespace(
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
                reaggregation_trigger_count=2,
            )
        )
    )
    context = SimpleNamespace(
        org_id="test-org",
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
    )
    coordinator = MagicMock()
    scope = _EntryFailureScope()
    coordinator.apply_scope.return_value = scope
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=context,  # type: ignore[arg-type]
        agent_version="v1",
        effect_coordinator=coordinator,
    )
    aggregator.get_clusters = MagicMock(return_value={0: user_playbooks})  # type: ignore[method-assign]
    state = MagicMock()
    state.get_cluster_fingerprints.return_value = {}
    aggregator._create_state_manager = MagicMock(return_value=state)  # type: ignore[method-assign]
    aggregator._generate_playbooks_with_source_clusters = MagicMock(  # type: ignore[method-assign]
        return_value=[]
    )
    monkeypatch.setattr(aggregator_module, "record_usage_event", lambda **_kw: None)

    with pytest.raises(_OriginalEntryError, match="effect transaction entry failed"):
        aggregator.run(PlaybookAggregatorRequest(agent_version="v1", rerun=True))

    assert scope.exit_called is False
