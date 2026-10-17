"""Disabled target policies must not start the scheduling infrastructure."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.playbook_optimizer import PlaybookOptimizationScheduler


@pytest.mark.parametrize(
    "kind,policy",
    [
        ("user_playbook", {"enabled": False, "user_playbooks_enabled": True}),
        ("user_playbook", {"enabled": True, "user_playbooks_enabled": False}),
        (
            "user_playbook",
            {
                "enabled": True,
                "optimize_user_playbooks": True,
                "auto_update_user_playbooks": False,
            },
        ),
        (
            "agent_playbook",
            {
                "enabled": True,
                "optimize_agent_playbooks": True,
                "auto_update_pending_agent_playbooks": False,
            },
        ),
    ],
)
def test_disabled_enqueue_does_not_construct_scheduler(kind, policy):
    config = PlaybookOptimizerConfig.model_validate(policy)
    configurator = Mock(
        get_config=lambda: SimpleNamespace(playbook_optimizer_config=config)
    )
    with patch.object(PlaybookOptimizationScheduler, "get_instance") as scheduler:
        if kind == "user_playbook":
            service = object.__new__(PlaybookGenerationService)
            service.configurator = configurator
            service._enqueue_user_playbook_optimization([Mock()])
        else:
            aggregator = object.__new__(PlaybookAggregator)
            aggregator.configurator = configurator
            aggregator._enqueue_playbook_optimization([Mock()])
    scheduler.assert_not_called()
