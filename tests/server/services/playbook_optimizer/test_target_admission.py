"""Disabled target policies must not start the scheduling infrastructure."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import AgentPlaybook, UserPlaybook
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.playbook_optimizer import PlaybookOptimizationScheduler


@pytest.mark.parametrize(
    "kind,policy",
    [
        ("user_playbook", {"metric_call_budget": 0, "user_playbooks_enabled": True}),
        ("agent_playbook", {"metric_call_budget": 0, "agent_playbooks_enabled": True}),
        ("user_playbook", {"metric_call_budget": 20, "user_playbooks_enabled": False}),
        # A positive work budget alone must leave the canonical user target disabled.
        ("user_playbook", {"metric_call_budget": 20}),
        (
            "agent_playbook",
            {
                "metric_call_budget": 20,
                "agent_playbooks_enabled": False,
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


@pytest.mark.parametrize("kind", ["user_playbook", "agent_playbook"])
def test_production_enqueue_keeps_custom_admission_policy(kind):
    config = PlaybookOptimizerConfig(
        metric_call_budget=20,
        user_playbooks_enabled=True,
        agent_playbooks_enabled=True,
        abort_cooldown_threshold=4,
        cooldown_after_aborts_seconds=60,
    )
    configurator = Mock(
        get_config=lambda: SimpleNamespace(playbook_optimizer_config=config)
    )
    context, client = Mock(org_id="policy-org"), Mock()
    with patch.object(PlaybookOptimizationScheduler, "get_instance") as get_scheduler:
        if kind == "user_playbook":
            service = object.__new__(PlaybookGenerationService)
            service.configurator = configurator
            service.request_context = context
            service.client = client
            service._enqueue_user_playbook_optimization(
                [
                    UserPlaybook(
                        user_playbook_id=17,
                        agent_version="v1",
                        request_id="r",
                        source_interaction_ids=[1],
                    )
                ]
            )
        else:
            aggregator = object.__new__(PlaybookAggregator)
            aggregator.configurator = configurator
            aggregator.request_context = context
            aggregator.client = client
            aggregator._enqueue_playbook_optimization(
                [
                    AgentPlaybook(
                        agent_playbook_id=17,
                        agent_version="v1",
                        content="candidate",
                    )
                ]
            )
    enqueue = get_scheduler.return_value.enqueue.call_args.kwargs
    assert enqueue["org_id"] == "policy-org"
    assert enqueue["target"].kind == kind
    policy = enqueue["policy"]
    assert policy.model_dump() == {
        "abort_cooldown_threshold": 4,
        "cooldown_after_aborts_seconds": 60,
    }
    config.cooldown_after_aborts_seconds = 999
    assert policy.cooldown_after_aborts_seconds == 60
    with pytest.raises(ValueError, match="frozen"):
        policy.cooldown_after_aborts_seconds = 999
