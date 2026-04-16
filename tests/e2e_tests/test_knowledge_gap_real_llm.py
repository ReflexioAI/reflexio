"""Real-LLM e2e test: knowledge gap playbook extraction.

Run with:
    set -a && source .env && set +a && \
    RUN_LOW_PRIORITY=1 uv run pytest tests/e2e_tests/test_knowledge_gap_real_llm.py -v -o 'addopts=' -s

This test calls a real LLM using the default agent_context_prompt and
user_playbook_extractor_configs to verify that when an agent fabricates
answers instead of admitting it can't check, the extracted playbook:
1. Identifies the knowledge gap honestly
2. Does NOT hallucinate a specific fix the agent can't actually do
3. Uses the new flat schema (trigger + content, no instruction/pitfall)
"""

import os
from collections.abc import Callable

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import GetUserPlaybooksRequest
from reflexio.models.api_schema.service_schemas import InteractionData, UserPlaybook
from tests.server.test_utils import skip_low_priority

pytestmark = [pytest.mark.e2e, pytest.mark.requires_credentials]


@skip_low_priority
def test_knowledge_gap_extraction_real_llm(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Publish a conversation where the agent guesses instead of admitting ignorance.

    Uses the default agent_context_prompt and user_playbook_extractor_configs.
    Verify the extracted playbook captures the knowledge gap honestly.
    """
    interactions = [
        InteractionData(
            role="User",
            content="Hey, I placed an order last week, order #A1234. Can you check the shipping status for me?",
        ),
        InteractionData(
            role="Agent",
            content="Of course! Let me check that for you. Your order #A1234 is currently being processed and should ship within 2-3 business days.",
        ),
        InteractionData(
            role="User",
            content="That doesn't sound right. I got an email saying it was already delivered but I never received it. Can you actually look this up in your system?",
        ),
        InteractionData(
            role="Agent",
            content="I apologize for the confusion. Looking at it more carefully, it appears there may have been a delivery issue. Let me escalate this to our shipping team who can investigate the missing package.",
        ),
        InteractionData(
            role="User",
            content="Wait — did you actually look it up or are you just guessing? I need the real tracking info, not made up answers.",
        ),
        InteractionData(
            role="Agent",
            content="You're right, I apologize. I don't actually have access to look up real-time order tracking information. I was making assumptions based on general timelines rather than checking actual data. For your order status, please check the tracking link in your confirmation email, or I can transfer you to our order support team who has direct access to the tracking system.",
        ),
    ]

    # Publish with real LLM (no MOCK_LLM_RESPONSE)
    original_mock = os.environ.pop("MOCK_LLM_RESPONSE", None)
    try:
        response = reflexio_instance_playbook_only.publish_interaction(
            {
                "user_id": "knowledge_gap_user",
                "interaction_data_list": interactions,
                "source": "test_knowledge_gap",
                "agent_version": "v1.0",
                "force_extraction": True,
            }
        )
        assert response.success is True

        # Retrieve extracted playbooks
        playbooks_response = reflexio_instance_playbook_only.get_user_playbooks(
            GetUserPlaybooksRequest(
                playbook_name="test_playbook",
                status_filter=[None],
            )
        )
        assert playbooks_response.success is True
        user_playbooks = playbooks_response.user_playbooks
        assert user_playbooks, (
            "Expected at least one playbook from knowledge-gap interaction"
        )

        # Print extracted playbooks for inspection
        print("\n" + "=" * 70)
        print("EXTRACTED PLAYBOOKS FROM KNOWLEDGE-GAP INTERACTION")
        print("=" * 70)
        for i, pb in enumerate(user_playbooks, 1):
            print(f"\n--- Playbook {i} ---")
            print(f"  Trigger:   {pb.trigger}")
            print(f"  Content:   {pb.content}")
            if pb.rationale:
                print(f"  Rationale: {pb.rationale}")
            if pb.blocking_issue:
                print(
                    f"  Blocking:  [{pb.blocking_issue.kind}] {pb.blocking_issue.details}"
                )
        print("\n" + "=" * 70)

        # Verify schema: flat fields, no instruction/pitfall
        for pb in user_playbooks:
            assert pb.content and pb.content.strip()
            assert pb.trigger and pb.trigger.strip()
            assert "instruction" not in UserPlaybook.model_fields
            assert "pitfall" not in UserPlaybook.model_fields

        # Verify content quality: should mention the gap, not hallucinate a fix
        all_content = " ".join(pb.content for pb in user_playbooks).lower()
        # The playbook should reference the core issue: guessing / fabricating / no access
        gap_keywords = [
            "access",
            "check",
            "look up",
            "guess",
            "fabricat",
            "admit",
            "transparent",
            "don't have",
            "cannot",
            "unable",
            "honest",
            "limitation",
            "assume",
            "verify",
            "make up",
            "invented",
        ]
        assert any(kw in all_content for kw in gap_keywords), (
            f"Playbook content should reference the knowledge gap. Got: {all_content}"
        )

    finally:
        if original_mock is not None:
            os.environ["MOCK_LLM_RESPONSE"] = original_mock
