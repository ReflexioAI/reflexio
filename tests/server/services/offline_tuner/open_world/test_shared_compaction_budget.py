"""Wider offline leaves retain native admission and immutable journal budgets."""

from dataclasses import replace
from hashlib import sha256

import pytest

from reflexio.server.llm.input_budget import InputBudgetExceededError
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ProviderRequestGuardError,
)
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
    EvidenceSource,
)
from reflexio.server.services.input_compression_llm import plan_with_client
from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
    SharedCompactionJournal,
)
from reflexio.server.services.offline_tuner.open_world.shared_compaction_preparer import (
    _compression_budget,
)
from tests.server.services.offline_tuner.open_world.test_shared_compaction import (
    MemoryCheckpoints,
)
from tests.server.services.offline_tuner.open_world.test_shared_compaction import (
    no_network as no_network,
)


@pytest.mark.parametrize(
    ("context", "expected"),
    [(1_000_000, 131_072), (204_800, 102_400), (32_768, 16_384)],
)
def test_task_budget_is_context_bounded_without_changing_shared_limits(
    context, expected
):
    budget = _compression_budget(1234, context)
    assert budget.input_tokens == budget.section_tokens == expected
    assert budget.output_tokens == 1234
    assert (budget.max_sections, budget.max_calls, budget.max_source_bytes) == (
        48,
        128,
        16 * 1024 * 1024,
    )
    assert CompressionBudget(output_tokens=1234).input_tokens == 32_768


def plan(text, budget, *, context=1_000_000):
    return plan_with_client(
        [
            EvidenceSource(
                source_id="source",
                request_id="request",
                interaction_id=1,
                field="content",
                speaker="user",
                source_order=0,
                text=text,
            )
        ],
        scope={"org_id": "org", "project_id": "project", "user_id": "user"},
        policy="offline_tuning",
        budget=budget,
        client=LiteLLMClient(
            LiteLLMConfig(
                model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[]
            )
        ),
        context_limits={"minimax/MiniMax-M3": context},
    )


def test_wider_leaves_cover_unique_source_under_the_same_section_and_call_caps():
    # Unique text prevents duplicate-passage savings from hiding the boundary.
    # A smaller section count reproduces the production limit without a private
    # multi-megabyte fixture. Actual native leaf request checks remain enabled.
    text = "".join(sha256(str(i).encode()).hexdigest() for i in range(2048))
    old = replace(CompressionBudget(output_tokens=512), max_sections=2, max_calls=3)
    with pytest.raises(CompressionError, match="source section budget"):
        plan(text, old)
    new = replace(_compression_budget(512, 1_000_000), max_sections=2, max_calls=3)
    prepared = plan(text, new)
    assert not prepared.compression.passthrough
    assert 0 < len(prepared.compression.section_digests) <= 2
    assert max(prepared.compression.section_input_tokens) <= new.input_tokens


def test_task_ceiling_cannot_bypass_native_context_and_output_reservation():
    text = "".join(sha256(str(i).encode()).hexdigest() for i in range(512))
    with pytest.raises(InputBudgetExceededError):
        plan(text, _compression_budget(512, 1_000_000), context=16_384)


def test_budget_change_cannot_rebind_an_existing_journal():
    storage = MemoryCheckpoints()
    journal = SharedCompactionJournal(
        storage,
        job_id=7,
        fence=1,
        identity="a" * 64,
        before_call=lambda: None,
        now=lambda: 1,
        limit=128,
    )
    journal.bind(plan("Original evidence", CompressionBudget(output_tokens=512)))
    saved = dict(storage.rows)
    with pytest.raises(ProviderRequestGuardError, match="frozen plan changed"):
        journal.bind(plan("Original evidence", _compression_budget(512, 1_000_000)))
    assert storage.rows == saved
