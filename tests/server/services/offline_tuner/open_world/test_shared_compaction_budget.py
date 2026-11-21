"""Wider offline leaves retain native admission and immutable journal budgets."""

from copy import deepcopy
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
    CompressionNodeInput,
    EvidenceSource,
)
from reflexio.server.services.input_compression_llm import plan_with_client
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    Root,
    SharedReceipt,
    receipt,
)
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


def plan(text, budget, *, context=1_000_000, _admission_version=None):
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
        _admission_version=_admission_version,
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


def _legacy_journal(*, calls=127):
    storage = MemoryCheckpoints()
    legacy = plan(
        "Original evidence",
        CompressionBudget(output_tokens=512),
        _admission_version="v6",
    )
    current = plan("Original evidence", CompressionBudget(output_tokens=512))
    assert legacy.witness != current.witness
    journal = SharedCompactionJournal(
        storage,
        job_id=7,
        fence=1,
        identity="a" * 64,
        before_call=lambda: None,
        now=lambda: 1,
        limit=128,
    )
    root = Root(
        source_digest=legacy.compression.source_digest,
        plan_witness=legacy.witness,
        request_contract_digest=legacy.request_contract_digest,
        limit=128,
        calls=calls,
    )
    storage.rows[(7, "a" * 64)] = receipt("a" * 64, root, complete=False)
    return journal, storage, legacy, current


def test_exact_legacy_witness_keeps_the_spent_physical_allowance():
    journal, storage, legacy, current = _legacy_journal()
    saved = deepcopy(storage.rows)

    assert journal.bind(current, legacy_plan=lambda: legacy) is legacy
    assert storage.rows == saved
    journal.inspect(CompressionNodeInput("b" * 64, "c" * 64))
    journal.guard({"model": "test"})
    root = SharedReceipt.model_validate(storage.rows[(7, "a" * 64)]).payload
    assert isinstance(root, Root) and root.calls == root.limit == 128
    with pytest.raises(ProviderRequestGuardError, match="allowance exhausted"):
        journal.guard({"model": "test"})
    assert SharedReceipt.model_validate(storage.rows[(7, "a" * 64)]).payload == root


@pytest.mark.parametrize(
    "change", ["missing", "refunded", "source", "limit", "reopened"]
)
def test_legacy_rebuild_cannot_recreate_or_weaken_a_saved_root(change):
    journal, storage, legacy, current = _legacy_journal()
    if change == "reopened":
        storage.rows[(7, "a" * 64)]["complete"] = True

    def rebuild():
        row = storage.rows[(7, "a" * 64)]
        if change == "missing":
            del storage.rows[(7, "a" * 64)]
        elif change == "refunded":
            row["payload"]["calls"] -= 1
            row["batches"] -= 1
        elif change == "source":
            row["payload"]["source_digest"] = "f" * 64
        elif change == "limit":
            row["payload"]["limit"] -= 1
        else:
            row["complete"] = False
        return legacy

    with pytest.raises(ProviderRequestGuardError, match="legacy root"):
        journal.bind(current, legacy_plan=rebuild)
    assert journal.plan is None
    if change == "missing":
        assert (7, "a" * 64) not in storage.rows


def test_legacy_rebuild_accepts_only_monotonic_call_advancement():
    journal, storage, legacy, current = _legacy_journal()

    def rebuild():
        storage.rows[(7, "a" * 64)]["payload"]["calls"] = 128
        storage.rows[(7, "a" * 64)]["batches"] = 128
        return legacy

    assert journal.bind(current, legacy_plan=rebuild) is legacy
    journal.inspect(CompressionNodeInput("b" * 64, "c" * 64))
    with pytest.raises(ProviderRequestGuardError, match="allowance exhausted"):
        journal.guard({"model": "test"})


def test_unknown_saved_witness_is_not_a_legacy_admission_override():
    journal, storage, legacy, current = _legacy_journal()
    storage.rows[(7, "a" * 64)]["payload"]["plan_witness"] = "f" * 64
    saved = deepcopy(storage.rows)
    with pytest.raises(ProviderRequestGuardError, match="frozen plan changed"):
        journal.bind(current, legacy_plan=lambda: legacy)
    assert storage.rows == saved and journal.plan is None


def test_new_root_uses_current_plan_without_constructing_legacy():
    journal, storage, _legacy, current = _legacy_journal()
    storage.rows.clear()

    def forbidden():
        pytest.fail("New admission must not build a legacy plan")

    assert journal.bind(current, legacy_plan=forbidden) is current
    root = SharedReceipt.model_validate(storage.rows[(7, "a" * 64)]).payload
    assert (
        isinstance(root, Root)
        and root.plan_witness == current.witness
        and root.calls == 0
    )


@pytest.mark.parametrize(
    ("model", "transport", "native"),
    [
        ("openclaw/anthropic/claude-sonnet-4-6", None, True),
        ("anthropic/claude-sonnet-4-6", "openclaw", True),
        ("openclaw/anthropic/claude-sonnet-4-6", "openai", False),
    ],
)
def test_shared_native_identity_admission_precedes_dispatch_and_receipt_debit(
    monkeypatch, model, transport, native
):
    from copy import deepcopy
    from unittest.mock import Mock

    import litellm

    from reflexio.server.llm import input_budget
    from reflexio.server.services.input_compression import CompressionNodeInput
    from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
        JournaledCompressionClient,
    )

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
    journal.inspect(CompressionNodeInput(key="b" * 64, input_refs_digest="c" * 64))
    saved = deepcopy(storage.rows)
    monkeypatch.setattr(input_budget.litellm, "token_counter", lambda **_: 100)
    monkeypatch.setattr(
        input_budget.litellm,
        "get_model_info",
        lambda *_args, **_kwargs: {
            "max_input_tokens": 1000000,
            "max_output_tokens": 32,
        },
    )
    response = litellm.ModelResponse(
        model="requested-model-echo" if native else "served-http-model",
        choices=[
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    )
    if native:
        response._hidden_params = {"reflexio_provider": "openclaw"}
    dispatch = Mock(return_value=response)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    client = JournaledCompressionClient(
        LiteLLMConfig(model=model, max_tokens=32, fallback_models=["gpt-4o-mini"]),
        journal,
    )
    options = {"custom_llm_provider": transport} if transport else {}
    with journal.provider_scope():
        if native:
            with pytest.raises(ProviderRequestGuardError, match="OpenClaw"):
                client.generate_chat_response_with_provenance(
                    [{"role": "user", "content": "synthetic evidence"}], **options
                )
            dispatch.assert_not_called()
            assert storage.rows == saved
        else:
            result = client.generate_chat_response_with_provenance(
                [{"role": "user", "content": "synthetic evidence"}], **options
            )
            assert result.provenance.model_name == "served-http-model"
            assert dispatch.call_count == 1
