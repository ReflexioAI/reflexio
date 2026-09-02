"""What does consolidation do when evidence contradicts a stale avoidance rule?

Nobody currently knows, and the answer decides whether the self-sealing-rule design
needs a "receiver" at all. See `docs/superpowers/specs/2026-08-29-self-sealing-avoidance-rules.md`.

The gap: a playbook says "don't call API A, it's broken". A later session shows API A
working. For that correction to land, some pipeline has to retire or amend the stale
rule. Every other candidate is already ruled out -- the review service re-reads a frozen
evidence window, the offline tuner is dark in every deployment mode, aggregation retires
only empty clusters, GC only ages tombstones. Consolidation is the last one standing.

It has exactly four decisions (`components/consolidator.py:277-385`):

    unify          -- can archive the incumbent (`archive_existing_ids`). CLOSES the loop.
    reject_new     -- "the new candidate is redundant; an existing row supersedes it".
                      The correction is DISCARDED and the stale rule survives.
    differentiate  -- both survive; the agent now holds contradictory guidance.
    independent    -- both survive.

And the `unify` docstring instructs the model NOT to merge rules that contradict on the
same situation -- which is exactly what a correction does. So the one decision that would
close the loop is the one the prompt discourages.

This probe measures which decision actually fires. The assertion is deliberately narrow:
only `reject_new` is treated as a failure, because discarding a correction as "redundant"
is wrong under any reading and would be a defect wider than this design. `differentiate`
and `independent` are recorded, not failed -- they leave the loop open, which is a design
finding rather than a bug in consolidation.

Uses `_consolidation_decisions`, which the component documents as the render+call+parse
seam "for downstream eval providers" -- no hybrid search, no apply, no storage.

Costs real API calls. Lives under `e2e_tests/` because that is the only path
`llm_mock._is_e2e_test_run` exempts from the session-wide `litellm.completion` patch;
from anywhere else the model answers from the mock and the parse fails in ways that look
like a consolidation regression and are not.

Run:

    set -a && source .env && set +a && \\
    RUN_LOW_PRIORITY=1 uv run pytest \\
      tests/e2e_tests/test_consolidation_stale_rule_correction_real_llm.py \\
      -o 'addopts=' -n 0 -v -s
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.test_support.llm_credentials import (
    real_generation_provider,
    real_provider_key,
)
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from tests.server.test_utils import skip_low_priority

# Same gate as the reviewer manifest probe: provider-agnostic, because the model comes
# from `resolve_model_name(ModelRole.GENERATION)`. `real_generation_provider` rather than
# `os.getenv` -- the credential floor pins a placeholder key when none is set, and a
# plain getenv check would let this run against a credential that authenticates nothing.
_CONSOLIDATION_CAPABLE = frozenset({"openai", "anthropic", "minimax"})

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_credentials,
]


@contextmanager
def _only_real_provider_keys():
    """Hide provider keys that are absent, empty, or the placeholder, for one call.

    Passing a model to ``LiteLLMConfig`` does NOT pin it: ``_resolve_primary_model``
    (``_litellm_text_generation.py:646``) overrides the config model whenever a
    ``model_role`` is supplied, and the consolidator always supplies
    ``ModelRole.GENERATION``. Resolution then auto-detects from the environment -- where
    a bare ``ANTHROPIC_API_KEY=`` still counts as a DETECTED provider.

    On this machine that produced `claude-sonnet-5` with no key behind it, and every case
    failed with `x-api-key header is required` -- a failure that reads like a
    consolidation regression and is not. Removing the non-real keys for the duration of
    the call makes auto-detection agree with the credential actually held.

    ``patch.dict`` restores the original environment on exit, so the pops are scoped.
    """
    with patch.dict(os.environ, {}, clear=False):
        for env_var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            if not real_provider_key(env_var):
                os.environ.pop(env_var, None)
        yield


def _require_real_provider() -> None:
    """Skip unless a consolidation-capable provider holds a real key.

    Checked at call time, never at import: credentials are not settled when this module
    is imported -- the session fixtures load ``.env`` afterwards -- so a module-level
    check reads the ambient environment and picks the wrong provider.
    """
    if real_generation_provider(_CONSOLIDATION_CAPABLE) is None:
        pytest.skip(
            "no real key for a consolidation-capable provider "
            "(openai/anthropic/minimax) is set"
        )

_INCUMBENT_ID = 5512


def _incumbent(rationale: str) -> UserPlaybook:
    """The stale avoidance rule, as extraction would have written it."""
    return UserPlaybook(
        user_playbook_id=_INCUMBENT_ID,
        user_id="probe-user",
        agent_version="v1",
        request_id="req-incumbent",
        playbook_name="probe",
        content="Do not call billing.export_invoice; build the PDF manually instead.",
        trigger="user asks to export an invoice",
        rationale=rationale,
        created_at=1_772_409_600,  # 2026-03-02
        source="api",
        source_interaction_ids=[1],
    )


def _correction(content: str, rationale: str) -> UserPlaybook:
    """The candidate a later successful session would produce."""
    return UserPlaybook(
        user_playbook_id=0,
        user_id="probe-user",
        agent_version="v1",
        request_id="req-correction",
        playbook_name="probe",
        content=content,
        trigger="user asks to export an invoice",
        rationale=rationale,
        created_at=1_788_000_000,  # ~2026-08
        source="api",
        source_interaction_ids=[2],
    )


# Three shapes of the same correction. They are separated because the decision may well
# hinge on how explicitly the candidate contradicts the incumbent, and "which shape gets
# discarded" is the actionable output -- an aggregate would hide it.
_CASES = [
    pytest.param(
        "undated_incumbent",
        _incumbent("The invoice export returned a server error."),
        _correction(
            "Use billing.export_invoice to export an invoice as a PDF.",
            "The invoice export completed successfully and produced the PDF.",
        ),
        id="undated_incumbent",
    ),
    pytest.param(
        "dated_incumbent",
        _incumbent(
            "As of 2026-03-02, billing.export_invoice returned a server error."
        ),
        _correction(
            "Use billing.export_invoice to export an invoice as a PDF.",
            "The invoice export completed successfully and produced the PDF.",
        ),
        id="dated_incumbent",
    ),
    pytest.param(
        "explicit_contradiction",
        _incumbent(
            "As of 2026-03-02, billing.export_invoice returned a server error."
        ),
        _correction(
            "Use billing.export_invoice to export an invoice as a PDF; the earlier "
            "server error no longer occurs.",
            "The invoice export that previously failed now completes successfully.",
        ),
        id="explicit_contradiction",
    ),
]


def _decide(incumbent: UserPlaybook, correction: UserPlaybook):
    """Run the real consolidation decision step over one NEW + one EXISTING row."""
    _require_real_provider()
    assert_litellm_unpatched()

    request_context = MagicMock()
    request_context.prompt_manager = PromptManager()
    # `api_key_config` MUST be a real None, not a MagicMock attribute.
    # `BaseDeduplicator.__init__` (deduplication_utils.py:190-195) resolves its OWN
    # model via
    #     resolve_model_name(GENERATION, api_key_config=
    #         request_context.configurator.get_config().api_key_config)
    # A MagicMock answers every attribute truthily, so resolution reads it as a fully
    # populated APIKeyConfig, decides anthropic is configured, and returns
    # `claude-sonnet-5` -- with no key behind it. Every case then fails on auth, which
    # looks like a consolidation regression and is not. Pinning this to None lets
    # resolution fall through to real env auto-detection.
    request_context.configurator.get_config.return_value.api_key_config = None

    # Both the client and the call live inside the isolation context: the config's
    # ``model`` is required but is overridden by ``model_role`` at call time, so what
    # actually decides the model is the environment auto-detection at that moment.
    with _only_real_provider_keys():
        model = resolve_model_name(ModelRole.GENERATION)
        consolidator = PlaybookConsolidator(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model=model)),
        )
        print(f"\n[T6] model={model}")
        output = consolidator._consolidation_decisions(
            new_playbooks=[correction],
            existing_playbooks=[incumbent],
        )
    return output.decisions


@skip_low_priority
@pytest.mark.parametrize(("case_id", "incumbent", "correction"), _CASES)
def test_correction_to_a_stale_rule_is_not_discarded(
    case_id: str,
    incumbent: UserPlaybook,
    correction: UserPlaybook,
) -> None:
    """A correction must not be thrown away as redundant.

    Asserted per-case rather than as a rate: which shape of correction gets discarded is
    the diagnostic, and an aggregate would let one failure hide behind two passes.

    The decision is printed unconditionally because this test is a measurement first --
    a green run still carries the finding (`unify` closes the loop, `differentiate` and
    `independent` leave it open and require a receiver in a later phase).
    """
    decisions = _decide(incumbent, correction)

    kinds = [d.kind for d in decisions]
    archived = [
        idx
        for d in decisions
        if d.kind == "unify"
        for idx in getattr(d, "archive_existing_ids", [])
    ]
    print(
        f"\n[T6:{case_id}] decisions={kinds} "
        f"archives_incumbent={bool(archived)} raw_archive_ids={archived}"
    )

    assert decisions, (
        f"[{case_id}] consolidation returned no decision at all; the correction "
        "silently vanished rather than being adjudicated"
    )
    assert "reject_new" not in kinds, (
        f"[{case_id}] the correction was rejected as REDUNDANT while the stale "
        "avoidance rule survives. A system that discards evidence contradicting a "
        "stale rule cannot self-correct, and this is a defect independent of the "
        "self-sealing-rule design."
    )


@skip_low_priority
def test_unrelated_candidate_is_not_merged_into_the_stale_rule() -> None:
    """Control: an unrelated candidate must not unify with the invoice rule.

    Without this, a run where the model unifies everything would read as "the loop
    closes" when it is really just collapsing distinct lessons. This is the negative
    control that makes a `unify` result in the cases above interpretable.
    """
    decisions = _decide(
        _incumbent("As of 2026-03-02, billing.export_invoice returned a server error."),
        _correction(
            "Confirm the recipient's email address before sending a receipt.",
            "The receipt was sent to a stale address the user had already corrected.",
        ),
    )

    kinds = [d.kind for d in decisions]
    print(f"\n[T6:control_unrelated] decisions={kinds}")

    assert decisions, "control returned no decision"
    assert "unify" not in kinds, (
        "an unrelated candidate was unified with the invoice-export rule, so a "
        "`unify` result in the contradiction cases cannot be read as the loop closing"
    )
