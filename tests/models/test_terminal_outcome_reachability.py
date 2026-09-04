"""``OptimizationTerminalOutcome`` retains members no path can write.

Phase 7 removed the four replay-named members of the union. Seven more were
reachable only through that same replay arm and died with it, but removing them
would narrow the tenant CHECK a second time -- a one-way door this branch has
already spent once -- and would abort the validating migration on the first
organization still holding a historical row. So they are RETAINED, exactly the
way ``offline_tuner_legacy`` is retained in ``OptimizerKind``.

Retention only stays honest if it is recorded. These assertions pin the retained
set so it cannot silently grow: a new member added to the union without a writer
lands in the reachable half and fails the second test, which is the prompt to
show that the new outcome can actually be written.
"""

from __future__ import annotations

from typing import get_args

from reflexio.models.api_schema.domain.entities import (
    RETAINED_UNREACHABLE_TERMINAL_OUTCOMES,
    OptimizationTerminalOutcome,
)
from reflexio.server.services.storage.sqlite_storage.playbook._optimization import (
    _TERMINAL_OUTCOMES_BY_OPTIMIZER,
)

# The twelve outcomes that survive Phase 7 with a path that can reach them. Six
# are written by the stage-advance allowlist below; the other six are written
# elsewhere and are named here with their writer so the split is auditable.
_REACHABLE_TERMINAL_OUTCOMES = frozenset(
    {
        # commit_user_playbook_publication (tenant 20260830020000:1293)
        "applied",
        # the same routine's CAS-lost branch (tenant 20260830020000:1147)
        "incumbent_changed",
        # the retention sweep's raw DML (reflexio_ext _retired_table_purge.py)
        "generation_failed",
        # the governance erasure path
        "governance_erased",
        # the regeneration fence: reflexio_ext open_world/runner.py:251 calls
        # _converge_terminal_failure with it, and the TENANT stage-advance RPC's
        # 'failed' arm assigns it (tenant 20260830020000:325-327). It is
        # deliberately absent from the SQLite allowlist below -- SQLite carries
        # no open-world fence -- which is why it is named here rather than left
        # to the `writable <=` assertion to cover.
        "regeneration_fenced",
        # the invocation-slot pin: reflexio_ext open_world/runner.py:262-264
        # calls _converge_terminal_failure with it on
        # OpenWorldInvocationSlotExhaustedError -- the attempt made NO provider
        # call because every row identity its question could occupy is owned by
        # another job. The TENANT stage-advance RPC's 'failed' arm assigns it
        # (tenant 20260903010000:129-131); the 'abstained' arm deliberately does
        # not, since nothing was judged. Absent from the SQLite allowlist below
        # for the same reason as 'regeneration_fenced' -- SQLite carries no
        # open-world invocation table, so no slot can be pinned -- which is why
        # it is named here rather than left to the `writable <=` assertion.
        "invocation_slot_pinned",
        # stage-advance: 'failed'
        "infrastructure_failure",
        "analyst_unqualified",
        "stale_incumbent",
        "governance_invalidated",
        # stage-advance: 'abstained'
        "no_grounded_hypothesis",
        "heldout_evidence_failed",
    }
)


def test_the_retained_unreachable_set_is_exactly_these_seven() -> None:
    """Pinned by value, so growing the set is an edit somebody has to make here.

    ``deployment_unsupported`` is the one that looks alive to grep: the same
    spelling is also ``OfflineTunerUnavailableReason``, a config-enablement
    rejection code with dozens of live references on a different type. Those
    references say nothing about this vocabulary.
    """
    assert (
        frozenset(
            {
                "insufficient_negative_evidence",
                "insufficient_positive_evidence",
                "insufficient_coverage",
                "deployment_unsupported",
                "candidate_regressed",
                "candidate_did_not_improve",
                "publication_failed",
            }
        )
        == RETAINED_UNREACHABLE_TERMINAL_OUTCOMES
    )
    assert len(RETAINED_UNREACHABLE_TERMINAL_OUTCOMES) == 7


def test_the_union_is_exactly_the_reachable_set_plus_the_retained_set() -> None:
    """Every member is classified: reachable, or retained-and-recorded.

    A member added to the union with neither a writer nor a retention rationale
    fails here rather than accumulating quietly.
    """
    members = frozenset(get_args(OptimizationTerminalOutcome))

    assert members >= RETAINED_UNREACHABLE_TERMINAL_OUTCOMES
    assert members >= _REACHABLE_TERMINAL_OUTCOMES
    assert (
        members - RETAINED_UNREACHABLE_TERMINAL_OUTCOMES == _REACHABLE_TERMINAL_OUTCOMES
    )
    assert len(members) == 19
    assert len(_REACHABLE_TERMINAL_OUTCOMES) == 12


def test_no_retained_outcome_is_writable_through_the_stage_advance_allowlist() -> None:
    """The SQLite writer allowlist is the mirror of the tenant advance RPC.

    If a retained member ever appears in it, the member is no longer
    unreachable and this file's premise is wrong -- so the assertion is the
    thing that would catch a silent revival, not just the growth of the set.
    """
    writable = {
        outcome
        for stages in _TERMINAL_OUTCOMES_BY_OPTIMIZER.values()
        for outcomes in stages.values()
        for outcome in outcomes
    }

    assert not (writable & RETAINED_UNREACHABLE_TERMINAL_OUTCOMES)
    assert writable <= _REACHABLE_TERMINAL_OUTCOMES
