"""``OptimizationTerminalOutcome`` retains members no path can write.

The union has no replay-named members. Seven more were reachable only through
that same replay arm and have no writer, but removing them would narrow the
tenant CHECK again -- a one-way door -- and would abort the validating
migration on the first organization still holding a historical row. So they
are RETAINED, exactly the
way ``offline_tuner_legacy`` is retained in ``OptimizerKind``.

Retention only stays honest if it is recorded. These assertions pin the retained
set so it cannot silently grow: a new member added to the union without a writer
lands in the reachable half and fails the second test, which is the prompt to
show that the new outcome can actually be written.
"""

from __future__ import annotations

from typing import get_args

from reflexio.models.api_schema.domain.entities import (
    PENDING_WRITER_TERMINAL_OUTCOMES,
    RETAINED_UNREACHABLE_TERMINAL_OUTCOMES,
    OptimizationTerminalOutcome,
)
from reflexio.server.services.storage.sqlite_storage.playbook._optimization import (
    _TERMINAL_OUTCOMES_BY_OPTIMIZER,
)

# The sixteen outcomes with a path that can reach them, each named with its
# writer so the split is auditable. The open-world members are also in the
# SQLite stage-advance allowlist (OPEN_WORLD_FAILED_TERMINAL_OUTCOMES /
# OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES), which the `writable <=` assertion
# below checks.
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
        # stage-advance 'failed': the attempt-start regeneration fence
        # (currently dormant; see open_world/fence.py) refused the attempt
        # before it spent anything.
        "regeneration_fenced",
        # stage-advance 'failed': the attempt made NO provider call because
        # every row identity its question could occupy is owned by another job
        # (OpenWorldRefusal.SLOT_PINNED). The 'abstained' arm deliberately does
        # not accept it, since nothing was judged.
        "invocation_slot_pinned",
        # stage-advance: 'failed'
        "infrastructure_failure",
        "analyst_unqualified",
        "stale_incumbent",
        "governance_invalidated",
        # stage-advance: 'abstained'
        "no_grounded_hypothesis",
        "heldout_evidence_failed",
        # stage-advance 'abstained': the analyst reached no grounded hypothesis
        # AND the frozen bundle's skipped-session receipt is non-empty.
        "evidence_view_incomplete",
        # stage-advance 'failed': the frozen bundle declares an evidence-bundle
        # schema version this image cannot derive, so it is unloadable and no
        # retry can help (OpenWorldRefusal.BUNDLE_SCHEMA_UNSUPPORTED). The
        # 'abstained' arm deliberately does not accept it, since the refusal is
        # before derivation and nothing was judged.
        "bundle_schema_unsupported",
        # stage-advance 'failed': the evidence freeze's stranded-job retake
        # terminalizes a prior day's stranded row with it, BEFORE the rolled
        # discovery/attempt keys are compared.
        "abandoned",
        # stage-advance 'failed': a second tick's attempt was refused at the
        # manifest-membership check, having made no provider call
        # (OpenWorldRefusal.MANIFEST_CONFLICT). The 'abstained' arm
        # deliberately does not accept it, since nothing was judged.
        "evidence_manifest_conflict",
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
    """Every member is classified: reachable, retained, or pending a writer.

    A member added to the union with none of a writer, a retention rationale, or
    a recorded pending-writer reason fails here rather than accumulating
    quietly.

    ``PENDING_WRITER_TERMINAL_OUTCOMES`` is the third bucket. It exists because
    the tenant CHECK is shared between two projects and has to land as one
    migration, so the Literal can run ahead of the code that writes a member.
    That is a TODO with a name and an owner, not a licence to add unwritten
    outcomes -- moving a member out of it when its writer lands is the edit
    this assertion is here to force.
    """
    members = frozenset(get_args(OptimizationTerminalOutcome))

    assert members >= RETAINED_UNREACHABLE_TERMINAL_OUTCOMES
    assert members >= _REACHABLE_TERMINAL_OUTCOMES
    assert members >= PENDING_WRITER_TERMINAL_OUTCOMES
    # The three classifications are disjoint, or "classified" would not mean
    # anything: a member cannot be both retired and not yet born.
    assert not (RETAINED_UNREACHABLE_TERMINAL_OUTCOMES & _REACHABLE_TERMINAL_OUTCOMES)
    assert not (PENDING_WRITER_TERMINAL_OUTCOMES & _REACHABLE_TERMINAL_OUTCOMES)
    assert not (
        PENDING_WRITER_TERMINAL_OUTCOMES & RETAINED_UNREACHABLE_TERMINAL_OUTCOMES
    )
    assert (
        members
        - RETAINED_UNREACHABLE_TERMINAL_OUTCOMES
        - PENDING_WRITER_TERMINAL_OUTCOMES
        == _REACHABLE_TERMINAL_OUTCOMES
    )
    assert len(members) == 23
    assert len(_REACHABLE_TERMINAL_OUTCOMES) == 16
    # Empty by design: a member stays pending only until its writer lands.
    # Asserting zero forces each shared-migration handoff to clean up after
    # itself.
    assert len(PENDING_WRITER_TERMINAL_OUTCOMES) == 0


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
    # A pending-writer member in the SQLite allowlist would mean the writer has
    # in fact landed and the member was never moved out of the pending set.
    assert not (writable & PENDING_WRITER_TERMINAL_OUTCOMES)
    assert writable <= _REACHABLE_TERMINAL_OUTCOMES
