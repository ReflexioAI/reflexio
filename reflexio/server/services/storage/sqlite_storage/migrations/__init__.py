"""Ordered SQLite schema steps and the ``PRAGMA user_version`` ledger.

``MIGRATION_STEPS`` is the whole schema history of a local database, in the
order it runs. A database stamped ``user_version == N`` has applied every step
whose version is ``<= N``; ``run_pending_migrations`` applies the rest, stamping
after each one, so a crash resumes at the step that did not finish. A database
at ``user_version == 0`` (every database created before the ledger, and every
fresh one) runs the full sequence once.

Rules for changing this list:

* Append only. Never reorder, renumber, insert or delete a step: a stored
  version names a position in this sequence, and a database already past a step
  never runs it again.
* A step must stay idempotent, because a database at version 0 may already hold
  any prefix of its effects (it ran the same code on every open before the
  ledger existed).
* A step that derives its target from code (for example the optimizer job
  ``CHECK`` literals built from ``OptimizationTerminalOutcome``) no longer
  re-runs when that code changes. Widening such an input needs a NEW appended
  step; ``test_schema_ledger.py`` pins the inputs that exist today.
* Ordering constraints are encoded by position. Steps 3-7 backfill columns that
  the base DDL (step 8) indexes, so they must precede it: on an existing
  database the table predates the column and ``executescript(_DDL)`` would raise
  "no such column", leaving the database permanently stuck. Steps after 8 may
  assume every base table exists.

Each step is the name of a ``SQLiteSchemaMigrationsMixin`` method taking no arguments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._steps import SQLiteSchemaMigrationsMixin

logger = logging.getLogger(__name__)

MIGRATION_STEPS: tuple[tuple[int, str], ...] = (
    (1, "_migrate_feedback_schema"),
    (2, "_migrate_interactions_schema"),
    # Steps 3-7: columns the base DDL indexes or constrains, backfilled first.
    (3, "_migrate_eval_result_user_id"),
    (4, "_migrate_retrieved_learning_interaction_identity"),
    (5, "_migrate_playbook_optimization_job_columns"),
    (6, "_migrate_request_retrieval_experiment"),
    (7, "_backfill_columns_indexed_by_base_ddl"),
    (8, "_apply_base_ddl"),
    (9, "_apply_attention_ddl"),
    (10, "_migrate_session_outcomes_schema"),
    # Needs the sqlite-vec extension; without it the ledger stops stamping here.
    (11, "_create_and_backfill_vec_tables"),
    (12, "_apply_publication_receipts_migration"),
    (13, "_apply_publication_preparations_migration"),
    (14, "_apply_open_world_migration"),
    (15, "_migrate_agent_runs_schema"),
    (16, "_migrate_pending_tool_calls_schema"),
    (17, "_migrate_expanded_terms"),
    (18, "_migrate_unicode_lexical_indexes"),
    (19, "_migrate_tags"),
    (20, "_migrate_profile_source_interaction_ids"),
    (21, "_migrate_interaction_window_indexes"),
    (22, "_migrate_agentic_signals"),
    (23, "_migrate_agent_playbook_source_windows"),
    (24, "_migrate_request_evaluation_only"),
    (25, "_migrate_request_session_id_required"),
    (26, "_migrate_shadow_comparison_verdicts"),
    (27, "_migrate_user_playbook_polarity"),
    (28, "_migrate_lineage"),
    (29, "_migrate_retired_at"),
    (30, "_migrate_lineage_event_table"),
    (31, "_migrate_playbook_diagnosis"),
    (32, "_migrate_playbook_optimization_candidate_metadata"),
    (33, "_migrate_user_playbook_publication_staging_columns"),
    (34, "_migrate_user_playbook_publication_results_shape"),
    (35, "_classify_legacy_playbook_optimization_jobs"),
    (36, "_enforce_playbook_optimization_job_constraints"),
    (37, "_enforce_playbook_optimization_artifact_constraints"),
    (38, "_migrate_retire_profile_change_logs"),
    (39, "_migrate_retire_playbook_aggregation_change_logs"),
    (40, "_init_stall_state"),
    (41, "_apply_extraction_stream_migration"),
    # Fresh databases gain windows after the run/receipt schema; this installs
    # their guards.
    (42, "_apply_publication_receipts_migration"),
)

LATEST_VERSION = MIGRATION_STEPS[-1][0]


def stored_version(storage: SQLiteSchemaMigrationsMixin) -> int:
    """Return the schema version stamped on the open database."""
    with storage._lock:
        return int(storage.conn.execute("PRAGMA user_version").fetchone()[0])


def _stamp(storage: SQLiteSchemaMigrationsMixin, version: int) -> None:
    """Raise the stored version to ``version`` under one write transaction.

    The read and the write share ``BEGIN IMMEDIATE`` so a concurrent opener
    that is further along cannot be lowered between them.
    """
    with storage._lock:
        storage.conn.commit()
        storage.conn.execute("BEGIN IMMEDIATE")
        try:
            if stored_version(storage) < version:
                storage.conn.execute(f"PRAGMA user_version = {version}")
            storage.conn.commit()
        except BaseException:
            storage.conn.rollback()
            raise


def run_pending_migrations(storage: SQLiteSchemaMigrationsMixin) -> None:
    """Apply every step above the stored version, stamping after each one.

    A step returning ``False`` could not run in this process (sqlite-vec is not
    loaded). Later steps still run, as they did before the ledger, but nothing
    from that step on is stamped, so a process that can run it does.
    """
    current = stored_version(storage)
    if current >= LATEST_VERSION:
        return
    stamping = True
    for version, method_name in MIGRATION_STEPS:
        if version <= current:
            continue
        if getattr(storage, method_name)() is False:
            stamping = False
        if stamping:
            _stamp(storage, version)
