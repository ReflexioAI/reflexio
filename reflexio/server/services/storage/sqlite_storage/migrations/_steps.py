"""Schema steps applied by the ``user_version`` ledger; the order lives in ``migrations/__init__.py``."""

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from reflexio.server.services.storage.session_outcome_identity import (
    OUTCOME_ALLOWED_VALUES,
    OUTCOME_FINALIZATION_RULE,
    outcome_contract_digest,
)

from .._ddl import (
    _DDL,
    _RETIRED_ARTIFACT_KIND_LITERALS,
    _RETIRED_OPTIMIZER_JOB_LITERALS,
    _TERMINAL_OUTCOME_CHECK,
    _TERMINAL_OUTCOME_LITERALS,
    PROVISIONAL_STAGING_COLUMNS,
    PUBLICATION_RESULTS_DDL,
)
from .._row_mappers import _json_loads
from .._search_ranking import _UNICODE_LEXICAL_INDEX_VERSION
from .._session_trajectory import (
    _SESSION_OUTCOME_MIGRATION_BATCH_SIZE,
    _legacy_session_outcome_id,
    _prefetch_canonical_session_trajectory_digests,
)
from .._stall_state import init_stall_state_table
from .._subject_write_gate import (
    backfill_user_playbook_subject_refs,
    init_subject_write_barrier_table,
)
from ..migrations import run_pending_migrations

logger = logging.getLogger(__name__)


class SQLiteSchemaMigrationsMixin:
    """The ordered schema steps of a local SQLite database."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _has_sqlite_vec: bool
    embedding_dimensions: int
    _vec_upsert: Any
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _subject_ref_for_user_id: Callable[[str], str]

    # ------------------------------------------------------------------
    # DDL / migration
    # ------------------------------------------------------------------

    def migrate(self) -> bool:
        """Apply every schema step above the database's ``PRAGMA user_version``."""
        run_pending_migrations(self)
        # The vec tables depend on the loadable extension, not only on the
        # ledger: an extension that was missing when the database was stamped
        # still needs them once it loads.
        if self._has_sqlite_vec and not self._vec_tables_present():
            self._create_and_backfill_vec_tables()
        return True

    def _vec_tables_present(self) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
                "('interactions_vec', 'profiles_vec', 'user_playbooks_vec', "
                "'agent_playbooks_vec', 'playbook_aggregation_clusters_vec')"
            ).fetchone()
        return int(row[0]) == 5

    def _backfill_columns_indexed_by_base_ddl(self) -> None:
        with self._lock:
            session_outcome_columns = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(session_outcomes)"
                ).fetchall()
            }
            if (
                session_outcome_columns
                and "governance_subject_ref" not in session_outcome_columns
            ):
                self.conn.execute(
                    "ALTER TABLE session_outcomes "
                    "ADD COLUMN governance_subject_ref TEXT"
                )
            # Displacement columns. Added here rather than left to `_DDL`,
            # because `CREATE TABLE IF NOT EXISTS` is a no-op against a database
            # that already has the table -- so an upgraded install would keep
            # the old shape and every read of `is_inferred` would raise
            # "no such column" at runtime rather than at startup.
            #
            # No CHECK on the added columns: SQLite cannot add a table-level
            # constraint with ALTER TABLE, and the pairing is enforced by the
            # single write path that sets them. Fresh databases get the CHECK
            # from `_DDL`.
            #
            # EVERY PRE-EXISTING ROW BECOMES `is_inferred = 0`, AND THAT IS NOT
            # A GUESS. The only writer of inferred outcomes is the offline
            # tuner's outcome bridge, and the bridge ships in the SAME release
            # as this column -- it reaches storage exclusively through the
            # `is_inferred=True` keyword that did not exist until now. So no
            # row written before this migration can be an inferred one, and
            # there is nothing to backfill.
            #
            # BACKFILLING BY LABEL WOULD BE ACTIVELY WRONG, not merely
            # unnecessary. The bridge stamps
            # `label = 'inferred_from_agent_success_evaluation'`, but `label`
            # is a field on `SetSessionOutcomeRequest` -- the PUBLIC request
            # body -- so any customer can send that exact string. Promoting
            # rows by label would hand a caller a way to mark their own outcome
            # displaceable, which is the precise thing keeping `is_inferred`
            # off the request model exists to prevent. Provenance that a
            # customer can forge is not provenance.
            if session_outcome_columns and "is_inferred" not in session_outcome_columns:
                self.conn.execute(
                    "ALTER TABLE session_outcomes "
                    "ADD COLUMN is_inferred INTEGER NOT NULL DEFAULT 0"
                )
            if (
                session_outcome_columns
                and "superseded_outcome" not in session_outcome_columns
            ):
                self.conn.execute(
                    "ALTER TABLE session_outcomes ADD COLUMN superseded_outcome TEXT"
                )
            # Cutover columns. NULL is the legacy whole-session meaning, so
            # every pre-existing row is already correct without a backfill.
            if (
                session_outcome_columns
                and "trajectory_through_request_id" not in session_outcome_columns
            ):
                self.conn.execute(
                    "ALTER TABLE session_outcomes "
                    "ADD COLUMN trajectory_through_request_id TEXT"
                )
            eval_result_columns = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(agent_success_evaluation_result)"
                ).fetchall()
            }
            for column, column_type in (
                ("trajectory_through_request_id", "TEXT"),
                ("trajectory_interaction_count", "INTEGER"),
            ):
                if eval_result_columns and column not in eval_result_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_success_evaluation_result "
                        f"ADD COLUMN {column} {column_type}"
                    )

    def _apply_base_ddl(self) -> None:
        # Import lazily: the aggregation mixin shares this base and importing its
        # package while ``_base`` is still initializing would create a cycle.
        from ..playbook._aggregation import init_playbook_aggregation_tables

        with self._lock:
            cur = self.conn.cursor()
            # executescript does not open a transaction: otherwise each schema
            # statement commits separately. Roll back the whole base DDL on error.
            with self.conn:
                cur.executescript("BEGIN;\n" + _DDL + "\nCOMMIT;")
            init_subject_write_barrier_table(self.conn)
            init_playbook_aggregation_tables(self.conn)
            self.conn.commit()
            backfill_user_playbook_subject_refs(self.conn, self.org_id)

    def _apply_attention_ddl(self) -> None:
        from ..profiles._attention import ATTENTION_DDL

        with self._lock:
            self.conn.executescript(ATTENTION_DDL)

    def _create_and_backfill_vec_tables(self) -> None:
        if self._has_sqlite_vec:
            self._create_vec_tables()
            self._migrate_vec_tables()
            self._migrate_playbook_aggregation_agent_centroids()

    def _apply_publication_receipts_migration(self) -> None:
        from .._publication_receipts import migrate_publication_receipts

        migrate_publication_receipts(self.conn, org_id=self.org_id)

    def _apply_publication_preparations_migration(self) -> None:
        from .._publication_preparations import migrate_publication_preparations

        migrate_publication_preparations(self.conn)

    def _apply_open_world_migration(self) -> None:
        from .._open_world import migrate_open_world

        with self._lock:
            migrate_open_world(self.conn)

    def _init_stall_state(self) -> None:
        init_stall_state_table(self.conn)

    def _apply_extraction_stream_migration(self) -> None:
        from .._extraction_stream import migrate_extraction_stream

        migrate_extraction_stream(self.conn)

    def _migrate_playbook_diagnosis(self) -> None:
        """Add diagnostic evidence without changing optimizer jobs or old results."""
        with self._lock:
            columns = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(retrieved_learning_evaluation)"
                )
            }
            if {
                "diagnosis",
                "evaluated_playbook_digest",
                "diagnosis_evidence_complete",
            } <= columns:
                return
            # The initialization locks are process-local. Lock the database
            # before inspecting columns so concurrent workers cannot both ALTER.
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                columns = {
                    row["name"]
                    for row in self.conn.execute(
                        "PRAGMA table_info(retrieved_learning_evaluation)"
                    )
                }
                for name, definition in (
                    ("diagnosis", "TEXT"),
                    ("evaluated_playbook_digest", "TEXT"),
                    ("diagnosis_evidence_complete", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    if name not in columns:
                        self.conn.execute(
                            f"ALTER TABLE retrieved_learning_evaluation ADD COLUMN {name} {definition}"
                        )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _migrate_unicode_lexical_indexes(self) -> None:
        """Backfill the trigger-maintained Unicode FTS sidecars exactly once."""
        version_sentinel_rowid = -_UNICODE_LEXICAL_INDEX_VERSION
        with self._lock:
            version_row = self.conn.execute(
                "SELECT rowid FROM interactions_unicode_fts WHERE rowid = ?",
                (version_sentinel_rowid,),
            ).fetchone()
            if version_row is not None:
                return

            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for index_table in (
                    "interactions_unicode_fts",
                    "profiles_unicode_fts",
                    "user_playbooks_unicode_fts",
                    "agent_playbooks_unicode_fts",
                ):
                    self.conn.execute(f"DELETE FROM {index_table}")  # noqa: S608

                self.conn.execute(
                    """INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
                       SELECT i.rowid, reflexio_unicode_lexical_index(
                           COALESCE(i.content, '') || ' ' ||
                           COALESCE(i.user_action_description, '')
                       )
                       FROM interactions i"""
                )
                self.conn.execute(
                    """INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
                       SELECT p.rowid, reflexio_unicode_lexical_index(
                           COALESCE(p.content, '') || ' ' ||
                           COALESCE(p.expanded_terms, '')
                       )
                       FROM profiles p"""
                )
                self.conn.execute(
                    """INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
                       SELECT up.rowid, reflexio_unicode_lexical_index(
                           COALESCE(up.trigger, '') || ' ' ||
                           COALESCE(up.content, '') || ' ' ||
                           COALESCE(up.rationale, '') || ' ' ||
                           COALESCE(up.source, '')
                       )
                       FROM user_playbooks up"""
                )
                self.conn.execute(
                    """INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
                       SELECT ap.rowid, reflexio_unicode_lexical_index(
                           COALESCE(ap.trigger, '') || ' ' ||
                           COALESCE(ap.content, '') || ' ' ||
                           COALESCE(ap.rationale, '')
                       )
                       FROM agent_playbooks ap"""
                )
                self.conn.execute(
                    "INSERT INTO interactions_unicode_fts(rowid, search_ngrams) "
                    "VALUES (?, '')",
                    (version_sentinel_rowid,),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _create_vec_tables(self) -> None:
        """Create vec0 virtual tables for each entity that stores embeddings."""
        dim = self.embedding_dimensions
        vec_ddl = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS interactions_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS profiles_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS playbook_aggregation_clusters_vec USING vec0(
                embedding float[{dim}] distance_metric=cosine
            );
        """
        with self._lock:
            self.conn.executescript(vec_ddl)
            self.conn.commit()

    def _migrate_playbook_aggregation_agent_centroids(self) -> None:
        """Replace legacy user-vector means with canonical agent embeddings."""
        rows = self.conn.execute(
            "SELECT c.cluster_id, c.index_rowid, a.embedding "
            "FROM playbook_aggregation_cluster c JOIN agent_playbooks a "
            "ON a.agent_playbook_id=c.agent_playbook_id "
            "WHERE c.vector_sum IS NOT NULL AND c.index_rowid IS NOT NULL "
            "AND a.embedding IS NOT NULL "
            "AND trim(a.embedding) NOT IN ('', '[]')"
        ).fetchall()
        with self._lock:
            for row in rows:
                try:
                    embedding = json.loads(row[2])
                except (json.JSONDecodeError, TypeError):
                    continue
                if (
                    not isinstance(embedding, list)
                    or len(embedding) != self.embedding_dimensions
                    or not all(isinstance(value, int | float) for value in embedding)
                ):
                    continue
                self.conn.execute(
                    "UPDATE playbook_aggregation_cluster SET centroid=?, "
                    "vector_sum=NULL, embedding_dimension=? WHERE cluster_id=?",
                    (json.dumps(embedding), len(embedding), str(row[0])),
                )
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    (int(row[1]),),
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                    "VALUES (?, ?)",
                    (int(row[1]), json.dumps(embedding)),
                )
            self.conn.commit()

    def _migrate_vec_tables(self) -> None:
        """Backfill vec tables from existing embedding TEXT columns (idempotent)."""
        entity_map = [
            ("interactions", "interactions_vec", "interaction_id"),
            ("profiles", "profiles_vec", "profile_id"),
            ("user_playbooks", "user_playbooks_vec", "user_playbook_id"),
            ("agent_playbooks", "agent_playbooks_vec", "agent_playbook_id"),
        ]
        for main_table, vec_table, _id_col in entity_map:
            row = self._fetchone(f"SELECT COUNT(*) as cnt FROM {vec_table}")
            if row and row["cnt"] > 0:
                continue  # Already populated
            rows = self._fetchall(
                f"SELECT rowid AS rid, embedding FROM {main_table} WHERE embedding IS NOT NULL"
            )
            for r in rows:
                emb = _json_loads(r["embedding"])
                if emb:
                    self._vec_upsert(vec_table, r["rid"], emb)

    def _session_outcomes_schema_is_canonical(self) -> bool:
        table_info = self.conn.execute("PRAGMA table_info(session_outcomes)").fetchall()
        columns = {str(row["name"]): row for row in table_info}
        table = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'session_outcomes'"
        ).fetchone()
        table_sql = (table["sql"] if table is not None else "") or ""
        identity_columns = {
            "outcome_id",
            "outcome_revision",
            "outcome_contract_digest",
            "finalized_trajectory_digest",
        }
        expected_columns = identity_columns | {
            "user_id",
            "session_id",
            "outcome",
            "occurred_at",
            "source",
            "label",
            "value",
            "metadata",
            "governance_subject_ref",
            "created_at",
        }
        subject_column = columns.get("governance_subject_ref")
        has_empty_subject_default = subject_column is not None and subject_column[
            "dflt_value"
        ] in ("''", '""')
        return (
            expected_columns.issubset(columns)
            and "'unknown'" in table_sql
            and subject_column is not None
            and int(subject_column["notnull"]) == 1
            and not has_empty_subject_default
        )

    def _migrate_session_outcomes_schema(self) -> None:
        """Rebuild legacy outcome rows with immutable v1 finalization identities."""
        with self._lock:
            if (
                self._session_outcomes_schema_is_canonical()
                and self.conn.execute(
                    "SELECT 1 FROM session_outcomes WHERE governance_subject_ref IS NULL "
                    "OR trim(governance_subject_ref) = '' LIMIT 1"
                ).fetchone()
                is None
            ):
                return
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                if self._session_outcomes_schema_is_canonical():
                    rows_missing_subject_ref = self.conn.execute(
                        """SELECT user_id, session_id FROM session_outcomes
                           WHERE governance_subject_ref IS NULL
                              OR trim(governance_subject_ref) = ''"""
                    ).fetchall()
                    self.conn.executemany(
                        """UPDATE session_outcomes SET governance_subject_ref = ?
                           WHERE user_id = ? AND session_id = ?""",
                        [
                            (
                                self._subject_ref_for_user_id(str(row["user_id"])),
                                row["user_id"],
                                row["session_id"],
                            )
                            for row in rows_missing_subject_ref
                        ],
                    )
                    self.conn.commit()
                    return

                columns = {
                    row["name"]
                    for row in self.conn.execute("PRAGMA table_info(session_outcomes)")
                }
                self.conn.execute(
                    "ALTER TABLE session_outcomes RENAME TO session_outcomes_legacy"
                )
                self.conn.execute(
                    """CREATE TABLE session_outcomes (
                        outcome_id TEXT NOT NULL UNIQUE,
                        outcome_revision INTEGER NOT NULL CHECK (outcome_revision >= 1),
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
                        occurred_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        label TEXT,
                        value REAL,
                        metadata TEXT,
                        outcome_contract_digest TEXT NOT NULL,
                        finalized_trajectory_digest TEXT NOT NULL,
                        governance_subject_ref TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        -- Carried here as well as in `_DDL`, because this
                        -- rebuild runs AFTER the ALTER TABLE upgrade in
                        -- `init` and would otherwise drop both columns again
                        -- on any database still needing the identity
                        -- migration. Defaulting is correct rather than merely
                        -- convenient: this rebuild only fires on a schema
                        -- predating the identity columns, which predates the
                        -- outcome bridge, so no legacy row can be inferred.
                        is_inferred INTEGER NOT NULL DEFAULT 0
                            CHECK (is_inferred IN (0, 1)),
                        superseded_outcome TEXT,
                        trajectory_through_request_id TEXT,
                        CHECK (superseded_outcome IS NULL OR is_inferred = 0),
                        PRIMARY KEY (user_id, session_id)
                    )"""
                )
                last_user_id: str | None = None
                last_session_id: str | None = None
                while True:
                    if last_user_id is None:
                        legacy_rows = self.conn.execute(
                            """SELECT * FROM session_outcomes_legacy
                               ORDER BY user_id, session_id
                               LIMIT ?""",
                            (_SESSION_OUTCOME_MIGRATION_BATCH_SIZE,),
                        ).fetchall()
                    else:
                        legacy_rows = self.conn.execute(
                            """SELECT * FROM session_outcomes_legacy
                               WHERE user_id > ?
                                  OR (user_id = ? AND session_id > ?)
                               ORDER BY user_id, session_id
                               LIMIT ?""",
                            (
                                last_user_id,
                                last_user_id,
                                last_session_id,
                                _SESSION_OUTCOME_MIGRATION_BATCH_SIZE,
                            ),
                        ).fetchall()
                    if not legacy_rows:
                        break
                    trajectory_digests = _prefetch_canonical_session_trajectory_digests(
                        self.conn,
                        [
                            str(row["session_id"])
                            for row in legacy_rows
                            if "finalized_trajectory_digest" not in columns
                            or not row["finalized_trajectory_digest"]
                        ],
                    )
                    for row in legacy_rows:
                        source = str(row["source"])
                        subject_ref = (
                            row["governance_subject_ref"]
                            if "governance_subject_ref" in columns
                            else None
                        )
                        if subject_ref is None or not str(subject_ref).strip():
                            subject_ref = self._subject_ref_for_user_id(
                                str(row["user_id"])
                            )
                        existing_outcome_id = (
                            row["outcome_id"] if "outcome_id" in columns else None
                        )
                        existing_contract_digest = (
                            row["outcome_contract_digest"]
                            if "outcome_contract_digest" in columns
                            else None
                        )
                        existing_trajectory_digest = (
                            row["finalized_trajectory_digest"]
                            if "finalized_trajectory_digest" in columns
                            else None
                        )
                        self.conn.execute(
                            """INSERT INTO session_outcomes (
                                outcome_id, outcome_revision, user_id, session_id,
                                outcome, occurred_at, source, label, value, metadata,
                                outcome_contract_digest, finalized_trajectory_digest,
                                governance_subject_ref, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                existing_outcome_id
                                or _legacy_session_outcome_id(
                                    str(row["user_id"]), str(row["session_id"])
                                ),
                                (
                                    row["outcome_revision"]
                                    if "outcome_revision" in columns
                                    and row["outcome_revision"] is not None
                                    else 1
                                ),
                                row["user_id"],
                                row["session_id"],
                                row["outcome"],
                                row["occurred_at"],
                                source,
                                row["label"] if "label" in columns else None,
                                row["value"] if "value" in columns else None,
                                row["metadata"] if "metadata" in columns else None,
                                existing_contract_digest
                                or outcome_contract_digest(
                                    source=source,
                                    schema_version=1,
                                    allowed_values=OUTCOME_ALLOWED_VALUES,
                                    finalization_rule=OUTCOME_FINALIZATION_RULE,
                                ),
                                existing_trajectory_digest
                                or trajectory_digests[str(row["session_id"])],
                                subject_ref,
                                row["created_at"],
                            ),
                        )
                    last_user_id = str(legacy_rows[-1]["user_id"])
                    last_session_id = str(legacy_rows[-1]["session_id"])
                    if len(legacy_rows) < _SESSION_OUTCOME_MIGRATION_BATCH_SIZE:
                        break
                self.conn.execute("DROP TABLE session_outcomes_legacy")
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_occurred_at "
                    "ON session_outcomes(occurred_at)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_session_id "
                    "ON session_outcomes(session_id)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_source_outcome "
                    "ON session_outcomes(source, outcome)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_label ON session_outcomes(label)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_subject_ref "
                    "ON session_outcomes(governance_subject_ref)"
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _migrate_interactions_schema(self) -> None:
        """Add new columns to existing interactions table if missing."""
        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(interactions)")
            columns = {row[1] for row in cur.fetchall()}

        if not columns:
            return

        if "expert_content" not in columns:
            logger.info("Adding expert_content column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN expert_content TEXT NOT NULL DEFAULT ''"
                )
                self.conn.commit()

        if "citations" not in columns:
            logger.info("Adding citations column to interactions table.")
            with self._lock:
                self.conn.execute("ALTER TABLE interactions ADD COLUMN citations TEXT")
                self.conn.commit()

        if "image_encoding" not in columns:
            logger.info("Adding image_encoding column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN image_encoding TEXT NOT NULL DEFAULT ''"
                )
                self.conn.commit()

        if "retrieved_learnings" not in columns:
            logger.info("Adding retrieved_learnings column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN retrieved_learnings TEXT"
                )
                self.conn.commit()

        if "token_count" not in columns:
            logger.info("Adding token_count column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN token_count INTEGER"
                )
                self.conn.commit()

    def _migrate_feedback_schema(self) -> None:
        """Drop old-schema feedback/playbook tables so _DDL can recreate them.

        Checks for two migration scenarios:
        1. Old column layout (missing ``trigger``) -- drop data tables + FTS.
        2. Old FTS column name (``feedback_content`` instead of ``search_text``)
           -- drop only the FTS tables so they are recreated with the new column.

        Also handles migration from old table names (raw_feedbacks/feedbacks)
        to new names (user_playbooks/agent_playbooks), renames
        feedback_aggregation_change_logs to playbook_aggregation_change_logs,
        and renames columns on related tables (skills, profiles,
        playbook_aggregation_change_logs).

        Since SQLite is used only for local development, data loss is acceptable.
        """
        # Check for old table names and rename if needed
        with self._lock:
            old_tables = {
                row[0]
                for row in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        if "raw_feedbacks" in old_tables and "user_playbooks" not in old_tables:
            logger.warning(
                "Detected old table names (raw_feedbacks/feedbacks). "
                "Dropping old tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS raw_feedbacks_fts;
                    DROP TABLE IF EXISTS feedbacks_fts;
                    DROP TABLE IF EXISTS raw_feedbacks;
                    DROP TABLE IF EXISTS feedbacks;
                """)
                self.conn.commit()

        if (
            "feedback_aggregation_change_logs" in old_tables
            and "playbook_aggregation_change_logs" not in old_tables
        ):
            logger.warning(
                "Renaming table feedback_aggregation_change_logs → playbook_aggregation_change_logs."
            )
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE feedback_aggregation_change_logs RENAME TO playbook_aggregation_change_logs"
                )
                self.conn.commit()

        # Migrate renamed columns on related tables (skills, profiles, change_logs)
        self._migrate_renamed_columns()

        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(user_playbooks)")
            columns = {row[1] for row in cur.fetchall()}

        # Table doesn't exist yet -- nothing to migrate
        if not columns:
            return

        # Scenario 1: old data schema (missing trigger column — pre-flattening)
        if "trigger" not in columns:
            logger.warning(
                "Detected old playbook schema (missing trigger column). "
                "Dropping playbook tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS user_playbooks_fts;
                    DROP TABLE IF EXISTS agent_playbooks_fts;
                    DROP TABLE IF EXISTS user_playbooks;
                    DROP TABLE IF EXISTS agent_playbooks;
                """)
                self.conn.commit()
            return

        # Scenario 2: old FTS column name (feedback_content -> search_text)
        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(user_playbooks_fts)")
            fts_columns = {row[1] for row in cur.fetchall()}

        if fts_columns and "search_text" not in fts_columns:
            logger.warning(
                "Detected old FTS column name. "
                "Dropping FTS tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS user_playbooks_fts;
                    DROP TABLE IF EXISTS agent_playbooks_fts;
                """)
                self.conn.commit()

    def _migrate_renamed_columns(self) -> None:
        """Rename columns on tables affected by the feedback→playbook rename.

        Handles: skills (feedback_name→playbook_name, raw_feedback_ids→user_playbook_ids),
        profiles (profile_content→content), playbook_aggregation_change_logs (feedback_name→playbook_name).

        Since SQLite is used only for local development, we drop and recreate if needed.
        """
        renames = [
            ("skills", "feedback_name", "playbook_name"),
            ("skills", "raw_feedback_ids", "user_playbook_ids"),
            ("profiles", "profile_content", "content"),
            ("playbook_aggregation_change_logs", "feedback_name", "playbook_name"),
        ]

        for table, old_col, new_col in renames:
            with self._lock:
                try:
                    cols = {
                        row[1]
                        for row in self.conn.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()  # noqa: S608
                    }
                except Exception:  # noqa: S112
                    continue  # Table doesn't exist yet

                if not cols:
                    continue  # Table doesn't exist

                if old_col in cols and new_col not in cols:
                    logger.info(
                        "Renaming column %s.%s -> %s",
                        table,
                        old_col,
                        new_col,
                    )
                    try:
                        self.conn.execute(
                            f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}"  # noqa: S608
                        )
                        self.conn.commit()
                    except Exception as e:
                        logger.warning(
                            "Could not rename %s.%s -> %s: %s. "
                            "Dropping table so it can be recreated.",
                            table,
                            old_col,
                            new_col,
                            e,
                        )
                        self.conn.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
                        self.conn.commit()

    def _migrate_expanded_terms(self) -> None:
        """Add expanded_terms column if missing (for databases created before this feature)."""
        for table in ("profiles", "user_playbooks", "agent_playbooks"):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "expanded_terms" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN expanded_terms TEXT")
                logger.info("Added expanded_terms column to %s", table)
        self.conn.commit()

    def _migrate_tags(self) -> None:
        """Add tags column if missing."""
        for table in (
            "profiles",
            "user_playbooks",
            "agent_playbooks",
            "agent_success_evaluation_result",
        ):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "tags" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN tags TEXT")
                logger.info("Added tags column to %s", table)
        self.conn.commit()

    def _migrate_profile_source_interaction_ids(self) -> None:
        """Add profile source interaction ids for provenance on existing DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(profiles)").fetchall()
        }
        if "source_interaction_ids" not in cols:
            self.conn.execute(
                "ALTER TABLE profiles ADD COLUMN source_interaction_ids TEXT"
            )
            logger.info("Added source_interaction_ids column to profiles")
        self.conn.commit()

    def _migrate_interaction_window_indexes(self) -> None:
        """Add composite indexes used by sliding-window provenance lookups."""
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_user_created_at_desc "
            "ON interactions(user_id, created_at DESC, interaction_id DESC)"
        )
        self.conn.commit()

    def _migrate_agent_runs_schema(self) -> None:
        """Add resumable-agent run columns if missing from existing SQLite DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(_agent_runs)").fetchall()
        }
        if not cols:
            return
        if "max_steps_remaining" not in cols:
            self.conn.execute(
                "ALTER TABLE _agent_runs ADD COLUMN max_steps_remaining INTEGER"
            )
            logger.info("Added max_steps_remaining column to _agent_runs")
        for name, definition in (
            ("progress_stage", "TEXT"),
            ("last_progress_at", "TEXT"),
            ("recovery_operation_id", "TEXT"),
            ("recovery_history", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            if name not in cols:
                self.conn.execute(
                    f"ALTER TABLE _agent_runs ADD COLUMN {name} {definition}"
                )
        # Triggers cover every writer, including legacy callers and SQL claims.
        # They never backfill old rows and never treat generic updated_at as progress.
        stage = """CASE NEW.status WHEN 'running' THEN 'generation_started'
          WHEN 'agent_completed' THEN 'output_saved' WHEN 'finalizing' THEN 'finalization_started'
          WHEN 'finalized' THEN 'completed' WHEN 'finalized_pending_tool' THEN 'completed'
          ELSE NEW.status END"""
        for event, condition in (
            ("INSERT", "1"),
            (
                "UPDATE OF status, recovery_operation_id",
                "NEW.status IS NOT OLD.status OR NEW.recovery_operation_id IS NOT OLD.recovery_operation_id",
            ),
        ):
            suffix = "insert" if event == "INSERT" else "update"
            self.conn.execute(f"""CREATE TRIGGER IF NOT EXISTS agent_run_progress_{suffix}
              AFTER {event} ON _agent_runs WHEN {condition}
              BEGIN UPDATE _agent_runs SET progress_stage={stage},
                last_progress_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now'),
                recovery_history=CASE WHEN NEW.recovery_operation_id IS NULL THEN recovery_history
                  ELSE json_insert(recovery_history, '$[#]', json_object(
                    'operation_id', NEW.recovery_operation_id, 'event', NEW.status,
                    'at', strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))) END
                WHERE id=NEW.id; END""")
        self.conn.execute("""CREATE TRIGGER IF NOT EXISTS agent_run_receipt_progress
          AFTER INSERT ON _agent_run_finalization_receipts BEGIN
          UPDATE _agent_runs SET progress_stage='results_committed',
            last_progress_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now')
          WHERE id=NEW.run_id; END""")
        self.conn.execute("""CREATE TRIGGER IF NOT EXISTS agent_output_receipt_progress
          AFTER INSERT ON _agent_output_receipts BEGIN
          UPDATE _agent_runs SET progress_stage='results_committed',
            last_progress_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now')
          WHERE id=NEW.run_id; END""")
        self.conn.commit()

    def _migrate_pending_tool_calls_schema(self) -> None:
        """Add pending-tool-call columns if missing from existing SQLite DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(_pending_tool_calls)"
            ).fetchall()
        }
        if not cols:
            return
        if "superseded_by" not in cols:
            self.conn.execute(
                "ALTER TABLE _pending_tool_calls ADD COLUMN superseded_by TEXT"
            )
            logger.info("Added superseded_by column to _pending_tool_calls")
        self.conn.commit()

    def _migrate_agentic_signals(self) -> None:
        """Add source_span/notes/reader_angle columns if missing.

        Backfill-safe: columns are nullable with no default. Applies to both
        the profiles and user_playbooks tables — the agentic extraction
        pipeline populates them per-row; classic extraction leaves them NULL.
        """
        for table in ("profiles", "user_playbooks"):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col in ("source_span", "notes", "reader_angle"):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")  # noqa: S608
                    logger.info("Added %s column to %s", col, table)
        self.conn.commit()

    def _migrate_user_playbook_polarity(self) -> None:
        """Drop the legacy ``polarity`` column from ``user_playbooks`` if present.

        Polarity is retired under Option B (orientation lives in rule wording and
        is LLM-judged, never a stored field). This mirrors the Supabase drop
        migration and brings databases created while the column existed back in
        line with the current schema, which no longer defines it.
        """
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(user_playbooks)").fetchall()
        }
        if not cols:
            return
        if "polarity" in cols:
            self.conn.execute("ALTER TABLE user_playbooks DROP COLUMN polarity")
            logger.info("Dropped legacy polarity column from user_playbooks")
        self.conn.commit()

    def _migrate_lineage(self) -> None:
        """Add merged_into/superseded_by forward-pointer columns if missing.

        Backfill-safe: columns are nullable with no default. INTEGER for playbook
        tables (int foreign-key pointers), TEXT for profiles (str profile_id pointers).
        """
        int_tables = {"user_playbooks": "INTEGER", "agent_playbooks": "INTEGER"}
        str_tables = {"profiles": "TEXT"}
        for table, coltype in {**int_tables, **str_tables}.items():
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col in ("merged_into", "superseded_by"):
                if col not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"  # noqa: S608
                    )
                    logger.info("Added %s column to %s", col, table)
        self.conn.commit()

    def _migrate_retired_at(self) -> None:
        """Add nullable ``retired_at INTEGER`` GC column to tombstone-bearing tables.

        Backfill-safe: column is nullable with no default. Existing tombstones
        will have ``retired_at = NULL`` (conservative — GC T2 uses ``retired_at``
        as the age signal, so old tombstones without it are simply not yet eligible
        by the new clock; ops can backfill via T4 if needed).
        Also creates the covering index for GC queries (idempotent).
        """
        with self._lock:
            for table in ("profiles", "user_playbooks", "agent_playbooks"):
                cols = {
                    row["name"]
                    for row in self.conn.execute(
                        f"PRAGMA table_info({table})"  # noqa: S608
                    ).fetchall()
                }
                if "retired_at" not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN retired_at INTEGER"  # noqa: S608
                    )
                    logger.info("Added retired_at column to %s", table)
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_retired_at "  # noqa: S608
                    f"ON {table}(status, retired_at)"
                )
            self.conn.commit()

    def _migrate_lineage_event_table(self) -> None:
        """Create the lineage_event table + index for existing databases (idempotent)."""
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS lineage_event (
                    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id           TEXT NOT NULL,
                    entity_type      TEXT NOT NULL,
                    entity_id        TEXT NOT NULL,
                    op               TEXT NOT NULL,
                    prov_relation    TEXT NOT NULL DEFAULT '',
                    source_ids       TEXT NOT NULL DEFAULT '[]',
                    actor            TEXT NOT NULL DEFAULT '',
                    request_id       TEXT NOT NULL DEFAULT '',
                    reason           TEXT NOT NULL DEFAULT '',
                    created_at       INTEGER NOT NULL,
                    model_name       TEXT,
                    provider         TEXT,
                    UNIQUE (org_id, entity_type, entity_id, op, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_lineage_entity
                    ON lineage_event (entity_type, entity_id);
            """)
            existing_cols = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(lineage_event)"
                ).fetchall()
            }
            for col in (
                "from_status",
                "to_status",
                "status_namespace",
                "model_name",
                "provider",
            ):
                if col not in existing_cols:
                    self.conn.execute(
                        f"ALTER TABLE lineage_event ADD COLUMN {col} TEXT"  # noqa: S608
                    )
                    logger.info("Added %s column to lineage_event", col)
            self.conn.commit()

    def _migrate_playbook_optimization_candidate_metadata(self) -> None:
        """Add metadata_json to legacy optimizer candidate tables when missing."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_candidates)"
            ).fetchall()
        }
        if not cols:
            return
        if "metadata_json" not in cols:
            self.conn.execute(
                "ALTER TABLE playbook_optimization_candidates "
                "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
            logger.info(
                "Added metadata_json column to playbook_optimization_candidates"
            )
        self.conn.commit()

    def _migrate_playbook_optimization_job_columns(self) -> None:
        """Add durable optimizer identity, lease, stage, and digest columns."""
        existing_cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_jobs)"
            ).fetchall()
        }
        if not existing_cols:
            return
        columns = {
            "optimizer_kind": "TEXT",
            "discovery_key": "TEXT",
            "attempt_key": "TEXT",
            "lease_owner": "TEXT",
            "lease_fence": "INTEGER NOT NULL DEFAULT 0",
            "lease_expires_at": "INTEGER",
            "stage": "TEXT",
            "terminal_outcome": "TEXT",
            "expected_population_manifest_digest": "TEXT",
            "generation_selection_manifest_digest": "TEXT",
            "replay_manifest_digest": "TEXT",
            "candidate_content_digest": "TEXT",
            "search_projection_digest": "TEXT",
            "publication_scope_digest": "TEXT",
        }
        for column, definition in columns.items():
            if column not in existing_cols:
                self.conn.execute(
                    f"ALTER TABLE playbook_optimization_jobs "
                    f"ADD COLUMN {column} {definition}"  # noqa: S608
                )
        self.conn.commit()

    def _migrate_user_playbook_publication_staging_columns(self) -> None:
        """Add the frozen incumbent CAS and provisional fields to legacy staging."""
        existing_cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(user_playbook_publication_staging)"
            ).fetchall()
        }
        if not existing_cols:
            return
        columns = {
            "incumbent_content_digest": "TEXT",
            "incumbent_trigger": "TEXT",
            "incumbent_semantic_digest": "TEXT",
            **PROVISIONAL_STAGING_COLUMNS,
        }
        for column, definition in columns.items():
            if column not in existing_cols:
                self.conn.execute(
                    "ALTER TABLE user_playbook_publication_staging "
                    f"ADD COLUMN {column} {definition}"  # noqa: S608
                )
        self.conn.commit()

    def _migrate_user_playbook_publication_results_shape(self) -> None:
        """Rebuild a result table created before provisional publication.

        Its ``outcome`` CHECK admits only the GEPA outcomes and it lacks the
        lifecycle and retry-identity columns. SQLite cannot widen a CHECK in
        place, so the table is recreated from ``PUBLICATION_RESULTS_DDL`` and
        every row copied; a GEPA row is valid under the new CHECK unchanged.
        A no-op when the table is absent or already current.
        """
        with self._lock:
            columns = [
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(user_playbook_publication_results)"
                ).fetchall()
            ]
            if not columns or "deployment_lifecycle_id" in columns:
                return
            kept = ", ".join(columns)
            self.conn.commit()
            self.conn.execute("PRAGMA foreign_keys=OFF")
            try:
                with self.conn:
                    self.conn.execute("BEGIN")
                    self.conn.execute(
                        "ALTER TABLE user_playbook_publication_results"
                        " RENAME TO user_playbook_publication_results_previous"
                    )
                    self.conn.execute(PUBLICATION_RESULTS_DDL)
                    self.conn.execute(
                        f"INSERT INTO user_playbook_publication_results ({kept})"  # noqa: S608
                        f" SELECT {kept} FROM user_playbook_publication_results_previous"
                    )
                    self.conn.execute(
                        "DROP TABLE user_playbook_publication_results_previous"
                    )
            finally:
                self.conn.execute("PRAGMA foreign_keys=ON")

    def _classify_legacy_playbook_optimization_jobs(self) -> None:
        """Classify legacy optimizer history without choosing ambiguous rule order."""
        columns = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_jobs)"
            ).fetchall()
        }
        if "optimizer_kind" not in columns:
            return
        index_statements = (
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_discovery
               ON playbook_optimization_jobs(optimizer_kind, discovery_key)
               WHERE status IN ('pending', 'running')
                 AND discovery_key IS NOT NULL""",
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_attempt
               ON playbook_optimization_jobs(optimizer_kind, attempt_key)
               WHERE status IN ('pending', 'running')
                 AND attempt_key IS NOT NULL""",
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_target
               ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
               WHERE status IN ('pending', 'running')""",
        )
        # The constraint rebuild emits equivalent declarations with different
        # whitespace. These three statements have no whitespace inside their
        # quoted literals, so whitespace folding cannot change an accepted value.
        expected_indexes = {
            statement.split()[6]: " ".join(
                statement.replace(" IF NOT EXISTS", "", 1).split()
            )
            for statement in index_statements
        }
        current_indexes = dict(
            self.conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
                "AND name IN ('uq_poj_active_discovery', 'uq_poj_active_attempt', 'uq_poj_active_target')"
            ).fetchall()
        )
        current_indexes = {
            name: " ".join(sql.split()) for name, sql in current_indexes.items()
        }
        if (
            current_indexes == expected_indexes
            and self.conn.execute(
                "SELECT 1 FROM playbook_optimization_jobs WHERE optimizer_kind IS NULL "
                "OR (optimizer_kind IN ('offline_tuner_legacy', 'optimizer_legacy_unknown') "
                "AND status IN ('pending', 'running')) LIMIT 1"
            ).fetchone()
            is None
        ):
            return
        for index_name in (
            "uq_poj_active_discovery",
            "uq_poj_active_attempt",
            "uq_poj_active_target",
        ):
            self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")  # noqa: S608
        # Every json_type() is guarded with CASE WHEN json_valid(...), never
        # `json_valid(x) AND json_type(x, ...)`. SQLite does not short-circuit
        # AND in a result-column expression: it evaluates json_type on a
        # non-JSON value and raises "malformed JSON". metadata_json is a plain
        # str the older schema never validated, and this runs in the storage
        # constructor, so one such row would stop the backend starting after
        # an upgrade. That is reproduced for the two job-level guards. The
        # candidate guard sits in a subquery WHERE, which does short-circuit
        # today, but WHERE evaluation order is not guaranteed -- it is CASE
        # too as hardening. CASE is lazy. Do not "simplify" any of it to AND.
        self.conn.execute(
            """
            WITH signatures AS (
                SELECT
                    jobs.job_id,
                    (
                        EXISTS (
                            SELECT 1
                            FROM playbook_optimization_events AS events
                            WHERE events.job_id = jobs.job_id
                              AND events.event_type LIKE 'offline_tuner_%'
                        )
                        OR CASE WHEN json_valid(jobs.metadata_json) THEN
                            json_type(jobs.metadata_json, '$.offline_tuner')
                                IS NOT NULL
                        ELSE 0 END
                        OR EXISTS (
                            SELECT 1
                            FROM playbook_optimization_candidates AS candidates
                            WHERE candidates.job_id = jobs.job_id
                              AND CASE WHEN json_valid(candidates.metadata_json) THEN (
                                  json_type(
                                      candidates.metadata_json,
                                      '$.offline_tuner_metrics'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.rollback_baseline'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.frozen_selection_set'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.proposed_edit'
                                  ) IS NOT NULL
                              ) ELSE 0 END
                        )
                    ) AS tuner_signature,
                    CASE WHEN json_valid(jobs.metadata_json) THEN (
                        json_type(jobs.metadata_json, '$.source_window_count')
                            IS NOT NULL
                        AND json_type(jobs.metadata_json, '$.train_window_count')
                            IS NOT NULL
                        AND json_type(
                            jobs.metadata_json,
                            '$.validation_window_count'
                        ) IS NOT NULL
                    ) ELSE 0 END AS gepa_signature
                FROM playbook_optimization_jobs AS jobs
                WHERE jobs.optimizer_kind IS NULL
            )
            UPDATE playbook_optimization_jobs
            SET optimizer_kind = CASE
                WHEN tuner_signature AND NOT gepa_signature
                    THEN 'offline_tuner_legacy'
                WHEN gepa_signature AND NOT tuner_signature THEN 'gepa'
                ELSE 'optimizer_legacy_unknown'
            END
            FROM signatures
            WHERE playbook_optimization_jobs.job_id = signatures.job_id
            """
        )
        self.conn.execute(
            """
            WITH ranked AS (
                SELECT
                    job_id,
                    row_number() OVER (
                        PARTITION BY optimizer_kind, target_kind, target_id
                        ORDER BY created_at, job_id
                    ) AS target_rank,
                    CASE WHEN discovery_key IS NOT NULL THEN row_number() OVER (
                        PARTITION BY optimizer_kind, discovery_key
                        ORDER BY created_at, job_id
                    ) END AS discovery_rank,
                    CASE WHEN attempt_key IS NOT NULL THEN row_number() OVER (
                        PARTITION BY optimizer_kind, attempt_key
                        ORDER BY created_at, job_id
                    ) END AS attempt_rank
                FROM playbook_optimization_jobs
                WHERE status IN ('pending', 'running')
            )
            UPDATE playbook_optimization_jobs
            SET status = 'skipped',
                decision_reason = 'retired_duplicate_legacy_active_job',
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = CAST(strftime('%s', 'now') AS INTEGER)
            WHERE job_id IN (
                SELECT job_id
                FROM ranked
                WHERE target_rank > 1
                   OR discovery_rank > 1
                   OR attempt_rank > 1
            )
            """
        )
        self.conn.execute(
            """
            UPDATE playbook_optimization_jobs
            SET status = 'skipped',
                decision_reason = 'retired_by_replay_redesign',
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = CAST(strftime('%s', 'now') AS INTEGER)
            WHERE optimizer_kind IN (
                    'offline_tuner_legacy',
                    'optimizer_legacy_unknown'
                )
              AND status IN ('pending', 'running')
            """
        )
        for statement in index_statements:
            self.conn.execute(statement)
        self.conn.commit()

    def _enforce_playbook_optimization_job_constraints(self) -> None:
        """Rebuild upgraded optimizer tables so legacy values receive fresh checks."""
        table_sql_row = self.conn.execute(
            """SELECT sql FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_jobs'"""
        ).fetchone()
        if table_sql_row is None:
            return
        table_sql = table_sql_row["sql"]
        # Every literal the FINAL constraints admit must appear, not just a
        # representative few. A predicate that names some of them cannot tell
        # "current" from "missing a kind": a schema whose optimizer_kind CHECK
        # omits 'gepa' carries no retired literal, satisfies a partial list,
        # and is retained forever with a constraint that rejects legitimate
        # rows. The artifact rebuild below already enumerates its full set;
        # this is that same rule.
        required_checks = (
            "CHECK (optimizer_kind IN",
            "CHECK (stage IS NULL OR stage IN",
            "CHECK (terminal_outcome IS NULL OR terminal_outcome IN",
            "'gepa'",
            "'offline_tuner_open_world'",
            "'offline_tuner_legacy'",
            "'optimizer_legacy_unknown'",
            "'governance_erased'",
            "'discovery_analyzed'",
            "'held_out_analyzed'",
            *_TERMINAL_OUTCOME_LITERALS,
        )
        if all(check in table_sql for check in required_checks) and not any(
            retired in table_sql for retired in _RETIRED_OPTIMIZER_JOB_LITERALS
        ):
            return
        foreign_keys_enabled = bool(
            self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        )
        self.conn.commit()
        if foreign_keys_enabled:
            self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            sequence_high_water = self._autoincrement_high_water(
                "playbook_optimization_jobs",
                "job_id",
            )
            self.conn.execute("DROP TABLE IF EXISTS playbook_optimization_jobs_new")
            self.conn.execute(
                """
            CREATE TABLE playbook_optimization_jobs_new (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimizer_kind TEXT NOT NULL DEFAULT 'optimizer_legacy_unknown'
                    CHECK (optimizer_kind IN (
                        'gepa',
                        'offline_tuner_open_world',
                        'offline_tuner_legacy',
                        'optimizer_legacy_unknown'
                    )),
                target_kind TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                best_candidate_id INTEGER,
                successor_target_id INTEGER,
                decision_reason TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                discovery_key TEXT,
                attempt_key TEXT,
                lease_owner TEXT,
                lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
                lease_expires_at INTEGER,
                stage TEXT CHECK (stage IS NULL OR stage IN (
                    'evidence_frozen',
                    'discovery_analyzed',
                    'candidate_generated',
                    'held_out_analyzed',
                    'publishing',
                    'applied',
                    'abstained',
                    'failed'
                )),
                __TERMINAL_OUTCOME_CHECK__,
                expected_population_manifest_digest TEXT,
                generation_selection_manifest_digest TEXT,
                replay_manifest_digest TEXT,
                candidate_content_digest TEXT,
                search_projection_digest TEXT,
                publication_scope_digest TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """.replace("__TERMINAL_OUTCOME_CHECK__", _TERMINAL_OUTCOME_CHECK)
            )
            # Phase 7 review finding P4-2. The pre-existing legacy sweep in
            # _classify_legacy_playbook_optimization_jobs retires every ACTIVE
            # 'offline_tuner_legacy' / 'optimizer_legacy_unknown' job, but it
            # runs BEFORE this relabel. Without this statement the relabel below
            # turns a running replay job into a running 'offline_tuner_legacy'
            # job -- a kind with no producer or consumer -- which then holds a
            # uq_poj_active_target / uq_poj_active_discovery / uq_poj_active_attempt
            # slot until the next storage open lets the sweep see it. Retire it
            # here with exactly the status, reason and lease clearing that sweep
            # uses, so one open is enough.
            self.conn.execute(
                """
                UPDATE playbook_optimization_jobs
                SET status = 'skipped',
                    decision_reason = 'retired_by_replay_redesign',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = CAST(strftime('%s', 'now') AS INTEGER)
                WHERE optimizer_kind = 'offline_tuner_replay'
                  AND status IN ('pending', 'running')
                """
            )
            self.conn.execute(
                """
            INSERT INTO playbook_optimization_jobs_new (
                job_id, optimizer_kind, target_kind, target_id, status,
                best_candidate_id, successor_target_id, decision_reason,
                metadata_json, discovery_key, attempt_key, lease_owner,
                lease_fence, lease_expires_at, stage, terminal_outcome,
                expected_population_manifest_digest,
                generation_selection_manifest_digest, replay_manifest_digest,
                candidate_content_digest, search_projection_digest,
                publication_scope_digest, created_at, updated_at
            ) SELECT
                job_id,
                -- Phase 7 remediation. The copy would otherwise fail the new
                -- CHECK on any row still carrying a retired literal.
                -- 'offline_tuner_legacy' is the accurate surviving label: these
                -- jobs WERE offline-tuner jobs, and OQ-1 option A keeps that
                -- literal admissible. stage and terminal_outcome are nullable,
                -- so their retired values become NULL rather than a wrong one.
                CASE WHEN optimizer_kind = 'offline_tuner_replay'
                     THEN 'offline_tuner_legacy'
                     ELSE optimizer_kind END,
                target_kind, target_id, status,
                best_candidate_id, successor_target_id, decision_reason,
                metadata_json, discovery_key, attempt_key, lease_owner,
                lease_fence, lease_expires_at,
                CASE WHEN stage IN ('replay_running', 'replay_evaluated')
                     THEN NULL ELSE stage END,
                CASE WHEN terminal_outcome IN (
                         'replay_unsupported', 'incomplete_replay_scope',
                         'insufficient_replay_cases', 'replay_inconclusive',
                         'replay_failed'
                     ) THEN NULL ELSE terminal_outcome END,
                expected_population_manifest_digest,
                generation_selection_manifest_digest, replay_manifest_digest,
                candidate_content_digest, search_projection_digest,
                publication_scope_digest, created_at, updated_at
            FROM playbook_optimization_jobs;
            """
            )
            self.conn.execute("DROP TABLE playbook_optimization_jobs")
            self.conn.execute(
                "ALTER TABLE playbook_optimization_jobs_new "
                "RENAME TO playbook_optimization_jobs"
            )
            self._restore_autoincrement_high_water(
                "playbook_optimization_jobs",
                sequence_high_water,
            )
            self.conn.execute(
                "CREATE INDEX idx_poj_target "
                "ON playbook_optimization_jobs(target_kind, target_id)"
            )
            self.conn.execute(
                "CREATE INDEX idx_poj_status ON playbook_optimization_jobs(status)"
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_discovery
                ON playbook_optimization_jobs(optimizer_kind, discovery_key)
                WHERE status IN ('pending', 'running') AND discovery_key IS NOT NULL"""
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_attempt
                ON playbook_optimization_jobs(optimizer_kind, attempt_key)
                WHERE status IN ('pending', 'running') AND attempt_key IS NOT NULL"""
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_target
                ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
                WHERE status IN ('pending', 'running')"""
            )
            violations = self.conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(
                    f"foreign key check failed after optimizer migration: {violations}"
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute(
                f"PRAGMA foreign_keys={'ON' if foreign_keys_enabled else 'OFF'}"
            )

    def _enforce_playbook_optimization_artifact_constraints(self) -> None:
        """Rebuild legacy artifact tables with the current kind allowlist."""
        table_sql_row = self.conn.execute(
            """SELECT sql FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_artifacts'"""
        ).fetchone()
        if table_sql_row is None:
            return
        table_sql = table_sql_row["sql"]
        artifact_kinds = (
            "'expected_population_manifest'",
            "'generation_selection'",
            "'candidate'",
            "'candidate_search_projection'",
            "'open_world_evidence_bundle'",
            "'open_world_discovery_memo'",
            "'open_world_candidate'",
            "'open_world_content_result'",
            "'open_world_trigger_result'",
            "'open_world_attempt_decision'",
        )
        if all(
            artifact_kind in table_sql for artifact_kind in artifact_kinds
        ) and not any(
            retired in table_sql for retired in _RETIRED_ARTIFACT_KIND_LITERALS
        ):
            return

        foreign_keys_enabled = bool(
            self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        )
        self.conn.commit()
        if foreign_keys_enabled:
            self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            sequence_high_water = self._autoincrement_high_water(
                "playbook_optimization_artifacts",
                "artifact_id",
            )
            self.conn.execute(
                "DROP TABLE IF EXISTS playbook_optimization_artifacts_new"
            )
            self.conn.execute(
                """
                CREATE TABLE playbook_optimization_artifacts_new (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
                        'expected_population_manifest',
                        'generation_selection',
                        'candidate',
                        'candidate_search_projection',
                        'open_world_evidence_bundle',
                        'open_world_discovery_memo',
                        'open_world_candidate',
                        'open_world_content_result',
                        'open_world_trigger_result',
                        'open_world_attempt_decision'
                    )),
                    content_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE (job_id, artifact_kind),
                    FOREIGN KEY (job_id)
                        REFERENCES playbook_optimization_jobs(job_id)
                        ON DELETE CASCADE
                )
                """
            )
            # artifact_kind is NOT NULL and part of UNIQUE (job_id,
            # artifact_kind), so there is no surviving value to map
            # 'replay_manifest' onto. Local development SQLite only -- no tenant
            # Postgres row is touched by this path.
            #
            # Phase 7 review finding P4-3: this is the OPPOSITE strategy from the
            # sibling optimizer_kind vocabulary in
            # _enforce_playbook_optimization_job_constraints, where OQ-1 option A
            # RETAINS 'offline_tuner_legacy' / 'optimizer_legacy_unknown'
            # permanently as historical labels precisely so historical rows
            # survive. Retention was available here too and was rejected: the
            # tenant contract (supabase/data/tenant/20260830020000, constraint
            # playbook_optimization_artifacts_artifact_kind_check) drops
            # 'replay_manifest' outright, so retaining it on SQLite alone would
            # leave the two backends disagreeing about the artifact
            # vocabulary -- the dual-expression divergence the rest of
            # Phase 7 exists to remove. The rows are destroyed rather than
            # migrated, so the deletion is logged with its count instead of being
            # silent.
            deleted_replay_manifests = self.conn.execute(
                "DELETE FROM playbook_optimization_artifacts "
                "WHERE artifact_kind = 'replay_manifest'"
            ).rowcount
            if deleted_replay_manifests > 0:
                logger.warning(
                    "Phase 7 artifact contraction destroyed %d "
                    "'replay_manifest' artifact row(s) with their content_json: "
                    "the retired kind has no surviving artifact_kind to map onto "
                    "and the tenant contract drops it as well",
                    deleted_replay_manifests,
                )
            self.conn.execute(
                """
                INSERT INTO playbook_optimization_artifacts_new (
                    artifact_id, job_id, artifact_kind, content_json,
                    content_digest, created_at, updated_at
                ) SELECT
                    artifact_id, job_id, artifact_kind, content_json,
                    content_digest, created_at, updated_at
                FROM playbook_optimization_artifacts
                """
            )
            self.conn.execute("DROP TABLE playbook_optimization_artifacts")
            self.conn.execute(
                "ALTER TABLE playbook_optimization_artifacts_new "
                "RENAME TO playbook_optimization_artifacts"
            )
            self._restore_autoincrement_high_water(
                "playbook_optimization_artifacts",
                sequence_high_water,
            )
            self.conn.execute(
                "CREATE INDEX idx_poa_job ON playbook_optimization_artifacts(job_id)"
            )
            violations = self.conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(
                    "foreign key check failed after optimizer artifact migration: "
                    f"{violations}"
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute(
                f"PRAGMA foreign_keys={'ON' if foreign_keys_enabled else 'OFF'}"
            )

    def _autoincrement_high_water(self, table_name: str, id_column: str) -> int:
        sequence_row = self.conn.execute(
            "SELECT MAX(seq) FROM sqlite_sequence WHERE name = ?",
            (table_name,),
        ).fetchone()
        maximum_row = self.conn.execute(
            f"SELECT MAX({id_column}) FROM {table_name}"  # noqa: S608
        ).fetchone()
        return max(sequence_row[0] or 0, maximum_row[0] or 0)

    def _restore_autoincrement_high_water(
        self,
        table_name: str,
        high_water: int,
    ) -> None:
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table_name,))
        if high_water:
            self.conn.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)",
                (table_name, high_water),
            )

    def _migrate_retire_profile_change_logs(self) -> None:
        """Retire the frozen ``profile_change_logs`` table via a reversible RENAME.

        Lineage B3 Task 8: the legacy change log is fully de-referenced (no
        readers, writers, or GDPR delete callers remain) and the view is served
        from reconstruction. We rename the table out of the way now and DROP it
        in a later migration after the recovery window — keeping the data
        recoverable in the interim.

        Idempotent: SQLite has no ``RENAME ... IF EXISTS``, so we guard on the
        source table's presence and no-op once it has been renamed. The
        ``CREATE TABLE`` for ``profile_change_logs`` was deleted from ``_DDL`` so
        ``executescript(_DDL)`` does not recreate an empty table after the rename.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("profile_change_logs",),
            ).fetchone()
            if row is None:
                return
            target_row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("profile_change_logs_retired_20260623",),
            ).fetchone()
            if target_row is not None:
                logger.info(
                    "Retired target profile_change_logs_retired_20260623 already exists;"
                    " skipping rename (idempotent no-op)."
                )
                return
            self.conn.execute(
                "ALTER TABLE profile_change_logs "
                "RENAME TO profile_change_logs_retired_20260623"
            )
            logger.info(
                "Renamed profile_change_logs to "
                "profile_change_logs_retired_20260623 (B3 Task 8 retirement)"
            )
            self.conn.commit()

    def _migrate_retire_playbook_aggregation_change_logs(self) -> None:
        """Retire the frozen ``playbook_aggregation_change_logs`` table via a reversible RENAME.

        Lineage Track B Task 4: the legacy change log is fully de-referenced (no
        readers, writers, or GDPR delete callers remain) and the view is served
        from reconstruction. We rename the table out of the way now and DROP it
        in a later migration after the recovery window — keeping the data
        recoverable in the interim.

        Idempotent: SQLite has no ``RENAME ... IF EXISTS``, so we guard on the
        source table's presence and no-op once it has been renamed. The
        ``CREATE TABLE`` for ``playbook_aggregation_change_logs`` was deleted from
        ``_DDL`` so ``executescript(_DDL)`` does not recreate an empty table after
        the rename.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("playbook_aggregation_change_logs",),
            ).fetchone()
            if row is None:
                return
            target_row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("playbook_aggregation_change_logs_retired_20260624",),
            ).fetchone()
            if target_row is not None:
                logger.info(
                    "Retired target playbook_aggregation_change_logs_retired_20260624"
                    " already exists; skipping rename (idempotent no-op)."
                )
                return
            self.conn.execute(
                "ALTER TABLE playbook_aggregation_change_logs "
                "RENAME TO playbook_aggregation_change_logs_retired_20260624"
            )
            logger.info(
                "Renamed playbook_aggregation_change_logs to "
                "playbook_aggregation_change_logs_retired_20260624 "
                "(Track B Task 4 retirement)"
            )
            self.conn.commit()

    def _migrate_agent_playbook_source_windows(self) -> None:
        """Add source window snapshots to existing agent source mappings."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(agent_playbook_source_user_playbooks)"
            ).fetchall()
        }
        if not cols:
            return
        if "source_interaction_ids" not in cols:
            self.conn.execute(
                "ALTER TABLE agent_playbook_source_user_playbooks "
                "ADD COLUMN source_interaction_ids TEXT NOT NULL DEFAULT '[]'"
            )
            logger.info(
                "Added source_interaction_ids column to "
                "agent_playbook_source_user_playbooks"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apsup_user "
            "ON agent_playbook_source_user_playbooks(user_playbook_id)"
        )
        self.conn.commit()

    def _migrate_request_evaluation_only(self) -> None:
        """Add evaluation_only column to requests for learning exclusion."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols:
            return
        if "evaluation_only" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN evaluation_only INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("Added evaluation_only column to requests")
        self.conn.commit()

    def _migrate_request_retrieval_experiment(self) -> None:
        """Add nullable retrieval experiment attribution to existing requests."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols:
            return
        if "retrieval_experiment_id" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN retrieval_experiment_id TEXT"
            )
        if "retrieval_experiment_arm" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN retrieval_experiment_arm TEXT"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment "
            "ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id)"
        )
        self.conn.commit()

    def _migrate_request_session_id_required(self) -> None:
        """Require non-empty session ids on ``requests``.

        SQLite cannot add a ``NOT NULL`` or ``CHECK`` constraint to an
        existing column, so existing databases are rebuilt in place. Historical
        null/blank sessions are intentionally backfilled per request to avoid
        inventing conversation groupings that were never recorded.
        """
        cols = {
            row["name"]: row
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols or "session_id" not in cols:
            return

        table_sql_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'requests'"
        ).fetchone()
        table_sql = (table_sql_row["sql"] if table_sql_row else "") or ""
        blank_count = self.conn.execute(
            "SELECT COUNT(*) FROM requests WHERE session_id IS NULL OR trim(session_id) = ''"
        ).fetchone()[0]
        has_required_schema = (
            bool(cols["session_id"]["notnull"])
            and "CHECK (trim(session_id) != '')" in table_sql
        )
        if has_required_schema and blank_count == 0:
            return

        evaluation_only_expr = (
            "COALESCE(evaluation_only, 0)" if "evaluation_only" in cols else "0"
        )
        retrieval_experiment_id_expr = (
            "retrieval_experiment_id" if "retrieval_experiment_id" in cols else "NULL"
        )
        retrieval_experiment_arm_expr = (
            "retrieval_experiment_arm" if "retrieval_experiment_arm" in cols else "NULL"
        )
        governance_subject_ref_expr = (
            "governance_subject_ref" if "governance_subject_ref" in cols else "NULL"
        )
        # NOTE: this rebuild hardcodes the full `requests` column set. If a
        # future migration adds a column to `requests`, it MUST be added here
        # too (and to the SELECT below) or the rebuild will silently drop it.
        self.conn.executescript(
            f"""
            CREATE TABLE requests_new (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                agent_version TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL CHECK (trim(session_id) != ''),
                evaluation_only INTEGER NOT NULL DEFAULT 0,
                retrieval_experiment_id TEXT,
                retrieval_experiment_arm TEXT,
                governance_subject_ref TEXT
            );
            INSERT INTO requests_new
                (
                    request_id,
                    user_id,
                    created_at,
                    source,
                    agent_version,
                    session_id,
                    evaluation_only,
                    retrieval_experiment_id,
                    retrieval_experiment_arm,
                    governance_subject_ref
                )
            SELECT
                request_id,
                user_id,
                created_at,
                COALESCE(source, ''),
                COALESCE(agent_version, ''),
                CASE
                    WHEN session_id IS NULL OR trim(session_id) = ''
                    THEN 'legacy-' || lower(hex(randomblob(16)))
                    ELSE trim(session_id)
                END,
                {evaluation_only_expr},
                {retrieval_experiment_id_expr},
                {retrieval_experiment_arm_expr},
                {governance_subject_ref_expr}
            FROM requests;
            DROP TABLE requests;
            ALTER TABLE requests_new RENAME TO requests;
            CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
            CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
            CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
            CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment
                ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id);
            CREATE INDEX IF NOT EXISTS idx_requests_governance_subject_ref
                ON requests(governance_subject_ref);
            """
        )
        self.conn.commit()
        logger.info("Migrated requests.session_id to required non-empty values")

    def _migrate_eval_result_user_id(self) -> None:
        """Add user_id to session evaluation results for per-user identity."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(agent_success_evaluation_result)"
            ).fetchall()
        }
        if not cols:
            return
        if "user_id" not in cols:
            self.conn.execute(
                "ALTER TABLE agent_success_evaluation_result "
                "ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
            logger.info("Added user_id column to agent_success_evaluation_result")
        index_sql = (
            "CREATE INDEX idx_eval_identity_created_at_desc "
            "ON agent_success_evaluation_result"
            "(user_id, session_id, evaluation_name, agent_version, created_at DESC)"
        )
        existing = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=? "
            "AND tbl_name='agent_success_evaluation_result'",
            ("idx_eval_identity_created_at_desc",),
        ).fetchone()
        # This declaration contains identifiers only; whitespace and keyword case
        # do not change its meaning. Other definitions retain actual replacement.
        if existing and " ".join(existing[0].split()).casefold() == (
            " ".join(index_sql.split()).casefold()
        ):
            return
        self.conn.execute("DROP INDEX IF EXISTS idx_eval_identity_created_at_desc")
        self.conn.execute(
            index_sql.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)
        )
        self.conn.commit()

    def _migrate_shadow_comparison_verdicts(self) -> None:
        """F1: create the shadow_comparison_verdicts table if missing.

        Idempotent; safe to run on every startup. The PRAGMA-LBYL guard
        avoids running the CREATE statements on every boot for
        already-migrated DBs. The ``CREATE TABLE IF NOT EXISTS`` in :data:`_DDL` will
        also create this table on a fresh database, so this helper is a
        no-op there; its purpose is explicit symmetry with the per-feature
        migration convention and a single named hook the disk/supabase
        backends in Tasks 6/7 can mirror.
        """
        cur = self.conn.execute("PRAGMA table_info(shadow_comparison_verdicts)")
        cols = cur.fetchall()
        if cols:
            return
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS shadow_comparison_verdicts (
                verdict_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id          TEXT    NOT NULL,
                session_id              TEXT    NOT NULL,
                agent_version           TEXT    NOT NULL,
                reflexio_is_request_1   INTEGER NOT NULL,
                better_request          TEXT    NOT NULL CHECK (better_request IN ('1','2','tie')),
                is_significantly_better INTEGER NOT NULL,
                comparison_reason       TEXT,
                judge_prompt_version    TEXT    NOT NULL,
                created_at              TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_session
                ON shadow_comparison_verdicts (session_id, agent_version);
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_created_at
                ON shadow_comparison_verdicts (created_at);
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_v
                ON shadow_comparison_verdicts (judge_prompt_version);
        """)
        self.conn.commit()
        logger.info("Created shadow_comparison_verdicts table (F1 migration)")

    def _migrate_retrieved_learning_interaction_identity(self) -> None:
        """Rebuild legacy session-learning verdicts for occurrence identity.

        Existing rows are preserved with nullable interaction fields. Fresh
        evaluator runs replace a session snapshot with fully attributed rows.
        """
        columns = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(retrieved_learning_evaluation)"
            ).fetchall()
        }
        if not columns or {
            "interaction_id",
            "interaction_created_at",
        }.issubset(columns):
            return
        self.conn.executescript("""
            CREATE TABLE retrieved_learning_evaluation_new (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                agent_version TEXT NOT NULL DEFAULT '',
                interaction_id INTEGER,
                interaction_created_at INTEGER,
                kind TEXT NOT NULL,
                learning_id TEXT NOT NULL,
                is_relevant INTEGER,
                relevance_reason TEXT NOT NULL DEFAULT '',
                impact TEXT,
                impact_reason TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                governance_subject_ref TEXT,
                UNIQUE (
                    user_id, session_id, interaction_id, kind, learning_id
                )
            );
            INSERT INTO retrieved_learning_evaluation_new (
                result_id, user_id, session_id, agent_version, kind,
                learning_id, is_relevant, relevance_reason, impact,
                impact_reason, created_at, governance_subject_ref
            )
            SELECT
                result_id, user_id, session_id, agent_version, kind,
                learning_id, is_relevant, relevance_reason, impact,
                impact_reason, created_at, governance_subject_ref
            FROM retrieved_learning_evaluation;
            DROP TABLE retrieved_learning_evaluation;
            ALTER TABLE retrieved_learning_evaluation_new
                RENAME TO retrieved_learning_evaluation;
            CREATE INDEX IF NOT EXISTS idx_rle_created_at_result_id
                ON retrieved_learning_evaluation(created_at DESC, result_id DESC);
            CREATE INDEX IF NOT EXISTS idx_rle_interaction_created_at
                ON retrieved_learning_evaluation(
                    interaction_created_at DESC, interaction_id DESC, result_id DESC
                );
            CREATE INDEX IF NOT EXISTS idx_rle_session_id
                ON retrieved_learning_evaluation(session_id);
            CREATE INDEX IF NOT EXISTS idx_rle_subject_ref
                ON retrieved_learning_evaluation(governance_subject_ref);
        """)
        self.conn.commit()
        logger.info(
            "Migrated retrieved-learning verdicts to interaction occurrence identity"
        )
