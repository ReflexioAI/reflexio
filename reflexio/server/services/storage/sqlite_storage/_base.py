"""
Base class and module-level helpers for SQLite storage.

Supports hybrid search combining FTS5 (BM25) with embedding cosine similarity
via Reciprocal Rank Fusion (RRF). Falls back to FTS-only when no embeddings
are available.

"""

import logging
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from reflexio.models.config_schema import (
    EMBEDDING_DIMENSIONS,
    APIKeyConfig,
    LLMConfig,
)
from reflexio.server import initialize_server_runtime
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)
from reflexio.server.services.embedding_text import embedding_input
from reflexio.server.services.storage.error import (
    StorageError,
)
from reflexio.server.services.storage.retention_mixin import RetentionMixin
from reflexio.server.services.storage.storage_base import BaseStorage

# isort: off
# Moved out of this module; kept importable from here for the rest of the
# package, its tests and enterprise code.
from ._connection import (
    SQLiteConnectionMixin as SQLiteConnectionMixin,
    _MINIMUM_SQLITE_VERSION as _MINIMUM_SQLITE_VERSION,
    _SQLITE_INITIALIZATION_LOCK_STRIPES as _SQLITE_INITIALIZATION_LOCK_STRIPES,
    _get_sqlite_initialization_lock as _get_sqlite_initialization_lock,
    _sqlite_initialization_locks as _sqlite_initialization_locks,
)
from ._ddl import (
    PROVISIONAL_STAGING_COLUMNS as PROVISIONAL_STAGING_COLUMNS,
    PUBLICATION_RESULTS_DDL as PUBLICATION_RESULTS_DDL,
    _DDL as _DDL,
    _RETIRED_ARTIFACT_KIND_LITERALS as _RETIRED_ARTIFACT_KIND_LITERALS,
    _RETIRED_OPTIMIZER_JOB_LITERALS as _RETIRED_OPTIMIZER_JOB_LITERALS,
    _TERMINAL_OUTCOME_CHECK as _TERMINAL_OUTCOME_CHECK,
    _TERMINAL_OUTCOME_LITERALS as _TERMINAL_OUTCOME_LITERALS,
)
from ._row_mappers import (
    _MAX_SAFE_EPOCH_TS as _MAX_SAFE_EPOCH_TS,
    _MIN_CONTEMPORARY_MILLISECOND_EPOCH_TS as _MIN_CONTEMPORARY_MILLISECOND_EPOCH_TS,
    _MIN_SAFE_EPOCH_TS as _MIN_SAFE_EPOCH_TS,
    _build_status_sql as _build_status_sql,
    _epoch_now as _epoch_now,
    _epoch_to_iso as _epoch_to_iso,
    _iso_now as _iso_now,
    _iso_to_epoch as _iso_to_epoch,
    _json_dumps as _json_dumps,
    _json_loads as _json_loads,
    _row_to_agent_playbook as _row_to_agent_playbook,
    _row_to_eval_result as _row_to_eval_result,
    _row_to_interaction as _row_to_interaction,
    _row_to_profile as _row_to_profile,
    _row_to_request as _row_to_request,
    _row_to_user_playbook as _row_to_user_playbook,
    _status_value as _status_value,
    parse_status as parse_status,
)
from ._search_ranking import (
    _FTS5_OPERATORS as _FTS5_OPERATORS,
    _FTS5_RESERVED as _FTS5_RESERVED,
    _TOKEN_RE as _TOKEN_RE,
    _UNICODE_LEXICAL_INDEX_VERSION as _UNICODE_LEXICAL_INDEX_VERSION,
    _UNICODE_LEXICAL_MAX_CANDIDATES as _UNICODE_LEXICAL_MAX_CANDIDATES,
    _UNICODE_LEXICAL_MAX_TERMS as _UNICODE_LEXICAL_MAX_TERMS,
    _cosine_similarity as _cosine_similarity,
    _effective_search_mode as _effective_search_mode,
    _normalize_lexical_text as _normalize_lexical_text,
    _rank_unicode_lexical_rows as _rank_unicode_lexical_rows,
    _sanitize_fts_query as _sanitize_fts_query,
    _true_rrf_merge as _true_rrf_merge,
    _unicode_lexical_candidate_limit as _unicode_lexical_candidate_limit,
    _unicode_lexical_fts_query as _unicode_lexical_fts_query,
    _unicode_lexical_index_text as _unicode_lexical_index_text,
    _unicode_lexical_runs as _unicode_lexical_runs,
    _unicode_lexical_score as _unicode_lexical_score,
    _unicode_lexical_terms as _unicode_lexical_terms,
    _uses_unicode_lexical_fallback as _uses_unicode_lexical_fallback,
    _vector_rank_rows as _vector_rank_rows,
    register_unicode_lexical_index_function as register_unicode_lexical_index_function,
)
from ._session_trajectory import (
    _SESSION_OUTCOME_MIGRATION_BATCH_SIZE as _SESSION_OUTCOME_MIGRATION_BATCH_SIZE,
    _TRAJECTORY_FETCH_SIZE as _TRAJECTORY_FETCH_SIZE,
    _canonical_session_trajectory_digest as _canonical_session_trajectory_digest,
    _canonical_session_trajectory_snapshot as _canonical_session_trajectory_snapshot,
    _legacy_session_outcome_id as _legacy_session_outcome_id,
    _prefetch_canonical_session_trajectory_digests as _prefetch_canonical_session_trajectory_digests,
    _session_prefix_state as _session_prefix_state,
    _through_prefix_predicate as _through_prefix_predicate,
)

# isort: on
from ._dataset_path import resolve_sqlite_db_path

# Moved out of this module; kept importable from here for the rest of the
# package, its tests and enterprise code.
from ._learning_policy_authority import (
    complete_sqlite_dataset_creation,
    reserve_new_sqlite_dataset,
    retain_private_memory_creation,
)
from .migrations._steps import SQLiteSchemaMigrationsMixin

logger = logging.getLogger(__name__)


class SQLiteStorageBase(
    RetentionMixin, SQLiteConnectionMixin, SQLiteSchemaMigrationsMixin, BaseStorage
):
    """SQLite-backed storage base class for local/self-hosted deployments."""

    supports_embedding: ClassVar[bool] = True

    # Chunked bulk-delete helpers provided by SQLiteDeletionMixin via the
    # composed SQLiteStorage MRO; declared here for clear_user_data's benefit.
    _delete_in_chunks: Any

    _delete_source_windows_for_user_playbook_ids: Any

    _subject_ref_for_user_id: Callable[[str], str]

    # The optimizer-job erasure belongs to OptimizationJobStoreMixin.
    _erase_optimization_jobs_for_user_locked: Callable[[str, str], int]

    # The exposure-ledger clear belongs to SQLiteOpenWorldMixin.
    _delete_exposure_events_for_user_locked: Callable[[str], int]

    # Raw-answer erasure belongs to SQLitePendingToolCallStoreMixin.
    _erase_pending_tool_calls_for_user_unlocked: Callable[[str], int]

    # FTS/vec index helpers provided by SQLiteFtsVecMixin via the composed
    # SQLiteStorage MRO; declared here for _migrate_vec_tables's benefit.
    _vec_upsert: Any

    _flush_index_op: Any

    def __init__(
        self,
        org_id: str,
        db_path: str | None = None,
        api_key_config: APIKeyConfig | None = None,
        llm_config: LLMConfig | None = None,
        enable_document_expansion: bool = False,
    ) -> None:
        initialize_server_runtime()
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
        from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
        from reflexio.server.llm.providers.embedding_service_provider import (
            resolve_service_configured_model,
        )

        from ._extraction_stream import register_window_delivery_validator

        super().__init__(org_id)
        self.api_key_config = api_key_config
        self._enable_document_expansion = enable_document_expansion

        # Checked before any filesystem work: resolving a derived path claims the
        # dataset, and an unsupported SQLite should fail without leaving a directory
        # or a claim behind.
        if sqlite3.sqlite_version_info < _MINIMUM_SQLITE_VERSION:
            detected_version = ".".join(map(str, sqlite3.sqlite_version_info))
            minimum_version = ".".join(map(str, _MINIMUM_SQLITE_VERSION))
            raise RuntimeError(
                f"SQLite {minimum_version} or newer is required; detected {detected_version}"
            )

        # Resolve db_path: explicit arg > a file derived from org_id under
        # LOCAL_STORAGE_PATH. An explicit path is used verbatim -- callers that point
        # several identities at one file (multi-tenant tests, benchmarks) depend on
        # that, and it is what keeps the residual commingling case observable.
        #
        # The import stays function-local: the test session patches
        # ``reflexio.server.LOCAL_STORAGE_PATH`` on the already-imported module, which
        # a module-level import would read past.
        if db_path is None:
            from reflexio.server import LOCAL_STORAGE_PATH

            db_path = resolve_sqlite_db_path(LOCAL_STORAGE_PATH, org_id)

        self.db_path = db_path
        self._lock = threading.RLock()
        self._scope_depth = 0
        self._scope_rollback_only = False
        self._deferred_index_ops: list[
            tuple[str, Any]
        ] = []  # (kind, args) flushed post-commit

        logger.info("SQLite Storage for org %s using db_path: %s", org_id, db_path)

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        initialization_lock = _get_sqlite_initialization_lock(db_path)

        # SQLite's journal-mode negotiation can fail immediately when another
        # connection is cold-starting the same file. Serialize that setup by
        # database path while allowing unrelated SQLite files to initialize in
        # parallel.
        with initialization_lock:
            reserve_new_sqlite_dataset(db_path, org_id)
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            if db_path == ":memory:":
                retain_private_memory_creation(self.conn, org_id)
            register_unicode_lexical_index_function(self.conn)
            register_window_delivery_validator(self.conn)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")

        # LLM client for embeddings
        self.embedding_model_name = resolve_service_configured_model(
            resolve_model_name(
                ModelRole.EMBEDDING,
                config_override=(
                    llm_config.embedding_model_name if llm_config else None
                ),
                api_key_config=self.api_key_config,
            )
        )
        self.embedding_dimensions = EMBEDDING_DIMENSIONS
        # Text-generation model for storage-time document expansion. The
        # shared self.llm_client is pinned to the EMBEDDING model, which is
        # not a chat model — expansion calls must override the model or they
        # fail (e.g. local/minilm-l6-v2 cannot serve completions).
        self._expansion_model_name = resolve_model_name(
            ModelRole.GENERATION,
            llm_config=llm_config,
            api_key_config=self.api_key_config,
        )

        litellm_config = LiteLLMConfig(
            model=self.embedding_model_name,
            temperature=0.0,
            api_key_config=self.api_key_config,
        )
        if llm_config and llm_config.generation_max_output_tokens is not None:
            litellm_config.max_tokens = llm_config.generation_max_output_tokens
        self.llm_client = LiteLLMClient(litellm_config)

        # Optionally load sqlite-vec for native KNN vector search
        self._has_sqlite_vec = self._try_load_sqlite_vec()

        # Migrations use an instance-local lock, so separate storage instances
        # for the same file also need the shared path lock.
        with initialization_lock:
            self.migrate()
            complete_sqlite_dataset_creation(self.conn, db_path, org_id)

    def _get_embedding(
        self, text: str, purpose: Literal["document", "query"] = "document"
    ) -> list[float]:
        """Generate an embedding with model-specific input formatting.

        Args:
            text: The text to embed.
            purpose: Either ``"document"`` (stored embeddings) or ``"query"``
                (search-time embeddings).

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            return self.llm_client.get_embedding(
                embedding_input(
                    text,
                    model_name=self.embedding_model_name,
                    purpose=purpose,
                ),
                self.embedding_model_name,
                self.embedding_dimensions,
                purpose=purpose,
            )
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable for %s text; continuing without vector: %s",
                purpose,
                exc,
            )
            return []

    def _should_expand_documents(self) -> bool:
        """Check if document expansion is enabled."""
        return self._enable_document_expansion

    def _expand_document(self, content: str) -> str | None:
        """Expand document content with synonyms for FTS recall.

        Uses DocumentExpander to generate synonym groups. Returns the
        expanded_terms string (e.g., "backup, sync; failure, error")
        or None on failure.

        Args:
            content (str): Document text to expand

        Returns:
            str or None: Expanded terms text, or None if expansion fails/disabled
        """
        if not content:
            return None
        try:
            from reflexio.server.prompt.prompt_manager import PromptManager
            from reflexio.server.services.pre_retrieval import DocumentExpander

            expander = DocumentExpander(
                llm_client=self.llm_client,
                prompt_manager=PromptManager(),
                model_name=self._expansion_model_name,
            )
            result = expander.expand(content)
            return result.expanded_text or None
        except Exception:
            logger.warning("Document expansion failed", exc_info=True)
            return None

    def _current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    # ------------------------------------------------------------------
    # Per-user data clear
    # ------------------------------------------------------------------

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        """Per-``user_id`` row deletion across all user-scoped tables.

        Overrides the BaseStorage default with an optimized SQL implementation.
        Removes interactions, user playbooks, profiles, requests, raw extraction
        runs and originated pending questions/answers scoped to the user.
        Shared dependent runs retain independent answers and lose stale claims.
        Every optimizer job whose stored content names the user -- as target
        owner, as a frozen governance subject, or as the user of an evidence
        session another user's playbook was tuned on -- is deleted with all of
        its rows: on SQLite this clear is the erasure.
        Intentionally does NOT touch ``agent_playbooks`` — they are
        the cross-project rollup of skills and have no ``user_id`` column.

        Also cleans up FTS and vector sidecars for the user's rows so
        subsequent searches don't surface deleted data.

        **Lineage-aware erasure for profiles and user_playbooks:**
        Rows that are tombstones (``merged_into`` or ``superseded_by`` is set)
        *or* are pointed-to by another row (``has_inbound_lineage_refs`` returns
        True) are **content-purged** (skeleton kept, body blanked) rather than
        hard-deleted. This preserves chain resolution across user erasures.
        Standalone rows with no lineage involvement are hard-deleted as before.

        The purge/delete decision delegates to
        ``BaseStorage._partition_purge_vs_delete`` so the logic is defined once
        and shared with the default ``clear_user_data`` implementation used by
        Supabase/Postgres backends.

        **Commit ordering (atomicity invariant):**
        Pending output/window finances transfer in the same owned writer as
        the raw-run, answer, window and user-row deletes. Borrowed writers are
        refused before effects. Deleting runs cascades private preparations and
        historical/accepted receipts; pending receipts retain only financial facts.
        The hard-deletes for interactions, requests, and the delete-sets of
        profiles/user_playbooks are committed in one transaction first. Then
        ``purge_content`` is called for each purge-eligible row — each call
        commits atomically on its own. This two-phase approach is required
        because ``purge_content`` issues its own ``conn.commit()``, and nesting
        it inside the outer transaction would prematurely flush the still-pending
        hard-DELETEs.

        Args:
            user_id (str): The user id whose rows should be deleted.

        Returns:
            dict[str, int]: Per-entity counts with keys ``session_outcomes``,
                ``interactions``, ``user_playbooks``, ``profiles``, ``requests``,
                ``purged_profiles``, ``purged_user_playbooks``,
                ``user_playbook_exposure_events`` (rows naming the user as
                viewer or as playbook owner) and ``optimization_jobs`` (jobs
                whose stored content names the user; their artifacts, analyst
                calls, attempt context and checkpoints go with them).
                ``profiles`` and ``user_playbooks`` reflect hard-deleted counts;
                purged rows are counted separately.
        """
        with self._lock:
            if self.in_commit_scope:
                raise StorageError("User clear requires its own transaction")
            with self.commit_scope():
                with self._stream_sql() as db:
                    self._handoff_erased_output_finances(db, user_id)
                    self._handoff_erased_window_finances(db, user_id)
                    db.query(
                        "DELETE FROM _agent_runs WHERE org_id=? AND user_id=?",
                        (self.org_id, user_id),
                    )
                self._erase_pending_tool_calls_for_user_unlocked(user_id)
                # ------------------------------------------------------------------
                # Phase 1: snapshot learning/source ids before touching their rows.
                # ------------------------------------------------------------------
                interaction_ids = [
                    r["interaction_id"]
                    for r in self.conn.execute(
                        "SELECT interaction_id FROM interactions WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                ]
                raw_upb_ids = [
                    r["user_playbook_id"]
                    for r in self.conn.execute(
                        "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                ]
                profile_rows = self.conn.execute(
                    "SELECT rowid, profile_id FROM profiles WHERE user_id = ?",
                    (user_id,),
                ).fetchall()

                # Build a rowid lookup for FTS/vec cleanup (SQLite-specific need).
                profile_rowid_by_id: dict[str, int] = {
                    r["profile_id"]: r["rowid"] for r in profile_rows
                }
                all_profile_ids = list(profile_rowid_by_id.keys())

                # ------------------------------------------------------------------
                # Phase 2: partition profiles and user_playbooks into purge vs delete.
                # Delegates to the shared BaseStorage helper so the decision logic
                # is defined once and reused by all backends.
                # ------------------------------------------------------------------
                purge_profile_ids, delete_profile_ids = self._partition_purge_vs_delete(
                    "profile", all_profile_ids
                )
                purge_upb_str_ids, delete_upb_str_ids = self._partition_purge_vs_delete(
                    "user_playbook", [str(uid) for uid in raw_upb_ids]
                )
                purge_upb_ids = [int(s) for s in purge_upb_str_ids]
                delete_upb_ids = [int(s) for s in delete_upb_str_ids]

                # Rowids for the delete-set only (purge_content handles its own cleanup).
                delete_profile_rowids = [
                    profile_rowid_by_id[pid]
                    for pid in delete_profile_ids
                    if pid in profile_rowid_by_id
                ]

                # ------------------------------------------------------------------
                # Phase 3: FTS and vector cleanup — only for the delete-sets
                # (purge_content handles its own index cleanup for purged rows).
                # Use _delete_in_chunks to stay under SQLite's SQLITE_MAX_VARIABLE_NUMBER
                # limit for large user datasets.
                # ------------------------------------------------------------------
                self._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
                self._delete_in_chunks("user_playbooks_fts", "rowid", delete_upb_ids)
                self._delete_in_chunks("profiles_fts", "profile_id", delete_profile_ids)

                if self._has_sqlite_vec:
                    self._delete_in_chunks("interactions_vec", "rowid", interaction_ids)
                    self._delete_in_chunks(
                        "user_playbooks_vec", "rowid", delete_upb_ids
                    )
                    self._delete_in_chunks(
                        "profiles_vec", "rowid", delete_profile_rowids
                    )

                # ------------------------------------------------------------------
                # Phase 4: hard-delete the delete-sets and all interactions/requests.
                # ------------------------------------------------------------------
                self.conn.execute(
                    "DELETE FROM extraction_windows WHERE user_id=?", (user_id,)
                )
                self.conn.execute(
                    "DELETE FROM extraction_cursors WHERE user_id=?", (user_id,)
                )
                self.conn.execute(
                    "DELETE FROM learning_work WHERE org_id=? AND user_id=?",
                    (self.org_id, user_id),
                )
                exposure_events_deleted = self._delete_exposure_events_for_user_locked(
                    user_id
                )
                # Every optimizer job holding this user's content goes, with its
                # artifacts, analyst calls, attempt context and checkpoints. A
                # SQLite database has no separate erasure path, so this clear is
                # the erasure; it runs before the playbooks it reads ownership
                # from, in the same transaction as every other delete here.
                optimization_jobs_deleted = (
                    self._erase_optimization_jobs_for_user_locked(
                        user_id, self._subject_ref_for_user_id(user_id)
                    )
                )
                interactions_cur = self.conn.execute(
                    "DELETE FROM interactions WHERE user_id = ?", (user_id,)
                )
                session_outcomes_cur = self.conn.execute(
                    "DELETE FROM session_outcomes WHERE user_id = ?",
                    (user_id,),
                )
                requests_cur = self.conn.execute(
                    "DELETE FROM requests WHERE user_id = ?", (user_id,)
                )
                upb_deleted_count = 0
                if delete_upb_ids:
                    # Clean up source-window join rows before deleting the parent rows.
                    self._delete_source_windows_for_user_playbook_ids(delete_upb_ids)
                    self._delete_in_chunks(
                        "user_playbooks", "user_playbook_id", delete_upb_ids
                    )
                    # rowcount not available from _delete_in_chunks; derive from list length
                    # (all ids came from a pre-snapshot so they exist at delete time).
                    upb_deleted_count = len(delete_upb_ids)
                profile_deleted_count = 0
                if delete_profile_ids:
                    self._delete_in_chunks("profiles", "profile_id", delete_profile_ids)
                    profile_deleted_count = len(delete_profile_ids)

            # Phase 5: content-purge the purge-sets WITHOUT releasing the lock,
            # so erase-eligible rows are never observable by another thread with
            # PII still intact between the hard-delete commit and the purge. Each
            # purge_content call self-commits; self._lock is an RLock so its
            # internal ``with self._lock`` re-acquires cleanly, and the commit
            # above already closed the outer transaction (no flush hazard).
            for pid in purge_profile_ids:
                self.purge_content(entity_type="profile", entity_id=str(pid))
            for upid in purge_upb_ids:
                self.purge_content(entity_type="user_playbook", entity_id=str(upid))
            # Another SQLiteStorage connection has a different Python lock.
            # Sweep access history only after preserved profile payloads are
            # blank, so an in-flight accepted access cannot leave old epochs.
            with self.commit_scope():
                self.conn.execute(
                    "DELETE FROM profile_access_epochs WHERE org_id=? AND user_id=?",
                    (self.org_id, user_id),
                )

        return {
            "session_outcomes": session_outcomes_cur.rowcount,
            "interactions": interactions_cur.rowcount,
            "user_playbooks": upb_deleted_count,
            "profiles": profile_deleted_count,
            "requests": requests_cur.rowcount,
            "purged_profiles": len(purge_profile_ids),
            "purged_user_playbooks": len(purge_upb_ids),
            "user_playbook_exposure_events": exposure_events_deleted,
            "optimization_jobs": optimization_jobs_deleted,
        }
