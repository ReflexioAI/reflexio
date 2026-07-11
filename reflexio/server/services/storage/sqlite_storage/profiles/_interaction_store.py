"""Interaction store CRUD methods for SQLite storage.

Extracted verbatim from ``_profiles.py`` (the InteractionStore bucket). Profile
CRUD lives in ``profiles._profile_store`` (``ProfileStoreMixin``); search methods
live in ``profiles._search`` (``ProfileSearchMixin``).
"""

import logging
import sqlite3
from typing import Any

from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    Interaction,
)
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)

from .._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _json_dumps,
    _row_to_interaction,
)

logger = logging.getLogger(__name__)


def _embed_text_for_interaction(content: str | None, action_desc: str | None) -> str:
    """Derive the text embedded for an interaction.

    Kept byte-for-byte identical to the ingest path
    (``add_user_interactions_bulk`` / ``prepare_interaction_embeddings``) so a
    backfilled vector matches a freshly-ingested one.
    """
    return "\n".join([content or "", action_desc or ""])


class InteractionStoreMixin:
    """Mixin providing interaction-store CRUD for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _fts_upsert: Any
    _vec_upsert: Any
    _delete_in_chunks: Any
    _has_sqlite_vec: bool
    llm_client: Any
    embedding_model_name: str
    embedding_dimensions: int
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any
    _own_transaction: Any

    # ------------------------------------------------------------------
    # CRUD — Interactions
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        rows = self._fetchall(
            "SELECT * FROM interactions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_interaction(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        rows = self._fetchall(
            "SELECT * FROM interactions WHERE user_id = ?", (user_id,)
        )
        return [_row_to_interaction(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_all_user_ids(self) -> list[str]:
        rows = self._fetchall("SELECT DISTINCT user_id FROM interactions")
        return sorted(r["user_id"] for r in rows)

    @SQLiteStorageBase.handle_exceptions
    def add_user_interaction(self, user_id: str, interaction: Interaction) -> None:  # noqa: ARG002
        # NOTE: distinct from the bulk/backfill derivation
        # (_embed_text_for_interaction). This legacy single-insert path embeds via
        # the purpose-prefixed _get_embedding ("search_document: ...") and its own
        # f-string, so it is intentionally NOT routed through the shared helper —
        # aligning it would silently change stored embedding values (e.g. null ->
        # literal "None"). The real bulk ingest + backfill share the helper.
        embedding = self._get_embedding(
            f"{interaction.content}\n{interaction.user_action_description}"
        )
        interaction.embedding = embedding
        self._insert_interaction(interaction)

    def _insert_interaction(self, interaction: Interaction) -> int:
        created_at_iso = _epoch_to_iso(interaction.created_at)
        subject_ref = self._subject_ref_for_user_id(interaction.user_id)
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                if interaction.interaction_id:
                    self.conn.execute(
                        """INSERT OR REPLACE INTO interactions
                           (interaction_id, user_id, content, request_id, created_at,
                            role, user_action, user_action_description,
                            interacted_image_url, image_encoding, shadow_content,
                            expert_content, tools_used, citations, retrieved_learnings,
                            embedding, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            interaction.interaction_id,
                            interaction.user_id,
                            interaction.content,
                            interaction.request_id,
                            created_at_iso,
                            interaction.role,
                            interaction.user_action.value,
                            interaction.user_action_description,
                            interaction.interacted_image_url,
                            interaction.image_encoding,
                            interaction.shadow_content,
                            interaction.expert_content,
                            _json_dumps(
                                [t.model_dump() for t in interaction.tools_used]
                            ),
                            _json_dumps(
                                [c.model_dump() for c in interaction.citations]
                            ),
                            _json_dumps(
                                [
                                    c.model_dump(exclude_none=True)
                                    for c in interaction.retrieved_learnings
                                ]
                            ),
                            _json_dumps(interaction.embedding),
                            subject_ref,
                        ),
                    )
                    iid = interaction.interaction_id
                else:
                    cur = self.conn.execute(
                        """INSERT INTO interactions
                           (user_id, content, request_id, created_at,
                            role, user_action, user_action_description,
                            interacted_image_url, image_encoding, shadow_content,
                            expert_content, tools_used, citations, retrieved_learnings,
                            embedding, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            interaction.user_id,
                            interaction.content,
                            interaction.request_id,
                            created_at_iso,
                            interaction.role,
                            interaction.user_action.value,
                            interaction.user_action_description,
                            interaction.interacted_image_url,
                            interaction.image_encoding,
                            interaction.shadow_content,
                            interaction.expert_content,
                            _json_dumps(
                                [t.model_dump() for t in interaction.tools_used]
                            ),
                            _json_dumps(
                                [c.model_dump() for c in interaction.citations]
                            ),
                            _json_dumps(
                                [
                                    c.model_dump(exclude_none=True)
                                    for c in interaction.retrieved_learnings
                                ]
                            ),
                            _json_dumps(interaction.embedding),
                            subject_ref,
                        ),
                    )
                    iid = cur.lastrowid or 0
                    interaction.interaction_id = iid
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise
        # Update FTS and vec
        self._fts_upsert(
            "interactions_fts",
            iid,
            content=interaction.content,
            user_action_description=interaction.user_action_description,
        )
        if interaction.embedding:
            self._vec_upsert("interactions_vec", iid, interaction.embedding)
        return iid

    @SQLiteStorageBase.handle_exceptions
    def add_user_interactions_bulk(
        self,
        user_id: str,  # noqa: ARG002
        interactions: list[Interaction],
        *,
        embeddings_prepared: bool = False,
    ) -> None:
        if not interactions:
            return
        if not embeddings_prepared:
            # Only generate embeddings for interactions that do not already have them.
            # This allows callers to pre-populate embeddings (e.g. via
            # prepare_interaction_embeddings) before opening a commit_scope so that no
            # network I/O occurs inside the transaction.
            to_embed = [i for i in interactions if not i.embedding]
            if to_embed:
                texts = [
                    _embed_text_for_interaction(i.content, i.user_action_description)
                    for i in to_embed
                ]
                try:
                    embeddings = self.llm_client.get_embeddings(
                        texts, self.embedding_model_name, self.embedding_dimensions
                    )
                except EmbeddingUnavailableError as exc:
                    logger.warning(
                        "Embedding unavailable for interaction bulk insert; "
                        "continuing without vectors: %s",
                        exc,
                    )
                    embeddings = [[] for _ in texts]
                for interaction, embedding in zip(to_embed, embeddings, strict=False):
                    interaction.embedding = embedding
        for interaction in interactions:
            self._insert_interaction(interaction)

    @SQLiteStorageBase.handle_exceptions
    def prepare_interaction_embeddings(self, interactions: list[Interaction]) -> None:
        """Pre-populate interaction.embedding for each interaction (no DB write).

        Generates embeddings in one batch call so the write path inside a
        commit_scope can skip the network round-trip.
        """
        if not interactions:
            return
        to_embed = [i for i in interactions if not i.embedding]
        if not to_embed:
            return
        texts = [
            _embed_text_for_interaction(i.content, i.user_action_description)
            for i in to_embed
        ]
        try:
            embeddings = self.llm_client.get_embeddings(
                texts, self.embedding_model_name, self.embedding_dimensions
            )
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable during prepare_interaction_embeddings; "
                "continuing without vectors: %s",
                exc,
            )
            embeddings = [[] for _ in texts]
        for interaction, embedding in zip(to_embed, embeddings, strict=False):
            interaction.embedding = embedding

    # ------------------------------------------------------------------
    # Missing-vector backfill (durability sweep)
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def iter_interactions_missing_vectors(self, limit: int) -> list[tuple[int, str]]:
        """Enumerate interactions whose embedding was never persisted.

        A degraded/failed embedding leaves the ``embedding`` column empty
        (``'[]'`` or NULL) and writes no ``interactions_vec`` row — equivalent
        conditions on the write path. Detecting on the ``embedding`` column is
        both the root-cause signal and portable across backends (the column
        exists everywhere; ``interactions_vec`` is a SQLite-vec detail).

        Args:
            limit: Maximum number of interactions to return.

        Returns:
            list[tuple[int, str]]: ``(interaction_id, embed_text)`` pairs, at
            most ``limit`` long, ordered by ``interaction_id`` for stable paging.
        """
        if limit <= 0:
            return []
        rows = self._fetchall(
            """SELECT interaction_id, content, user_action_description
               FROM interactions
               WHERE embedding IS NULL OR embedding = '[]'
               ORDER BY interaction_id
               LIMIT ?""",
            (limit,),
        )
        return [
            (
                r["interaction_id"],
                _embed_text_for_interaction(r["content"], r["user_action_description"]),
            )
            for r in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def backfill_missing_interaction_vectors(self, limit: int) -> int:
        """Re-embed and persist vectors for interactions missing them.

        Bounded by ``limit`` and idempotent: once the ``embedding`` column and
        ``interactions_vec`` row are written, the row no longer matches
        detection and is skipped next time. Embeds via the same batch call and
        text derivation the ingest path uses so backfilled vectors match
        freshly-ingested ones.

        Fail-safe: if the embedder is unavailable the batch call raises
        ``EmbeddingUnavailableError``; we log a single bounded WARN and return 0,
        leaving the rows for the next tick rather than crashing the caller or
        hot-looping a down embedder.

        Args:
            limit: Maximum number of interactions to re-embed this call.

        Returns:
            int: Number of interactions whose vector was backfilled.
        """
        if limit <= 0:
            return 0
        pairs = self.iter_interactions_missing_vectors(limit)
        if not pairs:
            return 0
        texts = [text for _, text in pairs]
        try:
            embeddings = self.llm_client.get_embeddings(
                texts, self.embedding_model_name, self.embedding_dimensions
            )
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable during missing-vector backfill; leaving "
                "%d interaction(s) for the next tick: %s",
                len(pairs),
                exc,
            )
            return 0
        backfilled = 0
        for (iid, _text), embedding in zip(pairs, embeddings, strict=False):
            if not embedding:
                # Embedder returned empty for this row — leave it for next tick.
                continue
            self._persist_backfilled_vector(iid, embedding)
            backfilled += 1
        return backfilled

    def _persist_backfilled_vector(
        self, interaction_id: int, embedding: list[float]
    ) -> None:
        """Write a re-embedded vector for one interaction (column + vec row).

        Updates the ``embedding`` TEXT column (read by the Python vector-rank
        search path) and upserts the ``interactions_vec`` row via the same
        ``_vec_upsert`` helper the insert path uses.

        Only commits when it owns the transaction (mirroring ``_insert_interaction``)
        so it can never flush a partial outer transaction if ever called inside an
        open ``commit_scope``; ``_vec_upsert`` likewise defers under an open scope.
        """
        with self._lock:
            own_txn = self._own_transaction()
            self.conn.execute(
                "UPDATE interactions SET embedding = ? WHERE interaction_id = ?",
                (_json_dumps(embedding), interaction_id),
            )
            if own_txn:
                self.conn.commit()
        self._vec_upsert("interactions_vec", interaction_id, embedding)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        with self._lock:
            row = self.conn.execute(
                "SELECT interaction_id FROM interactions WHERE user_id = ? AND interaction_id = ?",
                (request.user_id, request.interaction_id),
            ).fetchone()
            if row is None:
                return
            self.conn.execute(
                "DELETE FROM interactions_fts WHERE rowid = ?",
                (request.interaction_id,),
            )
            if self._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM interactions_vec WHERE rowid = ?",
                    (request.interaction_id,),
                )
            self.conn.execute(
                "DELETE FROM interactions WHERE user_id = ? AND interaction_id = ?",
                (request.user_id, request.interaction_id),
            )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_interactions_for_user(self, user_id: str) -> None:
        with self._lock:
            rows = self.conn.execute(
                "SELECT interaction_id FROM interactions WHERE user_id = ?", (user_id,)
            ).fetchall()
            if not rows:
                return
            ids = [r["interaction_id"] for r in rows]
            self._delete_in_chunks("interactions_fts", "rowid", ids)
            if self._has_sqlite_vec:
                self._delete_in_chunks("interactions_vec", "rowid", ids)
            self.conn.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_interactions(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM interactions_fts")
            if self._has_sqlite_vec:
                self.conn.execute("DELETE FROM interactions_vec")
            self.conn.execute("DELETE FROM interactions")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def count_all_interactions(self) -> int:
        row = self._fetchone("SELECT COUNT(*) as cnt FROM interactions")
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def delete_oldest_interactions(self, count: int) -> int:
        if count <= 0:
            return 0
        with self._lock:
            rows = self.conn.execute(
                "SELECT interaction_id FROM interactions ORDER BY created_at ASC LIMIT ?",
                (count,),
            ).fetchall()
            if not rows:
                return 0
            ids = [r["interaction_id"] for r in rows]
            self._delete_in_chunks("interactions_fts", "rowid", ids)
            if self._has_sqlite_vec:
                self._delete_in_chunks("interactions_vec", "rowid", ids)
            self._delete_in_chunks("interactions", "interaction_id", ids)
            self.conn.commit()
        return len(ids)
