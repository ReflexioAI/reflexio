from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    DeleteUserInteractionRequest,
    Interaction,
)


class InteractionStoreMixin:
    """Mixin for interaction-store CRUD methods."""

    @abstractmethod
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_all_user_ids(self) -> list[str]:
        """Return distinct user IDs that have stored interactions."""
        raise NotImplementedError

    @abstractmethod
    def add_user_interaction(self, user_id: str, interaction: Interaction) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_user_interactions_bulk(
        self,
        user_id: str,
        interactions: list[Interaction],
        *,
        embeddings_prepared: bool = False,
    ) -> None:
        """Add multiple user interactions with batched embedding generation.

        Args:
            user_id: The user ID
            interactions: List of interactions to add
            embeddings_prepared: When True, skip all embedding generation and write
                whatever embedding is already on each interaction (real or ``[]``).
                Use on the durable path after ``prepare_interaction_embeddings`` to
                ensure no network I/O occurs inside an open ``commit_scope``.
        """
        raise NotImplementedError

    def prepare_interaction_embeddings(self, interactions: list[Interaction]) -> None:  # noqa: ARG002
        """Pre-populate interaction.embedding for each interaction without writing to storage.

        Call this before opening a commit_scope to avoid a network round-trip inside
        the transaction.  Subclasses that generate embeddings client-side MUST override
        this to call get_embeddings and populate interaction.embedding before the scope
        is opened; backends where the embedding is generated server-side can leave this
        as a no-op.

        Args:
            interactions: Interactions whose embedding fields will be populated in-place.
        """
        return

    def iter_interactions_missing_vectors(
        self,
        limit: int,  # noqa: ARG002
    ) -> list[tuple[int, str]]:
        """Enumerate interactions whose embedding vector was never persisted.

        When an embedding call fails or degrades, the ingest path stores an empty
        embedding (and writes no vector row), returns success, and nothing ever
        re-embeds the row — so that interaction is invisible to vector/hybrid
        search forever. This method surfaces those rows so a background sweep can
        re-embed them (see ``backfill_missing_interaction_vectors``).

        Returns ``(interaction_id, embed_text)`` pairs where ``embed_text`` is
        derived exactly as the ingest path derives it, so a backfilled vector is
        byte-for-byte comparable to a freshly-ingested one.

        Safe default: returns ``[]`` — "this backend does not support backfill
        detection". Backends that store embeddings (SQLite here; Supabase and
        Postgres in the enterprise repo) MUST override this so their rows are
        actually recoverable. Bounded by ``limit`` per call.

        Args:
            limit: Maximum number of interactions to return.

        Returns:
            list[tuple[int, str]]: ``(interaction_id, embed_text)`` pairs, at
            most ``limit`` long. Empty when nothing needs backfilling or the
            backend does not support detection.
        """
        return []

    def backfill_missing_interaction_vectors(
        self,
        limit: int,  # noqa: ARG002
    ) -> int:
        """Re-embed and persist vectors for interactions missing them.

        Idempotent and bounded: enumerates up to ``limit`` interactions via
        ``iter_interactions_missing_vectors``, re-embeds their text using the
        same embedder + derivation the ingest path uses, and writes the vector
        back through the same code path a fresh insert uses. Once a vector
        exists the row is skipped on the next call.

        Fail-safe: if the embedder is unavailable the implementation must skip
        gracefully and leave the work for a later call (never raise, never
        hot-loop a down embedder).

        Safe default: returns ``0`` — "this backend does not support backfill".
        SQLite overrides this; the enterprise Supabase/Postgres backends must
        override it (and ``iter_interactions_missing_vectors``) for production.

        Args:
            limit: Maximum number of interactions to re-embed this call.

        Returns:
            int: Number of interactions whose vector was backfilled.
        """
        return 0

    @abstractmethod
    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_interactions_for_user(self, user_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_interactions(self) -> None:
        """Delete all interactions across all users."""
        raise NotImplementedError

    @abstractmethod
    def count_all_interactions(self) -> int:
        """Count total interactions across all users.

        Returns:
            int: Total number of interactions
        """
        raise NotImplementedError

    @abstractmethod
    def delete_oldest_interactions(self, count: int) -> int:
        """Delete the oldest N interactions based on created_at timestamp.

        Args:
            count (int): Number of oldest interactions to delete

        Returns:
            int: Number of interactions actually deleted
        """
        raise NotImplementedError
