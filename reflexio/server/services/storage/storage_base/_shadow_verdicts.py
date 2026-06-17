"""Storage interface for per-turn shadow comparison verdicts (F1).

Declares the methods every storage implementation MAY implement to expose the
``shadow_comparison_verdicts`` table used by the per-turn Reflexio-vs-Shadow
comparison feature.

The default implementations raise :class:`NotImplementedError`. Concrete
backends that support the feature (currently :class:`SQLiteStorage`) override
these methods via their own mixin. Backends that don't (currently
:class:`DiskStorage`, :class:`SupabaseStorage` — added in later tasks) will
surface a clear ``NotImplementedError`` at call time rather than
``AttributeError``.

This intentionally follows the :class:`StallStateMixin` precedent so adding
a new backend-optional feature does not break the instantiation contract of
existing backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.models.api_schema.eval_overview_schema import (
        ShadowComparisonVerdict,
    )


class ShadowVerdictsMixin:
    """Default no-op shadow_comparison_verdicts interface for storage backends.

    Subclasses that support the feature must override every method. The
    methods below raise :class:`NotImplementedError` so callers see a
    clear error message naming the backend rather than ``AttributeError``.
    """

    def save_shadow_comparison_verdict(
        self, verdict: ShadowComparisonVerdict
    ) -> ShadowComparisonVerdict:
        """Persist a verdict and return the row with the assigned ``verdict_id``.

        Args:
            verdict (ShadowComparisonVerdict): The verdict to persist. The
                ``verdict_id`` field is ignored on save — storage assigns
                the autoincrement key.

        Returns:
            ShadowComparisonVerdict: The same verdict with ``verdict_id``
                replaced by the storage-assigned primary key.

        Raises:
            NotImplementedError: When the backend does not support shadow
                comparison verdicts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support shadow_comparison_verdicts"
        )

    def get_shadow_comparison_verdict(
        self, verdict_id: int
    ) -> ShadowComparisonVerdict | None:
        """Fetch one verdict by its autoincrement primary key.

        Args:
            verdict_id (int): The storage-assigned key returned by
                :meth:`save_shadow_comparison_verdict`.

        Returns:
            ShadowComparisonVerdict | None: The verdict if present, else
                ``None``.

        Raises:
            NotImplementedError: When the backend does not support shadow
                comparison verdicts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support shadow_comparison_verdicts"
        )

    def get_shadow_comparison_verdicts(
        self,
        from_ts: int,
        to_ts: int,
        judge_prompt_version: str,
    ) -> list[ShadowComparisonVerdict]:
        """Fetch verdicts in ``[from_ts, to_ts]`` for one pinned prompt version.

        Args:
            from_ts (int): Inclusive lower bound of ``created_at``, in Unix
                epoch seconds (UTC).
            to_ts (int): Inclusive upper bound of ``created_at``, in Unix
                epoch seconds (UTC).
            judge_prompt_version (str): The pinned ``shadow_comparison``
                prompt version. Filtering by this prevents rubric-mixing in
                the dashboard headline metric — verdicts produced under an
                older rubric stay in storage but do not contaminate the
                current trend.

        Returns:
            list[ShadowComparisonVerdict]: Matching verdicts in chronological
                order (ascending ``created_at``).

        Raises:
            NotImplementedError: When the backend does not support shadow
                comparison verdicts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support shadow_comparison_verdicts"
        )

    def get_recent_shadow_comparison_verdicts(
        self,
        from_ts: int,
        to_ts: int,
        judge_prompt_version: str,
        limit: int,
    ) -> list[ShadowComparisonVerdict]:
        """Fetch newest verdicts in descending ``created_at`` order."""
        if limit <= 0:
            return []
        verdicts = self.get_shadow_comparison_verdicts(
            from_ts=from_ts,
            to_ts=to_ts,
            judge_prompt_version=judge_prompt_version,
        )
        return list(reversed(verdicts))[:limit]

    def delete_shadow_comparison_verdicts_by_session(self, session_id: str) -> int:
        """Remove all verdicts for one session.

        Used by the regen worker's delete-after-save pattern: a re-eval
        run computes fresh verdicts, persists them, then clears the prior
        ones to avoid double-counting.

        Args:
            session_id (str): The session whose verdicts should be removed.

        Returns:
            int: Number of rows deleted.

        Raises:
            NotImplementedError: When the backend does not support shadow
                comparison verdicts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support shadow_comparison_verdicts"
        )
