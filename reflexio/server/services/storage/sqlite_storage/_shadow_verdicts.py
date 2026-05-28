"""SQLite CRUD for ``shadow_comparison_verdicts`` (F1)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)

from ._base import SQLiteStorageBase, _epoch_to_iso

# Maximum epoch seconds that ``datetime.fromtimestamp`` can represent (year
# 9999-12-31). Callers passing sentinel "open" upper bounds like
# ``sys.maxsize`` or ``10**12`` would otherwise overflow ``_epoch_to_iso``;
# clamping to this value yields the same query semantics (≥ everything) with
# a valid ISO string.
_MAX_SAFE_EPOCH_TS = 253_402_300_799  # 9999-12-31T23:59:59Z


def _parse_dt(value: str) -> datetime:
    """
    Parse a stored ISO timestamp into a tz-aware ``datetime``.

    Rows inserted with a model-provided ``created_at`` carry an explicit
    offset (``...+00:00``). Rows that fell back to the ``CURRENT_TIMESTAMP``
    default are tz-naive (SQLite stores ``YYYY-MM-DD HH:MM:SS``); tag those
    as UTC to keep the dashboard math correct.

    Args:
        value (str): Stored timestamp text.

    Returns:
        datetime: A tz-aware ``datetime`` in UTC.
    """
    if value.endswith("Z") or "+" in value or "-" in value[10:]:
        return datetime.fromisoformat(value)
    # CURRENT_TIMESTAMP default — "YYYY-MM-DD HH:MM:SS" with a space, not "T".
    iso = value.replace(" ", "T")
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def _row_to_verdict(row: sqlite3.Row) -> ShadowComparisonVerdict:
    """
    Hydrate a ``ShadowComparisonVerdict`` from a SQLite row.

    Args:
        row (sqlite3.Row): Row from ``shadow_comparison_verdicts``.

    Returns:
        ShadowComparisonVerdict: Reconstructed verdict including the nested
            :class:`ShadowComparisonOutput`.
    """
    d = dict(row)
    return ShadowComparisonVerdict(
        verdict_id=d["verdict_id"],
        interaction_id=d["interaction_id"],
        session_id=d["session_id"],
        agent_version=d["agent_version"],
        reflexio_is_request_1=bool(d["reflexio_is_request_1"]),
        output=ShadowComparisonOutput(
            better_request=d["better_request"],
            is_significantly_better=bool(d["is_significantly_better"]),
            comparison_reason=d.get("comparison_reason"),
        ),
        judge_prompt_version=d["judge_prompt_version"],
        created_at=_parse_dt(d["created_at"]),
    )


class ShadowVerdictsMixin:
    """SQLite implementation of the shadow_comparison_verdicts contract."""

    # Attributes provided by SQLiteStorageBase via MRO; declared here for
    # pyright so the @handle_exceptions decorator sees correct types.
    conn: sqlite3.Connection
    _execute: Any
    _fetchone: Any
    _fetchall: Any

    @SQLiteStorageBase.handle_exceptions
    def save_shadow_comparison_verdict(
        self, verdict: ShadowComparisonVerdict
    ) -> ShadowComparisonVerdict:
        """
        Insert a verdict and return the persisted row.

        Args:
            verdict (ShadowComparisonVerdict): Verdict to persist. The
                ``verdict_id`` field is ignored; storage assigns the
                autoincrement primary key.

        Returns:
            ShadowComparisonVerdict: The verdict with ``verdict_id``
                populated from the autoincrement.
        """
        cur = self._execute(
            """INSERT INTO shadow_comparison_verdicts
               (interaction_id, session_id, agent_version,
                reflexio_is_request_1, better_request,
                is_significantly_better, comparison_reason,
                judge_prompt_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verdict.interaction_id,
                verdict.session_id,
                verdict.agent_version,
                1 if verdict.reflexio_is_request_1 else 0,
                verdict.output.better_request,
                1 if verdict.output.is_significantly_better else 0,
                verdict.output.comparison_reason,
                verdict.judge_prompt_version,
                verdict.created_at.isoformat(),
            ),
        )
        verdict_id = cur.lastrowid
        return verdict.model_copy(update={"verdict_id": verdict_id})

    @SQLiteStorageBase.handle_exceptions
    def get_shadow_comparison_verdict(
        self, verdict_id: int
    ) -> ShadowComparisonVerdict | None:
        """
        Fetch a verdict by its autoincrement primary key.

        Args:
            verdict_id (int): The verdict's storage-assigned key.

        Returns:
            ShadowComparisonVerdict | None: The verdict if found, else
                ``None``.
        """
        row = self._fetchone(
            "SELECT * FROM shadow_comparison_verdicts WHERE verdict_id = ?",
            (verdict_id,),
        )
        return _row_to_verdict(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_shadow_comparison_verdicts(
        self,
        from_ts: int,
        to_ts: int,
        judge_prompt_version: str,
    ) -> list[ShadowComparisonVerdict]:
        """
        Fetch verdicts in ``[from_ts, to_ts]`` for one pinned prompt version.

        Comparisons happen against stored ISO timestamps, which are
        lexicographically orderable when expressed in UTC — see
        :func:`_epoch_to_iso`.

        Args:
            from_ts (int): Inclusive lower bound, Unix epoch seconds (UTC).
            to_ts (int): Inclusive upper bound, Unix epoch seconds (UTC).
            judge_prompt_version (str): Pinned ``shadow_comparison`` prompt
                version. Filtering by this prevents rubric-mixing in the
                dashboard headline.

        Returns:
            list[ShadowComparisonVerdict]: Verdicts in chronological order
                (ascending ``created_at``).
        """
        from_iso = _epoch_to_iso(max(0, min(from_ts, _MAX_SAFE_EPOCH_TS)))
        to_iso = _epoch_to_iso(max(0, min(to_ts, _MAX_SAFE_EPOCH_TS)))
        rows = self._fetchall(
            """SELECT * FROM shadow_comparison_verdicts
               WHERE created_at >= ? AND created_at <= ?
                 AND judge_prompt_version = ?
               ORDER BY created_at ASC""",
            (from_iso, to_iso, judge_prompt_version),
        )
        return [_row_to_verdict(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_shadow_comparison_verdicts_by_session(self, session_id: str) -> int:
        """
        Delete all verdicts for one session.

        Args:
            session_id (str): The session whose verdicts should be removed.

        Returns:
            int: Number of rows deleted.
        """
        cur = self._execute(
            "DELETE FROM shadow_comparison_verdicts WHERE session_id = ?",
            (session_id,),
        )
        return cur.rowcount
