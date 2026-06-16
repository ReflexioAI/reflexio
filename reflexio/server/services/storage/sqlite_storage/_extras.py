"""Analytics and change log methods for SQLite storage."""

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal, cast

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    ImportedScore,
)
from reflexio.models.api_schema.retriever_schema import (
    InjectionStat,
    MemoryReviewCandidate,
    PlaybookApplicationStat,
)
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    PlaybookAggregationChangeLog,
    ProfileChangeLog,
)

from ._base import (
    SQLiteStorageBase,
    _epoch_now,
    _epoch_to_iso,
    _iso_to_epoch,
    _json_dumps,
    _json_loads,
    _row_to_interaction,
    _row_to_playbook_aggregation_change_log,
    _row_to_profile_change_log,
)

type _CitationKind = Literal["playbook", "profile"]
type _MemoryReviewSignal = Literal[
    "stale", "duplicate", "high_cost_low_cite", "supersedeable"
]


class ExtrasMixin:
    """Mixin providing analytics, change log, and misc operations."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    _execute: Any
    _fetchall: Any

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_interactions_by_request_ids(
        self, request_ids: list[str]
    ) -> list[Interaction]:
        if not request_ids:
            return []
        ph = ",".join("?" for _ in request_ids)
        rows = self._fetchall(
            f"SELECT * FROM interactions WHERE request_id IN ({ph}) ORDER BY created_at ASC",  # noqa: S608
            request_ids,
        )
        return [_row_to_interaction(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_interactions_by_ids(self, interaction_ids: list[int]) -> list[Interaction]:
        if not interaction_ids:
            return []
        ph = ",".join("?" for _ in interaction_ids)
        rows = self._fetchall(
            f"SELECT * FROM interactions WHERE interaction_id IN ({ph}) ORDER BY created_at ASC",  # noqa: S608
            interaction_ids,
        )
        return [_row_to_interaction(r) for r in rows]

    _fetchone: Any
    _fetchall: Any

    # ------------------------------------------------------------------
    # Dashboard / Analytics methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_dashboard_stats(self, days_back: int = 30) -> dict:
        current_time = _epoch_now()
        seconds_in_period = days_back * 24 * 60 * 60
        current_start = current_time - seconds_in_period
        previous_start = current_start - seconds_in_period

        current_start_iso = _epoch_to_iso(current_start)
        current_time_iso = _epoch_to_iso(current_time)
        previous_start_iso = _epoch_to_iso(previous_start)

        def count_in(table: str, time_col: str, start: Any, end: Any) -> int:
            row = self._fetchone(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_col} >= ? AND {time_col} <= ?",
                (start, end),
            )
            return row["cnt"] if row else 0

        def count_in_lt(table: str, time_col: str, start: Any, end: Any) -> int:
            row = self._fetchone(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_col} >= ? AND {time_col} < ?",
                (start, end),
            )
            return row["cnt"] if row else 0

        current_stats: dict[str, int | float] = {
            "total_interactions": count_in(
                "interactions", "created_at", current_start_iso, current_time_iso
            ),
            "total_profiles": count_in(
                "profiles", "last_modified_timestamp", current_start, current_time
            ),
            "total_playbooks": (
                count_in(
                    "user_playbooks", "created_at", current_start_iso, current_time_iso
                )
                + count_in(
                    "agent_playbooks", "created_at", current_start_iso, current_time_iso
                )
            ),
        }

        previous_stats: dict[str, int | float] = {
            "total_interactions": count_in_lt(
                "interactions", "created_at", previous_start_iso, current_start_iso
            ),
            "total_profiles": count_in_lt(
                "profiles", "last_modified_timestamp", previous_start, current_start
            ),
            "total_playbooks": (
                count_in_lt(
                    "user_playbooks",
                    "created_at",
                    previous_start_iso,
                    current_start_iso,
                )
                + count_in_lt(
                    "agent_playbooks",
                    "created_at",
                    previous_start_iso,
                    current_start_iso,
                )
            ),
        }

        # Success rates
        def calc_success_rate(rows: list[sqlite3.Row]) -> float:
            if not rows:
                return 0.0
            total = len(rows)
            success = sum(1 for r in rows if r["is_success"])
            return success / total * 100

        eval_current = self._fetchall(
            "SELECT is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at <= ?",
            (current_start_iso, current_time_iso),
        )
        eval_previous = self._fetchall(
            "SELECT is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at < ?",
            (previous_start_iso, current_start_iso),
        )
        current_stats["success_rate"] = calc_success_rate(eval_current)
        previous_stats["success_rate"] = calc_success_rate(eval_previous)

        # Time series
        interactions_ts = self._fetchall(
            "SELECT created_at FROM interactions WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )
        profiles_ts = self._fetchall(
            "SELECT last_modified_timestamp FROM profiles WHERE last_modified_timestamp >= ? AND last_modified_timestamp <= ? ORDER BY last_modified_timestamp",
            (current_start, current_time),
        )
        playbooks_ts = self._fetchall(
            "SELECT created_at FROM user_playbooks WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )
        evals_ts = self._fetchall(
            "SELECT created_at, is_success FROM agent_success_evaluation_result WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (current_start_iso, current_time_iso),
        )

        return {
            "current_period": current_stats,
            "previous_period": previous_stats,
            "interactions_time_series": [
                {"timestamp": _iso_to_epoch(r["created_at"]), "value": 1}
                for r in interactions_ts
            ],
            "profiles_time_series": [
                {"timestamp": r["last_modified_timestamp"], "value": 1}
                for r in profiles_ts
            ],
            "playbooks_time_series": [
                {"timestamp": _iso_to_epoch(r["created_at"]), "value": 1}
                for r in playbooks_ts
            ],
            "evaluations_time_series": [
                {
                    "timestamp": _iso_to_epoch(r["created_at"]),
                    "value": 100 if r["is_success"] else 0,
                }
                for r in evals_ts
            ],
        }

    @SQLiteStorageBase.handle_exceptions
    def get_playbook_application_stats(
        self, days_back: int = 30
    ) -> list[PlaybookApplicationStat]:
        """Return per-rule citation counts from the ``interactions`` table.

        Aggregates the JSON ``citations`` column over the look-back window and
        groups by ``(kind, real_id)``. Iteration is done in Python (rather
        than pushing into SQL via ``json_each``) because volumes are bounded
        per org and the resulting code is easier to maintain. Titles come
        from the citation rows themselves — they are captured at injection
        time when the rule is rendered into context.

        Args:
            days_back (int): Look-back window in days. Must be positive.

        Returns:
            list[PlaybookApplicationStat]: One row per cited ``(kind,
                real_id)``, sorted by ``applied_count`` descending and then
                by ``last_applied_at`` descending. Empty when no interactions
                in the window carry citations.
        """
        if days_back <= 0:
            return []

        current_time = _epoch_now()
        start_iso = _epoch_to_iso(current_time - days_back * 24 * 60 * 60)
        rows = self._fetchall(
            "SELECT interaction_id, created_at, citations FROM interactions "
            "WHERE created_at >= ? "
            "AND citations IS NOT NULL AND citations != '' AND citations != '[]' "
            "ORDER BY created_at DESC, interaction_id DESC",
            (start_iso,),
        )
        if not rows:
            return []

        aggregates: dict[tuple[_CitationKind, str], dict[str, Any]] = defaultdict(
            lambda: {
                "applied_count": 0,
                "title": "",
                "last_applied_at": None,
                "last_interaction_id": None,
            }
        )
        for row in rows:
            citations = _json_loads(row["citations"])
            if not isinstance(citations, list):
                continue
            seen_keys_in_interaction: set[tuple[_CitationKind, str]] = set()
            for c in citations:
                if not isinstance(c, dict):
                    continue
                kind = c.get("kind")
                real_id = c.get("real_id")
                if kind not in ("playbook", "profile") or not real_id:
                    continue
                key: tuple[_CitationKind, str] = (
                    cast(_CitationKind, kind),
                    str(real_id),
                )
                if key in seen_keys_in_interaction:
                    continue
                seen_keys_in_interaction.add(key)
                agg = aggregates[key]
                agg["applied_count"] += 1
                if agg["last_applied_at"] is None:
                    # rows ordered DESC, so the first time we see this key
                    # is the most recent occurrence
                    agg["last_applied_at"] = _iso_to_epoch(row["created_at"])
                    agg["last_interaction_id"] = row["interaction_id"]
                if not agg["title"]:
                    title = c.get("title") or ""
                    if isinstance(title, str) and title.strip():
                        agg["title"] = title.strip()

        stats = [
            PlaybookApplicationStat(
                real_id=real_id,
                kind=kind,
                title=agg["title"],
                applied_count=agg["applied_count"],
                last_applied_at=agg["last_applied_at"],
                last_interaction_id=agg["last_interaction_id"],
            )
            for (kind, real_id), agg in aggregates.items()
        ]
        stats.sort(
            key=lambda s: (
                -s.applied_count,
                -(s.last_applied_at if s.last_applied_at is not None else 0),
            )
        )
        return stats

    @SQLiteStorageBase.handle_exceptions
    def record_usage_event(
        self,
        *,
        org_id: str,
        event_name: str,
        event_category: str,
        user_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        pipeline: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        caller_type: str | None = None,
        count_value: int = 1,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        billing_input_tokens: int | None = None,
        platform_llm: bool | None = None,
        platform_storage: bool | None = None,
        duration_ms: int | None = None,
        error_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert one row into ``usage_events`` for the storage's org.

        Mirrors the ``UsageEvent`` dataclass in
        :mod:`reflexio.server.usage_metrics`. The :data:`org_id` argument
        is required for explicitness and multi-tenant flexibility; SQLite
        storage is per-org and the value should match ``self.org_id``.

        Args:
            org_id: Organisation id (must equal ``self.org_id``).
            event_name: Stable event name (e.g., ``"learning_injection"``).
            event_category: Stable event category (e.g., ``"application"``).
            user_id: Caller user id.
            request_id: Correlation id.
            session_id: Conversation id.
            pipeline: Logical pipeline tag.
            entity_type: ``"user_playbook"`` / ``"agent_playbook"`` /
                ``"profile"`` for per-entity events.
            entity_id: Storage id of the surfaced entity.
            caller_type: Caller classification.
            count_value: Multiplicity; default 1.
            prompt_tokens: Tokens for the rendered content.
            completion_tokens: Tokens for completions.
            billing_input_tokens: Input-anchored billed tokens.
            platform_llm: Whether the platform supplies the LLM.
            platform_storage: Whether the platform supplies storage.
            duration_ms: Wall-clock duration in milliseconds.
            error_kind: Error classification.
            metadata: Free-form per-event metadata (serialised as JSON).
        """
        # SQLite storage is per-org; the assertion keeps the call site honest.
        # Wrapped in ``StorageError`` so the ``@handle_exceptions`` decorator
        # does not double-wrap and so the caller sees a typed exception
        # rather than the raw ``ValueError``.
        if org_id != self.org_id:
            from reflexio.server.services.storage.error import StorageError
            raise StorageError(
                f"record_usage_event org_id mismatch: storage={self.org_id!r} "
                f"event={org_id!r}"
            )
        self._execute(
            """INSERT INTO usage_events
                 (org_id, user_id, request_id, session_id, pipeline,
                  entity_type, entity_id, event_name, event_category,
                  caller_type, count_value, prompt_tokens, completion_tokens,
                  billing_input_tokens, platform_llm, platform_storage,
                  duration_ms, error_kind, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                org_id,
                user_id or "",
                request_id or "",
                session_id or "",
                pipeline,
                entity_type,
                entity_id,
                event_name,
                event_category,
                caller_type,
                count_value,
                prompt_tokens,
                completion_tokens,
                billing_input_tokens,
                int(platform_llm) if platform_llm is not None else None,
                int(platform_storage) if platform_storage is not None else None,
                duration_ms,
                error_kind,
                _json_dumps(dict(metadata) if metadata else {}),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_injection_stats(
        self, days_back: int = 30
    ) -> list[InjectionStat]:
        """Per-entity injection rollup from ``usage_events``.

        Reads rows of ``event_name = "learning_injection"`` for
        ``self.org_id`` within the look-back window and groups by
        ``(entity_type, entity_id)`` to surface:

        - ``surfaced_count``: times the entity was injected
        - ``total_prompt_tokens``: sum of per-entity prompt tokens
        - ``first_injected_at`` / ``last_injected_at``: epoch seconds
        - ``last_session_id``: most-recent session that injected the entity

        Titles are NOT joined here. Callers needing the playbook / profile
        name should join with the corresponding tables using ``entity_id``.

        Args:
            days_back (int): Look-back window in days. Must be > 0.

        Returns:
            list[InjectionStat]: Sorted by ``surfaced_count`` DESC,
            then ``last_injected_at`` DESC. Empty when no events in the
            window.
        """
        if days_back <= 0:
            return []

        current_time = _epoch_now()
        start_iso = _epoch_to_iso(current_time - days_back * 24 * 60 * 60)
        rows = self._fetchall(
            """SELECT entity_type, entity_id, session_id, prompt_tokens,
                      created_at
                 FROM usage_events
                 WHERE event_name = 'learning_injection'
                   AND org_id = ?
                   AND created_at >= ?
                   AND entity_id IS NOT NULL
                 ORDER BY created_at DESC""",
            (self.org_id, start_iso),
        )
        if not rows:
            return []

        aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            entity_type = row["entity_type"] or ""
            entity_id = str(row["entity_id"])
            key: tuple[str, str] = (entity_type, entity_id)
            agg = aggregates.get(key)
            created_at_epoch = _iso_to_epoch(row["created_at"]) or 0
            if agg is None:
                agg = {
                    "surfaced_count": 0,
                    "distinct_sessions": set(),
                    "total_prompt_tokens": 0,
                    "first_injected_at": created_at_epoch,
                    "last_injected_at": created_at_epoch,
                    "last_session_id": row["session_id"] or "",
                }
                aggregates[key] = agg
            agg["surfaced_count"] += 1
            if row["session_id"]:
                agg["distinct_sessions"].add(row["session_id"])
            agg["total_prompt_tokens"] += int(row["prompt_tokens"] or 0)
            # rows are DESC, so the first occurrence is the most recent
            if created_at_epoch > agg["last_injected_at"]:
                agg["last_injected_at"] = created_at_epoch
                agg["last_session_id"] = row["session_id"] or ""
            if created_at_epoch < agg["first_injected_at"]:
                agg["first_injected_at"] = created_at_epoch

        stats = [
            InjectionStat(
                entity_type=entity_type,
                entity_id=entity_id,
                surfaced_count=agg["surfaced_count"],
                distinct_session_count=len(agg["distinct_sessions"]),
                total_prompt_tokens=agg["total_prompt_tokens"],
                first_injected_at=agg["first_injected_at"] or None,
                last_injected_at=agg["last_injected_at"] or None,
                last_session_id=agg["last_session_id"],
            )
            for (entity_type, entity_id), agg in aggregates.items()
        ]
        stats.sort(
            key=lambda s: (
                -s.surfaced_count,
                -(s.last_injected_at if s.last_injected_at is not None else 0),
            )
        )
        return stats

    @SQLiteStorageBase.handle_exceptions
    def get_memory_review_candidates(
        self,
        days_back: int = 60,
        user_id: str | None = None,
        include_all_users: bool = False,
    ) -> list[MemoryReviewCandidate]:
        """Surface user_playbooks flagged for memory review.

        Implements two signals in v1. ``duplicate`` and ``supersedeable``
        are reserved for a follow-up:

        - ``duplicate``: cosine-on-content is O(n²) per call and is
          better done as a periodic batch job (long-term answer for
          installations with thousands of playbooks).
        - ``supersedeable``: the aggregation change log records removed
          *agent* playbooks (``agent_playbook_id``), not the
          ``user_playbook_id`` this review keys on, so the signal can't
          be derived correctly from the current change-log schema. It is
          deferred until the source mapping is wired in.

        Archived playbooks (``status = 'archived'``) are excluded — once a
        data owner archives a candidate it must not reappear in the queue.

        Signals implemented:
          - ``stale``: not injected in ``days_back`` and created more
            than ``days_back`` ago.
          - ``high_cost_low_cite``: injected >= 3 times and citation
            rate (citations / injections) is < 0.5.

        Args:
            days_back (int): Look-back window in days. Must be > 0.
            user_id (str | None): User whose playbooks should be reviewed.
                Required unless ``include_all_users`` is true.
            include_all_users (bool): Explicit opt-in for org-wide review.

        Returns:
            list[MemoryReviewCandidate]: Sorted by ``-score``
            descending. Score encodes signal priority
            (``stale`` = 50-99, ``high_cost_low_cite`` = 30-49) so this
            sort groups by primary signal and orders within each
            group by strength. Empty when no candidates.
        """
        if days_back <= 0:
            return []
        if not include_all_users and user_id is None:
            return []

        current_time = _epoch_now()
        start_ts = current_time - days_back * 24 * 60 * 60
        start_ts_iso = _epoch_to_iso(start_ts)

        # Snapshot of current, non-archived user_playbooks.
        user_filter = ""
        params: list[Any] = []
        if not include_all_users:
            user_filter = " AND user_id = ?"
            params.append(user_id)
        playbook_rows = self._fetchall(
            """SELECT user_playbook_id, playbook_name, content, status,
                      created_at, source_span
                 FROM user_playbooks
                 WHERE COALESCE(status, '') != 'archived'"""
            + user_filter,
            tuple(params),
        )
        if not playbook_rows:
            return []

        # Build (entity_id → injection stats) for the look-back window.
        # Only ``user_playbook`` events are read: agent playbooks live in a
        # separate id space and are recorded under ``entity_type =
        # 'agent_playbook'`` to avoid id collisions here.
        injection_rows = self._fetchall(
            """SELECT entity_id,
                      COUNT(*) AS injection_count,
                      MAX(created_at) AS last_injected_at
                 FROM usage_events
                 WHERE org_id = ?
                   AND event_name = 'learning_injection'
                   AND entity_type = 'user_playbook'
                   AND created_at >= ?
                 GROUP BY entity_id""",
            (self.org_id, start_ts_iso),
        )
        injection_by_id: dict[str, dict[str, Any]] = {
            str(r["entity_id"]): {
                "injection_count": int(r["injection_count"] or 0),
                "last_injected_at": _iso_to_epoch(r["last_injected_at"]) if r["last_injected_at"] else None,
            }
            for r in injection_rows
        }

        # Citation counts from the existing applied-stats join.
        applied_rows = self._get_applied_counts_for_window(start_ts_iso)

        # Compose candidates.
        candidates: list[MemoryReviewCandidate] = []
        for row in playbook_rows:
            eid = str(row["user_playbook_id"])
            created_at_epoch = _iso_to_epoch(row["created_at"])
            if created_at_epoch is None or created_at_epoch <= 0:
                created_at_epoch = None
            inj = injection_by_id.get(
                eid, {"injection_count": 0, "last_injected_at": None}
            )
            cite_count = applied_rows.get(eid, 0)
            inj_count = inj["injection_count"]
            cite_per_inj = (cite_count / inj_count) if inj_count > 0 else 0.0

            signals: list[_MemoryReviewSignal] = []
            score = 0

            # stale — only meaningful when we have a real created_at.
            if (
                created_at_epoch is not None
                and inj_count == 0
                and (current_time - created_at_epoch) >= days_back * 24 * 60 * 60
            ):
                signals.append("stale")
                # Older = higher score; clamp to [50, 99] so stale
                # outranks high_cost_low_cite (30-49).
                age_days = max(
                    0, (current_time - created_at_epoch) // (24 * 60 * 60)
                )
                score = max(score, min(99, 50 + age_days // 7))

            # high_cost_low_cite
            if inj_count >= 3 and cite_per_inj < 0.5:
                signals.append("high_cost_low_cite")
                score = max(score, min(49, 30 + min(19, inj_count)))

            if not signals:
                continue

            title = row["playbook_name"] or (
                (row["content"] or "")[:80] + ("..." if len(row["content"] or "") > 80 else "")
            )
            candidates.append(
                MemoryReviewCandidate(
                    entity_type="user_playbook",
                    entity_id=eid,
                    title=title,
                    signals=signals,
                    score=score,
                    injection_count=inj_count,
                    citation_count=cite_count,
                    last_injected_at=inj["last_injected_at"],
                    last_cited_at=None,  # not currently exposed by get_playbook_application_stats
                    last_modified_at=created_at_epoch,
                )
            )

        candidates.sort(key=lambda c: -c.score)
        return candidates

    def _get_applied_counts_for_window(
        self, start_ts_iso: str
    ) -> dict[str, int]:
        """Return ``{user_playbook_id: citation_count}`` for the window.

        Reads ``interactions.citations`` (a JSON array of
        ``{"kind", "real_id"}`` references) in the look-back window
        and flattens to a count per user_playbook id. Reuses the same
        JSON-walk pattern as :meth:`get_playbook_application_stats`
        but with a different rollup shape (count per id vs. one row
        per cited rule) and its own ``SELECT`` because the two
        queries diverge on grouping and ordering.
        """
        rows = self._fetchall(
            """SELECT interaction_id, citations
                 FROM interactions
                 WHERE created_at >= ?
                   AND citations IS NOT NULL
                   AND citations != ''
                   AND citations != '[]'""",
            (start_ts_iso,),
        )
        counts: dict[str, int] = {}
        for row in rows:
            citations = _json_loads(row["citations"])
            if not isinstance(citations, list):
                continue
            seen: set[str] = set()
            for c in citations:
                if not isinstance(c, dict):
                    continue
                if c.get("kind") != "playbook":
                    continue
                rid = c.get("real_id")
                if rid is None or str(rid) in seen:
                    continue
                seen.add(str(rid))
                counts[str(rid)] = counts.get(str(rid), 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Statistics methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_profile_statistics(self) -> dict:
        current_ts = _epoch_now()
        expiring_soon_ts = current_ts + (7 * 24 * 60 * 60)

        rows = self._fetchall(
            "SELECT status, expiration_timestamp FROM profiles WHERE expiration_timestamp >= ?",
            (current_ts,),
        )
        stats = {
            "current_count": 0,
            "pending_count": 0,
            "archived_count": 0,
            "expiring_soon_count": 0,
        }
        for r in rows:
            s = r["status"]
            exp = r["expiration_timestamp"]
            if s is None:
                stats["current_count"] += 1
                if exp is not None and exp <= expiring_soon_ts:
                    stats["expiring_soon_count"] += 1
            elif s == "pending":
                stats["pending_count"] += 1
            elif s == "archived":
                stats["archived_count"] += 1
        return stats

    # ------------------------------------------------------------------
    # Profile Change Log methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_profile_change_log(self, profile_change_log: ProfileChangeLog) -> None:
        self._execute(
            """INSERT INTO profile_change_logs
               (user_id, request_id, created_at, added_profiles, removed_profiles, mentioned_profiles)
               VALUES (?,?,?,?,?,?)""",
            (
                profile_change_log.user_id,
                profile_change_log.request_id,
                profile_change_log.created_at,
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.added_profiles]
                ),
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.removed_profiles]
                ),
                _json_dumps(
                    [p.model_dump() for p in profile_change_log.mentioned_profiles]
                ),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_profile_change_logs(self, limit: int = 100) -> list[ProfileChangeLog]:
        rows = self._fetchall(
            "SELECT * FROM profile_change_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_profile_change_log(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_profile_change_log_for_user(self, user_id: str) -> None:
        self._execute("DELETE FROM profile_change_logs WHERE user_id = ?", (user_id,))

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profile_change_logs(self) -> None:
        self._execute("DELETE FROM profile_change_logs")

    # ------------------------------------------------------------------
    # Playbook Aggregation Change Log methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_playbook_aggregation_change_log(
        self, change_log: PlaybookAggregationChangeLog
    ) -> None:
        self._execute(
            """INSERT INTO playbook_aggregation_change_logs
               (created_at, playbook_name, agent_version, run_mode,
                added_playbooks, removed_playbooks, updated_playbooks)
               VALUES (?,?,?,?,?,?,?)""",
            (
                change_log.created_at,
                change_log.playbook_name,
                change_log.agent_version,
                change_log.run_mode,
                _json_dumps(
                    [fb.model_dump() for fb in change_log.added_agent_playbooks]
                ),
                _json_dumps(
                    [fb.model_dump() for fb in change_log.removed_agent_playbooks]
                ),
                _json_dumps(
                    [
                        {"before": e.before.model_dump(), "after": e.after.model_dump()}
                        for e in change_log.updated_agent_playbooks
                    ]
                ),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_playbook_aggregation_change_logs(
        self,
        playbook_name: str,
        agent_version: str,
        limit: int = 100,
    ) -> list[PlaybookAggregationChangeLog]:
        rows = self._fetchall(
            """SELECT * FROM playbook_aggregation_change_logs
               WHERE playbook_name = ? AND agent_version = ?
               ORDER BY created_at DESC LIMIT ?""",
            (playbook_name, agent_version, limit),
        )
        return [_row_to_playbook_aggregation_change_log(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_playbook_aggregation_change_logs(self) -> None:
        self._execute("DELETE FROM playbook_aggregation_change_logs")

    # ------------------------------------------------------------------
    # Evaluation-overview support (Plan B-backend)
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def count_sessions_with_shadow_content(self, from_ts: int, to_ts: int) -> int:
        """Count distinct sessions with at least one non-empty shadow interaction.

        Joins `interactions` with `requests` since `session_id` is on the
        request, not the interaction.

        Args:
            from_ts (int): Window start, unix epoch seconds.
            to_ts (int): Window end, unix epoch seconds.

        Returns:
            int: Distinct count of sessions in the window with shadow content.
        """
        rows = self._fetchall(
            """SELECT COUNT(DISTINCT r.session_id) AS n
               FROM interactions i
               JOIN requests r ON i.request_id = r.request_id
               WHERE COALESCE(i.shadow_content, '') != ''
                 AND r.session_id != ''
                 AND i.created_at >= ?
                 AND i.created_at <= ?""",
            (_epoch_to_iso(from_ts), _epoch_to_iso(to_ts)),
        )
        if not rows:
            return 0
        return int(rows[0][0] or 0)

    @SQLiteStorageBase.handle_exceptions
    def get_interactions_by_session(self, session_id: str) -> list[Interaction]:
        """Return interactions for a session, ordered by created_at.

        Joins `interactions` with `requests` so we can filter by
        Request.session_id.

        Args:
            session_id (str): The session whose interactions to fetch.

        Returns:
            list[Interaction]: Interactions in the session, possibly empty.
        """
        if not session_id:
            return []
        rows = self._fetchall(
            """SELECT i.*
               FROM interactions i
               JOIN requests r ON i.request_id = r.request_id
               WHERE r.session_id = ?
               ORDER BY i.created_at ASC""",
            (session_id,),
        )
        return [_row_to_interaction(r) for r in rows]

    # ------------------------------------------------------------------
    # Braintrust connector storage (Plan C-backend + Plan C-overview)
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_braintrust_connection(self, connection: BraintrustConnection) -> None:
        """Upsert the org's Braintrust connection.

        Args:
            connection (BraintrustConnection): Encrypted connection record.
        """
        self._execute(
            """INSERT INTO braintrust_connection
                 (org_id, api_key_enc, workspace_id, workspace_name,
                  project_ids, last_sync_ts, last_error)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(org_id) DO UPDATE SET
                 api_key_enc = excluded.api_key_enc,
                 workspace_id = excluded.workspace_id,
                 workspace_name = excluded.workspace_name,
                 project_ids = excluded.project_ids,
                 last_sync_ts = excluded.last_sync_ts,
                 last_error = excluded.last_error""",
            (
                connection.org_id,
                connection.api_key_enc,
                connection.workspace_id,
                connection.workspace_name,
                _json_dumps(connection.project_ids),
                connection.last_sync_ts,
                connection.last_error,
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_braintrust_connection(self, org_id: str) -> BraintrustConnection | None:
        """Fetch the org's Braintrust connection or None if not connected."""
        rows = self._fetchall(
            """SELECT api_key_enc, workspace_id, workspace_name, project_ids,
                      last_sync_ts, last_error
               FROM braintrust_connection
               WHERE org_id = ?""",
            (org_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return BraintrustConnection(
            org_id=org_id,
            api_key_enc=row[0],
            workspace_id=row[1],
            workspace_name=row[2] or "",
            project_ids=list(_json_loads(row[3]) or []),
            last_sync_ts=row[4],
            last_error=row[5],
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_braintrust_connection(self, org_id: str) -> None:
        """Delete the org's connection (idempotent)."""
        self._execute("DELETE FROM braintrust_connection WHERE org_id = ?", (org_id,))

    @SQLiteStorageBase.handle_exceptions
    def save_imported_scores(self, scores: list[ImportedScore]) -> None:
        """Upsert imported scores by (org_id, source, source_run_id, scorer_name)."""
        if not scores:
            return
        rows = [
            (
                s.org_id,
                s.source,
                s.source_run_id,
                s.session_id,
                s.scorer_name,
                s.value,
                s.ts,
            )
            for s in scores
        ]
        with self._lock:
            self.conn.executemany(
                """INSERT INTO imported_score
                     (org_id, source, source_run_id, session_id,
                      scorer_name, value, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(org_id, source, source_run_id, scorer_name)
                   DO UPDATE SET
                     session_id = excluded.session_id,
                     value = excluded.value,
                     ts = excluded.ts""",
                rows,
            )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def get_imported_scores(
        self, org_id: str, from_ts: int, to_ts: int
    ) -> list[ImportedScore]:
        """Return imported scores for the org in `[from_ts, to_ts]`."""
        rows = self._fetchall(
            """SELECT source, source_run_id, session_id, scorer_name, value, ts
               FROM imported_score
               WHERE org_id = ?
                 AND ts >= ?
                 AND ts <= ?
               ORDER BY ts ASC""",
            (org_id, from_ts, to_ts),
        )
        return [
            ImportedScore(
                org_id=org_id,
                source=cast(Literal["braintrust"], row[0]),
                source_run_id=row[1],
                session_id=row[2],
                scorer_name=row[3],
                value=float(row[4]),
                ts=int(row[5]),
            )
            for row in rows
        ]
