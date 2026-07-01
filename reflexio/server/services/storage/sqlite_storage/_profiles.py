"""Profile and interaction CRUD + search mixins for SQLite storage."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode

from ._base import (
    SQLiteStorageBase,
    _build_status_sql,
    _effective_search_mode,
    _epoch_now,
    _row_to_interaction,
    _row_to_profile,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
)


def _build_tags_sql(alias: str, tags: list[str] | None) -> tuple[str, list[Any]]:
    if not tags:
        return "", []
    placeholders = ",".join("?" for _ in tags)
    return (
        f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value IN ({placeholders}))",
        list(tags),
    )


class ProfileMixin:
    """Mixin providing profile + interaction search."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _fetchall: Any

    # ------------------------------------------------------------------
    # Search — Interactions & Profiles
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def search_interaction(
        self,
        search_interaction_request: SearchInteractionRequest,
        query_embedding: list[float] | None = None,
    ) -> list[Interaction]:
        req = search_interaction_request
        has_query = bool(req.query)
        match_count = req.most_recent_k or 10
        mode = _effective_search_mode(req.search_mode, query_embedding, req.query)

        conditions: list[str] = ["i.user_id = ?"]
        params: list[str | int | float] = [req.user_id]

        if req.request_id:
            conditions.append("i.request_id = ?")
            params.append(req.request_id)
        if req.start_time:
            conditions.append("i.created_at >= ?")
            params.append(req.start_time.timestamp())
        if req.end_time:
            conditions.append("i.created_at <= ?")
            params.append(req.end_time.timestamp())

        where_clause = " AND ".join(conditions)
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Vector-only: rank by embedding similarity
        if (
            mode in (SearchMode.VECTOR, SearchMode.HYBRID)
            and query_embedding
            and not has_query
        ):
            vector_limit = match_count * 10
            sql = f"""SELECT i.* FROM interactions i
                      WHERE {where_clause}
                      ORDER BY i.created_at DESC
                      LIMIT ?"""
            rows = self._fetchall(sql, (*params, vector_limit))
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        elif has_query:
            # FTS search (with optional HYBRID re-ranking)
            fts_query = _sanitize_fts_query(req.query)  # type: ignore[arg-type]
            fts_conditions = ["interactions_fts MATCH ?", *conditions]
            fts_where = " AND ".join(fts_conditions)
            fts_params: list[str | int | float] = [fts_query, *params, overfetch]
            sql = f"""SELECT i.* FROM interactions i
                      JOIN interactions_fts f ON i.interaction_id = f.rowid
                      WHERE {fts_where}
                      ORDER BY bm25(interactions_fts, 1.0, 2.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, tuple(fts_params))

            if mode == SearchMode.HYBRID and query_embedding:
                vec_limit = match_count * 10
                vec_sql = f"""SELECT i.* FROM interactions i
                              WHERE {where_clause}
                              ORDER BY i.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, (*params, vec_limit))
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "interaction_id",
                    match_count,
                )
            else:
                rows = fts_rows[:match_count]
        else:
            if req.most_recent_k:
                # No query — just fetch most recent interactions by time
                sql = f"""SELECT i.* FROM interactions i
                          WHERE {where_clause}
                          ORDER BY i.created_at DESC
                          LIMIT ?"""
                rows = self._fetchall(sql, (*params, req.most_recent_k))
                return [_row_to_interaction(r) for r in reversed(rows)]
            return []

        interactions = [_row_to_interaction(r) for r in rows]
        if req.most_recent_k:
            sorted_ints = sorted(interactions, key=lambda x: x.created_at, reverse=True)
            return list(reversed(sorted_ints[: req.most_recent_k]))
        return interactions

    @SQLiteStorageBase.handle_exceptions
    def search_user_profile(  # noqa: C901
        self,
        search_user_profile_request: SearchUserProfileRequest,
        status_filter: list[Status | None] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]

        req = search_user_profile_request
        match_count = req.top_k or 10
        current_ts = _epoch_now()
        has_query = bool(req.query)
        mode = _effective_search_mode(req.search_mode, query_embedding, req.query)
        has_embedding = query_embedding is not None
        logger.info(
            "Profile search: requested_mode=%s, effective_mode=%s, has_query=%s, has_embedding=%s, user_id=%s",
            req.search_mode,
            mode,
            has_query,
            has_embedding,
            req.user_id,
        )

        conditions: list[str] = ["p.expiration_timestamp >= ?"]
        params: list[object] = [current_ts]

        if req.user_id:
            conditions.append("p.user_id = ?")
            params.append(req.user_id)
        if req.start_time:
            conditions.append("p.last_modified_timestamp >= ?")
            params.append(int(req.start_time.timestamp()))
        if req.end_time:
            conditions.append("p.last_modified_timestamp <= ?")
            params.append(int(req.end_time.timestamp()))
        if req.source:
            conditions.append("LOWER(p.source) = LOWER(?)")
            params.append(req.source)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)
        tag_frag, tag_params = _build_tags_sql("p", req.tags)
        if tag_frag:
            conditions.append(tag_frag)
            params.extend(tag_params)

        where_clause = " AND ".join(conditions)
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC"""
            rows = self._fetchall(sql, tuple(params))
            logger.info(
                "VECTOR search: %d candidates fetched, ranking by embedding", len(rows)
            )
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        elif has_query:
            fts_query = _sanitize_fts_query(req.query)  # type: ignore[arg-type]
            sql = f"""SELECT p.* FROM profiles p
                      JOIN profiles_fts f ON p.profile_id = f.profile_id
                      WHERE profiles_fts MATCH ?
                      AND {where_clause}
                      ORDER BY bm25(profiles_fts, 0.0, 1.0)
                      LIMIT ?"""
            params_list: list[object] = [fts_query, *params, overfetch]
            fts_rows = self._fetchall(sql, tuple(params_list))
            logger.info("FTS search: %d results from BM25", len(fts_rows))

            if mode == SearchMode.HYBRID and query_embedding:
                logger.info("HYBRID merging FTS + vector results via RRF")
                vec_limit = match_count * 10
                vec_sql = f"""SELECT p.* FROM profiles p
                              WHERE {where_clause}
                              ORDER BY p.last_modified_timestamp DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, (*params, vec_limit))
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "profile_id",
                    match_count,
                )
            else:
                rows = fts_rows
        elif query_embedding:
            # HYBRID without query text: rank by embedding only
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC"""
            rows = self._fetchall(sql, tuple(params))
            logger.info(
                "HYBRID (no query text) search: %d candidates, ranking by embedding",
                len(rows),
            )
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        else:
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC
                      LIMIT ?"""
            params_list = [*params, overfetch]
            rows = self._fetchall(sql, tuple(params_list))

        profiles = [_row_to_profile(r) for r in rows]
        logger.info("Profile search: %d profiles before post-filtering", len(profiles))

        # Apply filters that can't easily go into SQL
        filtered: list[UserProfile] = []
        for profile in profiles:
            if req.custom_feature and (
                req.custom_feature.lower() not in str(profile.custom_features).lower()
            ):
                continue
            filtered.append(profile)
            if len(filtered) >= match_count:
                break
        return filtered
