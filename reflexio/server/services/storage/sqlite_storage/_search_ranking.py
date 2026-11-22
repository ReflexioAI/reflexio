"""FTS, unicode-lexical, vector and RRF ranking helpers for SQLite storage."""

import functools
import logging
import math
import re
import sqlite3
import unicodedata
from collections.abc import Sequence
from typing import Any

from reflexio.models.config_schema import (
    SearchMode,
)

from ._row_mappers import _json_loads

logger = logging.getLogger(__name__)

_FTS5_OPERATORS = frozenset({"OR", "AND", "NOT"})


_FTS5_RESERVED = _FTS5_OPERATORS | {"NEAR"}


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


_UNICODE_LEXICAL_MAX_TERMS = 16


_UNICODE_LEXICAL_MAX_CANDIDATES = 250


_UNICODE_LEXICAL_INDEX_VERSION = 1


def _normalize_lexical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _unicode_lexical_runs(text: str) -> tuple[str, ...]:
    """Split normalized text at punctuation and ASCII/non-ASCII boundaries."""
    normalized = _normalize_lexical_text(text)
    runs: list[str] = []
    current: list[str] = []
    current_is_ascii: bool | None = None
    for char in normalized:
        if not char.isalnum():
            if current:
                runs.append("".join(current))
                current = []
            current_is_ascii = None
            continue
        is_ascii = char.isascii()
        if current and is_ascii != current_is_ascii:
            runs.append("".join(current))
            current = []
        current.append(char)
        current_is_ascii = is_ascii
    if current:
        runs.append("".join(current))
    return tuple(runs)


@functools.lru_cache(maxsize=512)
def _unicode_lexical_terms(query: str) -> tuple[str, ...]:
    """Return bounded Unicode-aware terms used for query coverage scoring."""
    terms: list[str] = []
    # Match native FTS syntax: uppercase reserved operators are not source terms.
    source_terms = " ".join(
        token
        for token in _TOKEN_RE.findall(unicodedata.normalize("NFKC", query))
        if token not in _FTS5_RESERVED
    )
    for run in _unicode_lexical_runs(source_terms):
        if run.isascii() or len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(dict.fromkeys(terms))[:_UNICODE_LEXICAL_MAX_TERMS]


def _unicode_lexical_index_text(document: str | None) -> str:
    """Build normalized unigram/bigram tokens for the Unicode FTS sidecars."""
    if not document:
        return ""
    terms: list[str] = []
    for run in _unicode_lexical_runs(document):
        if run.isascii():
            terms.append(run)
            continue
        terms.extend(run)
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return " ".join(dict.fromkeys(terms))


def register_unicode_lexical_index_function(conn: sqlite3.Connection) -> None:
    """Register the Unicode lexical trigger function on a writer connection."""
    conn.create_function(
        "reflexio_unicode_lexical_index",
        1,
        _unicode_lexical_index_text,
        deterministic=True,
    )


def _unicode_lexical_fts_query(query: str) -> str:
    """Return an OR query over normalized Unicode lexical terms."""
    return " OR ".join(f'"{term}"' for term in _unicode_lexical_terms(query))


def _unicode_lexical_candidate_limit(result_limit: int) -> int:
    """Bound the indexed candidate pool before Python-side coverage scoring."""
    return min(max(result_limit * 5, 50), _UNICODE_LEXICAL_MAX_CANDIDATES)


def _uses_unicode_lexical_fallback(query: str) -> bool:
    return any(char.isalnum() and not char.isascii() for char in query)


def _unicode_lexical_score(document: str | None, query: str | None) -> float:
    """Score query-term coverage in a Unicode-normalized document."""
    if not document or not query:
        return 0.0
    terms = _unicode_lexical_terms(query)
    if not terms:
        return 0.0
    normalized_document = _normalize_lexical_text(document)
    matched = sum(term in normalized_document for term in terms)
    return matched / len(terms)


def _rank_unicode_lexical_rows(
    rows: Sequence[Any],
    query: str,
    text_columns: Sequence[str],
    limit: int,
) -> list[Any]:
    """Rank an already-bounded indexed candidate set by query-term coverage."""
    scored: list[tuple[float, int, Any]] = []
    for index, row in enumerate(rows):
        document = " ".join(str(row[column] or "") for column in text_columns)
        score = _unicode_lexical_score(document, query)
        if score > 0:
            scored.append((score, index, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _score, _index, row in scored[:limit]]


def _sanitize_fts_query(text: str) -> str:
    """Sanitize a query string for FTS5, defaulting to OR between tokens.

    Bare (unquoted) tokens preserve Porter stemming. Explicit OR/AND/NOT
    operators are passed through. A trailing ``*`` is appended to the last
    token for prefix matching.

    Args:
        text: Raw user query string (may contain FTS5 boolean operators like OR)

    Returns:
        FTS5-safe query string with stemming enabled and OR default
    """
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text))
    if not tokens:
        return '""'

    has_explicit_operator = any(t in _FTS5_OPERATORS for t in tokens)

    parts: list[str] = []
    for t in tokens:
        if t in _FTS5_OPERATORS:
            if not parts or parts[-1] in _FTS5_OPERATORS:
                continue
            parts.append(t)
        elif t in _FTS5_RESERVED:
            continue
        else:
            if not has_explicit_operator and parts and parts[-1] not in _FTS5_OPERATORS:
                parts.append("OR")
            parts.append(t)

    if parts and parts[-1] in _FTS5_OPERATORS:
        parts.pop()
    if not parts:
        return '""'

    # Append prefix wildcard to last token for partial-word matching
    parts[-1] = parts[-1] + "*"
    return " ".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 for degenerate inputs.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _effective_search_mode(
    mode: SearchMode,
    query_embedding: list[float] | None,
    query: str | None,
) -> SearchMode:
    """Downgrade search mode when the required embedding is unavailable.

    Args:
        mode: Requested search mode.
        query_embedding: Pre-computed query embedding, or None.
        query: Original query text. The fallback warning is suppressed when
            this is falsy, since an empty-query HYBRID/VECTOR request has no
            semantic intent to lose.

    Returns:
        The effective SearchMode — falls back to FTS when HYBRID/VECTOR lacks an embedding.
    """
    if mode in (SearchMode.HYBRID, SearchMode.VECTOR) and not query_embedding:
        if query:
            logger.warning(
                "Search mode '%s' requested but no query embedding provided — falling back to FTS",
                mode,
            )
        return SearchMode.FTS
    return mode


def _vector_rank_rows(
    rows: Sequence[Any],
    query_embedding: list[float],
    match_count: int,
    *,
    threshold: float,
) -> list[Any]:
    """Rank rows by cosine similarity to the query embedding.

    Args:
        rows: Candidate rows with stored embeddings.
        query_embedding: The query's embedding vector.
        match_count: Number of results to return.
        threshold: Strict minimum cosine similarity for the vector arm.

    Returns:
        Top ``match_count`` rows sorted by cosine similarity descending.
    """
    scored: list[tuple[Any, float]] = []
    for row in rows:
        raw_emb = row["embedding"] if "embedding" in row.keys() else None  # noqa: SIM118
        emb = _json_loads(raw_emb) if raw_emb else None
        if emb:
            sim = _cosine_similarity(query_embedding, emb)
            scored.append((row, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Diagnostic: log the full pre-cut score distribution so retrieval
    # misses are debuggable. Without this, the only signal callers see is
    # "K results returned" with no way to tell whether a relevant row
    # scored 0.39 (close, just below an upstream threshold filter) vs
    # 0.05 (semantic mismatch, no threshold tuning will save it). Top 10
    # is sufficient context; logs at INFO so it shows up in production
    # backend.log without an explicit debug flag. Cost is one log line
    # per vector search call (~200 bytes).
    if scored:
        top = [round(s, 3) for _, s in scored[:10]]
        passing = [(row, score) for row, score in scored if score > threshold]
        logger.info(
            "vector_rank: candidates=%d passing=%d threshold=%.3f "
            "match_count=%d top_scores=%s",
            len(scored),
            len(passing),
            threshold,
            match_count,
            top,
        )
        return [row for row, _ in passing[:match_count]]
    return []


def _true_rrf_merge(
    fts_rows: Sequence[Any],
    vec_rows: Sequence[Any],
    id_column: str,
    match_count: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 1.0,
) -> list[Any]:
    """Merge independent FTS and vector result sets via Reciprocal Rank Fusion.

    Unlike ``_rrf_rerank`` (which re-ranks FTS results only), this function
    takes two independently-produced result lists and unions them so that
    documents appearing in *either* modality can surface.

    Args:
        fts_rows: Rows from an FTS query, in BM25-ranked order.
        vec_rows: Rows from a vector query, in cosine-similarity order.
        id_column: Column name used as primary key to deduplicate rows.
        match_count: Number of results to return.
        rrf_k: RRF smoothing constant (default 60).
        vector_weight: Weight for vector similarity contribution.
        fts_weight: Weight for FTS contribution.

    Returns:
        Top ``match_count`` rows sorted by combined RRF score.
    """
    if not fts_rows and not vec_rows:
        return []

    # Collect unique rows by ID (first-seen wins for the Row object)
    row_by_id: dict[str | int, Any] = {}
    for row in (*fts_rows, *vec_rows):
        rid = row[id_column]
        if rid not in row_by_id:
            row_by_id[rid] = row

    # Build rank maps (1-based); missing entries get a penalty rank
    fts_rank: dict[str | int, int] = {
        row[id_column]: i + 1 for i, row in enumerate(fts_rows)
    }
    vec_rank: dict[str | int, int] = {
        row[id_column]: i + 1 for i, row in enumerate(vec_rows)
    }
    fts_penalty = len(fts_rows) + 1
    vec_penalty = len(vec_rows) + 1

    scored: list[tuple[Any, float]] = []
    for rid, row in row_by_id.items():
        f_rank = fts_rank.get(rid, fts_penalty)
        v_rank = vec_rank.get(rid, vec_penalty)
        score = fts_weight / (rrf_k + f_rank) + vector_weight / (rrf_k + v_rank)
        scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:match_count]]
