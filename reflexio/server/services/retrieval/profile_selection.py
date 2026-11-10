"""Profile Fusion V2: preserve native support and recover bounded lexical matches."""

from __future__ import annotations

import re
from typing import Protocol

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.config_schema import SearchMode
from reflexio.server import search_runtime
from reflexio.server.services.storage.error import ReadinessUnavailableError

# Exact source-only query tokenizer/stop list from the qualified Fusion V2.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)
_TERMS = re.compile(r"\w+", flags=re.UNICODE)


class ProfileRetrieval(Protocol):
    def __call__(
        self, *, query: str, top_k: int, search_mode: SearchMode
    ) -> list[UserProfile]:
        """Borrow native retrieval with the original caller scope and deadline."""
        ...


def lexical_query(query: str) -> str:
    """Unique non-stop words of length >=3, in source order, bounded to 2000 chars."""
    seen: set[str] = set()
    terms: list[str] = []
    length = 0
    for index, match in enumerate(_TERMS.finditer(query.casefold())):
        if index % 128 == 0:
            search_runtime.checkpoint()
        term = match.group()
        if term in _STOPWORDS or term in seen or len(term) < 3:
            continue
        seen.add(term)
        addition = len(term) + (4 if terms else 0)
        if length + addition > 2000:
            break
        terms.append(term)
        length += addition
    return " OR ".join(terms)


def select_profiles(
    *,
    items: list[UserProfile],
    request: UnifiedSearchRequest,
    query: str,
    effective_mode: SearchMode,
    query_embedding: list[float] | None,
    degraded: bool,
    floor_on: bool,
    temporal_reorder: bool,
    retrieve: ProfileRetrieval,
) -> list[UserProfile]:
    """Run one secondary retrieval for qualifying hybrid default-five requests.

    Sparse results preserve their native prefix and append unseen secondary
    identities. A full native five protects its first three and puts the first
    two FTS-ranked identities outside that prefix ahead of them. A small lexical
    pool preserves the native five. The native caller owns retrieval, eligibility,
    recency, session filtering and final served-state recording.
    """
    if (
        not items
        or degraded
        or (request.entity_types and "profiles" not in request.entity_types)
        or not request.user_id
        or request.threshold is not None
        or request.top_k not in (None, 5)
        or effective_mode != SearchMode.HYBRID
        or floor_on
        or temporal_reorder
        or query_embedding is None
    ):
        return items
    primary_ids = [item.profile_id for item in items]
    if len(items) > 5 or len(set(primary_ids)) != len(primary_ids):
        raise ReadinessUnavailableError("Invalid native profile selection identities")
    broad = len(items) == 5
    secondary_query = lexical_query(query) if broad else query
    if broad and not secondary_query:
        return items
    search_runtime.checkpoint()
    secondary = retrieve(
        query=secondary_query,
        top_k=20 if broad else 5,
        search_mode=SearchMode.FTS if broad else effective_mode,
    )
    search_runtime.checkpoint()
    secondary_ids = [item.profile_id for item in secondary]
    if len(set(secondary_ids)) != len(secondary_ids):
        raise ReadinessUnavailableError("Duplicate secondary profile identity")
    search_runtime.increment("retrieval.profile_secondary")
    if broad:
        if len(secondary) <= 5:
            return items
        protected = primary_ids[:3]
        head = [key for key in secondary_ids if key not in protected][:2]
        selected = head + protected if len(head) == 2 else primary_ids
    else:
        selected = list(primary_ids)
        present = set(selected)
        for key in secondary_ids:
            if key not in present:
                selected.append(key)
                present.add(key)
            if len(selected) == 5:
                break
    # Original native instances win if a secondary lookup returns the same ID.
    indexed = {item.profile_id: item for item in secondary}
    indexed.update({item.profile_id: item for item in items})
    search_runtime.checkpoint()
    return [indexed[key] for key in selected]
