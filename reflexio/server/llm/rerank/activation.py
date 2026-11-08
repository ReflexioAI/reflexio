"""Complete-source query/pair planning inside the cross-encoder owner."""

from __future__ import annotations

from typing import Any


def query_view(query: str, tokenizer: Any) -> str:
    """Preserve original characters; shorten only queries over 256 tokens."""
    encoding = tokenizer.encode(query, add_special_tokens=False)
    size = len(encoding.ids)
    if size <= 256:
        return query
    if len(encoding.offsets) != size:
        raise ValueError("Complete original content offsets required")
    previous_start = previous_end = -1
    for span in encoding.offsets:
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            raise ValueError("Complete offset pairs required")
        start, end = span
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= len(query)
            or start < previous_start
            or end < previous_end
        ):
            raise ValueError("Monotonic in-bounds original offsets required")
        previous_start, previous_end = start, end
    end = encoding.offsets[127][1]
    start = encoding.offsets[-128][0]
    if not 0 < end < start < len(query):
        raise ValueError("Nonoverlapping original character spans required")
    return query[:end] + "\n[…]\n" + query[start:]


def activation_plan(
    query: str, documents: list[str], tokenizer: Any
) -> tuple[str, list[int]]:
    """Return the query view and indices whose complete pair fits 512 tokens."""
    view = query_view(query, tokenizer)
    indices = []
    for index, document in enumerate(documents):
        size = len(tokenizer.encode(view, document, add_special_tokens=True).ids)
        if size <= 0:
            raise ValueError("A complete activation pair must contain tokens")
        if size <= 512:
            indices.append(index)
    return view, indices
