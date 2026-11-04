"""Lossless, request-bound addresses for original review-window text.

Addresses authenticate provenance only. No segmentation or lookup establishes
semantic support, factual origin, retention authority or relevance.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

_CURRENT: ContextVar[EvidenceBook | None] = ContextVar(
    "profile_source_evidence", default=None
)
_TEXT_FIELDS = ("content", "user_action_description")


def _pieces(text: str) -> Any:
    """Partition exactly, preferring whitespace; never trim or omit characters."""
    start = 0
    while start < len(text):
        stop = min(start + 360, len(text))
        if stop < len(text):
            boundary = max(
                text.rfind(" ", start + 180, stop), text.rfind("\n", start + 180, stop)
            )
            if boundary >= start + 180:
                stop = boundary + 1
        yield start, text[start:stop]
        start = stop


class EvidenceBook:
    """One complete source window, namespaced by its actual request and owner."""

    def __init__(self, window: Any, *, user_id: str, request_id: str, org_id: str):
        if not all(
            isinstance(value, str) and value for value in (user_id, request_id, org_id)
        ):
            raise ValueError(
                "Evidence requires actual nonempty request and owner identities"
            )
        if not window:
            raise ValueError("Evidence requires a complete original review window")
        digest = hashlib.sha256(
            json.dumps(
                {
                    "org_id": org_id,
                    "user_id": user_id,
                    "request_id": request_id,
                    "window": window,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self._references = {}
        self.rendered_window = deepcopy(window)
        seen = set()
        for session in self.rendered_window:
            session_id = session["session_id"]
            for interaction in session["interactions"]:
                identity = (session_id, interaction["interaction_id"])
                if identity in seen:
                    raise ValueError(
                        "Duplicate source interaction identity is ambiguous"
                    )
                seen.add(identity)
                if interaction["role"] not in {"user", "assistant", "system", "tool"}:
                    raise ValueError("Unknown original source role")
                for field in _TEXT_FIELDS:
                    text = interaction.get(field)
                    if text is None:
                        continue
                    if not isinstance(text, str):
                        raise ValueError("Original source text must be a string")
                    excerpts = []
                    for offset, piece in _pieces(text):
                        excerpt_id = f"E-{digest}-{len(self._references)}"
                        self._references[excerpt_id] = {
                            "session_id": session_id,
                            "interaction_id": interaction["interaction_id"],
                            "role": interaction["role"],
                            "quote": piece,
                        }
                        excerpts.append(
                            {"excerpt_id": excerpt_id, "offset": offset, "text": piece}
                        )
                    # Replaces each text field once, retaining all other original
                    # metadata, source roles and distinction between text/actions.
                    interaction[field] = {"excerpts": excerpts}

    def resolve(self, excerpt_id: str) -> dict:
        try:
            return dict(self._references[excerpt_id])
        except (KeyError, TypeError):
            raise ValueError("Unknown or wrong-request source excerpt_id") from None

    @contextmanager
    def bound(self) -> Any:
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)


def current_evidence_book() -> EvidenceBook:
    book = _CURRENT.get()
    if book is None:
        raise ValueError("No request-bound original source evidence is active")
    return book
