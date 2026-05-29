"""Polarity helpers for playbook orientation.

Extractor prompts teach the LLM to write either direct action rules or
avoidance rules, but they do not require a separate polarity output field.
This module derives the internal ``UserPlaybook.polarity`` value from the
written rule so downstream search, reflection, consolidation, and aggregation
can still keep action rules separate from avoidance rules.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from reflexio.models.api_schema.domain.entities import UserPlaybook

logger = logging.getLogger(__name__)

NEGATIVE_PREFIXES: tuple[str, ...] = ("Avoid", "Do not", "Don't", "Never")
NEGATIVE_EVIDENCE_TERMS: tuple[str, ...] = (
    "failed",
    "failure",
    "rejected",
    "refuted",
    "pushback",
    "pushed back",
    "self-corrected",
    "disliked",
)


def _content_fingerprint(content: str) -> str:
    """Stable short hash for log correlation without leaking raw text.

    Args:
        content (str): The playbook content.

    Returns:
        str: First 16 hex chars of the SHA-256 of the UTF-8-encoded content.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def looks_negative(content: str) -> bool:
    """Heuristic check: does the content's leading word look negative-framed?

    This is a framing signal. ``infer_playbook_polarity`` combines it with
    failure evidence before deriving internal polarity.

    Args:
        content (str): The playbook's content text.

    Returns:
        bool: True iff the stripped content starts with one of
        ``NEGATIVE_PREFIXES``.
    """
    stripped = content.lstrip()
    return any(stripped.startswith(p) for p in NEGATIVE_PREFIXES)


def infer_playbook_polarity(
    content: str,
    rationale: str | None = None,
) -> Literal["positive", "negative"]:
    """Derive playbook polarity from rule wording and failure evidence.

    Positive/actionable guidance is the default. Negative polarity is reserved
    for rules that are written as explicit avoidance guidance and whose
    rationale/content contains a failure signal.

    Args:
        content (str): The playbook content.
        rationale (str | None): Optional rationale supporting the playbook.

    Returns:
        Literal["positive", "negative"]: The derived internal polarity.
    """
    if not looks_negative(content):
        return "positive"

    evidence_text = f"{content}\n{rationale or ''}".lower()
    if any(term in evidence_text for term in NEGATIVE_EVIDENCE_TERMS):
        return "negative"
    return "positive"


def warn_if_polarity_content_mismatch(playbook: UserPlaybook) -> None:
    """Log a warning when content framing disagrees with declared polarity.

    Does not raise; does not block writes. Used to surface prompt drift
    in observability without taking corrective action.

    Args:
        playbook (UserPlaybook): The playbook about to be written.
    """
    content_looks_negative = looks_negative(playbook.content)
    declared_negative = playbook.polarity == "negative"
    if content_looks_negative != declared_negative:
        # Log only non-sensitive metadata: a stable fingerprint for
        # correlation, content length, and a boolean for which side
        # mismatched. Raw playbook content can carry PII / sensitive
        # instructions and must never reach centralized logging.
        logger.warning(
            "event=polarity_content_mismatch playbook_id=%s polarity=%s "
            "content_sha256_16=%s content_len=%d content_looks_negative=%s",
            playbook.user_playbook_id,
            playbook.polarity,
            _content_fingerprint(playbook.content),
            len(playbook.content),
            content_looks_negative,
        )
