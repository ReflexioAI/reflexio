"""Polarity helpers for playbook orientation.

Extractor prompts teach the LLM to write either direct action rules or
avoidance rules, but they do not require a separate polarity output field.
This module derives a playbook's orientation (positive action guidance vs.
negative avoidance guidance) from the written rule so downstream search,
reflection, consolidation, and aggregation can still keep action rules
separate from avoidance rules. Polarity is never stored on ``UserPlaybook``;
it is always derived from wording at read time.
"""

from __future__ import annotations

from typing import Literal

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


def looks_negative(content: str) -> bool:
    """Heuristic check: does the content's leading word look negative-framed?

    This is a framing signal. ``infer_playbook_polarity`` combines it with
    failure evidence before deriving polarity.

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
        Literal["positive", "negative"]: The derived polarity.
    """
    if not looks_negative(content):
        return "positive"

    evidence_text = f"{content}\n{rationale or ''}".lower()
    if any(term in evidence_text for term in NEGATIVE_EVIDENCE_TERMS):
        return "negative"
    return "positive"
