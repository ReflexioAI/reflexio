"""TEST-ONLY assertion oracle for classifying playbook rule orientation.

This module is **not production code**. Under Option B (consolidator-compose),
production no longer derives polarity mechanically — a playbook's orientation
(positive action guidance vs. negative avoidance guidance) lives in the rule
wording itself and is judged by the LLM, never computed from a heuristic and
never stored on ``UserPlaybook``.

The mechanical heuristic below is preserved purely as an **assertion oracle**
for e2e/eval tests: it lets those tests classify the orientation of rules in
controlled fixtures (e.g. "the surviving rows don't carry opposing polarity on
the same trigger", "a failure-derived playbook reads as avoid/negative"). Only
an LLM could classify orientation better, and reworking those e2e/eval
assertions is out of scope — but the heuristic must not live in production, so
it lives here under ``tests/``.

Importable from every test subtree (e2e, server/services, eval) as
``tests._polarity_oracle`` because ``tests`` is a package on ``sys.path``
(see ``tests/conftest.py``).
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

    This is a framing signal the test oracle ``infer_playbook_polarity``
    combines with failure evidence to classify a fixture's orientation. It is
    not a production polarity derivation (see the module docstring).

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
    """Classify a fixture's orientation from rule wording and failure evidence.

    TEST ORACLE ONLY (see module docstring) — production never derives polarity
    this way. Positive/actionable guidance is the default. Negative is reserved
    for rules written as explicit avoidance guidance whose rationale/content
    contains a failure signal.

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
