"""Soft consistency helpers for playbook polarity.

The source of truth for a playbook's polarity is the typed
``UserPlaybook.polarity`` field, populated via Pydantic structured output
in every LLM emission schema that writes playbooks. The functions in
this module check whether a playbook's free-form ``content`` framing
agrees with the declared polarity — used to produce a soft warning log
when the LLM drifts off the writing convention. Never used as the
classifier for branching logic.
"""

from __future__ import annotations

import logging

from reflexio.models.api_schema.domain.entities import UserPlaybook

logger = logging.getLogger(__name__)

NEGATIVE_PREFIXES: tuple[str, ...] = ("Avoid", "Do not", "Don't", "Never")


def looks_negative(content: str) -> bool:
    """Heuristic check: does the content's leading word look negative-framed?

    NOT the source of truth for polarity. Use ``UserPlaybook.polarity``.
    Used only by ``warn_if_polarity_content_mismatch`` to detect prompt
    drift.

    Args:
        content (str): The playbook's content text.

    Returns:
        bool: True iff the stripped content starts with one of
        ``NEGATIVE_PREFIXES``.
    """
    stripped = content.lstrip()
    return any(stripped.startswith(p) for p in NEGATIVE_PREFIXES)


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
        logger.warning(
            "event=polarity_content_mismatch playbook_id=%s polarity=%s content_starts=%r",
            playbook.user_playbook_id,
            playbook.polarity,
            playbook.content[:32],
        )
