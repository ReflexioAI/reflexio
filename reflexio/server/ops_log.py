"""The operational-event channel: informational lines that must always be visible.

Production keeps the root logger at WARNING (``configure_logging``), so an INFO
line is dropped. That pushed routine events -- a slow-publish timing line, a
leadership handover, the boot-time logging posture -- to WARNING just to be
seen, and about a third of production's WARNING stream became events that were
never problems, burying the ones that were.

``configure_logging`` pins this namespace to INFO in every profile, so an event
logged here at INFO is always emitted and WARNING keeps meaning "something is
wrong". Call sites keep their own message text (log-derived metric filters and
greps match on ``event=<name>``, not on the logger or the level).
"""

from __future__ import annotations

import logging

#: Reserved logger namespace. ``configure_logging`` pins it to INFO; do not
#: raise it, or every operational event disappears in production.
OPS_LOGGER_NAME = "reflexio.ops"


def get_ops_logger(subsystem: str) -> logging.Logger:
    """Return the operational-event logger for one subsystem.

    Log at INFO on it. WARNING and above belong on the subsystem's ordinary
    logger, because they mean something needs attention.

    Args:
        subsystem (str): A short dotted name, e.g. ``"publish_timing"``.

    Returns:
        logging.Logger: ``reflexio.ops.<subsystem>``.
    """
    return logging.getLogger(f"{OPS_LOGGER_NAME}.{subsystem}")
