"""Canonical identifiers for user profiles."""

from __future__ import annotations

import uuid


def new_profile_id() -> str:
    """Return a new profile identifier.

    Returns:
        str: A canonical UUIDv4 string.
    """
    return str(uuid.uuid4())
