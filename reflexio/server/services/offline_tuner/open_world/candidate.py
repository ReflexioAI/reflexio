"""Retired import name for the superseded full-rewrite tuner.

The sole active method emits an indexed Proposal and applies its exact edit in
code. This tombstone prevents historical scripts from silently running the old
engine; it holds no baseline prompt, model call or fallback implementation.
"""

from .models import OpenWorldCapabilityUnavailableError


class ContentOnlyCandidateGenerator:
    def __init__(self, **_kwargs: object) -> None:
        raise OpenWorldCapabilityUnavailableError(
            "Legacy full-rewrite tuner is retired; use the rule-evidence method"
        )
