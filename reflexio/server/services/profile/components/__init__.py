"""Profile service components."""

from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
    ProfileDeletionDirective,
    ProfileDuplicateGroup,
)

__all__ = [
    "ProfileConsolidator",
    "ProfileDeduplicationOutput",
    "ProfileDeletionDirective",
    "ProfileDuplicateGroup",
]
