"""UI-facing enums for API response models.

These mirror domain enum values but are independently owned by the UI layer.
Changes to domain enums do not automatically affect the API contract.
"""

from enum import Enum, StrEnum

__all__ = [
    "UserActionType",
    "ProfileTimeToLive",
    "PlaybookStatus",
    "Status",
    "RegularVsShadow",
]


class UserActionType(StrEnum):
    CLICK = "click"
    SCROLL = "scroll"
    TYPE = "type"
    NONE = "none"


class ProfileTimeToLive(StrEnum):
    ONE_DAY = "one_day"
    ONE_WEEK = "one_week"
    ONE_MONTH = "one_month"
    ONE_QUARTER = "one_quarter"
    ONE_YEAR = "one_year"
    INFINITY = "infinity"


class PlaybookStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Status(str, Enum):  # noqa: UP042 - CURRENT=None is not compatible with StrEnum
    CURRENT = None
    ARCHIVED = "archived"
    PENDING = "pending"
    ARCHIVE_IN_PROGRESS = "archive_in_progress"


class RegularVsShadow(StrEnum):
    REGULAR_IS_BETTER = "regular_is_better"
    REGULAR_IS_SLIGHTLY_BETTER = "regular_is_slightly_better"
    SHADOW_IS_BETTER = "shadow_is_better"
    SHADOW_IS_SLIGHTLY_BETTER = "shadow_is_slightly_better"
    TIED = "tied"
