from enum import Enum, StrEnum

from ..common import BlockingIssueKind  # noqa: F401

__all__ = [
    "UserActionType",
    "ProfileTimeToLive",
    "PlaybookStatus",
    "Status",
    "OperationStatus",
    "RegularVsShadow",
    "BlockingIssueKind",
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
    CURRENT = None  # None for current profile/playbook
    ARCHIVED = "archived"  # archived old profiles/playbooks
    PENDING = "pending"  # new profiles/playbooks that are not approved
    ARCHIVE_IN_PROGRESS = (
        "archive_in_progress"  # temporary status during downgrade operation
    )


class OperationStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RegularVsShadow(StrEnum):
    """
    This enum is used to indicate the relative performance of the regular and shadow versions of the agent.
    """

    REGULAR_IS_BETTER = "regular_is_better"
    REGULAR_IS_SLIGHTLY_BETTER = "regular_is_slightly_better"
    SHADOW_IS_BETTER = "shadow_is_better"
    SHADOW_IS_SLIGHTLY_BETTER = "shadow_is_slightly_better"
    TIED = "tied"
