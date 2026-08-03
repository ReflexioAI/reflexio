from enum import Enum, StrEnum
from typing import Literal

from ..common import BlockingIssueKind  # noqa: F401

__all__ = [
    "UserActionType",
    "ProfileTimeToLive",
    "PlaybookStatus",
    "Status",
    "OperationStatus",
    "RegularVsShadow",
    "SessionOutcomeKind",
    "SessionOutcomeFailureReason",
    "BlockingIssueKind",
    "PlaybookReviewReasonCode",
    "ReviewUserPlaybookReasonCode",
]

#: Why the candidate reviewer decided what it decided.
#:
#: Defined here rather than beside the reviewer because the public
#: ``ReviewUserPlaybookResult`` also has to name this vocabulary, and ``models``
#: must not import from ``server``. One definition, so the schema the reviewer
#: enforces and the schema the API advertises cannot drift.
#:
#: Adding a member is not self-contained: it must also appear in the active
#: ``playbook_candidate_review`` prompt's "reason_code is exactly one of" list,
#: or the model is never told the code exists.
PlaybookReviewReasonCode = Literal[
    "grounded_useful",
    "unsupported_evidence",
    "generic",
    "speculative",
    "unsupported_causality",
    "unseen_artifact",
    "redundant",
    "late_trigger",
    "compound",
    "internal_status",
    "absence_inference",
    "not_agent_decision",
]

#: What a re-review can report publicly: every reviewer code, plus the one the
#: review *service* raises on its own behalf when a row cannot be reviewed at
#: all. ``evidence_unavailable`` never comes from the model -- it accompanies
#: ``decision="skip"``, meaning the row's provenance was absent or invalid.
ReviewUserPlaybookReasonCode = Literal[PlaybookReviewReasonCode, "evidence_unavailable"]


class SessionOutcomeKind(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class SessionOutcomeFailureReason(StrEnum):
    CONFLICTING_FINALIZATION = "conflicting_finalization"
    UNKNOWN_SESSION = "unknown_session"
    OCCURRED_BEFORE_SESSION = "occurred_before_session"
    OCCURRED_IN_FUTURE = "occurred_in_future"
    AFTER_OUTCOME_WINDOW = "after_outcome_window"
    SUBJECT_NOT_WRITABLE = "subject_not_writable"
    STORAGE_ERROR = "storage_error"


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
    MERGED = "merged"  # tombstone: consolidated into a survivor (merged_into set)
    SUPERSEDED = (
        "superseded"  # tombstone: replaced by a new version (superseded_by set)
    )
    EXPIRED = "expired"  # tombstone: TTL elapsed while active (see expiry reclamation)


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
