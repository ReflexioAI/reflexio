"""UI-layer enums — re-export domain enums to keep type identity shared.

Previously this module declared duplicate StrEnum classes with the same
variants as the domain enums. That broke type identity for pyright — the
UI enum and the domain enum were seen as distinct types even though their
values matched. Re-exporting means ``reflexio.models.api_schema.ui.enums.UserActionType``
and ``reflexio.models.api_schema.domain.enums.UserActionType`` are the same
class, and converter functions don't need casts.
"""

from reflexio.models.api_schema.domain.enums import (
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    Status,
    UserActionType,
)

__all__ = [
    "PlaybookStatus",
    "ProfileTimeToLive",
    "RegularVsShadow",
    "Status",
    "UserActionType",
]
