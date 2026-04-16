from ._base import BaseStorageCore, matches_status_filter
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin
from ._share_links import ShareLinkMixin


class BaseStorage(
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
    ShareLinkMixin,
    BaseStorageCore,
):
    """Base class for storage."""

    pass


__all__ = ["BaseStorage", "PlaybookMixin", "ShareLinkMixin", "matches_status_filter"]
