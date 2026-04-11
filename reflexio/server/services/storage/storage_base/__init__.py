from ._base import BaseStorageCore, matches_status_filter
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin


class BaseStorage(
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
    BaseStorageCore,
):
    """Base class for storage."""

    pass


__all__ = ["BaseStorage", "PlaybookMixin", "matches_status_filter"]
