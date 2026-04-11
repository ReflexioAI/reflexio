from ._base import DiskStorageBase
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin


class DiskStorage(
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
    DiskStorageBase,
):
    """Disk-based storage with entity files and QMD search.

    Each entity is stored as a file with YAML frontmatter for metadata
    and a body for content.  Search operations are delegated to QMD
    (tobi/qmd) for BM25, vector, and hybrid search.
    """

    pass


__all__ = ["DiskStorage"]
