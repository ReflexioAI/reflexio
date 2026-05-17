"""Contract tests for the ``supports_embedding`` capability flag on storage backends.

The flag replaces the legacy ``hasattr(storage, "_get_embedding")`` duck-type
check used by ``unified_search_service`` and ``Reflexio._query_embedding``. A
backend that implements ``_get_embedding`` must declare ``supports_embedding =
True``; backends that don't must leave it at the ``False`` default.
"""

from reflexio.server.services.storage.disk_storage import DiskStorageBase
from reflexio.server.services.storage.sqlite_storage import SQLiteStorageBase
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._base import BaseStorageCore


def test_base_storage_core_defaults_to_false() -> None:
    """The default capability is False so a new backend that forgets to set
    the flag fails closed (no embedding lookup attempted)."""
    assert BaseStorageCore.supports_embedding is False
    assert BaseStorage.supports_embedding is False


def test_sqlite_storage_declares_embedding_support() -> None:
    """SQLite implements ``_get_embedding`` and must advertise it."""
    assert SQLiteStorageBase.supports_embedding is True
    assert hasattr(SQLiteStorageBase, "_get_embedding")


def test_disk_storage_does_not_declare_embedding_support() -> None:
    """Disk storage does not implement ``_get_embedding``; the flag stays False."""
    assert DiskStorageBase.supports_embedding is False
    assert not hasattr(DiskStorageBase, "_get_embedding")


def test_flag_and_method_agree_for_each_concrete_backend() -> None:
    """For every BaseStorage subclass, ``supports_embedding`` must match
    whether ``_get_embedding`` is actually defined. Prevents silent drift."""
    backends: list[type[BaseStorage]] = [SQLiteStorageBase, DiskStorageBase]
    for backend in backends:
        has_method = hasattr(backend, "_get_embedding")
        assert backend.supports_embedding is has_method, (
            f"{backend.__name__}: supports_embedding={backend.supports_embedding} "
            f"but _get_embedding is {'present' if has_method else 'absent'}"
        )
