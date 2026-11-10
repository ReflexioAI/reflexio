"""SQLite implements the whole ``OpenWorldStoragePort``.

Not by name alone: ``BaseStorage`` declares the provisional-publication
members as stubs that raise ``NotImplementedError``, so a ``hasattr`` check
would pass on a backend that implements none of them. Each method member must
resolve to a SQLite class, and pyright checks the structural assignment below.
Behaviour is the port contract's job
(``tests/server/services/storage/test_open_world_port_contract.py``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

from reflexio.server.services.offline_tuner import open_world_port
from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldStoragePort,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_SQLITE_PACKAGE = "reflexio.server.services.storage.sqlite_storage"


def _port_members() -> tuple[set[str], set[str]]:
    """The port's ``(methods, data members)``, across every consumer Protocol."""
    methods: set[str] = set()
    data: set[str] = set()
    for cls in inspect.getmro(OpenWorldStoragePort):
        if cls.__module__ != open_world_port.__name__ and not cls.__name__.startswith(
            "UserPlaybook"
        ):
            continue
        for name, value in vars(cls).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(value):
                methods.add(name)
        data |= {
            name
            for name in getattr(cls, "__annotations__", {})
            if not name.startswith("_")
        }
    return methods, data


def _defining_class(name: str) -> type:
    return next(cls for cls in SQLiteStorage.__mro__ if name in vars(cls))


def _as_port(storage: SQLiteStorage) -> OpenWorldStoragePort:
    """Pyright fails this file if SQLite stops satisfying the port structurally."""
    return storage


def test_every_port_method_is_a_sqlite_implementation() -> None:
    methods, _data = _port_members()

    assert "commit_user_playbook_provisional_publication" in methods
    assert "load_open_world_provisional_publication_retry" in methods
    assert "list_playbook_deployment_lifecycles" in methods
    not_sqlite = {
        name: _defining_class(name).__module__
        for name in sorted(methods)
        if not _defining_class(name).__module__.startswith(_SQLITE_PACKAGE)
    }
    assert not_sqlite == {}


def test_the_port_data_members_are_on_every_instance(tmp_path: Path) -> None:
    _methods, data = _port_members()
    storage = SQLiteStorage(org_id="port-surface", db_path=str(tmp_path / "p.db"))
    try:
        assert data == {"org_id", "llm_client"}
        assert all(hasattr(storage, name) for name in data)
        assert _as_port(storage) is storage
    finally:
        storage.conn.close()
