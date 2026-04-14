"""Reflexio library package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.lib.reflexio_lib import Reflexio

__all__ = ["Reflexio"]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> type:
    if name == "Reflexio":
        from reflexio.lib.reflexio_lib import Reflexio

        globals()["Reflexio"] = Reflexio  # cache after first access
        return Reflexio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
