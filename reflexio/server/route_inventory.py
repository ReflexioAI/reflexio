"""Enumerate the effective API routes of a FastAPI app or router.

Since FastAPI 0.137, ``router.routes`` / ``app.routes`` is an internal tree:
``include_router`` stores an included-router node rather than copying each
``APIRoute``, so iterating ``.routes`` directly misses every included handler.
FastAPI's supported replacement is ``fastapi.routing.iter_route_contexts``
(correct for included-route metadata from 0.143.1, the declared floor).

Use :func:`api_route_contexts` wherever code needs "every path operation this
router serves" -- the enterprise billing QPS scan and the route inventories in
tests. Read fields from the yielded ``ApiRouteContext`` (``ctx.path``, ``ctx.methods``,
``ctx.endpoint``, ``ctx.dependant``, ``ctx.body_field``): they carry the
include-level prefix and dependencies. ``ctx.original_route`` is the
un-prefixed route as declared, without the include-level dependencies.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute, RouteContext, iter_route_contexts


@dataclass(frozen=True)
class ApiRouteContext:
    """One served path operation, as FastAPI resolves it.

    ``path``, ``methods`` and ``endpoint`` are typed views of the wrapped
    ``RouteContext``; every other attribute (``dependant``, ``body_field``,
    ``response_model``, ``status_code``, ``tags``, ``name``...) is read from it.
    All of them include the include-level prefix and dependencies.
    """

    context: RouteContext

    @property
    def path(self) -> str:
        """The served path template, include prefixes applied."""
        return cast(str, self.context.path)

    @property
    def methods(self) -> frozenset[str]:
        """The HTTP methods this operation answers."""
        return frozenset(self.context.methods or ())

    @property
    def endpoint(self) -> Callable[..., Any]:
        """The handler function as declared."""
        return cast(Callable[..., Any], self.context.endpoint)

    @property
    def original_route(self) -> APIRoute:
        """The route as declared: no include prefix or include dependencies."""
        return cast(APIRoute, self.context.original_route)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.context, name)


def api_route_contexts(owner: FastAPI | APIRouter) -> Iterator[ApiRouteContext]:
    """Yield every path operation served by ``owner``.

    This module is the only place that reads ``.routes``; a source-scan guard
    in each repository fails on a raw ``.routes`` read elsewhere.

    Args:
        owner (FastAPI | APIRouter): The app or router to enumerate.

    Yields:
        ApiRouteContext: One per served path operation, with include-level
        prefixes and dependencies applied. Non-API routes (docs, OpenAPI,
        mounts) are skipped.
    """
    for context in iter_route_contexts(owner.routes):
        if isinstance(context.original_route, APIRoute):
            yield ApiRouteContext(context)


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_app_or_router(node: ast.expr) -> bool:
    name = _terminal_name(node)
    if name is None:
        return False
    lowered = name.lower()
    return lowered.endswith(("app", "router"))


def find_raw_route_reads(paths: Iterable[Path]) -> list[str]:
    """Return ``file:line`` for each ``.routes`` read on an app or router.

    The repositories' guard tests call this so that :func:`api_route_contexts`
    stays the only route enumerator. A receiver counts as an app or router
    when its terminal name (or called function's name) ends in ``app`` or
    ``router``, e.g. ``app.routes``, ``core_router.routes``,
    ``client.app.routes`` and ``create_app().routes``. Blind spots: a
    differently named variable holding a router, and ``getattr(x, "routes")``.

    Args:
        paths (Iterable[Path]): Python source files to scan. This module is
            skipped.

    Returns:
        list[str]: Offending locations, sorted.
    """
    own = Path(__file__).resolve()
    found: list[str] = []
    for path in paths:
        if path.resolve() == own:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found.extend(
            f"{path}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "routes"
            and _is_app_or_router(node.value)
        )
    return sorted(found)
