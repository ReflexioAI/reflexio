"""The OpenClaw plugin's request bodies must bind to the server's request models.

The plugin is TypeScript and talks HTTP directly, so nothing shares a type
with the server. Its vitest suite pins the exact bodies it sends in
``wire-requests.json``; this test checks that fixture against the request model
each FastAPI route actually declares. Every key must be a declared field at
every level: the models strip unknown keys rather than rejecting them
(``CapturesUnknownFields``), so a renamed field would otherwise publish
successfully and silently lose its data.
"""

from __future__ import annotations

import json
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]  # tests/integrations/ -> repo root
FIXTURE_PATH = (
    _REPO_ROOT
    / "reflexio/integrations/openclaw/plugin/tests/fixtures/wire-requests.json"
)


def _route_models() -> dict[str, type[BaseModel]]:
    from reflexio.server.api import app

    models: dict[str, type[BaseModel]] = {}
    for route in app.routes:
        if isinstance(route, APIRoute) and route.body_field is not None:
            annotation = route.body_field.field_info.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                models[route.path] = annotation
    return models


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """Return the pydantic model inside ``list[M]`` / ``M | None``, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (list, typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if (model := _nested_model(arg)) is not None:
                return model
    return None


def _undeclared_keys(payload: Any, model: type[BaseModel], path: str) -> list[str]:
    if isinstance(payload, list):
        return [
            key
            for i, item in enumerate(payload)
            for key in _undeclared_keys(item, model, f"{path}[{i}]")
        ]
    if not isinstance(payload, dict):
        return []
    fields = model.model_fields
    problems = [f"{path}.{key}" for key in payload if key not in fields]
    for key, value in payload.items():
        if key in fields and (child := _nested_model(fields[key].annotation)):
            problems.extend(_undeclared_keys(value, child, f"{path}.{key}"))
    return problems


_FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text())


@pytest.mark.parametrize("route_path", sorted(_FIXTURE))
def test_plugin_request_binds_to_route_model(route_path: str) -> None:
    models = _route_models()
    assert route_path in models, f"{route_path} has no request model on the server"  # noqa: S101
    model = models[route_path]
    payload = _FIXTURE[route_path]

    model.model_validate(payload)
    undeclared = _undeclared_keys(payload, model, model.__name__)
    assert not undeclared, f"keys the server would drop: {undeclared}"  # noqa: S101
