"""Bulk request bodies are validated off the event loop, with an unchanged contract.

A multi-MB publish validated on the loop starved ``/health/live`` long enough
for the ALB to replace the only production task (2026-11-10). These tests pin:

- the liveness probe answers while a large publish body is being validated;
- every body route whose model carries a list of nested models uses
  ``off_loop_body`` (derived from the app, not listed);
- the OpenAPI request bodies and component schemas equal a plain ``payload:
  Model`` declaration;
- the 422 for a bad body equals FastAPI's own, byte for byte.
"""

# No ``from __future__ import annotations``: the reference routes declare
# ``payload: model`` with a local variable, which FastAPI must see evaluated.

import json
import threading
import types
import typing
from typing import Any, cast

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel

from reflexio.server import offloop_body
from reflexio.server.api import _safe_request_validation_exception_handler, create_app
from reflexio.server.offloop_body import _dependency, off_loop_body

_HEADERS = {"User-Agent": "pytest-agent"}


def _bulk_fields(model: Any) -> list[str]:
    """Top-level fields typed as a list of models: per-element validation cost."""
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return []
    bulk = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if typing.get_origin(annotation) in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(annotation) if a is not type(None)]
            annotation = members[0] if len(members) == 1 else annotation
        args = typing.get_args(annotation)
        if (
            typing.get_origin(annotation) is list
            and args
            and isinstance(args[0], type)
            and issubclass(args[0], BaseModel)
        ):
            bulk.append(name)
    return bulk


def _off_loop_model(route: APIRoute) -> type[BaseModel] | None:
    """The model an ``off_loop_body`` dependency of this route validates, if any."""
    for dependency in route.dependant.dependencies:
        for body_param in dependency.body_params:
            model = body_param.field_info.annotation
            if dependency.call is _dependency(model):
                return cast(type[BaseModel], model)
    return None


def _body_routes() -> list[APIRoute]:
    return [
        route
        for route in create_app().routes
        if isinstance(route, APIRoute) and route.body_field is not None
    ]


def _bulk_routes() -> list[tuple[APIRoute, type[BaseModel]]]:
    found = []
    for route in _body_routes():
        model = route.body_field.field_info.annotation  # type: ignore[union-attr]
        if _bulk_fields(model):
            found.append((route, model))
    return found


def test_every_bulk_body_route_validates_off_loop() -> None:
    bulk = _bulk_routes()
    # The predicate must keep finding the routes this fix was written for.
    assert {"/api/publish_interaction", "/api/add_user_profile"} <= {
        route.path for route, _ in bulk
    }
    on_loop = [
        f"{route.path} ({model.__name__}: {_bulk_fields(model)})"
        for route, model in bulk
        if _off_loop_model(route) is not model
    ]
    assert not on_loop, (
        "These routes take a list of nested models in their body but let FastAPI "
        "validate it on the event loop; declare the body with off_loop_body(Model): "
        + ", ".join(on_loop)
    )


def _reference_app(model: type[BaseModel], *, off_loop: bool) -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(
        RequestValidationError,
        _safe_request_validation_exception_handler,  # type: ignore[arg-type]
    )
    if off_loop:

        @app.post("/body")
        async def body(
            payload: BaseModel = off_loop_body(model),  # noqa: B008
        ) -> dict[str, str]:
            return {"type": type(payload).__name__}

    else:

        @app.post("/body")
        async def body_plain(payload: model) -> dict[str, str]:  # type: ignore[valid-type]
            return {"type": type(payload).__name__}

    return app


def _off_loop_models() -> list[type[BaseModel]]:
    models = {_off_loop_model(route) for route in _body_routes()}
    return sorted((m for m in models if m is not None), key=lambda m: m.__name__)


@pytest.mark.parametrize("model", _off_loop_models(), ids=lambda m: m.__name__)
def test_openapi_body_matches_a_plain_declaration(model: type[BaseModel]) -> None:
    plain = _reference_app(model, off_loop=False).openapi()
    moved = _reference_app(model, off_loop=True).openapi()
    path = "/body"
    assert (
        moved["paths"][path]["post"]["requestBody"]
        == plain["paths"][path]["post"]["requestBody"]
    )
    assert moved["components"]["schemas"] == plain["components"]["schemas"]


def _publish_item(**overrides: Any) -> dict[str, Any]:
    return {"role": "user", "content": "hi", **overrides}


_BAD_BODIES: list[tuple[str, bytes | None, str | None]] = [
    ("missing_body", None, None),
    ("empty_object", b"{}", "application/json"),
    ("malformed_json", b'{"user_id": ', "application/json"),
    ("json_list", b"[1, 2]", "application/json"),
    ("json_string", b'"text"', "application/json"),
    ("text_plain", b'{"user_id": "u"}', "text/plain"),
    ("no_content_type", b'{"user_id": "u"}', None),
    ("non_finite_number", b'{"user_id": NaN, "session_id": "s"}', "application/json"),
    (
        "nested_type_errors",
        json.dumps(
            {
                "user_id": 5,
                "session_id": "s",
                "interaction_data_list": [
                    _publish_item(content=["not", "a", "string"]),
                    _publish_item(tools_used=[{"tool_data": 3}]),
                    "not an object",
                ],
                "agent_version": "x" * 1_001,
            }
        ).encode(),
        "application/json",
    ),
    (
        "model_validator_error",
        json.dumps(
            {"user_id": "u", "session_id": "s", "interaction_data_list": []}
        ).encode(),
        "application/json",
    ),
    (
        "valid",
        json.dumps(
            {
                "user_id": "u",
                "session_id": "s",
                "interaction_data_list": [
                    _publish_item(tools_used=[{"tool_name": "t"}])
                ],
            }
        ).encode(),
        "application/json",
    ),
]


@pytest.mark.parametrize("model", _off_loop_models(), ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    ("body", "content_type"),
    [(b, c) for _, b, c in _BAD_BODIES],
    ids=[name for name, _, _ in _BAD_BODIES],
)
@pytest.mark.asyncio
async def test_responses_match_fastapi_validation(
    model: type[BaseModel], body: bytes | None, content_type: str | None
) -> None:
    headers = dict(_HEADERS)
    if content_type is not None:
        headers["content-type"] = content_type
    answers = []
    for off_loop in (False, True):
        transport = httpx.ASGITransport(app=_reference_app(model, off_loop=off_loop))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            response = await c.post("/body", content=body, headers=headers)
        answers.append((response.status_code, response.content))
    assert answers[1] == answers[0]


@pytest.mark.asyncio
async def test_liveness_answers_while_a_large_publish_body_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real publish route, a ~5 MB body, and a validation held open.

    Validation is held until the probe has answered (bounded at 3 s), so the
    outcome does not depend on how fast this machine validates: on the loop,
    the probe cannot be served until validation returns, and the hold expires
    with ``probe_answered`` unset.
    """
    entered = threading.Event()
    probe_answered = threading.Event()
    seen: list[bool] = []
    real_validate = offloop_body.validate_body

    def held_validate(model: type[BaseModel], value: Any) -> BaseModel:
        entered.set()
        seen.append(probe_answered.wait(timeout=3))
        return real_validate(model, value)

    monkeypatch.setattr(offloop_body, "validate_body", held_validate)
    tools = [
        {"tool_name": f"t{i}", "status": "success", "tool_data": {"out": "z" * 5_200}}
        for i in range(70)
    ]
    body = json.dumps(
        {
            "user_id": "u",
            "session_id": "s",
            # Invalid last item: full validation runs, then a 422, so no
            # storage is needed behind the route.
            "interaction_data_list": [
                *(_publish_item(tools_used=tools) for _ in range(14)),
                _publish_item(role=["bad"]),
            ],
        }
    ).encode()
    assert len(body) > 5_000_000

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        anyio.create_task_group() as tg,
    ):
        publish_status: list[int] = []

        async def publish() -> None:
            response = await client.post(
                "/api/publish_interaction",
                content=body,
                headers={**_HEADERS, "content-type": "application/json"},
            )
            publish_status.append(response.status_code)

        tg.start_soon(publish)
        try:
            with anyio.fail_after(10):
                while not entered.is_set():
                    await anyio.sleep(0.01)
            with anyio.fail_after(1):
                probe = await client.get("/health/live", headers=_HEADERS)
            assert probe.status_code == 200
            probe_answered.set()
        finally:
            probe_answered.set()

    assert seen == [True]
    assert publish_status == [422]


def test_publish_route_declares_its_body_after_its_dependencies() -> None:
    """Body errors still come after auth and the billing gate, as before."""
    route = next(
        route
        for route in create_app().routes
        if isinstance(route, APIRoute) and route.path == "/api/publish_interaction"
    )
    calls = [dependency.call for dependency in route.dependant.dependencies]
    body = _dependency(_off_loop_model(route))  # type: ignore[arg-type]
    assert calls[-1] is body
