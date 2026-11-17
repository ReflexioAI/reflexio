"""Validate large JSON request bodies in a worker thread, not on the event loop.

FastAPI validates a declared body model inside its request handler, on the
event loop, even for a sync (``def``) endpoint: only the endpoint call itself
moves to the threadpool. A bulk-ingest body -- a list of nested models, up to
the 10 MB body limit -- costs per-element Python-level validation there, and
everything else on that loop waits, ``/health/live`` included. On 2026-11-10 a
multi-MB publish held the single production task's loop long enough for two
consecutive ALB liveness probes to time out, and ECS replaced the task.

:func:`off_loop_body` keeps the route's contract unchanged and moves only the
model validation:

- FastAPI still reads the body and decodes the JSON, so a malformed or missing
  body, the content-type rules and the ``json_invalid`` / ``missing`` 422s are
  FastAPI's own. ``json.loads`` is one C call that holds the GIL wherever it
  runs, so a thread would not shorten it.
- The declared annotation is the model itself, so the OpenAPI request body and
  component schemas are byte-identical to declaring ``payload: Model``. A
  pass-through validator only skips the loop-side validation.
- The model is validated in a worker thread exactly as FastAPI would
  (``from_attributes=True``, errors prefixed with ``"body"``, no ``url``), and a
  failure is raised as the same ``RequestValidationError``, so the 422 body is
  unchanged.
- It is a dependency, so it resolves where FastAPI validated the body before:
  after the route's other dependencies (auth, billing gate) and before the
  endpoint and its rate-limit decorator.

One difference: FastAPI reports body errors together with query-parameter
errors in one 422; a dependency that raises reports the body errors alone.
Both are 422s.

Which routes must use it is derived, not listed: a body model with a top-level
``list[BaseModel]`` field is bulk input
(``tests/server/test_offloop_body.py`` fails on such a route that does not).
"""

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

from fastapi import Body, Depends
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, PlainValidator, ValidationError

__all__ = ["off_loop_body", "validate_body"]


def _passthrough(value: Any) -> Any:
    return value


def validate_body[M: BaseModel](model: type[M], value: Any) -> M:
    """Validate a decoded body as FastAPI's single, non-embedded body param does.

    Args:
        model (type[M]): The declared body model.
        value (Any): The decoded JSON (or raw bytes for a non-JSON content type).

    Returns:
        M: The validated model.

    Raises:
        RequestValidationError: With FastAPI's error shape for this body.
    """
    try:
        return model.model_validate(value, from_attributes=True)
    except ValidationError as exc:
        raise RequestValidationError(
            [
                {**error, "loc": ("body", *error["loc"])}
                for error in exc.errors(include_url=False)
            ],
            body=value,
        ) from None


@functools.cache
def _dependency[M: BaseModel](
    model: type[M],
) -> Callable[..., Coroutine[Any, Any, M]]:
    async def validated_body(
        payload: Annotated[
            model,  # type: ignore[valid-type]
            Body(),
            PlainValidator(_passthrough, json_schema_input_type=model),
        ],
    ) -> M:
        return await asyncio.to_thread(validate_body, model, payload)

    validated_body.__name__ = f"validated_{model.__name__}"
    return validated_body


def off_loop_body(model: type[BaseModel]) -> Any:
    """Declare ``model`` as the request body, validated off the event loop.

    Use as a parameter default, after the route's other dependencies::

        payload: PublishUserInteractionRequest = off_loop_body(
            PublishUserInteractionRequest
        )

    Args:
        model (type[BaseModel]): The body model.

    Returns:
        Any: A ``Depends`` marker resolving to the validated model.
    """
    return Depends(_dependency(model))
