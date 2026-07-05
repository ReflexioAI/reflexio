"""Reflexio param resolution + best-effort background publishing for the wrapper."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.defaults import resolve_agent_version

if TYPE_CHECKING:
    from reflexio.client import ReflexioClient
    from reflexio.models.api_schema.domain.entities import InteractionData

logger = logging.getLogger(__name__)


class ReflexioParams(BaseModel):
    """Publish params for a wrapped LLM call.

    ``user_id`` and ``session_id`` are **required** to publish — construct this
    model (or pass an equivalent dict) per call. They may instead be supplied as
    wrap-time defaults to ``wrap_llm_client``; the requirement is enforced on the
    *merged* result, before the LLM call, whenever ``publish`` is True.

    Pass either a ``ReflexioParams`` instance or a plain ``dict`` as ``reflexio=``.
    Unknown keys are rejected (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_content: str | None = None
    source: str = ""
    agent_version: str = ""
    publish: bool = True
    publish_partial_stream: bool = False
    skip_aggregation: bool = False
    force_extraction: bool = False
    evaluation_only: bool = False

    @model_validator(mode="after")
    def _validate_flags(self) -> Self:
        # Mirror PublishUserInteractionRequest so contradictions fail at the call site.
        if self.evaluation_only and self.force_extraction:
            raise ValueError("evaluation_only cannot be combined with force_extraction")
        return self


# Canonical keys accepted in a ``reflexio`` dict (wrap-time and per-call).
ALLOWED_KEYS = frozenset(ReflexioParams.model_fields)

# Keys forwarded to ``publish_interaction`` as extraction flags.
_PASSTHROUGH_FLAGS = ("skip_aggregation", "force_extraction", "evaluation_only")

_REQUIRED_TO_PUBLISH = ("user_id", "session_id")


def to_param_dict(value: Any) -> dict[str, Any]:
    """Normalize a ``reflexio`` argument (``ReflexioParams`` | mapping | None) to a dict.

    For a ``ReflexioParams`` instance only the explicitly-set fields are returned,
    so wrap-time defaults and per-call overrides merge cleanly.
    """
    if value is None:
        return {}
    if isinstance(value, ReflexioParams):
        return value.model_dump(exclude_unset=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("reflexio= must be a ReflexioParams or a mapping")


def validate_keys(params: Mapping[str, Any]) -> None:
    """Reject unknown ``reflexio`` keys (typo protection)."""
    unknown = set(params) - ALLOWED_KEYS
    if unknown:
        raise TypeError(
            f"Unknown reflexio param(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(ALLOWED_KEYS)}"
        )


def merge_and_validate(defaults: Mapping[str, Any], call_reflexio: Any) -> dict[str, Any]:
    """Merge wrap-time defaults with a per-call ``reflexio`` and validate.

    Per-call values win key-by-key. Validation is synchronous (before the LLM
    call) so config mistakes fail fast at the call site rather than silently in
    the background publish worker.

    Raises:
        TypeError: ``call_reflexio`` is not a ``ReflexioParams``/mapping, or an
            unknown key is present.
        ValueError: when publishing (``publish`` is not ``False``) and a required
            field is missing, or ``evaluation_only``/``force_extraction`` conflict.
            (``pydantic.ValidationError`` is a subclass of ``ValueError``.)
    """
    merged = {**to_param_dict(defaults), **to_param_dict(call_reflexio)}
    validate_keys(merged)

    if merged.get("publish", True) is False:
        # Not publishing: identity is not required, but still reject contradictions.
        if merged.get("evaluation_only") and merged.get("force_extraction"):
            raise ValueError("evaluation_only cannot be combined with force_extraction")
        return merged

    missing = [k for k in _REQUIRED_TO_PUBLISH if not str(merged.get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"reflexio is missing required field(s) {missing} needed to publish. "
            "Provide user_id and session_id, or pass publish=False to skip publishing "
            "this call."
        )
    # Full model validation (types, flag combos, extra=forbid).
    ReflexioParams.model_validate(merged)
    return merged


class Publisher:
    """Submits interaction batches to Reflexio off the caller's critical path."""

    def __init__(self, reflexio_client: ReflexioClient) -> None:
        self._client = reflexio_client

    def publish(
        self, interactions: list[InteractionData], params: dict[str, Any]
    ) -> None:
        """Schedule a best-effort background publish. Never raises into the caller.

        No-ops on an empty batch or when identity is missing (already guarded
        upstream, re-checked here so the worker can never hit the server's
        empty-``session_id`` ``ValueError``).
        """
        if not interactions:
            return
        user_id = params.get("user_id", "")
        session_id = params.get("session_id", "")
        if not user_id or not session_id:
            logger.warning(
                "reflexio wrapper: missing user_id/session_id; skipping publish"
            )
            return

        client = self._client
        publish_kwargs: dict[str, Any] = {
            "source": params.get("source", ""),
            "agent_version": resolve_agent_version(params.get("agent_version", "")),
            "session_id": session_id,
            "wait_for_response": False,
        }
        for flag in _PASSTHROUGH_FLAGS:
            if flag in params:
                publish_kwargs[flag] = params[flag]

        def _job() -> None:
            try:
                client.publish_interaction(user_id, interactions, **publish_kwargs)
            except Exception:
                logger.exception("reflexio wrapper: background publish failed")

        client._thread_pool.submit(_job)
