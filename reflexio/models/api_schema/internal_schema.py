# ==============================
# Internal data models
# for only internal data representation
# ==============================
from typing import NamedTuple

from pydantic import BaseModel, PrivateAttr

from .domain import CitationKind
from .service_schemas import Interaction, Request


class RequestInteractionDataModel(BaseModel):
    # Runtime provenance from the missing-request loader; never a serialized
    # schema/config option or permission to replace a captured real request.
    _fallback_request: bool = PrivateAttr(default=False)
    arrival_order: bool = False
    session_id: str
    request: Request
    interactions: list[Interaction]


class SessionDescriptor(NamedTuple):
    """Lightweight identifier for a session in a time window.

    Used by regenerate workflows to enumerate sessions without loading
    full Request/Interaction payloads.
    """

    user_id: str
    session_id: str
    agent_version: str
    source: str


class SessionFirstRequest(NamedTuple):
    """Earliest request metadata for one session."""

    session_id: str
    user_id: str
    source: str
    created_at: int


class SessionCitation(NamedTuple):
    """One cited rule/profile occurrence, keyed by user/session."""

    user_id: str
    session_id: str
    kind: CitationKind
    real_id: str
    title: str
