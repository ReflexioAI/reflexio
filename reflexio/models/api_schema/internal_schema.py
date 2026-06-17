# ==============================
# Internal data models
# for only internal data representation
# ==============================
from typing import NamedTuple

from pydantic import BaseModel

from .service_schemas import Interaction, Request


class RequestInteractionDataModel(BaseModel):
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
    """One cited rule/profile occurrence, keyed by session."""

    session_id: str
    kind: str
    real_id: str
    title: str
