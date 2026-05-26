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
