# ==============================
# Internal data models
# for only internal data representation
# ==============================
from pydantic import BaseModel

from .service_schemas import Interaction, Request


class RequestInteractionDataModel(BaseModel):
    session_id: str
    request: Request
    interactions: list[Interaction]
