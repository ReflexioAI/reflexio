"""Optional transactional participant for durable external evidence admission.

Implementations must use the SAME storage commit_scope as GenerationService and
perform no network I/O here. Claim runs after the canonical user stream lock;
complete runs after canonical inserts and extraction admission. Either may raise
to roll back the entire transaction. This is a library seam, never an HTTP input.
For an existing canonical request, claim also runs before embedding preparation
to return its receipt without model work. Misses are claimed again in the final
admission transaction, preserving races and authorization fences.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from reflexio.models.api_schema.service_schemas import Interaction, Request


@dataclass(frozen=True)
class ExternalAdmissionReceipt:
    request_id: str
    interaction_ids: tuple[int, ...]


class ExternalAdmissionParticipant(Protocol):
    def claim(self, request_id: str, user_id: str) -> ExternalAdmissionReceipt | None:
        """Fence authorization and return an exact committed replay, if any."""
        ...

    def complete(
        self,
        request: Request,
        interactions: list[Interaction],
        admission: dict[str, Any],
    ) -> None:
        """Persist the receipt atomically; errors roll back canonical admission."""
        ...
