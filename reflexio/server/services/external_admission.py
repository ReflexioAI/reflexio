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


class ExternalAdmissionFencedError(Exception):
    """A participant's own fence was lost, so it refused this admission.

    Raise it (or a subclass) from ``claim`` or ``complete`` when the authority
    the participant admitted under -- a lease, a revision, an erasure
    generation, or the evidence it read -- changed before commit. The
    transaction rolls back, so nothing landed, and the participant's owner
    retries the work. It is an expected race, not a fault: GenerationService
    logs it as a WARNING (``event=publish_admission_fenced``) and re-raises it
    unchanged so the owner can handle its own subclass. Do not use it for
    validation errors or integrity conflicts; those must stay visible as
    failures.
    """


class ExternalAdmissionParticipant(Protocol):
    def claim(self, request_id: str, user_id: str) -> ExternalAdmissionReceipt | None:
        """Fence authorization and return an exact committed replay, if any.

        Raises ExternalAdmissionFencedError when the participant's fence was
        lost; any other exception is reported as a publish failure.
        """
        ...

    def complete(
        self,
        request: Request,
        interactions: list[Interaction],
        admission: dict[str, Any],
    ) -> None:
        """Persist the receipt atomically; errors roll back canonical admission."""
        ...
