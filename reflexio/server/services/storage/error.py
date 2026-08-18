from typing import Any


class StorageError(Exception):
    """
    Exception raised for storage errors
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"StorageError: {self.message}"


class SubjectWriteBarrierError(StorageError):
    """Raised when a write targets a subject with an active erasure barrier."""


class OptimizationJobLeaseLiveError(StorageError):
    """Raised when an optimizer recovery attempt finds a non-expired lease."""


class OptimizationJobIdentityConflictError(StorageError):
    """Raised when one durable optimizer identity resolves to conflicting jobs."""


class OptimizationArtifactIntegrityError(StorageError):
    """Raised when a durable optimizer artifact is malformed or conflicts."""


class OpenWorldQualificationConflictError(StorageError):
    """Raised when a cached qualification key resolves to a conflicting result."""


def require_non_empty_session_id(value: Any) -> str:
    """Return a stripped, non-empty request ``session_id`` or raise ``StorageError``.

    ``Request.session_id`` is a required non-empty field. Rows persisted before
    the session-id migration may still carry NULL/blank values; surfacing a
    typed storage error (rather than a raw Pydantic ``ValidationError`` deep in
    a read path) tells the operator to run the latest data migrations.

    Args:
        value (Any): The raw ``session_id`` value read from storage.

    Returns:
        str: The stripped, non-empty session id.

    Raises:
        StorageError: If ``value`` is missing or blank.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise StorageError(
        "requests.session_id is missing or empty; run the latest data migrations"
    )
