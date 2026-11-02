"""Optional execution admission immediately before external compute dispatch."""

from collections.abc import Callable
from contextvars import ContextVar

from reflexio.server.llm._litellm_types import ProviderRequestGuardError


class ComputeAdmissionError(ProviderRequestGuardError):
    def __init__(self, reason: Exception):
        super().__init__("External compute admission refused")
        self.reason = reason


external_compute_admission: ContextVar[Callable[[], None] | None] = ContextVar(
    "external_compute_admission", default=None
)


def admit_external_compute() -> None:
    guard = external_compute_admission.get()
    if guard is not None:
        try:
            guard()
        except Exception as error:
            raise ComputeAdmissionError(error) from error
