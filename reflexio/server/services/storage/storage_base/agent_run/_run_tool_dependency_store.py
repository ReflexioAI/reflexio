"""Backend-neutral attachment, listing and consumption of run-tool dependencies."""

from __future__ import annotations

from ._models import RunToolDependencyRecord


class RunToolDependencyStoreABC:
    """Backend-neutral run-tool-dependency store contract."""

    def attach_run_tool_dependency(
        self, record: RunToolDependencyRecord
    ) -> RunToolDependencyRecord:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def list_run_tool_dependencies(self, run_id: str) -> list[RunToolDependencyRecord]:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def consume_run_tool_dependencies(self, run_id: str) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")
