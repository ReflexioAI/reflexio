"""Shared abstractions for host-agent adapters.

Every host adapter implements `HostAgentAdapter` so the benchmark runner can
loop over phases/tasks without caring which agent is under test. The reflexio
injection point is the single `memory: str | None` argument on `run()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.gdpval.tokens import TokenStats


@dataclass
class AgentResult:
    """Everything one task run produces, regardless of which host executed it.

    `messages` is kept around so P1 runs can publish transcripts to reflexio
    via `ReflexioMemory.publish_trajectory()`. Each adapter normalizes its
    host's trajectory into a uniform list of `{"role": str, "content": str}`
    dicts before returning.
    """

    status: str
    iterations: int
    tool_calls: int
    tokens: TokenStats
    artifacts_dir: Path
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class HostAgentAdapter(ABC):
    """Contract every host-agent adapter must satisfy."""

    name: str

    @abstractmethod
    async def initialize(self, host_state_dir: Path) -> None:
        """Bind the host agent to an isolated memory/skills directory.

        P1 calls this with an empty directory. P2 and P3 call it with a fresh
        copy of the post-P1 snapshot so both phases start from bit-identical
        warm state.

        Args:
            host_state_dir (Path): Directory that will hold the host's native
                memory (OpenSpace skills, Hermes MEMORY.md, etc.).
        """

    @abstractmethod
    async def run(
        self,
        task: dict[str, Any],
        workspace: Path,
        memory: str | None,
    ) -> AgentResult:
        """Execute one GDPVal task through the host agent.

        Args:
            task (dict[str, Any]): Normalized GDPVal task dict (from task_loader).
            workspace (Path): Isolated working directory where the agent writes
                deliverables. Each task gets its own workspace.
            memory (str | None): Optional pre-rendered reflexio playbook block.
                `None` for P1 and P2 (reflexio OFF); non-empty string for P3.
                Each adapter decides how to inject it (prompt prepend vs system
                message).

        Returns:
            AgentResult: Uniform result record consumed by the benchmark runner.
        """

    @abstractmethod
    async def snapshot_state(self, dest: Path) -> None:
        """Persist the host's current native memory/skills to `dest`.

        Called once after P1 completes. P2 and P3 each copy from `dest` back
        into a fresh `host_state_dir` before starting.

        Args:
            dest (Path): Destination directory for the snapshot.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Release any resources the adapter is holding (clients, processes)."""
