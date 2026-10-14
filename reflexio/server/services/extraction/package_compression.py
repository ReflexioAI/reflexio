"""Bind shared compression to an already-admitted source-package journal.

This entry point does not admit a window, mark full processing coverage, run a
consumer or publish learnings. The caller must first freeze the complete native
consumer contract in its parent. Existing worker selection remains unchanged.
"""

from dataclasses import dataclass
from typing import Any

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.compression import ExtractionCompressionPlan
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.source_package import (
    SourcePackagePresentation,
    source_package_presentation,
)
from reflexio.server.services.input_compression import (
    CompressionNodeInput,
    CompressionResult,
)


@dataclass(frozen=True)
class CompressedSourcePackage:
    result: CompressionResult
    presentation: SourcePackagePresentation


def compress_source_package(
    plan: ExtractionCompressionPlan, coordinator: OverflowCoordinator
) -> CompressedSourcePackage:
    """Reuse exact selections and charge each new native dispatch durably.

    No caller-supplied cache can impersonate authored model output. Every saved
    choice carries its actual visible-input and native generation-contract
    witnesses, plus the actual attempts reserved while computing that node.
    Unknown usage after a crash remains consumed. Stage exhaustion fails rather
    than claiming source processing or publishing a partial learning.
    """
    descriptor = coordinator.package_protocol
    if (
        plan.source_digest != coordinator.source_digest
        or plan.witness != descriptor["plan_digest"]
        or plan.source_cache_digest != descriptor["source_cache_digest"]
        or list(plan.native.compression.section_digests)
        != descriptor["section_digests"]
    ):
        raise InputBudgetConfigurationError("Compression package plan changed")

    def authorize() -> None:
        with coordinator.storage.commit_scope():
            coordinator.validate_current_parent()
            coordinator.storage.validate_extraction_inputs(coordinator.window)
            coordinator.storage.validate_extraction_source_digest(
                coordinator.window, plan.source_digest
            )

    # Full original fields are read only at the operation boundaries. Individual
    # dispatches use the existing scoped CAS guard's lease/identity checks.
    authorize()
    active: CompressionNodeInput | None = None
    attempt_start = 0
    with (
        coordinator.provider_scope(),
        coordinator.attempt_stage("compress") as attempts,
    ):

        def inspect(node: CompressionNodeInput) -> None:
            nonlocal active, attempt_start
            active, attempt_start = node, len(attempts)

        def bindings(key: str) -> dict[str, Any]:
            if active is None or active.key != key:
                raise InputBudgetConfigurationError("Compression node binding changed")
            return {
                "kind": "source_selection",
                "protocol_version": 1,
                "source_digest": plan.source_digest,
                "plan_digest": plan.witness,
                "node_key": key,
                "input_refs_digest": active.input_refs_digest,
                "request_contract_digest": plan.native.request_contract_digest,
            }

        def load(key: str) -> dict[str, Any] | None:
            expected = bindings(key)
            saved = coordinator.completed_step(key)
            if saved is None:
                return None
            if any(saved.get(name) != value for name, value in expected.items()):
                raise InputBudgetConfigurationError("Saved compression node changed")
            # The core independently revalidates the canonical choice against
            # actual visible originals and this consumer's exact output size.
            return saved["output"]

        def save(key: str, output: dict[str, Any]) -> None:
            coordinator.complete_selection_step(
                key,
                {
                    **bindings(key),
                    "output": output,
                    "attempt_ids": attempts[attempt_start:],
                },
            )

        result = plan.execute(load=load, save=save, inspect_node=inspect)
    authorize()
    return CompressedSourcePackage(
        result, source_package_presentation(result.packet, sources=plan.sources)
    )
