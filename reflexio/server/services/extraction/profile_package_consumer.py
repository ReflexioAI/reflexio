"""Explicit consumption of an already-admitted profile source package.

No worker selects this entry point automatically. The caller owns admission,
frozen-context reconstruction, deduplication, publication and billing. Completing
this child never completes its parent or advances an extraction cursor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.tools import Tool
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.package_compression import (
    CompressedSourcePackage,
    compress_source_package,
)
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
)
from reflexio.server.services.extraction.profile_package_preparation import (
    ProfilePackagePreparation,
    pending_tool_policy_snapshot,
)
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    ResumableExtractionAgent,
    create_pending_info_tools_for_extractor_kind,
)
from reflexio.server.services.extraction.source_package import (
    source_package_presentation,
)

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient


def _validate_parent(
    preparation: ProfilePackagePreparation,
    coordinator: OverflowCoordinator,
) -> None:
    """Check the frozen consumer and human context before any model work."""
    descriptor = coordinator.package_protocol
    snapshot = preparation.snapshot
    expected_scope = {
        "org_id": coordinator.storage.org_id,
        "project_id": coordinator.window.project_id,
        "user_id": coordinator.window.user_id,
    }
    parent = coordinator.storage.get_agent_run(f"window:{coordinator.window.window_id}")
    prior = snapshot["prior_context"]
    answers = coordinator.checkpoint.get("answer_witnesses", {})
    if (
        coordinator.window.kind != "profile"
        or descriptor["contract_digest"] != preparation.contract_digest
        or descriptor["plan_digest"] != preparation.compression.witness
        or descriptor["source_cache_digest"]
        != preparation.compression.source_cache_digest
        or descriptor["section_digests"]
        != list(preparation.compression.native.compression.section_digests)
        or descriptor["consumer"] != snapshot["consumer"]
        or any(snapshot[key] != value for key, value in expected_scope.items())
        or parent is None
        or parent.binding.extractor_kind != "profile"
        or parent.binding.request_id != descriptor["scope"]["request_id"]
        or parent.binding.agent_version is not None
        or parent.binding.source_interaction_ids
        != [item["interaction_id"] for item in coordinator.window.manifest]
        or parent.binding.source != snapshot["source"]
        or parent.generation_request_snapshot.get("profile_package_preparation")
        != snapshot
        or parent.agent_context_snapshot != snapshot["agent_context"]
        or (
            prior is not None
            and any(
                answers.get(key) != value
                for key, value in prior["answer_witnesses"].items()
            )
        )
    ):
        raise InputBudgetConfigurationError("Profile package consumer binding changed")
    with coordinator.storage.commit_scope():
        coordinator.validate_current_parent()
        coordinator.storage.validate_extraction_inputs(coordinator.window)
        coordinator.storage.validate_extraction_source_digest(
            coordinator.window, coordinator.source_digest
        )
        coordinator.storage.validate_extraction_answers(
            coordinator.checkpoint.get("answer_witnesses", {})
        )
        coordinator.storage.validate_extraction_answer_dependencies(
            coordinator.window, coordinator.checkpoint.get("answer_dependencies", {})
        )


def validate_profile_package(
    preparation: ProfilePackagePreparation,
    package: CompressedSourcePackage,
    coordinator: OverflowCoordinator,
) -> dict[str, Any]:
    """Validate actual originals, frozen parent and processed package identity."""
    _validate_parent(preparation, coordinator)
    canonical = source_package_presentation(
        package.result.packet, sources=preparation.compression.sources
    )
    if (
        canonical != package.presentation
        or package.result.source_digest != coordinator.source_digest
        or canonical.source_digest != coordinator.source_digest
        or package.result.covered_source_ids
        != tuple(span.evidence_ref for span in preparation.compression.sources.spans)
    ):
        raise InputBudgetConfigurationError("Profile package source binding changed")
    return {
        "version": 1,
        "source_digest": coordinator.source_digest,
        "plan_digest": preparation.compression.witness,
        "contract_digest": preparation.contract_digest,
        "package_digest": canonical.package_digest,
        "section_digests": coordinator.package_protocol["section_digests"],
        "node_keys": list(package.node_keys),
        "covered_source_ids": list(package.result.covered_source_ids),
        "passthrough": package.result.passthrough,
    }


def _pending_tools(
    preparation: ProfilePackagePreparation,
    coordinator: OverflowCoordinator,
    pending_tool_config: PendingToolCallConfig | None,
) -> tuple[list[Tool], PendingToolCallToolContext | None]:
    tools_config = preparation.snapshot["pending_tool_config"]
    if tools_config is None:
        return [], None
    if pending_tool_config is None:
        raise InputBudgetConfigurationError("Profile pending tool secrets unavailable")
    pending_tool_policy_snapshot(pending_tool_config, tools_config)
    return create_pending_info_tools_for_extractor_kind(
        "profile"
    ), PendingToolCallToolContext(
        storage=coordinator.storage,
        run_id=f"window:{coordinator.window.window_id}",
        org_id=coordinator.storage.org_id,
        extractor_kind="profile",
        user_id=coordinator.window.user_id,
        config=pending_tool_config,
    )


@dataclass(frozen=True)
class ProfilePackageConsumerResult:
    package: CompressedSourcePackage
    gate_decision: bool
    agent: AgentRunResult | None


def consume_profile_package(
    *,
    preparation: ProfilePackagePreparation,
    coordinator: OverflowCoordinator,
    client: LiteLLMClient,
    pending_tool_config: PendingToolCallConfig | None = None,
    cache_only: bool = False,
) -> ProfilePackageConsumerResult:
    """Run one real gate and durable child; never publish or finalize a parent.

    Replay rebuilds the actual package from native selection receipts. A missing
    selection fails before a new dispatch when ``cache_only`` is requested.
    Human answers remain complete and every resumed model turn keeps its quota.
    """
    _validate_parent(preparation, coordinator)
    tools, tool_context = _pending_tools(preparation, coordinator, pending_tool_config)
    package = compress_source_package(
        preparation.compression, coordinator, cache_only=cache_only
    )
    witness = validate_profile_package(preparation, package, coordinator)
    checkpoint = coordinator.checkpoint
    previous = checkpoint.get("package_processing")
    if previous is not None and previous != witness:
        raise InputBudgetConfigurationError("Processed profile package changed")
    if previous is None:
        if cache_only:
            raise InputBudgetConfigurationError(
                "Profile package lacks processing receipt"
            )
        checkpoint["package_processing"] = witness
        coordinator.save_checkpoint(checkpoint)

    gate_messages = preparation.gate_messages(package.presentation.text)
    decision = True
    if gate_messages:
        key = coordinator.step_key(
            "profile-package-gate-v1", {"witness": witness, "messages": gate_messages}
        )
        saved = coordinator.completed_step(key)
        if saved is None:
            if cache_only:
                raise InputBudgetConfigurationError(
                    "Profile package lacks gate receipt"
                )
            with coordinator.profile_request(
                preparation, "gate", gate_messages, receipt_key=key
            ) as attempts:
                completion = client.generate_chat_response_with_provenance(
                    messages=gate_messages,
                    model=preparation.snapshot["consumer"]["ladders"]["gate"][0],
                )
            if not isinstance(completion.value, str) or (
                completion.value.strip().lower() not in {"true", "false"}
            ):
                raise InputBudgetConfigurationError(
                    "Profile package gate is not boolean"
                )
            decision = completion.value.strip().lower() == "true"
            coordinator.validate_profile_attempts("gate", attempts, key)
            coordinator.complete_step(
                key,
                {"decision": decision, "attempt_ids": attempts},
                completion.provenance,
            )
        else:
            output = saved["output"]
            if (
                not isinstance(output, dict)
                or set(output) != {"decision", "attempt_ids"}
                or type(output["decision"]) is not bool
            ):
                raise InputBudgetConfigurationError("Saved profile gate is not boolean")
            coordinator.validate_profile_attempts("gate", output["attempt_ids"], key)
            decision = output["decision"]
    if not decision:
        return ProfilePackageConsumerResult(package, False, None)

    parent = coordinator.storage.get_agent_run(f"window:{coordinator.window.window_id}")
    if parent is None:
        raise InputBudgetConfigurationError("Profile package parent unavailable")
    agent = ResumableExtractionAgent(client=client, storage=coordinator.storage)
    result = agent.start_profile_package(
        run=parent,
        preparation=preparation,
        package=package,
        coordinator=coordinator,
        extra_tools=tools,
        extra_tool_context=tool_context,
        cache_only=cache_only,
    )
    return ProfilePackageConsumerResult(package, True, result)
