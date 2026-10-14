"""Frozen native deduplication for an explicitly admitted profile package.

The ordinary consolidator still owns its semantic validator and defensive
fallback. This adapter records actual attempts and the host fallback separately;
it never labels an invalid first-parsed response as validated model output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ValidationError

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.deduplication_utils import EXTRACTION_CONTROL_ERRORS
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.profile_package_preparation import (
    ProfilePackagePreparation,
)
from reflexio.server.services.extraction.profile_package_state import (
    PROFILE_WITNESS_FIELDS,
    profile_witness,
)
from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
)


class _SavedDedupFallbackError(RuntimeError):
    """Re-enter the existing host fallback with its actual parsed provenance."""

    def __init__(self, provenance: ModelProvenance | None):
        super().__init__("Previously recorded profile deduplication fallback")
        self.first_parsed_provenance = provenance


class PackageProfileConsolidator(ProfileConsolidator):
    def __init__(
        self,
        *,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        coordinator: OverflowCoordinator,
        preparation: ProfilePackagePreparation,
        template_identity: dict[str, str],
    ):
        super().__init__(request_context=request_context, llm_client=llm_client)
        self.coordinator = coordinator
        self.preparation = preparation
        self.template_identity = dict(template_identity)
        self.model_name = preparation.snapshot["consumer"]["ladders"]["dedup"][0]

    def _validate_old(self) -> None:
        with self.coordinator.storage.commit_scope():
            self.coordinator.validate_current_parent()
            self.coordinator.storage.validate_extraction_historical_profiles(
                self.coordinator.window,
                self.coordinator.checkpoint.get("historical_profiles", {}),
            )

    def _retrieve_existing_profiles(
        self, new_profiles: list[UserProfile], user_id: str
    ) -> list[UserProfile]:
        if user_id != self.coordinator.window.user_id:
            raise InputBudgetConfigurationError("Profile deduplication owner changed")
        key = self.coordinator.candidate_key(
            [profile.model_dump(mode="json") for profile in new_profiles]
        )
        saved = self.coordinator.checkpoint.get("profile_dedup_inputs")
        if saved is None:
            old = super()._retrieve_existing_profiles(new_profiles, user_id)
            if any(profile.user_id != user_id for profile in old):
                raise InputBudgetConfigurationError("Historical profile owner changed")
            checkpoint = self.coordinator.checkpoint
            witnesses = checkpoint.setdefault("historical_profiles", {})
            for profile in old:
                previous = witnesses.setdefault(
                    profile.profile_id, profile_witness(profile)
                )
                if previous != profile_witness(profile):
                    raise InputBudgetConfigurationError("Historical profile changed")
            checkpoint["profile_dedup_inputs"] = {
                "new_digest": key,
                "old": [
                    profile.model_dump(mode="json", include=set(PROFILE_WITNESS_FIELDS))
                    for profile in old
                ],
                "template": self.template_identity,
            }
            self.coordinator.save_checkpoint(checkpoint)
        else:
            if (
                saved["new_digest"] != key
                or saved["template"] != self.template_identity
            ):
                raise InputBudgetConfigurationError(
                    "Frozen profile dedup inputs changed"
                )
            old = [UserProfile.model_validate(row) for row in saved["old"]]
            witnesses = self.coordinator.checkpoint.get("historical_profiles", {})
            if any(
                witnesses.get(row.profile_id) != profile_witness(row) for row in old
            ):
                raise InputBudgetConfigurationError("Frozen historical profile changed")
        self._validate_old()
        return old

    def _render_dedup_prompt(
        self,
        new_profiles: list[UserProfile],
        existing_profiles: list[UserProfile],
        *,
        template_identity: Mapping[str, str] | None = None,
    ) -> str:
        if (
            template_identity is not None
            and dict(template_identity) != self.template_identity
        ):
            raise InputBudgetConfigurationError("Profile dedup template changed")
        return super()._render_dedup_prompt(
            new_profiles, existing_profiles, template_identity=self.template_identity
        )

    def _call_native_dedup(
        self,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        validator: Callable[[BaseModel], list[str]],
    ) -> CompletionResult[Any]:
        key = self.coordinator.step_key(
            "profile-package-dedup-v1",
            {"contract": self.preparation.contract_digest, "messages": messages},
        )
        self._validate_old()
        saved = self.coordinator.checkpoint.get("profile_dedup_decision")
        if saved is not None:
            if not isinstance(saved, dict) or set(saved) != {
                "key",
                "attempt_ids",
                "disposition",
                "output",
                "provenance",
            }:
                raise InputBudgetConfigurationError(
                    "Malformed saved deduplication decision"
                )
            if saved["key"] != key:
                raise InputBudgetConfigurationError(
                    "Frozen profile dedup prompt changed"
                )
            self.coordinator.validate_profile_attempts(
                "profile_dedup", saved["attempt_ids"], key
            )
            try:
                provenance = (
                    ModelProvenance(**saved["provenance"])
                    if saved["provenance"] is not None
                    else None
                )
                output = (
                    ProfileDeduplicationOutput.model_validate(saved["output"])
                    if saved["output"] is not None
                    else None
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise InputBudgetConfigurationError(
                    "Malformed saved deduplication decision"
                ) from exc
            if saved["disposition"] == "validated":
                if output is None or validator(output) or provenance is None:
                    raise InputBudgetConfigurationError(
                        "Invalid saved deduplication decision"
                    )
                return CompletionResult(output, provenance)
            if saved["disposition"] != "host_fallback":
                raise InputBudgetConfigurationError(
                    "Unknown saved deduplication decision"
                )
            if output is not None:
                validator(
                    output
                )  # Preserve first-parsed, even when semantically invalid.
            raise _SavedDedupFallbackError(provenance)

        first: BaseModel | None = None

        def capture(output: BaseModel) -> list[str]:
            nonlocal first
            if first is None and isinstance(output, ProfileDeduplicationOutput):
                first = output
            return validator(output)

        attempts: list[str] = []
        try:
            with self.coordinator.profile_request(
                self.preparation, "dedup", messages, receipt_key=key
            ) as attempts:
                completion = super()._call_native_dedup(
                    messages, output_schema, capture
                )
        except EXTRACTION_CONTROL_ERRORS:
            raise
        except Exception as exc:
            # No dispatch means there is no paid receipt to cache. The native
            # consolidator can still apply its ordinary best-effort fallback.
            if attempts:
                self._save_decision(
                    key,
                    attempts,
                    "host_fallback",
                    first,
                    getattr(exc, "first_parsed_provenance", None),
                )
            raise
        if not isinstance(completion.value, ProfileDeduplicationOutput):
            self._save_decision(key, attempts, "host_fallback", None, None)
            raise _SavedDedupFallbackError(None)
        if capture(completion.value):
            raise InputBudgetConfigurationError(
                "Native deduplication bypassed validation"
            )
        self._save_decision(
            key, attempts, "validated", completion.value, completion.provenance
        )
        return completion

    def _save_decision(
        self,
        key: str,
        attempts: list[str],
        disposition: str,
        output: BaseModel | None,
        provenance: ModelProvenance | None,
    ) -> None:
        self.coordinator.validate_profile_attempts("profile_dedup", attempts, key)
        self._validate_old()
        checkpoint = self.coordinator.checkpoint
        checkpoint["profile_dedup_decision"] = {
            "key": key,
            "attempt_ids": list(attempts),
            "disposition": disposition,
            "output": output.model_dump(mode="json") if output is not None else None,
            "provenance": asdict(provenance) if provenance is not None else None,
        }
        self.coordinator.save_checkpoint(checkpoint)
