"""Versioned JSON representation of a computed window; no executable payloads."""

import time
from typing import Any

from pydantic_core import to_jsonable_python

from reflexio.models.api_schema.domain.entities import (
    LineageContext,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.base_generation_service import PreparedGenerationRun
from reflexio.server.services.deferred_learning_plan import (
    FinalizationResult,
    GenerationComputePlan,
    PlaybookWritePlan,
    ProfileWritePlan,
)


def encode_plan(plan: GenerationComputePlan | None, service: Any) -> dict[str, Any]:
    if plan is None:
        return {"version": 1, "skipped": True}
    return to_jsonable_python(
        {
            "version": 1,
            "skipped": False,
            "write_plan": plan.write_plan,
            "generated_count": plan.generated_count,
            "billable_count": plan.billable_count,
            "extraction_run_ids": plan.extraction_run_ids,
            "token_totals": plan.token_totals,
            "model_provenance": service._last_model_provenance,
            "stats": service._last_extractor_run_stats,
            "finalization_result": plan.finalization_result,
        }
    )


def decode_plan(
    data: dict[str, Any], kind: str, config: Any, user_id: str, service: Any
) -> GenerationComputePlan | None:
    if data["version"] != 1:
        raise ValueError("Unsupported extraction outcome version")
    if data["skipped"]:
        return None
    raw = data["write_plan"]
    write_plan = None
    if raw is not None:
        raw = dict(raw)
        raw["lineage_contexts"] = [
            LineageContext.model_validate(x) for x in raw["lineage_contexts"]
        ]
        if kind == "profile":
            raw["new_profiles"] = [
                UserProfile.model_validate(x) for x in raw["new_profiles"]
            ]
            write_plan = ProfileWritePlan(**raw)
        else:
            raw["new_playbooks"] = [
                UserPlaybook.model_validate(x) for x in raw["new_playbooks"]
            ]
            if raw.get("consolidation_provenance"):
                raw["consolidation_provenance"] = ModelProvenance(
                    **raw["consolidation_provenance"]
                )
            write_plan = PlaybookWritePlan(**raw)
    totals = RunTokenTotals(**data["token_totals"]) if data["token_totals"] else None
    service._last_extraction_run_ids = data["extraction_run_ids"]
    service._last_token_totals = totals
    service._last_model_provenance = (
        ModelProvenance(**data["model_provenance"])
        if data["model_provenance"]
        else None
    )
    service._last_extractor_run_stats = data["stats"]
    from reflexio.server.services.extractor_config_utils import get_extractor_name

    return GenerationComputePlan(
        prepared=PreparedGenerationRun(config, get_extractor_name(config), user_id),
        generated_count=data["generated_count"],
        billable_count=data["billable_count"],
        write_plan=write_plan,
        bookmark_advance=None,
        generation_start=time.perf_counter(),
        extraction_run_ids=data["extraction_run_ids"],
        token_totals=totals,
        finalization_result=FinalizationResult(**data["finalization_result"])
        if data.get("finalization_result")
        else None,
    )
