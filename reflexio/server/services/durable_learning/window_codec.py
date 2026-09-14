"""Versioned JSON representation of a computed window; no executable payloads.

**Adding a field to a persisted payload is a cross-version contract change.**
A window written by a new image can be decoded by an OLD one mid-rollout, and
``decode_plan`` used to splat ``token_totals`` straight into ``RunTokenTotals``
kwargs — so a new field raised ``TypeError`` inside ``window_executor.execute``,
unguarded, on a durable-learning path. Two defences, and they point in opposite
directions on purpose:

* :func:`decode_plan` now IGNORES keys it does not know, so any *future* field
  addition is safe to decode.
* :data:`_PERSISTED_TOKEN_FIELDS` pins what :func:`encode_plan` *writes*, so an
  old image (which does not have the tolerance above) never sees a key it cannot
  handle. Tolerance alone could not protect this first addition, because the
  image that must tolerate it is the one already deployed.

Consequence, stated rather than absorbed: the cache sub-buckets are not carried
across a durable resume and read back as 0 there. That path's token totals are
a fallback for the run-scoped accumulator, so the loss is bounded to resumed
windows. Widen ``_PERSISTED_TOKEN_FIELDS`` once no pre-cache-field image is
running.
"""

import time
from dataclasses import fields as dataclass_fields
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

#: The ``RunTokenTotals`` fields ``encode_plan`` is allowed to persist. See the
#: module docstring: this is a compatibility floor, not a description of the
#: dataclass, and it must not be regenerated from it.
_PERSISTED_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens")


def _encode_token_totals(totals: RunTokenTotals | None) -> dict[str, int] | None:
    """Serialise only the fields every deployed image can decode."""
    if totals is None:
        return None
    return {name: getattr(totals, name) for name in _PERSISTED_TOKEN_FIELDS}


def _decode_token_totals(raw: Any) -> RunTokenTotals | None:
    """Rebuild ``RunTokenTotals``, ignoring keys this build does not know.

    Unknown keys are dropped rather than raising, so a window written by a newer
    image decodes here instead of failing the whole execution.
    """
    if not raw:
        return None
    known = {f.name for f in dataclass_fields(RunTokenTotals)}
    return RunTokenTotals(**{k: v for k, v in raw.items() if k in known})


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
            "token_totals": _encode_token_totals(plan.token_totals),
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
    totals = _decode_token_totals(data["token_totals"])
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
