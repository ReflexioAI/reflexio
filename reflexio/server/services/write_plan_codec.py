"""One business write-plan JSON contract for durable window and output recovery.

Execution ownership never travels in this payload; callers must acquire and
validate their actual retained execution before applying a decoded plan.
"""

from copy import deepcopy
from typing import Any

from pydantic_core import to_jsonable_python

from reflexio.models.api_schema.domain.entities import (
    LineageContext,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.services.deferred_learning_plan import (
    PlaybookWritePlan,
    ProfileWritePlan,
)


def encode_write_plan(
    plan: ProfileWritePlan | PlaybookWritePlan | None,
) -> dict[str, Any] | None:
    return to_jsonable_python(plan) if plan is not None else None


def decode_write_plan(
    data: dict[str, Any] | None, kind: str
) -> ProfileWritePlan | PlaybookWritePlan | None:
    if kind not in {"profile", "playbook"}:
        raise ValueError("Unknown write-plan kind")
    if data is None:
        return None
    raw = deepcopy(data)
    raw["lineage_contexts"] = [
        LineageContext.model_validate(x) for x in raw["lineage_contexts"]
    ]
    if kind == "profile":
        raw["new_profiles"] = [
            UserProfile.model_validate(x) for x in raw["new_profiles"]
        ]
        return ProfileWritePlan(**raw)
    raw["new_playbooks"] = [
        UserPlaybook.model_validate(x) for x in raw["new_playbooks"]
    ]
    if raw.get("consolidation_provenance"):
        raw["consolidation_provenance"] = ModelProvenance(
            **raw["consolidation_provenance"]
        )
    return PlaybookWritePlan(**raw)
