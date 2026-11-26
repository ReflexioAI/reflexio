"""Read committed extraction receipts without widening new model authoring.

Historical TTLs remain visibility bounds with unknown lifecycle origin. Readers
preserve the saved field shape because durable candidate attribution hashes it.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, model_serializer, model_validator

from reflexio.models.api_schema.service_schemas import ProfileTimeToLive
from reflexio.server.services.profile.lifecycle import KEY
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
    StructuredProfilesOutput,
)


class _CommittedProfileAddItem(ProfileAddItem):
    time_to_live: ProfileTimeToLive

    @model_validator(mode="before")
    @classmethod
    def refuse_estimated_lifetime(cls, value: Any) -> Any:
        # Only this committed reader accepts the historical required enum.
        if isinstance(value, dict) and (
            value.get("explicit_storage_deadline") is not None
            or value.get("explicit_forget_quote") is not None
            or KEY in value
        ):
            raise ValueError(
                "Legacy output cannot acquire storage authority or lifecycle clocks"
            )
        return value

    @model_serializer(mode="wrap")
    def preserve_saved_fields(self, handler: Any) -> dict[str, Any]:
        values = handler(self)
        return {
            key: value
            for key, value in values.items()
            if key in self.model_fields_set or key in (self.model_extra or {})
        }


class _CommittedProfilesOutput(StructuredProfilesOutput):
    profiles: list[ProfileAddItem | _CommittedProfileAddItem] | None = None


def read_saved_structured_output[Output: BaseModel](
    schema: type[Output], payload: dict[str, Any]
) -> Output:
    """Validate trusted stored output; never use for a fresh model response."""
    if schema is StructuredProfilesOutput and any(
        isinstance(item, dict) and "time_to_live" in item
        for item in payload.get("profiles") or []
    ):
        return cast(Output, _CommittedProfilesOutput.model_validate(payload))
    return schema.model_validate(payload)
