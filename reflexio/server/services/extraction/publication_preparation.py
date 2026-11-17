"""Original resolved publication data, never a replacement execution owner."""

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC
from typing import Any

from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.services.deferred_learning_plan import (
    ExtractorBookmarkPosition,
    PlaybookWritePlan,
    ProfileWritePlan,
)
from reflexio.server.services.extraction.history_inputs import (
    HistoricalInputs,
    history_from_output,
    require_historical_inputs,
)
from reflexio.server.services.extraction.publication_receipts import (
    output_publication_origin,
    require_output_id,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    build_scope_hash,
)
from reflexio.server.services.write_plan_codec import (
    decode_write_plan,
    encode_write_plan,
)


def _require_bookmark(
    run: AgentRunRecord, data: Any
) -> ExtractorBookmarkPosition | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != {
        "extractor_name",
        "interaction_ids",
        "last_processed_timestamp",
        "user_id",
    }:
        raise ValueError("Saved bookmark is invalid")
    position = ExtractorBookmarkPosition(**data)
    ids = position.interaction_ids
    if (
        position.extractor_name != run.binding.extractor_kind
        or position.user_id != run.binding.user_id
        or not isinstance(ids, list)
        or any(type(value) is not int or value <= 0 for value in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(run.binding.source_interaction_ids)
        or position.last_processed_timestamp is not None
        and type(position.last_processed_timestamp) is not int
    ):
        raise ValueError("Saved bookmark binding changed")
    return position


def bookmark_from_output(run: AgentRunRecord) -> ExtractorBookmarkPosition:
    """Recover only the original initial-writer position, never today's source state."""
    output = run.committed_output
    if (
        not isinstance(output, dict)
        or output_publication_origin(output) != "initial_inline"
    ):
        raise ExtractionInputIntegrityError(
            "Bookmark requires an initial inline output"
        )
    try:
        publication = output["publication"]
        bookmark = _require_bookmark(run, publication["bookmark"])
    except (ValueError, TypeError, KeyError) as error:
        raise ExtractionInputIntegrityError(
            "Original initial bookmark is unavailable"
        ) from error
    if bookmark is None:
        raise ExtractionInputIntegrityError("Original initial bookmark is unavailable")
    return bookmark


@dataclass(frozen=True)
class SavedPublicationPreparation:
    write_plan: ProfileWritePlan | PlaybookWritePlan | None
    history_inputs: HistoricalInputs
    bookmark: ExtractorBookmarkPosition | None = None

    def payload(self, run: AgentRunRecord) -> dict[str, Any]:
        output = run.committed_output
        if not isinstance(output, dict):
            raise ExtractionInputIntegrityError("Preparation requires retained output")
        bookmark = asdict(self.bookmark) if self.bookmark is not None else None
        position = _require_bookmark(run, bookmark)
        if position is not None and position != bookmark_from_output(run):
            raise ExtractionInputIntegrityError(
                "Prepared bookmark changed its original position"
            )
        return {
            "version": 1,
            "output_id": require_output_id(output.get("publication", {}).get("id")),
            "output_digest": build_scope_hash(output),
            "kind": run.binding.extractor_kind,
            "write_plan": encode_write_plan(self.write_plan),
            "history_inputs": asdict(require_historical_inputs(self.history_inputs)),
            "bookmark": bookmark,
        }

    @classmethod
    def from_payload(
        cls, run: AgentRunRecord, data: Any
    ) -> "SavedPublicationPreparation":
        if not isinstance(data, dict) or set(data) != {
            "version",
            "output_id",
            "output_digest",
            "kind",
            "write_plan",
            "history_inputs",
            "bookmark",
        }:
            raise ExtractionInputIntegrityError("Saved publication preparation invalid")
        output = run.committed_output
        if not isinstance(output, dict):
            raise ExtractionInputIntegrityError("Preparation requires retained output")
        if (
            type(data["version"]) is not int
            or data["version"] != 1
            or data["output_id"]
            != require_output_id(output.get("publication", {}).get("id"))
            or data["output_digest"] != build_scope_hash(output)
            or data["kind"] != run.binding.extractor_kind
        ):
            raise ExtractionInputIntegrityError(
                "Saved publication preparation binding changed"
            )
        try:
            history = history_from_output(
                {"publication": {"history_inputs": data["history_inputs"]}}
            )
            plan = decode_write_plan(data["write_plan"], run.binding.extractor_kind)
            bookmark = _require_bookmark(run, data["bookmark"])
            if bookmark is not None and bookmark != bookmark_from_output(run):
                raise ValueError("Prepared bookmark changed its original position")
            if plan is not None:
                rows = (
                    plan.new_profiles
                    if isinstance(plan, ProfileWritePlan)
                    else plan.new_playbooks
                )
                if (
                    plan.request_id != run.binding.request_id
                    or isinstance(plan, ProfileWritePlan)
                    and plan.user_id != run.binding.user_id
                    or any(row.user_id != run.binding.user_id for row in rows)
                ):
                    raise ValueError("Publication subjects changed")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ExtractionInputIntegrityError(
                "Saved publication preparation invalid"
            ) from exc
        return cls(plan, history, bookmark)


def preparation_claim_digest(run: AgentRunRecord) -> str:
    if run.claimed_by is None or run.claimed_at is None:
        raise ExtractionInputIntegrityError(
            "Preparation requires an actual execution claim"
        )
    stamp = run.claimed_at
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    return build_scope_hash({"worker": run.claimed_by, "claimed_at": stamp.isoformat()})


@dataclass(frozen=True)
class PendingPreparationAttempt:
    identity: str
    output_id: str
    output_digest: str
    kind: str
    claim_digest: str
    failures_at_entry: int

    @classmethod
    def begin(cls, run: AgentRunRecord) -> "PendingPreparationAttempt":
        if type(run.finalization_attempts) is not int or run.finalization_attempts < 0:
            raise ExtractionInputIntegrityError(
                "Preparation failure budget unavailable"
            )
        output = run.committed_output
        if not isinstance(output, dict):
            raise ExtractionInputIntegrityError("Preparation requires retained output")
        return cls(
            uuid.uuid4().hex,
            require_output_id(output.get("publication", {}).get("id")),
            build_scope_hash(output),
            run.binding.extractor_kind,
            preparation_claim_digest(run),
            run.finalization_attempts,
        )

    def payload(self) -> dict[str, Any]:
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_payload(
        cls, run: AgentRunRecord, data: Any
    ) -> "PendingPreparationAttempt":
        if (
            not isinstance(data, dict)
            or set(data)
            != {
                "version",
                "identity",
                "output_id",
                "output_digest",
                "kind",
                "claim_digest",
                "failures_at_entry",
            }
            or type(data["version"]) is not int
            or data["version"] != 1
            or type(data["failures_at_entry"]) is not int
            or data["failures_at_entry"] < 0
        ):
            raise ExtractionInputIntegrityError("Pending preparation entry invalid")
        require_output_id(data["identity"])
        require_output_id(data["output_id"])
        if any(
            not isinstance(data[key], str)
            or len(data[key]) != 64
            or any(c not in "0123456789abcdef" for c in data[key])
            for key in ("output_digest", "claim_digest")
        ):
            raise ExtractionInputIntegrityError("Pending preparation entry invalid")
        output = run.committed_output
        if (
            not isinstance(output, dict)
            or data["output_id"] != output.get("publication", {}).get("id")
            or data["output_digest"] != build_scope_hash(output)
            or data["kind"] != run.binding.extractor_kind
        ):
            raise ExtractionInputIntegrityError(
                "Pending preparation output binding changed"
            )
        return cls(**{key: value for key, value in data.items() if key != "version"})
