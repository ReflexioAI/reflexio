"""Executed, credential-free structural controls for the Trigger component."""

from hashlib import sha256
from typing import Any

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    TRIGGER_QUALIFICATION_CHECK_NAMES,
)
from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    TriggerGate,
    TriggerMaintenanceResult,
    TriggerSourceFrame,
    run_trigger_maintenance,
)


def qualify_trigger_structure() -> dict[str, object]:
    frame = TriggerSourceFrame(
        incumbent_id=1,
        owner_id="u",
        incumbent_version="a" * 64,
        evidence_bundle_digest="b" * 64,
        final_content_artifact_digest="c" * 64,
        original_content="After a timeout, inspect the unknown result before retrying.",
        final_content="After a timeout, inspect the unknown result before retrying.",
        original_trigger="After confirmed success",
    )
    gate: dict[str, Any] = {
        "contrast": {
            "gap": "opposed_consultation_state",
            "missed_task": "Inspect an unknown result",
            "original_excludes_because": "Requires confirmed success",
            "content_actually_provides": "Inspection after timeout",
            "revision_admits_because": "Allows unknown result",
            "support": [
                {"field": "trigger", "quote": "confirmed success"},
                {"field": "content", "quote": "unknown result"},
            ],
            "missing_facts": [],
        },
        "revision": {
            "action": "rewrite",
            "trigger": "gate suggestion is not builder input",
            "reason": "State mismatch",
            "support": [{"field": "content", "quote": "unknown result"}],
        },
    }
    builder: dict[str, Any] = {
        "revisions": [
            {
                "index": 0,
                "revision": {
                    "action": "rewrite",
                    "trigger": "After timeout with an unknown result",
                    "reason": "Consult the actual help",
                    "support": [{"field": "content", "quote": "unknown result"}],
                },
            }
        ]
    }

    def run(
        *, source: TriggerSourceFrame = frame, damage: str | None = None
    ) -> tuple[TriggerMaintenanceResult, list[dict[str, object]]]:
        calls: list[dict[str, object]] = []

        def invoke(
            _system: str,
            payload: dict[str, object],
            schema: type[StrictStructuredOutput],
        ) -> StrictStructuredOutput:
            import copy

            calls.append(payload)
            raw = copy.deepcopy(gate if schema is TriggerGate else builder)
            if schema is TriggerGate and damage == "quote":
                raw["contrast"]["support"][0]["quote"] = "invented original"
            if schema is TriggerGate and damage == "keep":
                raw["revision"]["action"] = "keep"
            if schema is not TriggerGate and damage == "index":
                raw["revisions"][0]["index"] = 1
            return schema.model_validate(raw)

        return run_trigger_maintenance(source, invoke), calls

    repaired, calls = run()
    null_result, null_calls = run(
        source=frame.model_copy(update={"original_trigger": None})
    )
    checks = {
        "source_only_builder": len(calls) == 2
        and calls[1]
        == {
            "sources": [
                {
                    "index": 0,
                    "original_trigger": frame.original_trigger,
                    "content": frame.final_content,
                    "gap": "opposed_consultation_state",
                }
            ]
        },
        "trigger_only_revision": repaired.revision.revised_content
        == frame.original_content
        and repaired.revision.revised_trigger != frame.original_trigger,
        "null_preserved_without_inference": not null_calls
        and null_result.revision.revised_trigger is None,
        "source_frame_binding": repaired.source_digest == frame.digest
        and frame.model_copy(update={"final_content_artifact_digest": "d" * 64}).digest
        != frame.digest,
    }
    for name, damage in (
        ("invalid_quote_refused", "quote"),
        ("invalid_keep_refused", "keep"),
        ("wrong_index_refused", "index"),
    ):
        try:
            run(damage=damage)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    if set(checks) != set(TRIGGER_QUALIFICATION_CHECK_NAMES) or not all(
        checks.values()
    ):
        raise ValueError("native Trigger structural qualification failed")
    suite = {
        "suite": "native-trigger-maintenance-contract-v1",
        "frame": frame.model_dump(mode="json"),
        "gate": gate,
        "builder": builder,
        "checks": list(TRIGGER_QUALIFICATION_CHECK_NAMES),
    }
    return {
        "suite_digest": sha256(canonical_json_bytes(suite)).hexdigest(),
        "checks": checks,
        "passed": True,
    }
