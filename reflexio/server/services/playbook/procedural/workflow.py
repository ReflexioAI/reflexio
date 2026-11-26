"""The selected source-reader/compiler/full-result-review procedure.

No database, provider, lease or publication authority lives here. The caller
supplies a stage client, which may restore source-bound saved stage results.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from importlib.resources import files
from typing import Any, Protocol

from pydantic import BaseModel, Field, JsonValue, create_model

from . import codec, review
from .catalog import catalog
from .compilation import Compilation, compile_changes, proposals, witnesses
from .models import (
    Citation,
    ContractError,
    Create,
    Input,
    Maintenance,
    Observation,
    RemoveResource,
    Resource,
    ResultVerdict,
    Retire,
    Review,
    Revise,
    Text,
)

VERSION = "evidence_first_skill_compiler_v8_result_review"
INCUMBENT_FIELDS = (
    "user_playbook_id",
    "content",
    "trigger",
    "source_interaction_ids",
    "source_span",
    "rationale",
    "expanded_terms",
    "status",
    "playbook_status",
)


class StageClientProtocol(Protocol):
    def ask[T](
        self,
        stage: str,
        prompt: str,
        body: dict[str, Any],
        models: dict[str, type[BaseModel]],
        assemble: Callable[[list[tuple[str, Any]]], T],
        validate: Callable[[T], Any],
        *,
        evidence_choices: list[str] | None = None,
    ) -> T: ...


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def incumbent_state(row: dict[str, Any]) -> dict[str, Any]:
    return deepcopy({key: row.get(key) for key in INCUMBENT_FIELDS})


def _prompt(name: str) -> str:
    return files(__package__).joinpath("prompts", name + ".md").read_text()


def run_pipeline(
    data: Input, visible_segments: dict[int, list[str]], client: StageClientProtocol
) -> dict[str, Any]:
    """Compile only unanimously accepted related changes; keep all others intact."""
    body, quotes, _ = catalog(data, visible_segments)
    incumbents = {f"P{i}": deepcopy(row) for i, row in enumerate(data.existing, 1)}
    identities = [row.get("user_playbook_id") for row in incumbents.values()]
    if any(
        type(identity) is not int or identity <= 0 for identity in identities
    ) or len(identities) != len(set(identities)):
        raise ContractError(
            "Procedural maintenance needs unique native incumbent identities"
        )
    if any(row.get("user_id") != data.owner for row in incumbents.values()):
        raise ContractError("Procedural incumbents must belong to the source owner")
    body["existing"] = {
        key: {
            **incumbent_state(row),
            "document": document.model_dump(mode="json")
            if (document := codec.read_existing(row["content"])) is not None
            else None,
        }
        for key, row in incumbents.items()
    }
    body["representation"] = (
        "Code assembles the stored Document. A create_procedure/revise_procedure name "
        "is the stable slug and its entry argument supplies the single SKILL.md. "
        "Do not return a nested Document/pages object or duplicate the entry through "
        "attach_resource. Only optional references/, scripts/ and assets/ files use "
        "resource declarations in the same document/group. Keep essential supported "
        "conditions in entry. Scripts still require exact visible source bytes and "
        "matched successful execution evidence. Files remain inert stored text."
    )

    def observations_from(batch: list[tuple[str, Any]]) -> list[Observation]:
        observations = [Observation.model_validate(item) for _, item in batch]
        for observation in observations:
            witnesses(observation.evidence, quotes)
        return observations

    observations = (
        client.ask(
            "source",
            _prompt("source"),
            body,
            {"record_observation": Observation},
            observations_from,
            lambda _: None,
            evidence_choices=list(quotes),
        )
        if data.sources
        else []
    )
    raw_observations = [item.model_dump(mode="json") for item in observations]
    # The reader can discover useful locations without pre-authoring the answer.
    body["source_observations"] = [
        {
            "topic": "Source-linked learning",
            "evidence": item.evidence,
            "observation": "Read the original cited source before authoring a rule.",
        }
        for item in observations
    ]

    def compile_(value: Maintenance) -> Compilation:
        return compile_changes(value, incumbents, quotes, visible_segments)

    declarations: dict[str, type[BaseModel]] = {
        "create_procedure": Create,
        "revise_procedure": Revise,
        "retire_procedure": Retire,
        "attach_resource": Resource,
        "remove_resource": RemoveResource,
        "cite_evidence": Citation,
    }
    produced = (
        client.ask(
            "maintain",
            _prompt("compile") + "\n\n" + _prompt("source_conditions"),
            body,
            declarations,
            lambda batch: proposals(batch, incumbents, quotes),
            compile_,
            evidence_choices=list(quotes),
        )
        if observations
        else Maintenance(changes=[])
    )
    compiled = compile_(produced)
    groups = list(dict.fromkeys(change.group for change in produced.changes))
    proposal_digest = digest(
        {
            "records": compiled.records,
            "keep_existing": compiled.keep_existing,
            "retired_ids": compiled.retired_ids,
        }
    )
    verdict = None
    if groups:
        payload = {
            "source": body,
            "changes": compiled.changes,
            "proposed_records": compiled.records,
            "keep_existing": compiled.keep_existing,
            "retired_ids": compiled.retired_ids,
        }
        choices: list[JsonValue] = [f"U{i}" for i in range(len(produced.changes))]
        unit_type = create_model(
            "BoundResultVerdict",
            __base__=ResultVerdict,
            unit=(Text, Field(json_schema_extra={"enum": choices})),
        )

        def validate_verdict(value: Review) -> None:
            if len(value.groups) != len(groups) or {
                v.group for v in value.groups
            } != set(groups):
                raise ContractError("Review every proposed group exactly once")

        verdict = client.ask(
            "review",
            _prompt("review"),
            review.prepare(payload),
            {"review_result": unit_type},
            lambda batch: review.reduce(payload, [item for _, item in batch]),
            validate_verdict,
        )
    accepted = review.accepted_groups(verdict) if verdict else set()
    selected = Maintenance(
        changes=[change for change in produced.changes if change.group in accepted]
    )
    result = compile_(selected)
    original_indices = [
        i for i, change in enumerate(produced.changes) if change.group in accepted
    ]
    aliases = {f"W{i}": f"W{original}" for i, original in enumerate(original_indices)}
    rendered = {aliases[key]: value for key, value in result.records.items()}
    plan = {
        "selected": [aliases[key] for key in result.selected],
        "keep_existing": result.keep_existing,
    }
    return {
        "method_version": VERSION,
        **(
            {"source_projection": "no_visible_native_evidence"}
            if not data.sources
            else {}
        ),
        "source_observations": raw_observations,
        "downstream_source_observations": body["source_observations"],
        "produced": produced.model_dump(mode="json"),
        "accepted_changes": selected.model_dump(mode="json"),
        "proposed_sha256": proposal_digest,
        "group_review": verdict.model_dump(mode="json") if verdict else None,
        "accepted_groups": sorted(accepted),
        "rendered": rendered,
        "plan": plan,
        "retired_ids": result.retired_ids,
        "reviewed_sha256": digest(
            {
                "records": rendered,
                "keep_existing": result.keep_existing,
                "retired_ids": result.retired_ids,
            }
        ),
        "old_state": {
            str(row["user_playbook_id"]): digest(incumbent_state(row))
            for row in incumbents.values()
        },
        "accepted": plan["selected"],
        "outcome": "accept" if accepted else "empty" if not groups else "reject",
    }
