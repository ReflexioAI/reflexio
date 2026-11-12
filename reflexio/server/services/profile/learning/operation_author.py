"""Coherent author units with code-owned span and destination accounting.

The author chooses meaning and source authority. Code authenticates coordinates,
derives old-clause dispositions and passes the unchanged substantive compiler.
Unselected old text is never silently republished or dropped: full coverage remains
an existing compiler obligation for every retired row.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Annotated, Any, Literal, LiteralString, NoReturn

from pydantic import ConfigDict, Field, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from reflexio.models.structured_output import StrictStructuredOutput


class Strict(StrictStructuredOutput):
    model_config = ConfigDict(extra="forbid")


class Evidence(Strict):
    excerpt_id: str
    stance: Literal[
        "assertion",
        "endorsement",
        "current_reaffirmation",
        "retention_instruction",
        "forget_instruction",
        "historical_quote",
        "draft",
    ]


NewID = Annotated[str, Field(pattern=r"^NEW-(?:0|[1-9][0-9]*)$")]


class Target(Strict):
    row_id: str
    start_token: Annotated[
        int, Field(strict=True, ge=0, description="Shown token index, inclusive.")
    ]
    end_token: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            description="Shown token index, exclusive; may equal token_count.",
        ),
    ]


class Retention(Strict):
    source_new_id: NewID
    rule: str
    boundary: Literal["stop_at", "through"]
    evidence: list[Evidence] = Field(min_length=1)


class Carry(Strict):
    action: Literal["carry"]
    target: Target
    explicit_retention: Retention | None = Field(
        default=None,
        description=(
            "Null preserves old retention. Only a user-specified "
            "duration/deadline permits a change."
        ),
    )


class Assert(Strict):
    action: Literal["assert"]
    source_new_id: NewID
    content: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    replaces: list[Target] = Field(default_factory=list)
    explicit_retention: Retention | None = Field(
        default=None,
        description=(
            "Usually null: estimated NEW.ttl and event dates are not user storage instructions."
        ),
    )


class Unit(Strict):
    clauses: list[Annotated[Carry | Assert, Field(discriminator="action")]] = Field(
        min_length=1, max_length=12
    )
    search_terms: Any = None


class Forget(Strict):
    source_new_id: NewID
    targets: list[Target] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Group(Strict):
    new_ids: list[NewID] = Field(min_length=1)
    writes: list[Unit]
    forget: list[Forget] = Field(default_factory=list)


class OperationOutput(Strict):
    snapshot: str
    groups: list[Group]


_CURRENT: ContextVar[dict | None] = ContextVar("profile_operation_author", default=None)


@contextmanager
def operation_scope() -> Any:
    token = _CURRENT.set(context := {})
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def current_book() -> Any:
    context = _CURRENT.get()
    if context is None or "book" not in context:
        raise ValueError("No request-bound operation snapshot")
    return context["book"]


def fail(code: LiteralString, *locations: tuple[str | int, ...]) -> NoReturn:
    """Only static codes and numeric schema paths reach native safe repair."""
    errors: list[InitErrorDetails] = [
        {"type": PydanticCustomError(code, code), "loc": loc, "input": None}
        for loc in locations
    ]
    raise ValidationError.from_exception_data("OperationOutput", errors)


class OperationBook:
    def __init__(self, payload: Any):
        self.payload = deepcopy(payload)
        self.rows = {r["id"]: r for r in payload["EXISTING"]}
        self.new = {r["id"]: r for r in payload["NEW"]}
        if len(self.rows) != len(payload["EXISTING"]) or len(self.new) != len(
            payload["NEW"]
        ):
            raise ValueError("Duplicate operation row identity")
        self.snapshot = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.tokens = {
            key: list(re.finditer(r"\S+", row["content"]))
            for key, row in self.rows.items()
        }

    def rendered(self) -> Any:
        payload = deepcopy(self.payload)
        payload["snapshot"] = self.snapshot
        for row in payload["EXISTING"]:
            row["token_positions"] = {
                str(i): m.group() for i, m in enumerate(self.tokens[row["id"]])
            }
            row["token_count"] = len(self.tokens[row["id"]])
        return payload

    def resolve(self, target: Any) -> Any:
        if target.row_id not in self.rows:
            raise ValueError("Foreign or unknown operation row")
        tokens = self.tokens[target.row_id]
        if not 0 <= target.start_token < target.end_token <= len(tokens):
            raise ValueError("Operation token range outside current row")
        return tokens[target.start_token].start(), tokens[target.end_token - 1].end()

    def parts(self, row_id: Any, begin: Any, end: Any) -> Any:
        """Preserve authenticated stored clause geometry and its independent origins."""
        text = self.rows[row_id]["content"]
        clauses = self.rows[row_id].get("clauses")
        if clauses is None:
            return [(begin, end)]
        if not isinstance(clauses, list) or not clauses:
            raise ValueError("Invalid incumbent clause metadata")
        ranges, cursor = [], 0
        for clause in clauses:
            literal = clause.get("content") if isinstance(clause, dict) else None
            if not isinstance(literal, str) or not literal.strip():
                raise ValueError("Invalid incumbent clause content")
            start = text.find(literal, cursor)
            if start < 0 or text[cursor:start].strip():
                raise ValueError("Incumbent clause metadata differs from current text")
            stop = start + len(literal)
            if max(begin, start) < min(end, stop):
                ranges.append((max(begin, start), min(end, stop)))
            cursor = stop
        if text[cursor:].strip():
            raise ValueError("Incomplete incumbent clause metadata")
        return ranges

    def compile(self, value: Any) -> Any:  # noqa: C901 -- one atomic operation-to-plan translation
        output = OperationOutput.model_validate(value)
        if output.snapshot != self.snapshot:
            fail("operation_snapshot_mismatch", ("snapshot",))
        owners, row_owners, result = {}, {}, []
        for gi, group in enumerate(output.groups):
            for ni, identity in enumerate(group.new_ids):
                loc = ("groups", gi, "new_ids", ni)
                if identity not in self.new or identity in owners:
                    fail("operation_new_identity_not_exclusive", loc)
                owners[identity] = gi
            writes, dispositions, edits = [], [], {}
            flat_index = 0

            def reference(identity: Any, loc: Any, *, group: Any = group) -> Any:
                if identity not in group.new_ids:
                    fail("operation_new_reference_outside_group", loc)

            def retention(value: Any, loc: Any, *, reference: Any = reference) -> Any:
                if value is None:
                    return None
                reference(value.source_new_id, loc)
                return {
                    "new_id": value.source_new_id,
                    "rule": value.rule,
                    "boundary": value.boundary,
                    "evidence": [e.model_dump() for e in value.evidence],
                }

            def target(
                value: Any, loc: Any, *, gi: Any = gi, edits: Any = edits
            ) -> Any:
                try:
                    start, end = self.resolve(value)
                except ValueError:
                    fail("operation_target_outside_snapshot", loc)
                if value.row_id in row_owners and row_owners[value.row_id][0] != gi:
                    fail(
                        "operation_incumbent_multiple_groups",
                        row_owners[value.row_id][1],
                        loc,
                    )
                row_owners.setdefault(value.row_id, (gi, loc))
                for a, z, previous in edits.setdefault(value.row_id, []):
                    if max(a, start) < min(z, end):
                        fail("operation_overlapping_targets", previous, loc)
                edits[value.row_id].append((start, end, loc))
                return start, end

            for ui, unit in enumerate(group.writes):
                clauses = []
                for ci, clause in enumerate(unit.clauses):
                    loc = ("groups", gi, "writes", ui, "clauses", ci)
                    retained = retention(
                        clause.explicit_retention, (*loc, "explicit_retention")
                    )
                    if isinstance(clause, Carry):
                        begin, end = target(clause.target, (*loc, "target"))
                        for a, z in self.parts(clause.target.row_id, begin, end):
                            literal = self.rows[clause.target.row_id]["content"][a:z]
                            clauses.append(
                                {
                                    "content": literal,
                                    "origin": {
                                        "kind": "carry",
                                        "existing_id": clause.target.row_id,
                                    },
                                    "retention": retained,
                                }
                            )
                            dispositions.append(
                                {
                                    "existing_id": clause.target.row_id,
                                    "literal": literal,
                                    "disposition": "retained",
                                    "write_indexes": [flat_index],
                                    "evidence": [],
                                    "_position": a,
                                }
                            )
                            flat_index += 1
                    else:
                        reference(clause.source_new_id, (*loc, "source_new_id"))
                        evidence = [e.model_dump() for e in clause.evidence]
                        clauses.append(
                            {
                                "content": clause.content,
                                "origin": {
                                    "kind": "assert",
                                    "new_id": clause.source_new_id,
                                    "evidence": evidence,
                                },
                                "retention": retained,
                            }
                        )
                        for ti, replacement in enumerate(clause.replaces):
                            begin, end = target(replacement, (*loc, "replaces", ti))
                            dispositions.append(
                                {
                                    "existing_id": replacement.row_id,
                                    "literal": self.rows[replacement.row_id]["content"][
                                        begin:end
                                    ],
                                    "disposition": "replaced",
                                    "write_indexes": [flat_index],
                                    "evidence": evidence,
                                    "_position": begin,
                                }
                            )
                        flat_index += 1
                if len(clauses) > 12:
                    fail("operation_unit_clause_limit", ("groups", gi, "writes", ui))
                writes.append({"clauses": clauses, "search_terms": unit.search_terms})
            for fi, forgotten in enumerate(group.forget):
                loc = ("groups", gi, "forget", fi)
                reference(forgotten.source_new_id, (*loc, "source_new_id"))
                for ti, selection in enumerate(forgotten.targets):
                    begin, end = target(selection, (*loc, "targets", ti))
                    dispositions.append(
                        {
                            "existing_id": selection.row_id,
                            "literal": self.rows[selection.row_id]["content"][
                                begin:end
                            ],
                            "disposition": "forgotten",
                            "write_indexes": [],
                            "evidence": [e.model_dump() for e in forgotten.evidence],
                            "_position": begin,
                        }
                    )
            for row_id, ranges in edits.items():
                cursor = 0
                for begin, end, location in sorted(ranges):
                    if self.rows[row_id]["content"][cursor:begin].strip():
                        fail("operation_incomplete_row_coverage", location)
                    cursor = end
                if self.rows[row_id]["content"][cursor:].strip():
                    fail("operation_incomplete_row_coverage", ranges[-1][2])
            # The semantic compiler expects source occurrence order, independent
            # of the author's chosen order of coherent output units.
            dispositions.sort(
                key=lambda d: (list(self.rows).index(d["existing_id"]), d["_position"])
            )
            for disposition in dispositions:
                disposition.pop("_position")
            result.append(
                {
                    "new_ids": group.new_ids,
                    "retire_ids": list(edits),
                    "writes": writes,
                    "clause_dispositions": dispositions,
                }
            )
        if set(owners) != set(self.new):
            fail("operation_new_coverage", ("groups",))
        return {"groups": result}


GUIDANCE = """Author useful, self-contained memory UNITS from original_window and fallible NEW
proposals.
Return the declared snapshot/groups schema. Existing memory is prior context, not fresh fact
evidence.
Each NEW belongs exactly once in groups.new_ids, including redundant proposals. A group
may address multiple overlapping proposals with ONE supported change; do not repeat the edit.
Use an empty writes/forget group to reject unsupported, redundant or already represented NEWs.

Each unit's clauses are in final reading order; code joins them with spaces. Combine only
related owner/entity/context details with compatible retention. A singleton is lawful.
CARRY selects exact unchanged old text by target. ASSERT writes a supported new/corrected
claim once, with source_new_id and original excerpt evidence; replaces selects only the old
text it actually corrects. Code derives replacement evidence, old literals and destination
indexes; do NOT supply those separately. FORGET selects old text only for actual explicit
forget authority, with source evidence. Updating a draft/task field is a scoped correction,
not automatically a request to erase unrelated real-world knowledge. Conditions, negation,
relationships, uncertainty and the distinction between user facts and fictional/draft content
matter.

Targets use shown token_positions keys: start inclusive, end exclusive; end may equal token_count.
Never use character offsets or invent row/NEW/excerpt IDs. Omit unchanged rows entirely.
For EACH touched old row, explicitly cover ALL non-whitespace old text with carry,
assert.replaces or forget targets exactly once, in any authored order. No automatic residual
is published. Preserve unaffected context in coherent carry clauses. Never target a partial
phrase for replacement with an entire restated fact unless all its old meaning is accounted for.
Code proves exact geometry, never semantic support; the unchanged substantive compiler still runs.

Evidence uses actual excerpt_id and stance: assertion, endorsement, current_reaffirmation,
retention_instruction, forget_instruction, historical_quote or draft. ASSERT/replacement needs
user assertion/endorsement/current_reaffirmation supporting the entire claim. FORGET needs
user forget_instruction. Assistant drafts need actual claim-specific user endorsement;
endorsing fictional tone does not endorse fictional biography. Retention evidence alone is
not factual evidence; cite both stances when a source supplies both kinds of authority.

explicit_retention is usually null: preserve existing old constraints and use existing estimated
lifetime for new facts. NEW.ttl, event dates and generic remember are NOT user storage instructions.
Only actual specified duration/deadline permits {source_new_id,rule,boundary,evidence}.
rule uses one_day/one_week/one_month/one_quarter/one_year/infinity or whole-second timezone-bearing
RFC3339 deadline; boundary stop_at means absent AT deadline, through means visible AT it.
No silent ingestion-time anchoring, extended unrelated expiry or unbounded duplicate of a
renewed fact.
A factual confirmation with a new supported as-of date can require ASSERT/replaces;
CARRY plus explicit_retention changes lifetime only and preserves the old factual confirmation.
An event date passing does not establish attendance/cancellation. Preserve relevant historical
context.
Optional unit search_terms never supplies evidence; invalid metadata cannot invalidate a lawful
core write.

Choose the semantic operation before translating it into fields. A NEW describing a
user command is a proposal for an ACTION, not a factual assertion to copy into storage.
For a field-removal command, writes contain only surviving supported memory. Put removal
in the GROUP'S forget list, never in writes[].clauses: that union permits carry/assert only.
A carry-only unit is lawful; it needs no new assertion or renewed factual support.
Removing a trailing field may also remove its connector (such as "along") so the carried
residual is meaningful. Select the complete surviving phrase, not a dangling prefix.

Generic partial-forget illustration, not an instruction about any actual input:
old EXISTING-0 is exactly "Cycles on Sundays along River Road." with token positions
0:Cycles 1:on 2:Sundays 3:along 4:River 5:Road.; actual user source asks to forget only
the route. NEW-0 describes that request. Using the ACTUAL snapshot and ACTUAL source ID,
the group shape is:
{"new_ids":["NEW-0"],"writes":[{"clauses":[{"action":"carry","target":
{"row_id":"EXISTING-0","start_token":0,"end_token":3}}]}],"forget":[{
"source_new_id":"NEW-0","targets":[{"row_id":"EXISTING-0","start_token":3,
"end_token":6}],"evidence":[{"excerpt_id":"ACTUAL_SOURCE_ID",
"stance":"forget_instruction"}]}]}
Code carries "Cycles on Sundays" with its old factual origin/retention, retires the old
row, and forgets only "along River Road." There is NO assert clause saying that a route
was removed. Do not copy illustration IDs or positions onto different actual text.
If a source instead corrects ONE preference in a multi-preference row, preserve the other
preference explicitly; a whole-row replaces target is not authority to lose its unrelated
meaning. Choose self-contained carry plus corrected assertion, or a fully supported
restatement that preserves both meanings. These are model semantic duties, not facts
proved by geometry or a valid stance label.

Repair codes: operation_incumbent_multiple_groups means put all targets for that row in one
group; operation_overlapping_targets means remove duplicate edits/redundant proposed changes;
operation_new_coverage means account for every NEW exactly once;
operation_new_reference_outside_group
means include that actual NEW in the group; operation_target_outside_snapshot means use valid shown
row/token keys; operation_snapshot_mismatch means copy the exact current snapshot.
operation_incomplete_row_coverage means cover every unaffected old span explicitly with CARRY;
operation_unit_clause_limit means split the unit: stored clause boundaries can expand a carry
selection into several independently attributed clauses (maximum twelve per unit).
"""
