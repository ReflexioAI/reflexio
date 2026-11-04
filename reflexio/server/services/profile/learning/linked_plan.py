"""Source-bound linked profile changes with explicit retention lineage.

Uses the native gates, source extraction, retrieval, bounded model repairs,
embeddings and durable transaction. The planner may split/rewrite supported facts;
identity validation is structural, not a semantic supportedness guarantee.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections import Counter
from typing import Any, Literal

VERSION = "source-retention-lineage-v3"
EXTRACTION_GUIDANCE = """
Extract useful directly supported facts/preferences/constraints/plans from the original
conversation, preserving who, relationship, context, negation, uncertainty and time.
Assistant suggestions, drafting instructions, quotes and hypothetical examples are
not endorsed user biography. Do not require a special memory request for useful facts.
Write independently changing facts separately. A repetition is not a renewal. Include
applicable correction and retention instructions in the verbatim source_span, not
only the fact: e.g. an explicitly changed plan to remember for one day has one_day
TTL even if its subject concerns a longer event. Choose the stated retention period;
otherwise use appropriate native TTL. Preserve explicit forgetting as native
'Requested removal of ...' directives; do not turn them into biographical assertions.
""".strip()
RECONCILIATION_GUIDANCE = """
Candidate v3 replaces the preceding merge policy with a linked change plan. Use only
NEW assertions and their supplied source evidence plus actual EXISTING memory.
Every NEW ID occurs exactly once across groups, unique_ids or native deletions.
A group owns all its item_ids exclusively: one or more NEW plus zero or more EXISTING.
Use kept_existing_ids for old rows that remain completely unchanged. Every other
EXISTING in the group is retired ONLY together with all valid replacement/residual
writes. Every write has a unique write_id, content, support_ids, retention_origin,
retention_id and time_to_live. support_ids are nonempty IDs from that group whose
actual text/evidence supports this entire write. Preserve subject, entities,
relationships, conditions, scope and uncertainty in content itself.
- unchanged: unchanged/repeated existing fact or preserved residual clause; set
  retention_id to its supporting EXISTING ID. Code preserves its exact old modified
  time, expiry and TTL. Repetition alone never renews it. No unrelated new fact may
  renew an old component. For an unchanged complete row, keep it and emit no write.
- changed: genuine new or corrected assertion; retention_id is a supporting NEW ID.
  Choose TTL from actual NEW content/source evidence, correcting an upstream enum
  if necessary. Explicit 'one day' retention means one_day, not one_week.
- renewed: explicit new retention instruction for the SAME fact; retention_id is a
  supporting NEW ID and support_ids must also contain its EXISTING prior assertion.
  Choose the explicitly requested TTL. A mere repeated mention is NOT renewed.
Do not merge components with different clocks into one fresh write. A partial
correction to 'pickup Tuesday; hardcover only' becomes a changed pickup write plus
an unchanged hardcover residual retaining the old clock, and retires the compound
old row. Preserve all unrelated supported clauses. A correction in another time,
entity or context need not supersede the old fact. Keep genuinely distinct contexts.
A group with no writes can only keep all its EXISTING rows (e.g. exact repetition).
Use native deletions only for a canonical NEW Requested removal of marker from an
explicit whole-fact forget request. For partial forgetting,
consume the NEW forget directive in a group, retire the compound old row and write
only supported unaffected residuals with unchanged retention. Never save a forget
marker, forgotten clause or invented fallback fact. Invalid groups are inert as a
whole: old state remains, involved NEW proposals are withheld, unrelated valid work
continues. Code checks structure; you must judge complete semantic support and loss.
unique_ids are independently new, already correct NEW atoms requiring no rewrite or
TTL correction. Use a NEW-only group when atomic splitting or TTL correction is needed.
Optional write_search_terms is [{target_id: write_id OR unique NEW ID, search_terms:
[str,...]}]. Supply concise entity/context-preserving synonyms and natural retrieval
paraphrases. Cover both2–4synonyms/related phrasings per key and attribute/category
fact-key terms (including category terms for named products/entities), while preserving
entity/context/negation and avoiding unsupported factual additions. It is search
metadata, never fact evidence or retention authorization.
Malformed/absent metadata does not reject otherwise valid factual decisions.
""".strip()
TTL_VALUES = Literal[
    "one_day", "one_week", "one_month", "one_quarter", "one_year", "infinity"
]


def search_metadata(raw: Any, allowed: set[str]) -> dict[str, list[str]]:
    if not isinstance(raw, list):
        return {}
    counts = Counter(
        r.get("target_id")
        for r in raw
        if isinstance(r, dict) and isinstance(r.get("target_id"), str)
    )
    result = {}
    for r in raw:
        if not isinstance(r, dict) or set(r) != {"target_id", "search_terms"}:
            continue
        target, terms = r["target_id"], r["search_terms"]
        if not isinstance(target, str) or target not in allowed or counts[target] != 1:
            continue
        if (
            isinstance(terms, list)
            and terms
            and len(terms) <= 12
            and all(isinstance(t, str) and 0 < len(t.strip()) <= 240 for t in terms)
        ):
            result[target] = list(dict.fromkeys(t.strip() for t in terms))
    return result


# Keep the measured nested model namespace through faithful transfer.
def linked_classes(base: Any) -> Any:  # noqa: C901
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

    cm = importlib.import_module(
        "reflexio.server.services.profile.components.consolidator"
    )
    importlib.import_module("reflexio.server.services.profile.components.extractor")

    class PlannedWrite(BaseModel):
        write_id: str = Field(min_length=1, max_length=80)
        content: str = Field(min_length=1)
        support_ids: list[str] = Field(min_length=1)
        retention_origin: Literal["unchanged", "changed", "renewed"]
        retention_id: str
        time_to_live: TTL_VALUES
        model_config = ConfigDict(extra="forbid")

    class ChangeGroup(BaseModel):
        item_ids: list[str] = Field(min_length=1)
        kept_existing_ids: list[str]
        writes: list[PlannedWrite]
        model_config = ConfigDict(extra="forbid")

    class ChangeOutput(cm.ProfileDeduplicationOutput):
        duplicate_groups: list[ChangeGroup] = Field(default_factory=list)
        _search_terms: object = PrivateAttr(default=None)

        @model_validator(mode="wrap")
        @classmethod
        def separate_metadata(cls, value: Any, handler: Any) -> Any:
            if not isinstance(value, dict):
                return handler(value)
            core = dict(value)
            metadata = core.pop("write_search_terms", None)
            output = handler(core)
            output._search_terms = metadata
            return output

        @classmethod
        def model_json_schema(cls, *args, **kwargs) -> Any:
            schema = copy.deepcopy(super().model_json_schema(*args, **kwargs))
            schema["properties"]["write_search_terms"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_id": {"type": "string"},
                        "search_terms": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["target_id", "search_terms"],
                },
            }
            return schema

    class LinkedConsolidator(base):
        def _get_output_schema_class(self) -> Any:
            return ChangeOutput

        def _format_profiles_with_prefix(self, profiles: Any, prefix: Any) -> Any:
            # All fields originate in actual native retrieval/extraction objects.
            return json.dumps(
                [
                    {
                        "id": f"{prefix}-{i}",
                        "content": p.content,
                        "source_span": p.source_span,
                        "notes": p.notes,
                        "modified": p.last_modified_timestamp,
                        "expires": p.expiration_timestamp,
                        "ttl": p.profile_time_to_live.value,
                        "source": p.source,
                        "source_interaction_ids": p.source_interaction_ids,
                    }
                    for i, p in enumerate(profiles)
                ],
                ensure_ascii=False,
            )

        def _render_dedup_prompt(self, *args, **kwargs) -> Any:
            return (
                super()._render_dedup_prompt(*args, **kwargs)
                + "\n\n"
                + RECONCILIATION_GUIDANCE
            )

        def _retrieve_existing_profiles(self, new_profiles: Any, user_id: Any) -> Any:
            self._existing_atoms = super()._retrieve_existing_profiles(
                new_profiles, user_id
            )
            return self._existing_atoms

        def deduplicate(self, new_profiles: Any, user_id: Any, request_id: Any) -> Any:
            self._owner_id = user_id
            self._new_atoms = [p.model_copy(deep=True) for p in new_profiles]
            self._existing_atoms = []
            self._last_output = None
            self._builder_ran = False
            self.enrichment_audit = []
            self.enrichment_diagnostics = []
            result = super().deduplicate(self._new_atoms, user_id, request_id)
            # Native's no-group fast path returns every NEW atom even after an
            # exhausted invalid first-parsed response. Apply exact unique coverage
            # and metadata here too, without another model/retrieval request.
            if not self._builder_ran and self._last_output is not None:
                return self._build_deduplicated_results(
                    self._new_atoms,
                    self._existing_atoms,
                    self._last_output,
                    user_id,
                    request_id,
                )
            if self._new_atoms and not self._builder_ran and self._last_output is None:
                self.enrichment_diagnostics.append(
                    "planner_no_usable_output: NEW proposals withheld; prior rows unchanged"
                )
                return [], [], []
            return result

        def _call_native_dedup(
            self, messages: Any, output_schema: Any, validator: Any
        ) -> Any:
            first_parsed = None

            def validate(output: Any) -> Any:
                nonlocal first_parsed
                errors = validator(output)
                if isinstance(output, ChangeOutput):
                    if first_parsed is None:
                        first_parsed = output
                    self._last_output = output
                    errors += self._plan_errors(
                        output, self._new_atoms, self._existing_atoms
                    )
                return errors

            try:
                completion = super()._call_native_dedup(
                    messages, output_schema, validate
                )
            except Exception:
                self._last_output = first_parsed
                raise
            self._last_output = (
                completion.value if isinstance(completion.value, ChangeOutput) else None
            )
            return completion

        @staticmethod
        def _identities(output: Any) -> Any:
            refs = [
                cm.parse_item_id(i) for g in output.duplicate_groups for i in g.item_ids
            ]
            refs += [cm.parse_item_id(i) for i in output.unique_ids]
            for d in output.deletions:
                refs += [cm.parse_item_id(d.new_id)]
                refs += [cm.parse_item_id(i) for i in d.existing_ids]
            return Counter(i for i in refs if i is not None)

        def _group_errors(
            self,
            group: Any,
            new: Any,
            old: Any,
            counts: Any,
            write_counts: Any,
            user_id: Any,
        ) -> Any:
            errors = []
            members = [cm.parse_item_id(i) for i in group.item_ids]
            row_ids = Counter(p.profile_id for p in [*new, *old])

            def valid(identity: Any) -> Any:
                if identity is None:
                    return False
                kind, index = identity
                rows = new if kind == "NEW" else old
                return (
                    kind in {"NEW", "EXISTING"}
                    and 0 <= index < len(rows)
                    and (
                        rows[index].user_id == user_id and bool(rows[index].profile_id)
                    )
                    and row_ids[rows[index].profile_id] == 1
                )

            if not all(valid(i) and counts[i] == 1 for i in members):
                return ["Group IDs must be valid, same-owner and exclusively owned."]
            member_set = set(members)
            if not any(i[0] == "NEW" for i in members):
                errors.append("Each change group needs a NEW source/trigger.")
            kept = [cm.parse_item_id(i) for i in group.kept_existing_ids]
            if (
                len(set(kept)) != len(kept)
                or any(
                    i not in member_set or i[0] != "EXISTING"
                    for i in kept
                    if i is not None
                )
                or any(i is None for i in kept)
            ):
                errors.append("kept_existing_ids must be distinct EXISTING members.")
            retired = {i for i in members if i[0] == "EXISTING"} - set(kept)
            if retired and not group.writes:
                errors.append(
                    "Retirement requires linked supported writes; use deletions for forget."
                )
            for write in group.writes:
                support = [cm.parse_item_id(i) for i in write.support_ids]
                origin = cm.parse_item_id(write.retention_id)
                if len(set(support)) != len(support) or any(
                    i not in member_set for i in support
                ):
                    errors.append(
                        "Write support must be distinct IDs inside its group."
                    )
                if origin is None or origin not in support:
                    errors.append("Retention origin must be a supporting group member.")
                elif write.retention_origin == "unchanged" and origin[0] != "EXISTING":
                    errors.append("Unchanged retention must come from EXISTING memory.")
                elif write.retention_origin != "unchanged" and origin[0] != "NEW":
                    errors.append("Fresh retention must come from NEW evidence.")
                if write.retention_origin == "unchanged" and origin in kept:
                    errors.append(
                        "Do not duplicate a kept incumbent in an unchanged write."
                    )
                if write.retention_origin == "renewed" and not any(
                    i is not None and i[0] == "EXISTING" for i in support
                ):
                    errors.append(
                        "Renewal needs an EXISTING assertion and NEW instruction."
                    )
                if not write.content.strip() or write.content.strip().startswith(
                    cm._DELETION_MARKER_PREFIX
                ):
                    errors.append("A write must be a fact, not a forget marker.")
                if (
                    write_counts[write.write_id] != 1
                    or cm.parse_item_id(write.write_id) is not None
                ):
                    errors.append(
                        "write_id must be globally unique and not a native item ID."
                    )
            # Every retired row must remain linked to a replacement/residual;
            # links are structural, not proof that every old clause was preserved.
            supported = {
                cm.parse_item_id(i) for w in group.writes for i in w.support_ids
            }
            if not retired.issubset(supported):
                errors.append(
                    "Every retired EXISTING needs linked replacement/residual support."
                )
            return errors

        @staticmethod
        def _valid_identity(
            identity: Any, kind: Any, new: Any, old: Any, counts: Any, user_id: Any
        ) -> Any:
            rows = new if kind == "NEW" else old
            row_ids = Counter(p.profile_id for p in [*new, *old])
            return (
                identity is not None
                and identity[0] == kind
                and 0 <= identity[1] < len(rows)
                and rows[identity[1]].user_id == user_id
                and bool(rows[identity[1]].profile_id)
                and row_ids[rows[identity[1]].profile_id] == 1
                and counts[identity] == 1
            )

        def _deletion_errors(
            self, deletion: Any, new: Any, old: Any, counts: Any, user_id: Any
        ) -> Any:
            author = cm.parse_item_id(deletion.new_id)
            targets = [cm.parse_item_id(i) for i in deletion.existing_ids]
            if not self._valid_identity(author, "NEW", new, old, counts, user_id):
                return [
                    "Deletion NEW must be a valid, same-owner, exclusively owned identity."
                ]
            if not targets or any(
                not self._valid_identity(i, "EXISTING", new, old, counts, user_id)
                for i in targets
            ):
                return [
                    "Deletion targets must be nonempty valid, same-owner, "
                    "exclusively owned EXISTING IDs."
                ]
            if (
                not new[author[1]]
                .content.lstrip()
                .startswith(cm._DELETION_MARKER_PREFIX)
            ):
                return [
                    "Deletion NEW must be an actual Requested removal of directive marker."
                ]
            return []

        def _plan_errors(self, output: Any, new: Any, old: Any) -> Any:
            counts = self._identities(output)
            writes = Counter(
                w.write_id for g in output.duplicate_groups for w in g.writes
            )
            owner = getattr(self, "_owner_id", new[0].user_id if new else "")
            errors = []
            for group in output.duplicate_groups:
                errors.extend(
                    self._group_errors(group, new, old, counts, writes, owner)
                )
            for deletion in output.deletions:
                errors.extend(self._deletion_errors(deletion, new, old, counts, owner))
            errors.extend(
                "unique_ids must be valid same-owner exclusively owned NEW IDs."
                for raw in output.unique_ids
                if not self._valid_identity(
                    cm.parse_item_id(raw), "NEW", new, old, counts, owner
                )
            )
            return errors

        def _build_deduplicated_results(
            self,
            new_profiles: Any,
            existing_profiles: Any,
            dedup_output: Any,
            user_id: Any,
            request_id: Any,
        ) -> Any:
            self._builder_ran = True
            new, old = new_profiles, existing_profiles
            counts = self._identities(dedup_output)
            write_counts = Counter(
                w.write_id for g in dedup_output.duplicate_groups for w in g.writes
            )
            targets = {
                w.write_id for g in dedup_output.duplicate_groups for w in g.writes
            }
            targets |= {f"NEW-{i}" for i in range(len(new))}
            metadata = search_metadata(
                getattr(dedup_output, "_search_terms", None), targets
            )
            handled, retired, seen, superseded, result = set(), [], set(), [], []
            self.consolidated_output_indices = set()

            def resolve(identity: Any) -> Any:
                kind, index = identity
                return (new if kind == "NEW" else old)[index]

            def valid(identity: Any, kind: Any) -> Any:
                return self._valid_identity(identity, kind, new, old, counts, user_id)

            deletions = [
                d
                for d in dedup_output.deletions
                if not self._deletion_errors(d, new, old, counts, user_id)
            ]
            self._apply_deletion_directives(
                deletions,
                new_profiles=new,
                existing_profiles=old,
                handled_new_indices=handled,
                existing_ids_to_delete=retired,
                seen_delete_ids=seen,
                superseded_profiles=superseded,
            )
            for group in dedup_output.duplicate_groups:
                # Even exhausted invalid plans cannot fall through as unchecked
                # fresh proposals beside the unchanged old rows.
                for raw in group.item_ids:
                    item = cm.parse_item_id(raw)
                    if (
                        item is not None
                        and item[0] == "NEW"
                        and 0 <= item[1] < len(new)
                    ):
                        handled.add(item[1])
                if self._group_errors(group, new, old, counts, write_counts, user_id):
                    continue
                members = [cm.parse_item_id(i) for i in group.item_ids]
                kept = {cm.parse_item_id(i) for i in group.kept_existing_ids}
                pending = []
                for write in group.writes:
                    support = [resolve(cm.parse_item_id(i)) for i in write.support_ids]
                    origin = resolve(cm.parse_item_id(write.retention_id))
                    unchanged = write.retention_origin == "unchanged"
                    ttl = (
                        origin.profile_time_to_live
                        if unchanged
                        else cm.ProfileTimeToLive(write.time_to_live)
                    )
                    timestamp = origin.last_modified_timestamp
                    expiry = (
                        origin.expiration_timestamp
                        if unchanged
                        else cm.calculate_expiration_timestamp(timestamp, ttl)
                    )
                    span, notes, angle = cm._merged_provenance(support)
                    profile = cm.UserProfile(
                        profile_id=cm.new_profile_id(),
                        user_id=user_id,
                        content=write.content.strip(),
                        last_modified_timestamp=timestamp,
                        expiration_timestamp=expiry,
                        profile_time_to_live=ttl,
                        generated_from_request_id=request_id,
                        source=origin.source,
                        status=origin.status,
                        source_span=span,
                        notes=notes,
                        reader_angle=angle,
                        source_interaction_ids=sorted(
                            {i for p in support for i in p.source_interaction_ids}
                        ),
                        extractor_names=self._merge_extractor_names(support),
                        custom_features=self._merge_custom_features(support),
                        expanded_terms=", ".join(metadata[write.write_id])
                        if write.write_id in metadata
                        else None,
                    )
                    lineage = [str(p.profile_id) for p in support if p in old]
                    pending.append((profile, lineage, write.write_id))
                # Compilation of the entire linked group precedes any retirement.
                for profile, lineage, target in pending:
                    self.consolidated_output_indices.add(len(result))
                    result.append(profile)
                    self.lineage_sources_by_profile_id[profile.profile_id] = lineage
                    self.enrichment_audit.append(
                        {
                            "target": target,
                            "profile_id": profile.profile_id,
                            "status": "attached"
                            if target in metadata
                            else "missing_or_invalid",
                        }
                    )
                for identity in members:
                    if identity[0] == "EXISTING" and identity not in kept:
                        self._mark_existing_for_deletion(
                            f"EXISTING-{identity[1]}", old, retired, seen, superseded
                        )
            for raw in dedup_output.unique_ids:
                identity = cm.parse_item_id(raw)
                if not valid(identity, "NEW") or identity[1] in handled:
                    continue
                atom = resolve(identity).model_copy(deep=True)
                target = f"NEW-{identity[1]}"
                if target in metadata:
                    atom.expanded_terms = ", ".join(metadata[target])
                result.append(atom)
                handled.add(identity[1])
                self.enrichment_audit.append(
                    {
                        "target": target,
                        "profile_id": atom.profile_id,
                        "status": "attached"
                        if target in metadata
                        else "missing_or_invalid",
                    }
                )
            return cm._strip_deletion_markers(result), retired, superseded

    return ChangeOutput, LinkedConsolidator
