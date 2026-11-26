"""Bind flat declarations to complete source-backed, related-change groups."""

from dataclasses import dataclass
from typing import Any

from . import codec
from .models import ContractError, Maintenance, Quotation


@dataclass(frozen=True)
class Compilation:
    records: dict[str, dict[str, Any]]
    keep_existing: list[int]
    retired_ids: list[int]
    changes: list[dict[str, Any]]

    @property
    def selected(self) -> list[str]:
        return list(self.records)


def witnesses(text: str, quotes: dict[str, Quotation]) -> list[str]:
    keys = text.split()
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= quotes.keys():
        raise ContractError(
            "Evidence must be unique exact available witness IDs separated by spaces"
        )
    return keys


def proposals(
    batch: list[tuple[str, Any]],
    incumbents: dict[str, dict[str, Any]],
    quotes: dict[str, Quotation],
) -> Maintenance:
    """Translate flat declarations to the existing complete transaction contract."""
    changes, documents, resources, citations = [], {}, [], []
    for name, item in batch:
        if name == "cite_evidence":
            citations.append(item)
            continue
        if name in {"attach_resource", "remove_resource"}:
            resources.append((name, item))
            continue
        support = witnesses(item.evidence, quotes)
        consume = [] if name == "create_procedure" else [item.target]
        if consume and consume[0] not in incumbents:
            raise ContractError("Existing target must be an exact available P handle")
        if name == "retire_procedure":
            changes.append(
                {
                    "group": item.group,
                    "operation": "retire",
                    "consume": consume,
                    "document": None,
                    "trigger": None,
                    "support": support,
                }
            )
            continue
        pages = []
        if consume:
            old = codec.read_existing(incumbents[consume[0]]["content"])
            pages = [p.model_copy(deep=True) for p in old.pages[1:]] if old else []
        document = codec.Document(
            name=item.name,
            pages=[codec.Page(path="SKILL.md", content=item.entry), *pages],
        )
        if item.name in documents:
            raise ContractError(
                "Each proposed document needs one complete entry declaration"
            )
        change = {
            "group": item.group,
            "operation": "upsert",
            "consume": consume,
            "document": document.model_dump(mode="json"),
            "trigger": item.trigger,
            "support": support,
        }
        documents[item.name] = change
        changes.append(change)
    touched = set()
    for operation, item in resources:
        change = documents.get(item.document_name)
        if item.path == "SKILL.md":
            raise ContractError(
                "SKILL.md is supplied by entry; resource edits are optional files only"
            )
        if change is None or change["group"] != item.group:
            raise ContractError(
                "Resource needs a complete entry in the same document/group"
            )
        identity = (item.document_name, item.path)
        if identity in touched:
            raise ContractError(
                "A resource can be changed only once per complete batch"
            )
        touched.add(identity)
        support = witnesses(item.evidence, quotes)
        change["support"] = list(dict.fromkeys([*change["support"], *support]))
        pages = change["document"]["pages"]
        found = next((i for i, p in enumerate(pages) if p["path"] == item.path), None)
        if operation == "remove_resource":
            if found is None:
                raise ContractError("Cannot remove an unavailable incumbent resource")
            pages.pop(found)
        else:
            page = codec.Page(path=item.path, content=item.content).model_dump(
                mode="json"
            )
            if found is None:
                pages.append(page)
            else:
                pages[found] = page
    for citation in citations:
        related = [change for change in changes if change["group"] == citation.group]
        if not related:
            raise ContractError(
                "Additional evidence needs an existing proposed change group"
            )
        support = witnesses(citation.evidence, quotes)
        for change in related:
            change["support"] = list(dict.fromkeys([*change["support"], *support]))
    return Maintenance.model_validate({"changes": changes})


def compile_changes(
    produced: Maintenance,
    incumbents: dict[str, dict[str, Any]],
    quotes: dict[str, Quotation],
    visible: dict[int, list[str]],
) -> Compilation:
    pack = codec.encode
    unpack = codec.read_existing
    document_identity = codec.identity
    consumed, rendered, retired, changes = set(), {}, [], []
    for number, change in enumerate(produced.changes):
        if (
            len(change.consume) != len(set(change.consume))
            or not set(change.consume) <= incumbents.keys()
        ):
            raise ContractError("Consume unique existing procedure handles")
        if consumed.intersection(change.consume):
            raise ContractError(
                "A procedure can be consumed only once in a transaction"
            )
        if (
            len(change.support) != len(set(change.support))
            or not set(change.support) <= quotes.keys()
        ):
            raise ContractError("Changes require unique available source witnesses")
        consumed.update(change.consume)
        old = [incumbents[key] for key in change.consume]
        old_ids = [row["user_playbook_id"] for row in old]
        if change.operation == "retire":
            if not old or change.document is not None or change.trigger is not None:
                raise ContractError(
                    "Retirement names existing procedures and has no new document"
                )
            retired.extend(old_ids)
            changes.append(change.model_dump(mode="json"))
            continue
        if change.document is None or not change.trigger or not change.trigger.strip():
            raise ContractError(
                "An upsert requires a complete procedure document and trigger"
            )
        if codec.transient_handle(change.trigger):
            raise ContractError("Trigger cannot expose ephemeral source handles")
        content = pack(change.document)
        existing_documents = [unpack(row["content"]) for row in old]
        maintained_paths = {
            document_identity(d) for d in existing_documents if d is not None
        }
        if (
            maintained_paths
            and document_identity(change.document) not in maintained_paths
        ):
            raise ContractError("Revision must retain an existing procedure identity")
        if len(old) == 1 and (content, change.trigger) == (
            old[0]["content"],
            old[0].get("trigger"),
        ):
            raise ContractError(
                "Unchanged guidance must use NOOP, not replace its identity"
            )
        pieces = list(
            dict.fromkeys(
                (quotes[s].interaction_id, quotes[s].text) for s in change.support
            )
        )
        old_spans = list(
            dict.fromkeys(row["source_span"] for row in old if row.get("source_span"))
        )
        spans = list(dict.fromkeys([*old_spans, *(text for _, text in pieces)]))
        identifiers = list(
            dict.fromkeys(
                [
                    *(i for row in old for i in row.get("source_interaction_ids", [])),
                    *(i for i, _ in pieces),
                ]
            )
        )
        identity = f"W{number}"
        rendered[identity] = {
            "content": content,
            "trigger": change.trigger,
            "expanded_terms": None,
            "source_interaction_ids": identifiers,
            "source_span": "\n\n".join(spans),
            "pieces": [{"interaction_id": i, "text": text} for i, text in pieces]
            if not old
            else None,
            "replaces": old_ids,
        }
        changes.append(change.model_dump(mode="json"))
    current = [row for key, row in incumbents.items() if key not in consumed]
    paths = [
        document_identity(d)
        for row in current
        if (d := unpack(row["content"])) is not None
    ]
    paths += [
        document_identity(c.document)
        for c in produced.changes
        if c.operation == "upsert" and c.document is not None
    ]
    if len(paths) != len(set(paths)):
        raise ContractError(
            "Final library cannot contain duplicate procedure identities"
        )
    # New scripts must be copied from cited native-visible source. Whether the
    # observed execution succeeded remains the independent semantic review's job.
    for change in produced.changes:
        if change.document is None:
            continue
        old_scripts = {
            page.content
            for key in change.consume
            if (document := codec.read_existing(incumbents[key]["content"])) is not None
            for page in document.pages
            if page.path.startswith("scripts/")
        }
        cited_sources = {quotes[ref].interaction_id for ref in change.support}
        for page in change.document.pages:
            if (
                page.path.startswith("scripts/")
                and page.content not in old_scripts
                and not any(
                    page.content in piece
                    for source_id in cited_sources
                    for piece in visible.get(source_id, [])
                )
            ):
                raise ContractError(
                    "New script bytes must occur in the visible original source"
                )
    return Compilation(
        rendered, [row["user_playbook_id"] for row in current], retired, changes
    )
