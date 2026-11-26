"""Canonical inert skill files retained inside one ordinary playbook content field."""

import json
import posixpath
import re

from reflexio.server.services.playbook.playbook_evidence import (
    contains_call_local_turn_ref,
)

from .models import NAME_PATTERN, PAGE_PATTERN, ContractError, Document, Page

PREFIX = "<!-- reflexio-skill-bundle-v1 "


def transient_handle(text: str) -> bool:
    return contains_call_local_turn_ref(text) or bool(
        re.search(r"\b(?:EXCERPT|SEGMENT)_[A-Za-z0-9_]+", text)
    )


def link_targets(text: str) -> list[str]:
    definitions = {}
    for label, target in re.findall(r"(?m)^ {0,3}\[([^\]]+)\]:\s*<?([^\s>]+)>?", text):
        key = label.strip().casefold()
        if key in definitions:
            raise ContractError("Wiki reference labels must be unique")
        definitions[key] = target
    for label, ref in re.findall(r"\[([^\]]+)\]\[([^\]]*)\]", text):
        if (ref or label).strip().casefold() not in definitions:
            raise ContractError("Wiki reference-style link has no definition")
    inline = re.compile(
        r"\[[^\]]*\]\(\s*<?([^\s)>]+)>?"
        r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
    )
    if re.search(r"\]\s*\(", inline.sub("", text)):
        raise ContractError("Unsupported wiki link syntax must not bypass validation")
    return [*inline.findall(text), *definitions.values()]


def identity(document: Document) -> str:
    return document.name


def validate(document: Document) -> None:
    if len(document.name) > 64 or re.fullmatch(NAME_PATTERN, document.name) is None:
        raise ContractError("Bundle name must be a safe lowercase slug")
    if any(re.fullmatch(PAGE_PATTERN, page.path) is None for page in document.pages):
        raise ContractError(
            "Bundle page paths must remain in the permitted resource namespace"
        )
    paths = [p.path for p in document.pages]
    if paths[0] != "SKILL.md" or len(paths) != len(set(paths)):
        raise ContractError("Bundle needs one SKILL.md entry and unique resource paths")
    for page in document.pages:
        if "\x00" in page.content or transient_handle(page.content):
            raise ContractError(
                "Bundle content cannot contain NUL or transient source handles"
            )
        if page.path.startswith("scripts/"):
            continue  # Code syntax is not Markdown; script support is checked separately.
        for target in link_targets(page.content):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                if not target.startswith(("https://", "http://")):
                    raise ContractError("Bundle external references must use HTTP")
                continue
            path, _, anchor = target.partition("#")
            resolved = (
                posixpath.normpath(posixpath.join(posixpath.dirname(page.path), path))
                if path
                else page.path
            )
            if resolved not in paths:
                raise ContractError(
                    "Bundle link must resolve inside its complete document"
                )
            if anchor:
                text = next(p.content for p in document.pages if p.path == resolved)
                headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", text)
                slugs = {
                    re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-")
                    for h in headings
                }
                if anchor not in slugs:
                    raise ContractError("Bundle reference names an unavailable section")


def encode(document: Document) -> str:
    validate(document)
    header = {
        "name": document.name,
        "pages": [{"path": p.path, "chars": len(p.content)} for p in document.pages],
    }
    return (
        PREFIX
        + json.dumps(header, separators=(",", ":"))
        + " -->\n"
        + "".join(f"\n## {p.path}\n\n{p.content}" for p in document.pages)
    )


def decode(content: str) -> Document | None:
    if not content.startswith(PREFIX):
        return None
    try:
        header, body = content.split(" -->\n", 1)
        metadata = json.loads(header[len(PREFIX) :])
        pages = []
        for row in metadata["pages"]:
            marker = f"\n## {row['path']}\n\n"
            count = row["chars"]
            if type(count) is not int or count < 1 or not body.startswith(marker):
                raise ValueError("Invalid resource framing")
            body = body[len(marker) :]
            pages.append(Page(path=row["path"], content=body[:count]))
            body = body[count:]
        document = Document(name=metadata["name"], pages=pages)
        if body or encode(document) != content:
            raise ValueError("Noncanonical resource carrier")
        return document
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("Invalid persisted skill bundle") from exc


def read_existing(content: str) -> Document | None:
    """Read a native incumbent after a length-changing ordinary content edit.

    Canonical model-authored outputs still use strict ``decode``. Existing public
    editors and tuners edit the complete stored string, so its original framing
    counts may be stale. A read view is safe only when its original manifest and
    uniquely located complete section markers partition every body byte. This
    never changes persisted bytes, hashes, indexes, or resources.
    """
    try:
        return decode(content)
    except ContractError:
        try:
            header, body = content.split(" -->\n", 1)
            metadata = json.loads(header[len(PREFIX) :])
            if not isinstance(metadata, dict) or set(metadata) != {"name", "pages"}:
                raise ValueError("Invalid incumbent resource manifest")
            rows = metadata["pages"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("Missing incumbent resource manifest")
            markers: list[str] = []
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or set(row) != {"path", "chars"}
                    or not isinstance(row["path"], str)
                    or re.fullmatch(PAGE_PATTERN, row["path"]) is None
                    or type(row["chars"]) is not int
                    or row["chars"] < 1
                ):
                    raise ValueError("Invalid incumbent resource framing")
                marker = f"\n## {row['path']}\n\n"
                if body.count(marker) != 1:
                    raise ValueError("Ambiguous or missing incumbent resource marker")
                markers.append(marker)
            positions = [body.index(marker) for marker in markers]
            if positions[0] != 0 or positions != sorted(positions):
                raise ValueError("Incumbent resource order or framing changed")
            pages = [
                Page(
                    path=row["path"],
                    content=body[
                        position + len(marker) : positions[index + 1]
                        if index + 1 < len(positions)
                        else len(body)
                    ],
                )
                for index, (row, marker, position) in enumerate(
                    zip(rows, markers, positions, strict=True)
                )
            ]
            document = Document(name=metadata["name"], pages=pages)
            validate(document)
            if (
                "".join(
                    marker + page.content
                    for marker, page in zip(markers, pages, strict=True)
                )
                != body
            ):
                raise ValueError("Incumbent resource bytes were not preserved")
            return document
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Existing skill bundle cannot be read unambiguously"
            ) from error
