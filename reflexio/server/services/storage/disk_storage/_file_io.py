"""File I/O: serialize Pydantic models to/from files with YAML frontmatter.

Each entity is stored as a file where:
- YAML frontmatter contains metadata fields
- Content body contains the main content field (if any)

Embedding vectors are excluded — they're stored as sidecar .json files.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Fields that go into the content body (one per entity type).
# Key = class name, value = field name whose value becomes the body.
_CONTENT_FIELDS: dict[str, str] = {
    "UserProfile": "content",
    "Interaction": "content",
    "UserPlaybook": "content",
    "AgentPlaybook": "content",
    # Request, AgentSuccessEvaluationResult, etc. → metadata-only (no body)
}

# Fields always excluded from frontmatter (stored separately or not at all).
_EXCLUDED_FIELDS: frozenset[str] = frozenset({"embedding"})


def _serialize_value(value: Any) -> Any:
    """Recursively convert enums, lists, and dicts to YAML-friendly primitives."""
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return value.value
    return value


def serialize_entity(entity: BaseModel) -> str:
    """Convert a Pydantic model to YAML frontmatter format.

    Args:
        entity: The Pydantic model instance to serialize.

    Returns:
        A string with ``---`` YAML frontmatter and an optional body.
    """
    class_name = type(entity).__name__
    content_field = _CONTENT_FIELDS.get(class_name)

    # Build frontmatter dict — exclude content field and embeddings
    exclude_keys = _EXCLUDED_FIELDS | ({content_field} if content_field else set())
    data = entity.model_dump()
    frontmatter: dict[str, Any] = {}
    for key, value in data.items():
        if key in exclude_keys:
            continue
        frontmatter[key] = _serialize_value(value)

    # Serialize frontmatter
    yaml_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    ).rstrip("\n")

    # Build body
    body = ""
    if content_field:
        body = str(data.get(content_field, "") or "")

    parts = [f"---\n{yaml_str}\n---"]
    if body:
        parts.append(body)
    return "\n\n".join(parts) + "\n"


def deserialize_entity(text: str, model_class: type[T]) -> T:  # noqa: UP047
    """Parse YAML frontmatter format back into a Pydantic model.

    Args:
        text: The text (``---`` fenced YAML + optional body).
        model_class: The Pydantic model class to instantiate.

    Returns:
        An instance of *model_class* populated from the frontmatter + body.

    Raises:
        ValueError: If the text doesn't contain valid YAML frontmatter.
    """
    text = text.strip()
    if not text.startswith("---"):
        raise ValueError("Entity file does not start with YAML frontmatter (---)")

    # Find the closing --- delimiter after the opening one.
    # The opening "---" is at position 0; search for "\n---\n" or "\n---" at EOF
    # starting after the first line to avoid matching the opener.
    close_idx = text.find("\n---\n", 3)
    if close_idx == -1:
        # Check if the closing --- is at the very end of the text
        if text.endswith("\n---"):
            close_idx = len(text) - 4  # position of the \n before ---
        else:
            raise ValueError("Malformed YAML frontmatter: missing closing ---")

    yaml_text = text[3 : close_idx + 1]  # after opening "---", up to the newline
    body = text[close_idx + 4 :].strip()  # after closing "---\n"

    data: dict[str, Any] = yaml.safe_load(yaml_text) or {}

    # Inject body into the content field
    class_name = model_class.__name__
    content_field = _CONTENT_FIELDS.get(class_name)
    if content_field and body:
        data[content_field] = body

    # Ensure embedding gets a default (since we exclude it from frontmatter)
    if "embedding" not in data and "embedding" in model_class.model_fields:
        data["embedding"] = []

    return model_class.model_validate(data)


def serialize_embedding(embedding: list[float]) -> str:
    """Serialize an embedding vector to a JSON string for sidecar storage."""
    return json.dumps(embedding)


def deserialize_embedding(text: str) -> list[float]:
    """Deserialize an embedding vector from a JSON sidecar file."""
    return json.loads(text)
