"""
Prompt management using file system prompt bank with markdown frontmatter files.
"""

import logging
import re
from pathlib import Path
from typing import Any

from .prompt_schema import Prompt

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n(.*)", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from a markdown file using stdlib only.

    Args:
        raw (str): Raw file content with optional ``---`` delimited frontmatter.

    Returns:
        tuple[dict[str, Any], str]: Parsed metadata dict and the body content.

    Raises:
        ValueError: If frontmatter is missing or malformed.
    """
    if not (m := _FRONTMATTER_RE.match(raw)):
        raise ValueError("Missing or malformed YAML frontmatter")

    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value.startswith("[") and value.endswith("]"):
            # Simple list: [a, b, c]
            meta[key] = [
                v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()
            ]
        elif value.startswith("- "):
            # First item of a block list on the same line — shouldn't happen in our format
            meta[key] = [value[2:].strip()]
        elif value.lower() in ("true", "false"):
            meta[key] = value.lower() == "true"
        elif value == "" or value.lower() == "null":
            meta[key] = None
        else:
            meta[key] = value.strip("'\"")

    # Handle block-style lists (- item per line)
    current_key: str | None = None
    for line in m.group(1).splitlines():
        stripped = line.strip()
        if ":" in line and not stripped.startswith("- "):
            current_key = line.partition(":")[0].strip()
        elif stripped.startswith("- ") and current_key:
            if not isinstance(meta.get(current_key), list):
                meta[current_key] = []
            meta[current_key].append(stripped[2:].strip().strip("'\""))

    return meta, m.group(2)


class PromptManager:
    """Prompt management using file system prompt bank."""

    def __init__(
        self,
        prompt_bank_path: str | None = None,
        version_override: dict[str, str] | None = None,
    ):
        """
        Initialize the PromptManager.

        Args:
            prompt_bank_path (str, optional): Path to the prompt bank directory.
            version_override (dict[str, str], optional): Map of prompt_id → version string to override the active version.
        """
        if prompt_bank_path is None:
            current_dir = Path(__file__).parent
            self.prompt_bank_path = current_dir / "prompt_bank"
        else:
            self.prompt_bank_path = Path(prompt_bank_path)

        self.version_override = version_override

        if not self.prompt_bank_path.exists():
            logger.warning("Prompt bank path does not exist: %s", self.prompt_bank_path)

        self._cache: dict[str, Prompt] = {}

    # ==============================
    # Public methods
    # ==============================

    def render_prompt(self, prompt_id: str, variables: dict[str, Any]) -> str:
        """
        Render prompt template with variables.

        Args:
            prompt_id (str): ID of the prompt.
            variables (dict[str, Any]): Variables to substitute in template.

        Returns:
            str: Rendered prompt content.

        Raises:
            ValueError: If prompt not found or template rendering fails.
        """
        version = (
            self.version_override.get(prompt_id) if self.version_override else None
        )
        prompt = self._get_prompt(prompt_id, version)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        missing_vars = set(prompt.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(
                f"Missing required variables {missing_vars} for prompt {prompt_id}"
            )

        try:
            return prompt.content.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Missing required variable {e} for prompt {prompt_id}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error rendering prompt {prompt_id}: {e}") from e

    def list_versions(self, prompt_id: str) -> list[str]:
        """
        List all versions of a prompt.

        Args:
            prompt_id (str): ID of the prompt.

        Returns:
            list[str]: List of version strings.
        """
        prompt_dir = self.prompt_bank_path / prompt_id
        if not prompt_dir.is_dir():
            return []
        return [
            p.name.removeprefix("v").removesuffix(".prompt.md")
            for p in sorted(prompt_dir.glob("v*.prompt.md"))
        ]

    def get_active_version(self, prompt_id: str) -> str | None:
        """
        Get the active version for a prompt (considering overrides).

        Args:
            prompt_id (str): ID of the prompt.

        Returns:
            str | None: The active version string, or None if prompt not found.
        """
        if self.version_override and prompt_id in self.version_override:
            return self.version_override[prompt_id]
        return self._find_active_version(prompt_id)

    def get_all_prompt_ids(self) -> list[str]:
        """
        Get list of all available prompt IDs.

        Returns:
            list[str]: List of prompt IDs.
        """
        if not self.prompt_bank_path.exists():
            return []
        return [
            item.name
            for item in self.prompt_bank_path.iterdir()
            if item.is_dir() and any(item.glob("v*.prompt.md"))
        ]

    # ==============================
    # Private methods
    # ==============================

    def _load_prompt(self, prompt_id: str, version: str) -> Prompt | None:
        """Load a single prompt file by prompt_id and version string."""
        path = self.prompt_bank_path / prompt_id / f"v{version}.prompt.md"
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Prompt file not found: %s", path)
            return None
        except Exception as e:
            logger.error("Error reading prompt file %s: %s", path, e)
            return None

        try:
            meta, content = _parse_frontmatter(raw)
        except ValueError as e:
            logger.error("Error parsing frontmatter in %s: %s", path, e)
            return None

        return Prompt(
            active=meta.get("active", False),
            description=meta.get("description"),
            changelog=meta.get("changelog"),
            variables=meta.get("variables", []),
            content=content,
        )

    def _find_active_version(self, prompt_id: str) -> str | None:
        """Scan .prompt.md files to find the latest one with active: true."""
        prompt_dir = self.prompt_bank_path / prompt_id
        if not prompt_dir.is_dir():
            return None

        def _semver_key(p: Path) -> tuple[int, ...]:
            """Parse 'vX.Y.Z.prompt.md' into a comparable tuple."""
            try:
                return tuple(int(x) for x in p.stem.removesuffix(".prompt").split("."))
            except ValueError:
                return (0,)

        latest: str | None = None
        for path in sorted(
            prompt_dir.glob("v*.prompt.md"), key=_semver_key, reverse=True
        ):
            try:
                raw = path.read_text(encoding="utf-8")
                meta, _ = _parse_frontmatter(raw)
                if meta.get("active"):
                    latest = path.name.removeprefix("v").removesuffix(".prompt.md")
                    break
            except (ValueError, OSError):
                continue
        return latest

    def _get_prompt(self, prompt_id: str, version: str | None = None) -> Prompt | None:
        """
        Get prompt, using cache for active prompts.

        Args:
            prompt_id (str): ID of the prompt.
            version (str, optional): Specific version to load.

        Returns:
            Prompt | None: The prompt, or None if not found.
        """
        if version:
            return self._load_prompt(prompt_id, version)

        if prompt_id in self._cache:
            return self._cache[prompt_id]

        active_version = self._find_active_version(prompt_id)
        if not active_version:
            logger.warning("No active version found for prompt %s", prompt_id)
            return None

        prompt = self._load_prompt(prompt_id, active_version)
        if prompt:
            self._cache[prompt_id] = prompt
        return prompt
