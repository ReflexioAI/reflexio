"""
Prompt management using file system prompt bank with markdown frontmatter files.
"""

import logging
import re
from collections.abc import Sequence
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
        prompt_bank_path: str | Path | None = None,
        version_override: dict[str, str] | None = None,
        extra_prompt_bank_paths: Sequence[str | Path] | None = None,
    ):
        """
        Initialize the PromptManager.

        Args:
            prompt_bank_path (str | Path, optional): Path to the primary prompt bank directory.
            version_override (dict[str, str], optional): Map of prompt_id → version string to override the active version.
            extra_prompt_bank_paths (Sequence[str | Path], optional): Additional prompt bank directories.
        """
        if prompt_bank_path is None:
            current_dir = Path(__file__).parent
            self.prompt_bank_path = current_dir / "prompt_bank"
        else:
            self.prompt_bank_path = Path(prompt_bank_path)
        self.prompt_bank_paths = [self.prompt_bank_path]
        if extra_prompt_bank_paths:
            self.prompt_bank_paths.extend(
                Path(path) for path in extra_prompt_bank_paths
            )

        self.version_override = version_override

        for path in self.prompt_bank_paths:
            if not path.exists():
                logger.warning("Prompt bank path does not exist: %s", path)

        self._cache: dict[str, Prompt] = {}
        self._validate_unique_prompt_ids()

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
        versions: list[str] = []
        for prompt_dir in self._prompt_dirs(prompt_id):
            versions.extend(
                p.name.removeprefix("v").removesuffix(".prompt.md")
                for p in sorted(prompt_dir.glob("v*.prompt.md"))
            )
        return versions

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
        prompt_ids: list[str] = []
        seen: set[str] = set()
        for prompt_bank_path in self.prompt_bank_paths:
            if not prompt_bank_path.exists():
                continue
            for item in prompt_bank_path.iterdir():
                if (
                    item.is_dir()
                    and item.name not in seen
                    and any(item.glob("v*.prompt.md"))
                ):
                    prompt_ids.append(item.name)
                    seen.add(item.name)
        return prompt_ids

    # ==============================
    # Private methods
    # ==============================

    def _prompt_dirs(self, prompt_id: str) -> list[Path]:
        """Return configured prompt directories for a prompt ID."""
        return [
            prompt_bank_path / prompt_id
            for prompt_bank_path in self.prompt_bank_paths
            if (prompt_bank_path / prompt_id).is_dir()
        ]

    def _validate_unique_prompt_ids(self) -> None:
        """Reject prompt IDs that appear in more than one configured prompt bank."""
        locations: dict[str, list[Path]] = {}
        for prompt_bank_path in self.prompt_bank_paths:
            if not prompt_bank_path.exists():
                continue
            for item in prompt_bank_path.iterdir():
                if item.is_dir() and any(item.glob("v*.prompt.md")):
                    locations.setdefault(item.name, []).append(item)

        duplicates = {
            prompt_id: paths for prompt_id, paths in locations.items() if len(paths) > 1
        }
        if duplicates:
            details = "; ".join(
                f"{prompt_id}: {', '.join(str(path) for path in paths)}"
                for prompt_id, paths in sorted(duplicates.items())
            )
            raise ValueError(f"Duplicate prompt_id found in prompt banks: {details}")

    def _load_prompt(self, prompt_id: str, version: str) -> Prompt | None:
        """Load a single prompt file by prompt_id and version string."""
        path: Path | None = None
        for prompt_dir in self._prompt_dirs(prompt_id):
            candidate = prompt_dir / f"v{version}.prompt.md"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            logger.warning("Prompt file not found: %s v%s", prompt_id, version)
            return None

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
        prompt_dirs = self._prompt_dirs(prompt_id)
        if not prompt_dirs:
            return None

        def _semver_key(p: Path) -> tuple[int, ...]:
            """Parse 'vX.Y.Z.prompt.md' into a comparable tuple.

            Strips the leading 'v' before int parsing so versions sort by
            their numeric semver tuple rather than collapsing to a single
            fallback key. Without the prefix strip every version evaluates
            to the (0,) fallback (because ``int('v1')`` raises) and the
            "latest active" search becomes order-of-glob-dependent.
            """
            try:
                return tuple(
                    int(x)
                    for x in p.stem.removeprefix("v").removesuffix(".prompt").split(".")
                )
            except ValueError:
                return (0,)

        latest: str | None = None
        for prompt_dir in prompt_dirs:
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
            if latest:
                break
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
