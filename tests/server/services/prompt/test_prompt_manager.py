"""
Unit tests for PromptManager
"""

import tempfile
from pathlib import Path

import pytest

import reflexio.server.prompt as prompt
from reflexio.server.prompt.prompt_manager import PromptManager, _parse_frontmatter
from reflexio.server.prompt.prompt_schema import Prompt


def _write_prompt_md(
    directory: Path,
    version: str,
    content: str,
    variables: list[str],
    *,
    active: bool = False,
    description: str | None = None,
) -> None:
    """Helper to write a v{version}.prompt.md file with frontmatter."""
    lines = ["---"]
    if active:
        lines.append("active: true")
    if description:
        lines.append(f'description: "{description}"')
    lines.append("variables:")
    lines.extend(f"  - {v}" for v in variables)
    lines.append("---")
    lines.append("")
    (directory / f"v{version}.prompt.md").write_text("\n".join(lines) + content)


class TestParseFrontmatter:
    """Tests for the standalone _parse_frontmatter helper."""

    def test_basic(self):
        raw = "---\nactive: true\nvariables:\n  - a\n  - b\n---\nHello {a} {b}"
        meta, body = _parse_frontmatter(raw)
        assert meta["active"] is True
        assert meta["variables"] == ["a", "b"]
        assert body == "Hello {a} {b}"

    def test_missing_frontmatter(self):
        with pytest.raises(ValueError, match="Missing or malformed"):
            _parse_frontmatter("No frontmatter here")

    def test_description_field(self):
        raw = (
            '---\nactive: false\ndescription: "Some desc"\nvariables:\n  - x\n---\nBody'
        )
        meta, _ = _parse_frontmatter(raw)
        assert meta["description"] == "Some desc"
        assert meta["active"] is False


class TestPromptManager:
    """Test cases for PromptManager"""

    @pytest.fixture
    def temp_prompt_bank(self):
        """Create a temporary prompt bank directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_bank_path = Path(temp_dir) / "prompt_bank"
            prompt_bank_path.mkdir()

            # Create test prompt directory with two versions
            test_dir = prompt_bank_path / "test_prompt"
            test_dir.mkdir()

            _write_prompt_md(
                test_dir,
                "1.0.0",
                "This is a test prompt with {variable1} and {variable2}",
                ["variable1", "variable2"],
                active=True,
                description="Test prompt for unit testing",
            )
            _write_prompt_md(
                test_dir,
                "0.9.0",
                "Old test prompt with {variable1}",
                ["variable1"],
            )

            yield str(prompt_bank_path)

    @pytest.fixture
    def prompt_manager(self, temp_prompt_bank):
        """Create a PromptManager instance for testing."""
        return PromptManager(temp_prompt_bank)

    def test_get_prompt_success(self, prompt_manager):
        """Test successful retrieval of active prompt."""
        result = prompt_manager._get_prompt("test_prompt")
        assert result is not None
        assert isinstance(result, Prompt)
        assert (
            result.content == "This is a test prompt with {variable1} and {variable2}"
        )
        assert result.active is True

    def test_get_prompt_specific_version(self, prompt_manager):
        """Test retrieval of specific prompt version."""
        result = prompt_manager._get_prompt("test_prompt", "0.9.0")
        assert result is not None
        assert result.content == "Old test prompt with {variable1}"

    def test_get_prompt_not_found(self, prompt_manager):
        """Test get_prompt when prompt doesn't exist."""
        assert prompt_manager._get_prompt("nonexistent_prompt") is None

    def test_get_prompt_no_active_version(self, temp_prompt_bank):
        """Test get_prompt when no version has active: true."""
        no_active_dir = Path(temp_prompt_bank) / "no_active"
        no_active_dir.mkdir()
        _write_prompt_md(no_active_dir, "1.0.0", "Content", ["x"])

        pm = PromptManager(temp_prompt_bank)
        assert pm._get_prompt("no_active") is None

    def test_render_prompt_success(self, prompt_manager):
        """Test successful prompt rendering."""
        result = prompt_manager.render_prompt(
            "test_prompt", {"variable1": "value1", "variable2": "value2"}
        )
        assert result == "This is a test prompt with value1 and value2"

    def test_render_prompt_missing_variable(self, prompt_manager):
        """Test render_prompt with missing template variable."""
        with pytest.raises(ValueError, match="Missing required variable"):
            prompt_manager.render_prompt("test_prompt", {"variable2": "value2"})

    def test_render_prompt_not_found(self, prompt_manager):
        """Test render_prompt when prompt doesn't exist."""
        with pytest.raises(ValueError, match="Prompt nonexistent_prompt not found"):
            prompt_manager.render_prompt("nonexistent_prompt", {})

    def test_render_prompt_missing_another_variable(self, prompt_manager):
        """Test render_prompt fails when any template variable is missing."""
        with pytest.raises(ValueError, match="Missing required variable"):
            prompt_manager.render_prompt("test_prompt", {"variable1": "value1"})

    def test_list_versions(self, prompt_manager):
        """Test listing prompt versions."""
        result = prompt_manager.list_versions("test_prompt")
        assert set(result) == {"1.0.0", "0.9.0"}

    def test_list_versions_not_found(self, prompt_manager):
        """Test list_versions when prompt doesn't exist."""
        assert prompt_manager.list_versions("nonexistent_prompt") == []

    def test_get_active_version(self, prompt_manager):
        """Test getting the active version."""
        assert prompt_manager.get_active_version("test_prompt") == "1.0.0"

    def test_get_active_version_with_override(self, temp_prompt_bank):
        """Test version override takes precedence."""
        pm = PromptManager(temp_prompt_bank, version_override={"test_prompt": "0.9.0"})
        assert pm.get_active_version("test_prompt") == "0.9.0"

    def test_get_all_prompt_ids(self, temp_prompt_bank):
        """Test retrieval of all prompt IDs."""
        another_dir = Path(temp_prompt_bank) / "another_prompt"
        another_dir.mkdir()
        _write_prompt_md(another_dir, "1.0.0", "Content", ["x"], active=True)

        pm = PromptManager(temp_prompt_bank)
        assert set(pm.get_all_prompt_ids()) == {"test_prompt", "another_prompt"}

    def test_get_all_prompt_ids_empty(self):
        """Test get_all_prompt_ids when no prompts exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty = Path(temp_dir) / "empty"
            empty.mkdir()
            assert PromptManager(str(empty)).get_all_prompt_ids() == []

    def test_caching_behavior(self, prompt_manager):
        """Test that active prompt caching works correctly."""
        result1 = prompt_manager._get_prompt("test_prompt")
        result2 = prompt_manager._get_prompt("test_prompt")
        assert result1 is result2

    def test_version_override_bypasses_cache(self, temp_prompt_bank):
        """Test that explicit version requests bypass the cache."""
        pm = PromptManager(temp_prompt_bank, version_override={"test_prompt": "0.9.0"})
        result = pm.render_prompt("test_prompt", {"variable1": "val"})
        assert result == "Old test prompt with val"

    def test_integration_render_multiple_prompts(self, prompt_manager):
        """Integration test: render multiple prompts with caching."""
        variables = {"variable1": "test_value", "variable2": "another_value"}
        result1 = prompt_manager.render_prompt("test_prompt", variables)
        result2 = prompt_manager.render_prompt("test_prompt", variables)
        assert result1 == result2
        assert result1 == "This is a test prompt with test_value and another_value"

    def test_all_prompt_md_files_valid(self):
        """Test that all .prompt.md files in prompt_bank have valid frontmatter."""
        current_dir = Path(prompt.__file__).parent
        prompt_bank_path = (current_dir / "prompt_bank").resolve()

        if not prompt_bank_path.exists():
            pytest.skip("prompt_bank directory not found")

        md_files = list(prompt_bank_path.rglob("v*.prompt.md"))
        assert md_files, "No .prompt.md files found in prompt_bank"

        errors = []
        for md_file in md_files:
            raw = md_file.read_text(encoding="utf-8")
            try:
                meta, content = _parse_frontmatter(raw)
            except ValueError as e:
                errors.append(f"{md_file.relative_to(prompt_bank_path)}: {e}")
                continue

            if "variables" not in meta:
                errors.append(
                    f"{md_file.relative_to(prompt_bank_path)}: missing 'variables'"
                )
            elif not isinstance(meta["variables"], list):
                errors.append(
                    f"{md_file.relative_to(prompt_bank_path)}: 'variables' must be a list"
                )

            if not content.strip():
                errors.append(
                    f"{md_file.relative_to(prompt_bank_path)}: empty prompt content"
                )

        if errors:
            pytest.fail("Validation errors:\n" + "\n".join(errors))

    def test_exactly_one_active_per_prompt(self):
        """Test that each prompt directory has exactly one active version."""
        current_dir = Path(prompt.__file__).parent
        prompt_bank_path = (current_dir / "prompt_bank").resolve()

        if not prompt_bank_path.exists():
            pytest.skip("prompt_bank directory not found")

        errors = []
        for prompt_dir in sorted(prompt_bank_path.iterdir()):
            if not prompt_dir.is_dir():
                continue
            md_files = list(prompt_dir.glob("v*.prompt.md"))
            if not md_files:
                continue

            active_count = 0
            for md_file in md_files:
                raw = md_file.read_text(encoding="utf-8")
                try:
                    meta, _ = _parse_frontmatter(raw)
                    if meta.get("active"):
                        active_count += 1
                except ValueError:
                    pass

            if active_count != 1:
                errors.append(
                    f"{prompt_dir.name}: {active_count} active versions (expected 1)"
                )

        if errors:
            pytest.fail("Active version errors:\n" + "\n".join(errors))
