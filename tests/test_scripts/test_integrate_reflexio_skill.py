import re
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPOSITORY_ROOT / "skills" / "integrate-reflexio"
LOCAL_MARKDOWN_LINK = re.compile(
    r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)"
)
REQUIRED_FILES = {
    Path("SKILL.md"),
    Path("agents/openai.yaml"),
    Path("references/http-api.md"),
    Path("references/python-client.md"),
}
TARGET_LOCATIONS = {
    "codex": Path(".agents/skills/integrate-reflexio"),
    "claude-code": Path(".claude/skills/integrate-reflexio"),
    "cursor": Path(".agents/skills/integrate-reflexio"),
}


def _assert_portable_skill(skill_root: Path) -> None:
    relative_files = {
        path.relative_to(skill_root) for path in skill_root.rglob("*") if path.is_file()
    }
    assert relative_files >= REQUIRED_FILES
    assert not any(path.is_symlink() for path in skill_root.rglob("*"))

    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\n")
    frontmatter = skill_text.split("---\n", maxsplit=2)[1]
    assert "name: integrate-reflexio\n" in frontmatter
    assert "description:" in frontmatter

    resolved_root = skill_root.resolve()
    for markdown_file in skill_root.rglob("*.md"):
        markdown = markdown_file.read_text(encoding="utf-8")
        for relative_target in LOCAL_MARKDOWN_LINK.findall(markdown):
            resolved_target = (markdown_file.parent / relative_target).resolve()
            assert resolved_target.is_relative_to(resolved_root)
            assert resolved_target.exists(), (
                f"{markdown_file.relative_to(skill_root)} links to missing "
                f"{relative_target}"
            )


def test_integrate_reflexio_skill_is_portable(tmp_path: Path) -> None:
    _assert_portable_skill(SKILL_ROOT)

    for coding_agent, target_location in TARGET_LOCATIONS.items():
        copied_skill = tmp_path / coding_agent / target_location
        shutil.copytree(SKILL_ROOT, copied_skill)
        _assert_portable_skill(copied_skill)
