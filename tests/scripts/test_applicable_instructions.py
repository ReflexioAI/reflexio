"""Instruction-file resolution for changed files (scripts/applicable_instructions.py)."""

import subprocess
from pathlib import Path

import pytest

from scripts import applicable_instructions as ai


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git argv in a temp repo
        ["git", *args],  # noqa: S607 - git resolved from PATH by design
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _write(root / "AGENTS.md", "# Root\nroot rule\n")
    _write(root / "README.md", "readme\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _rel(paths: list[Path], root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in paths]


def _assert_precedence(chain: list[str]) -> None:
    """Nearest directory first; an override replaces its directory's AGENTS.md."""
    assert chain == [
        "pkg/sub/AGENTS.override.md",
        "pkg/AGENTS.md",
        "AGENTS.md",
    ]


def _precedence_tree(repo: Path) -> Path:
    _write(repo / "pkg" / "AGENTS.md", "pkg rule\n")
    _write(repo / "pkg" / "sub" / "AGENTS.md", "sub rule\n")
    _write(repo / "pkg" / "sub" / "AGENTS.override.md", "override rule\n")
    return _write(repo / "pkg" / "sub" / "mod.py", "x = 1\n")


def test_root_only(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ai.main(["README.md"], cwd=repo) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "== AGENTS.md",
        "applies to:",
        "  README.md",
        "--",
        "  1: # Root",
        "  2: root rule",
    ]


def test_nested_scoped_file(repo: Path) -> None:
    _write(repo / "pkg" / "AGENTS.md", "pkg rule\n")
    changed = _write(repo / "pkg" / "deep" / "mod.py", "x = 1\n")
    chain = ai.instruction_chain(changed, repo)
    assert _rel(chain, repo) == ["pkg/AGENTS.md", "AGENTS.md"]


def test_override_precedence(repo: Path) -> None:
    changed = _precedence_tree(repo)
    _assert_precedence(_rel(ai.instruction_chain(changed, repo), repo))


def test_precedence_check_fails_when_filename_order_is_reversed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = _precedence_tree(repo)
    monkeypatch.setattr(
        ai, "INSTRUCTION_FILENAMES", tuple(reversed(ai.INSTRUCTION_FILENAMES))
    )
    with pytest.raises(AssertionError):
        _assert_precedence(_rel(ai.instruction_chain(changed, repo), repo))


def test_precedence_check_fails_when_directory_order_is_reversed(
    repo: Path,
) -> None:
    changed = _precedence_tree(repo)
    with pytest.raises(AssertionError):
        _assert_precedence(_rel(ai.instruction_chain(changed, repo)[::-1], repo))


def test_shared_files_are_deduplicated(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo / "pkg" / "AGENTS.md", "pkg rule\n")
    _write(repo / "pkg" / "a.py", "a\n")
    _write(repo / "pkg" / "b.py", "b\n")
    assert ai.main(["pkg/a.py", "pkg/b.py", "README.md"], cwd=repo) == 0
    out = capsys.readouterr().out
    assert out.count("== AGENTS.md") == 1
    assert out.count("== pkg/AGENTS.md") == 1
    root_group = out.split("== AGENTS.md", 1)[1]
    assert "  pkg/a.py\n  pkg/b.py\n  README.md\n" in root_group

    assert ai.main(["pkg/a.py", "pkg/b.py", "--paths-only"], cwd=repo) == 0
    assert capsys.readouterr().out == "pkg/AGENTS.md\nAGENTS.md\n"


def test_no_changed_files(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ai.main(["--base", "main"], cwd=repo) == 0
    assert capsys.readouterr().out == "No changed files.\n"


def test_default_mode_uses_diff_against_merge_base(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo / "pkg" / "AGENTS.md", "pkg rule\n")
    _write(repo / "pkg" / "mod.py", "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    assert ai.main(["--base", "main", "--paths-only"], cwd=repo) == 0
    assert capsys.readouterr().out == "pkg/AGENTS.md\nAGENTS.md\n"


def test_default_mode_keeps_non_ascii_paths_exact(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Git C-quotes non-ASCII paths in line output ("na\303\257ve/..."), which
    # would point the walk at a directory that does not exist.
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo / "naïve" / "AGENTS.md", "scoped rule\n")
    _write(repo / "naïve" / "mod.py", "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    assert ai.main(["--base", "main", "--paths-only"], cwd=repo) == 0
    assert capsys.readouterr().out == "naïve/AGENTS.md\nAGENTS.md\n"


def test_submodule_file_uses_submodule_then_superproject(
    tmp_path: Path, repo: Path
) -> None:
    sub_src = tmp_path / "sub_src"
    sub_src.mkdir()
    _git(sub_src, "init", "-q", "-b", "main")
    _git(sub_src, "config", "user.email", "test@example.com")
    _git(sub_src, "config", "user.name", "test")
    _write(sub_src / "AGENTS.md", "sub repo rule\n")
    _write(sub_src / "lib" / "x.py", "x\n")
    _git(sub_src, "add", "-A")
    _git(sub_src, "commit", "-q", "-m", "init")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(sub_src),
        "vendor/sub",
    )
    inner = repo / "vendor" / "sub"
    assert ai.checkout_root(inner) == repo.resolve()
    chain = ai.instruction_chain((inner / "lib" / "x.py").resolve(), repo.resolve())
    assert _rel(chain, repo.resolve()) == ["vendor/sub/AGENTS.md", "AGENTS.md"]


def test_not_a_git_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert ai.main(["x.py"], cwd=outside) == 2
    assert "not inside a git repository" in capsys.readouterr().err


def test_changed_symlink_keeps_its_own_path_and_scope(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo / "pkg" / "AGENTS.md", "pkg rule\n")
    _write(repo / "target" / "AGENTS.md", "target rule\n")
    (repo / "pkg" / "link").symlink_to(Path("..") / "target", target_is_directory=True)
    (repo / "pkg" / "outside").symlink_to(repo.parent, target_is_directory=True)
    assert ai.main(["pkg/link", "pkg/outside"], cwd=repo) == 0
    out = capsys.readouterr().out
    assert "pkg/link" in out
    assert "pkg/outside" in out
    assert "target/AGENTS.md" not in out
    assert ai.main(["--paths-only", "pkg/link"], cwd=repo) == 0
    assert capsys.readouterr().out == "pkg/AGENTS.md\nAGENTS.md\n"
