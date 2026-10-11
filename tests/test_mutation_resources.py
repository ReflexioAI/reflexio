"""Mutation workspaces must retain resources required by the baseline tests."""

import tomllib
from pathlib import Path
from types import SimpleNamespace

from mutmut.utils import file_utils

import reflexio
from reflexio.cli import env_loader


def test_mutation_workspace_retains_env_template(tmp_path: Path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    resources = tomllib.loads((project / "pyproject.toml").read_text())["tool"][
        "mutmut"
    ]["also_copy"]
    template = (project / ".env.example").read_text()
    (tmp_path / ".env.example").write_text(template)
    package = tmp_path / "reflexio"
    package.mkdir()
    (package / "__init__.py").touch()
    (tmp_path / "mutants").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        file_utils, "config", lambda: SimpleNamespace(also_copy=resources)
    )
    file_utils.copy_also_copy_files()

    # CLI template fallback follows the relocated package, not the CWD.
    monkeypatch.setattr(
        reflexio, "__file__", str(tmp_path / "mutants/reflexio/__init__.py")
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert env_loader._find_env_example("reflexio.data") == template
