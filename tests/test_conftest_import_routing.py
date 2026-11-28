"""The test bootstrap must preserve normal and mutmut source imports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("relocated", [False, True])
@pytest.mark.parametrize("preinserted", [False, True])
def test_bootstrap_imports_from_its_own_project(
    tmp_path: Path, relocated: bool, preinserted: bool
):
    project = tmp_path / "project"
    mutants = project / "mutants"
    for root, marker in ((project, "original"), (mutants, "mutated")):
        package = root / "reflexio"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"SOURCE = {marker!r}\n")

    root = mutants if relocated else project
    bootstrap = root / "tests" / "conftest.py"
    bootstrap.parent.mkdir()
    bootstrap.write_text(Path(__file__).with_name("conftest.py").read_text())
    # Mutmut removes the original project from sys.path and inserts mutants.
    # Execute the real bootstrap before its first application import, without
    # loading server fixtures or credentials into this isolated sentinel package.
    code = """
import ast
import sys
from pathlib import Path

bootstrap = Path(sys.argv[1])
if sys.argv[4] == 'preinserted':
    sys.path.insert(0, sys.argv[2])
tree = ast.parse(bootstrap.read_text())
prefix = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('reflexio'):
        break
    prefix.append(node)
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(bootstrap), 'exec'),
     {'__file__': str(bootstrap)})
import reflexio
assert reflexio.SOURCE == sys.argv[3], (reflexio.SOURCE, reflexio.__file__)
assert Path(reflexio.__file__).resolve() == Path(sys.argv[2]) / 'reflexio/__init__.py'
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter/code and pytest-owned paths
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(bootstrap),
            str(root),
            "mutated" if relocated else "original",
            "preinserted" if preinserted else "empty",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutant", ["stats", ""])
def test_mutation_bootstrap_does_not_create_user_env(tmp_path: Path, mutant: str):
    project = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    home.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = os.environ.copy()
    for key in ("REFLEXIO_ENV_FILE", "REFLEXIO_LOG_DIR", "LOCAL_STORAGE_PATH"):
        env.pop(key, None)
    env.update(HOME=str(home), TMPDIR=str(scratch), MUTANT_UNDER_TEST=mutant)
    code = """
import os
import runpy
import sys
from pathlib import Path

project = Path(sys.argv[1])
sys.path.insert(0, str(project))
runpy.run_path(str(project / 'tests/conftest.py'))
assert 'reflexio.server' in sys.modules
assert not (Path.home() / '.reflexio/.env').exists()
selected = Path(os.environ['REFLEXIO_ENV_FILE'])
assert selected.is_relative_to(Path(os.environ['TMPDIR']))
assert selected.exists() and selected.read_text() == ''
"""
    result = subprocess.run(  # noqa: S603 — fixed code and pytest-owned home
        [sys.executable, "-I", "-c", code, str(project)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (home / ".reflexio" / ".env").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_judge_settings_in_developer_dotenv_do_not_leak(tmp_path: Path, legacy: bool):
    project = Path(__file__).resolve().parents[1]
    developer_env = tmp_path / ".env"
    developer_contents = (
        ("REFLEXIO_SEARCH_JUDGE=off\n" if legacy else "")
        + "REFLEXIO_SEARCH_JUDGE_API_KEY=fixture-judge\nOPENAI_API_KEY=fixture-credential\n"
    )
    developer_env.write_text(developer_contents)
    env = os.environ.copy()
    for key in (
        "REFLEXIO_ENV_FILE",
        "PYTHON_DOTENV_DISABLED",
        "MUTANT_UNDER_TEST",
        "OPENAI_API_KEY",
    ):
        env.pop(key, None)
    env.update(LITELLM_MODE="DEV", LITELLM_LOCAL_MODEL_COST_MAP="true")
    code = """
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

project = Path(sys.argv[1])
sys.path.insert(0, str(project))
bootstrap = runpy.run_path(str(project / 'tests/conftest.py'))
bootstrap["pytest_configure"](SimpleNamespace(args=["tests"]))
assert os.environ.get("OPENAI_API_KEY") == "fixture-credential"
assert "REFLEXIO_SEARCH_JUDGE" not in os.environ
assert "REFLEXIO_SEARCH_JUDGE_API_KEY" not in os.environ
from reflexio.cli import env_loader
_load_dotenv_pruned = env_loader._load_dotenv_pruned
env_loader.load_reflexio_env()
assert "REFLEXIO_SEARCH_JUDGE_API_KEY" not in os.environ
mode_home = Path("mode-home")
mode_home.mkdir()
(mode_home / ".env.self_host").write_text("REFLEXIO_SEARCH_JUDGE=off" + chr(10))
env_loader._USER_ENV_DIR = mode_home
Path(".env.self_host").write_text("MODE_FIXTURE=prefix-${REFLEXIO_SEARCH_JUDGE}" + chr(10))
env_loader.load_existing_reflexio_env_for_mode(cli_mode="self_host")
assert os.environ["MODE_FIXTURE"] == "prefix-off"
assert "REFLEXIO_SEARCH_JUDGE" not in os.environ
fixture = Path("fixture.env")
fixture.write_text("REFLEXIO_SEARCH_JUDGE=jev\\n")
try:
    _load_dotenv_pruned(fixture)
except ValueError as error:
    assert "REFLEXIO_SEARCH_JUDGE" in str(error)
else:
    raise AssertionError("A different source must retain strict validation")
"""
    result = subprocess.run(  # noqa: S603 — fixed code and a pytest-owned dotenv
        [sys.executable, "-I", "-c", code, str(project)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert developer_env.read_text() == developer_contents
