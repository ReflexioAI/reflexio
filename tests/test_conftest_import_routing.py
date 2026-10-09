"""The test bootstrap must preserve normal and mutmut source imports."""

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
