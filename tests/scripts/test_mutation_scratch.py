"""Real engine workers must release owned scratch and reject storage exhaustion."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode",
    ["mutation", "normal", "sqlite-full", "enospc", "teardown", "cleanup-failure"],
)
def test_actual_engine_scratch_lifecycle(tmp_path: Path, mode: str):
    root = Path(__file__).resolve().parents[2]
    source = (root / "tests/conftest.py").read_text()
    functions = {
        "pytest_sessionfinish",
        "pytest_runtest_makereport",
        "pytest_runtest_teardown",
    }
    lines = source.splitlines()
    hooks = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            hooks.append("\n".join(lines[first - 1 : node.end_lineno]))
    assert len(hooks) == 3
    project = tmp_path / "project"
    tests = project / "tests"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        "import errno, os, shutil, sqlite3\nimport pytest\n"
        "from collections.abc import Iterator\n" + "\n\n".join(hooks)
    )
    if mode == "cleanup-failure":
        with (tests / "conftest.py").open("a") as stream:
            stream.write("""
production_rmtree = shutil.rmtree
class CleanupControl:
    @staticmethod
    def rmtree(path):
        if "__mutmut_" in os.environ.get("MUTANT_UNDER_TEST", ""):
            raise PermissionError(errno.EACCES, "owned cleanup refused")
        production_rmtree(path)
shutil = CleanupControl
""")
    (project / "number.py").write_text("def identity(n):\n    return n + 0\n")
    (tests / "test_number.py").write_text(
        """import errno, os, sqlite3
from pathlib import Path
import pytest
from number import identity

@pytest.fixture(scope="session", autouse=True)
def finalizer_check(tmp_path_factory):
    path = tmp_path_factory.getbasetemp() / "session-owned"
    path.write_bytes(b"x" * 65536)
    yield
    assert path.is_file(), "scratch deleted before session fixture teardown"
    marker = Path(__file__).resolve().parents[2] / "finalized.txt"
    with marker.open("a") as stream:
        stream.write("done\\n")

def test_identity(tmp_path):
    result = identity(1)
    (tmp_path / "payload").write_bytes(b"x" * 65536)
    if "__mutmut_" in os.environ.get("MUTANT_UNDER_TEST", ""):
        if os.environ["SCRATCH_CONTROL"] == "sqlite-full":
            try:
                with sqlite3.connect(":memory:") as db:
                    db.execute("PRAGMA max_page_count=2")
                    db.execute("CREATE TABLE scratch (value BLOB)")
                    db.execute("INSERT INTO scratch VALUES (?)", (b"x" * 65536,))
            except sqlite3.OperationalError as error:
                assert error.sqlite_errorcode == sqlite3.SQLITE_FULL
                raise RuntimeError("wrapped storage failure") from error
        if os.environ["SCRATCH_CONTROL"] == "enospc":
            raise RuntimeError("wrapped storage failure") from OSError(errno.ENOSPC, "full")
    assert result == 1
"""
    )
    if mode == "teardown":
        with (project / "number.py").open("a") as stream:
            stream.write("\ndef late(n):\n    return n + 0\n")
        test_source = (tests / "test_number.py").read_text()
        test_source = test_source.replace(
            "from number import identity", "from number import identity, late"
        )
        test_source = test_source.replace(
            "def test_identity(tmp_path):",
            "@pytest.fixture\ndef late_teardown():\n    yield\n    assert late(1) == 1\n\ndef test_identity(tmp_path, late_teardown):",
        )
        test_source += "\ndef test_unrelated():\n    pass\n"
        (tests / "test_number.py").write_text(test_source)
    (project / "pyproject.toml").write_text(
        '[tool.mutmut]\nprocess_isolation = "forkserver"\n'
        'source_paths = ["number.py"]\n'
        'pytest_add_cli_args_test_selection = ["tests/"]\n'
        'pytest_add_cli_args = ["-o", "addopts=", "--timeout=10"]\n'
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {**os.environ, "TMPDIR": str(scratch), "SCRATCH_CONTROL": mode}
    for key in ("MUTANT_UNDER_TEST", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH"):
        env.pop(key, None)
    if mode == "normal":
        executable = sys.executable
        commands = [["-m", "pytest", "-o", "addopts=", "tests/"]]
    else:
        executable = shutil.which("mutmut")
        commands = [["run", "--max-children", "2"], ["export-cicd-stats"]]
    assert executable is not None
    for command in commands:
        result = subprocess.run(  # noqa: S603 - installed tools, owned project
            [executable, *command],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finalized = (
        project.parent / "finalized.txt"
        if mode == "normal"
        else project / "finalized.txt"
    )
    assert finalized.is_file()
    retained = list(scratch.rglob("payload"))
    if mode == "normal":
        assert retained, "normal pytest retention must remain unchanged"
        return
    if mode == "cleanup-failure":
        assert retained, "failed cleanup must retain evidence"
    else:
        assert not retained, "os._exit workers must not accumulate test databases"
    stats = json.loads((project / "mutants/mutmut-cicd-stats.json").read_text())
    if mode in ("sqlite-full", "enospc", "cleanup-failure"):
        assert stats["suspicious"] > 0
        assert stats["killed"] == 0
    else:
        assert stats["killed"] > 0 and stats["survived"] > 0
        assert stats["suspicious"] == 0
    if mode == "teardown":
        associations = json.loads((project / "mutants/mutmut-stats.json").read_text())[
            "tests_by_mangled_function_name"
        ]
        assert set(associations["number.x_late"]) == {
            "tests/test_number.py::test_identity"
        }
