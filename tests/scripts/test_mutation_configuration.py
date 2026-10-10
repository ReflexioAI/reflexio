"""The weekly engine must load actual source targets and serial, keyless pytest."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_mutmut_loads_current_configuration(monkeypatch):
    from mutmut.configuration import config, reset_config

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    reset_config()
    settings = config()
    reset_config()
    assert settings.source_paths
    assert settings.debug
    assert all(path.is_file() for path in settings.source_paths)
    assert settings.pytest_add_cli_args_test_selection == ["tests/"]
    assert settings.pytest_add_cli_args[:2] == ["-o", "addopts="]
    assert "not requires_credentials" in " ".join(settings.pytest_add_cli_args)
    assert {Path("reflexio/"), Path("skills/"), Path("docs/lib/methods/")} <= set(
        settings.also_copy
    )
    assert all(
        path.exists()
        for path in (Path("reflexio/"), Path("skills/"), Path("docs/lib/methods/"))
    )


@pytest.mark.parametrize("hash_seed", [0, 6])
@pytest.mark.parametrize("malformed", [False, True], ids=["measured", "huge-int"])
def test_actual_engine_orders_fast_checks_and_keeps_survivor_tests(
    tmp_path, malformed, hash_seed
):
    """A full mutant test set stays intact while pytest -x fails fast."""
    engine = shutil.which("mutmut")
    assert engine is not None
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "number.py").write_text(
        "def increment(n):\n    return n + 1\n\ndef identity(n):\n    return n + 0\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    # Execute the exact production hook in the real engine, without importing
    # unrelated server/native runtimes into this portable mutation canary.
    conftest = (root / "tests/conftest.py").read_text()
    hook = next(
        node
        for node in ast.parse(conftest).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pytest_collection_modifyitems"
    )
    hook_source = ast.get_source_segment(conftest, hook)
    assert hook_source is not None
    (tests / "conftest.py").write_text(
        "import json, math, os\nimport pytest\nfrom pathlib import Path\n"
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent\n" + hook_source + "\n"
    )
    if malformed:
        with (tests / "conftest.py").open("a") as stream:
            stream.write(
                """
production_hook = pytest_collection_modifyitems
def pytest_collection_modifyitems(items):
    if "__mutmut_" in os.environ.get("MUTANT_UNDER_TEST", ""):
        path = PROJECT_ROOT / "mutmut-stats.json"
        data = json.loads(path.read_text())
        data["duration_by_test"] = {item.nodeid: 10 ** 400 for item in items}
        path.write_text(json.dumps(data))
        with (PROJECT_ROOT.parent / "collected.jsonl").open("a") as stream:
            stream.write(json.dumps([os.environ["MUTANT_UNDER_TEST"],
                                     [item.name for item in items]]) + "\\n")
    production_hook(items)
"""
            )
    (tests / "test_number.py").write_text(
        """import json
import os
import time
from pathlib import Path
from number import increment, identity

def record(name):
    mutant = os.environ.get("MUTANT_UNDER_TEST", "")
    if "__mutmut_" in mutant:
        path = Path(__file__).resolve().parents[2] / "executed.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps([mutant, name]) + "\\n")

def test_a_slow():
    record("slow")
    time.sleep(0.05)
    assert identity(1) == 1
    assert increment(1) == 2

def test_z_fast():
    record("fast")
    assert identity(1) == 1
    assert increment(1) == 2
"""
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mutmut]\nprocess_isolation = "forkserver"\n'
        'source_paths = ["number.py"]\n'
        'pytest_add_cli_args_test_selection = ["tests/"]\n'
        'pytest_add_cli_args = ["-o", "addopts=", "--timeout=10"]\n'
    )
    for command in (["run", "--max-children", "1"], ["export-cicd-stats"]):
        result = subprocess.run(  # noqa: S603 - installed engine, fixed arguments
            [engine, *command],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
    stats = json.loads((tmp_path / "mutants/mutmut-cicd-stats.json").read_text())
    assert stats["killed"] > 0 and stats["survived"] > 0
    assert stats["killed"] + stats["survived"] == stats["total"]
    assert stats["segfault"] == stats["suspicious"] == 0
    executed = {}
    for line in (tmp_path / "executed.jsonl").read_text().splitlines():
        mutant, name = json.loads(line)
        executed.setdefault(mutant, []).append(name)
    metadata = json.loads((tmp_path / "mutants/number.py.meta").read_text())
    evaluated = metadata["exit_code_by_key"]
    assert set(executed) == {key for key, code in evaluated.items() if code in (0, 1)}
    collected = {}
    if malformed:
        for line in (tmp_path / "collected.jsonl").read_text().splitlines():
            mutant, names = json.loads(line)
            collected[mutant] = [
                "slow" if name == "test_a_slow" else "fast" for name in names
            ]
    for mutant, names in executed.items():
        # Mutmut selects associated tests from a set, so their CLI/collection
        # order may differ from source order. Preserve that actual input order.
        expected = collected[mutant] if malformed else ["fast", "slow"]
        assert names[0] == expected[0], (mutant, names)
        if evaluated[mutant] == 0:
            assert names == expected, (mutant, names)
