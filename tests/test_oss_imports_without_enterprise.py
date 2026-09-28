"""This package must import with `reflexio_ext` absent.

Five docstrings in this package assert the invariant -- `server/auth.py` says
"OSS code must never import from ``reflexio_ext``", and
`server/deployment_profile.py` calls itself "an OSS public seam: it imports
nothing from ``reflexio_ext``". Until now nothing checked any of them.

Existing CI does not cover it, and that is measured rather than assumed: the only
workflow in this repository is `mem0-compat.yml`, whose `installed-artifact` job
runs `scripts/verify_mem0_artifact.py` -- which imports `reflexio.ReflexioClient`
and `reflexio.mem0` and **never imports `reflexio.server`**. `reflexio.server` is
where the application wiring lives and where a leak would actually appear.

Why this matters concretely: `reflexio` is published to PyPI. An import of
`reflexio_ext` from here is not a style question, it is an `ImportError` on a
user's machine, at import time, with no workaround available to them.

Why a CHILD interpreter rather than `sys.meta_path` in this process: by the time
pytest runs this file, `reflexio.server` is already imported and cached in
`sys.modules`, so a blocker installed here would never be consulted.
`test_the_in_process_shortcut_would_not_have_worked` pins that.

Two scoping facts, both measured, because they bound what this file adds:

* **A top-level leak in `reflexio/server/__init__.py` is already caught** -- by
  `tests/conftest.py`, which imports `reflexio.server`, so the whole OSS suite dies
  at conftest load with `ModuleNotFoundError` and pytest exits **4**. Loud, but a
  usage error rather than a named failure. What this file adds is the SUBMODULE
  case: a leak in, say, `server/auth.py` leaves conftest importable, and nothing
  else notices. Both shapes were confirmed by mutation.
* **In the OSS venv `reflexio_ext` is not installed at all**, so in the lane that
  runs these tests the finder is belt-and-braces. It is what keeps the file honest
  when the suite is run from the ENTERPRISE venv, where `reflexio_ext` imports
  fine: measured there, a submodule leak still fails and names
  `reflexio.server.api`. It is also why nothing here asserts the module is absent
  -- an earlier version did, and that made the suite red from that venv on
  perfectly correct code.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy_guard

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The public entry points a user or an operator actually imports. `reflexio.server`
#: is the one that matters here: it is absent from the mem0 contract check, and it
#: is where `configure_logging`, the API app and the extension seams live.
_ENTRY_POINTS = (
    "reflexio",
    "reflexio.server",
    "reflexio.server.api",
    "reflexio.cli.app",
    "reflexio.models.config_schema",
)


def _console_script_modules() -> tuple[str, ...]:
    """The module behind each `[project.scripts]` entry, read from pyproject.toml.

    DERIVED rather than listed, so a console script added later is covered without
    anyone remembering this file. That matters here: the installed entry point is
    `reflexio.cli.__main__:main`, and importing `reflexio.cli.app` does NOT load
    `reflexio.cli.__main__` -- measured. So a leak in the real entry module would
    have left every check in this file green while `reflexio --help` died on an OSS
    install. Raised by Codex on ReflexioAI/reflexio#554.
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    targets = config["project"]["scripts"]
    assert targets, "pyproject.toml declares no console scripts -- path or key moved"
    return tuple(sorted({target.split(":", 1)[0] for target in targets.values()}))


def _blocker_for(blocked: str) -> str:
    """A `sys.meta_path` finder that rejects `blocked` AND everything under it.

    The prefix half is the point: a real leak reads
    `from reflexio_ext.server.x import y` far more often than it is a bare
    top-level import, so a finder matching only the exact name would miss the
    realistic shape. `test_the_prefix_branch_is_what_blocks_a_submodule` pins it,
    with a control -- getting that test to mean anything took two tries.

    Parameterised by name so the mechanism tests below can aim it at a module that
    IS importable here -- see their docstrings for why that matters.
    """
    return f'''
import sys
from importlib.abc import MetaPathFinder


class _Blocker(MetaPathFinder):
    # `find_spec`, NOT `find_module`. The `find_module`/`load_module` protocol was
    # deprecated in 3.4 and REMOVED in 3.12, so a finder written that way is a
    # silent no-op on this interpreter -- every import sails through and the check
    # reports success having blocked nothing. The first version of this file did
    # exactly that, and `test_the_blocker_mechanism_intercepts_an_installed_module`
    # is what caught it.
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "{blocked}" or fullname.startswith("{blocked}."):
            raise ImportError(f"{blocked} is not installed (blocked): {{fullname}}")
        return None


sys.meta_path.insert(0, _Blocker())
'''


def _import_in_child(
    *modules: str, blocked: str = "reflexio_ext", preimport: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Import `modules` in a fresh interpreter with `blocked` unimportable.

    `preimport` runs BEFORE the finder is installed. Only needed to make the
    finder's prefix branch reachable -- see
    `test_the_prefix_branch_is_reachable_and_necessary`.
    """
    prelude = f"import {preimport}\n" if preimport else ""
    return _run_child(
        prelude
        + _blocker_for(blocked)
        + "\n"
        + textwrap.dedent("\n".join(f"import {module}" for module in modules))
    )


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    """Run `script` in a fresh interpreter, with the success marker appended."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", script + '\nprint("IMPORTS_OK")\n'],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        # Under the 120s pytest-timeout the `os-unit-tests` lane applies, a
        # 180s value here could never fire -- the generic timeout would win and
        # say nothing about which child hung. Real cost is well under a second.
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_each_entry_point_imports_without_enterprise(entry_point: str) -> None:
    """Parametrized, so a failure names WHICH entry point leaked."""
    result = _import_in_child(entry_point)

    assert result.returncode == 0, (
        f"`import {entry_point}` requires reflexio_ext, so it raises ImportError on "
        f"a PyPI install.\n\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "IMPORTS_OK" in result.stdout


@pytest.mark.parametrize("module", _console_script_modules())
def test_each_installed_console_script_imports_without_enterprise(module: str) -> None:
    """The INSTALLED entry points, which are not the same modules as above.

    `_ENTRY_POINTS` is what a library consumer imports; this is what `pip install`
    puts on the PATH. A leak in one of these breaks the command outright, and the
    module list is read from `pyproject.toml` rather than restated here.
    """
    result = _import_in_child(module)

    assert result.returncode == 0, (
        f"the installed console script module {module!r} cannot be imported with "
        f"reflexio_ext absent, so `reflexio` is broken on an OSS install:\n"
        f"{result.stderr}"
    )


def test_the_blocker_and_not_the_venv_is_what_stops_the_import() -> None:
    """These tests must not rest on `reflexio_ext` happening to be uninstalled.

    `os-unit-tests` in the superproject's `ci-fast.yml` runs this suite with
    `working-directory: open_source/reflexio`, so `uv run` selects the OSS venv,
    where `reflexio_ext` is genuinely absent. A leak fails there with a plain
    `ModuleNotFoundError` whatever the finder does -- which means the finder could
    be broken and every test above would still pass in CI.

    So the assertion is not "the import failed", it is "the import failed FOR THE
    FINDER'S REASON". `(blocked)` appears only in the message the finder raises.

    Measured, both directions: run from the ENTERPRISE venv -- where
    `reflexio_ext` imports fine -- with a submodule leak in the tree, the
    entry-point tests above still fail and name `reflexio.server.api`. An earlier
    version of this test asserted the module was absent from the environment, which
    made the OSS suite red for anyone running it from that venv: a guard that fails
    on correct code, which is how a guard gets switched off.
    """
    blocked = _import_in_child("reflexio_ext")

    assert blocked.returncode != 0, (
        "the finder let `import reflexio_ext` through entirely"
    )
    assert "(blocked)" in blocked.stderr, (
        "`import reflexio_ext` failed, but not because the finder stopped it -- in "
        "the OSS venv it is simply not installed, so a broken finder would look "
        f"identical here. stderr={blocked.stderr!r}"
    )


def test_the_blocker_mechanism_intercepts_an_installed_module() -> None:
    """Anti-vacuity, aimed at a module that IS importable here.

    Aiming it at `reflexio_ext` proved nothing in the OSS venv: the import failed
    because the module is absent, not because the finder rejected it, so a typo in
    the finder would have passed. `httpx` is installed, so a failure here can only
    be the finder doing its job -- and the control proves it imports when unblocked.
    """
    control = _import_in_child("httpx", blocked="nothing_at_all")
    assert control.returncode == 0, (
        f"httpx is not importable here, so it cannot serve as the control:\n"
        f"{control.stderr}"
    )

    blocked = _import_in_child("httpx", blocked="httpx")
    assert blocked.returncode != 0, "the finder did not intercept an installed module"
    assert "blocked" in blocked.stderr


#: A stdlib package that does NOT eagerly import the submodule named beside it.
#: The choice is load-bearing and was measured: `import httpx` puts `httpx._client`
#: in `sys.modules` itself, so `import httpx._client` afterwards is a cache hit and
#: no finder is consulted at all. `json` does not import `json.tool`.
_LAZY_PARENT, _LAZY_CHILD = "json", "json.tool"


def _prefix_probe(finder: str) -> subprocess.CompletedProcess[str]:
    """Import `json.tool` with `json` ALREADY imported before `finder` goes in.

    Pre-importing the parent is what makes the prefix branch reachable: with it in
    `sys.modules`, `import json.tool` skips the parent entirely and the finder is
    consulted for the dotted name alone.
    """
    return _run_child(f"import {_LAZY_PARENT}\n{finder}\nimport {_LAZY_CHILD}\n")


def test_the_prefix_branch_is_what_blocks_a_submodule() -> None:
    """The `startswith` half of the finder, with the control that makes it mean something.

    An earlier version of this test blocked `httpx` and imported `httpx._client`,
    expecting that to exercise the prefix branch. It did not, for two independent
    reasons, and mutation is what exposed them -- narrowing the finder to
    `fullname == blocked` left all nine tests green:

    1. `import a.b` imports `a` first, so the EXACT-name branch already caught it.
    2. `httpx/__init__.py` imports `._client` itself, so after a pre-import of the
       parent the submodule is a `sys.modules` hit and NO finder runs.

    So the probe pre-imports a parent that stays lazy, and the narrowed finder is
    run as a control. Without the control this test would pass against a finder
    that rejects everything.
    """
    full = _blocker_for(_LAZY_PARENT)
    narrowed = full.replace(
        f'if fullname == "{_LAZY_PARENT}" or fullname.startswith("{_LAZY_PARENT}."):',
        f'if fullname == "{_LAZY_PARENT}":',
    )
    assert "startswith" not in narrowed, (
        "the narrowed finder was not produced -- the condition line moved, so this "
        "test would be comparing a finder against itself"
    )

    blocked = _prefix_probe(full)
    assert blocked.returncode != 0, (
        f"the prefix branch failed to block {_LAZY_CHILD} with {_LAZY_PARENT} "
        f"already imported -- the one case it exists for. stderr={blocked.stderr!r}"
    )

    allowed = _prefix_probe(narrowed)
    assert allowed.returncode == 0, (
        "the CONTROL failed: the exact-name-only finder also blocked "
        f"{_LAZY_CHILD}, so the assertion above is not evidence about the prefix "
        f"branch. stderr={allowed.stderr!r}"
    )


def test_the_in_process_shortcut_would_not_have_worked() -> None:
    """Why this file spawns a child, kept as a check rather than a comment.

    `reflexio.server` is already in this process's `sys.modules` -- pytest imported
    it long before now. Installing a blocker here and re-importing is a no-op: the
    cached module is returned and the blocker is never consulted. A version of this
    guard written that way would pass with a leak sitting in the tree -- whenever
    `reflexio_ext` is importable, which is every enterprise venv.
    """
    assert "reflexio.server" in sys.modules, (
        "reflexio.server is NOT already imported, so the premise of this file's "
        "child-interpreter design no longer holds -- re-check whether an in-process "
        "blocker would now be valid before simplifying"
    )
