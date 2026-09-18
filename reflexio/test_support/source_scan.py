"""Enumerate the package's OWN Python sources for source-scanning guards.

Several guards walk `reflexio/**/*.py` looking for a call pattern. A bare
`rglob("*.py")` from the package root does not mean "this package's source": it
means "every .py file currently on disk under this directory", and that includes
gitignored build output.

`reflexio/integrations/openclaw/plugin/` grows a `.venv` the moment anyone builds
that plugin. It is gitignored, so CI never sees it and the guards pass there --
but on a developer machine it turns a ~200-file scan into ~17,400, dragging in
sympy, torch and litellm. Two ways that bites, both observed:

- `test_lineage_silent_bulk_delete_callers_only_remove_ineligible_rows` uses a
  recursive `ast.NodeVisitor`. `sympy/polys/numberfields/resolvent_lookup.py`
  nests 568 levels deep, and at ~2 interpreter frames per AST level that clears
  the 1000-frame recursion limit: `RecursionError`, in a test about playbook
  deletion.
- `joblib/test/test_func_inspect_special_encoding.py` is deliberately not UTF-8,
  so any guard calling `read_text(encoding="utf-8")` raises `UnicodeDecodeError`.

Both failures name the scanning guard rather than the venv, so they read as
defects in code that is fine. Scanning the package's own sources removes the
whole class.
"""

from collections.abc import Iterator
from pathlib import Path

#: Directory names that are never this package's source. `.`-prefixed entries
#: (`.venv`, `.tox`, `.mypy_cache`, `.git`) are excluded wholesale; these are the
#: ones that carry no leading dot.
_NON_SOURCE_DIRS = frozenset(
    {"site-packages", "node_modules", "__pycache__", "build", "dist"}
)


def _is_source_dir(name: str) -> bool:
    return not name.startswith(".") and name not in _NON_SOURCE_DIRS


def package_source_files(root: Path) -> Iterator[Path]:
    """Yield the package's own ``.py`` files beneath ``root``, sorted.

    Excluded directories are PRUNED during the walk rather than filtered out
    afterwards. Filtering `rglob` still enumerates and sorts every path in the
    virtualenv first, which is most of the cost this exists to avoid — the
    guards would stop parsing 17,000 third-party files but still pay to walk
    them. `Path.walk` (3.12+) lets the excluded names be dropped from
    `dirnames` in place, so traversal never descends into them at all.

    Args:
        root (Path): Directory to scan, normally the package root.

    Yields:
        Path: Each first-party Python source file, in sorted order.
    """
    found: list[Path] = []
    for directory, dirnames, filenames in root.walk():
        # In-place, because `Path.walk` reads this list back to decide where to
        # descend. Rebinding the name would prune nothing.
        dirnames[:] = [name for name in dirnames if _is_source_dir(name)]
        found.extend(directory / name for name in filenames if name.endswith(".py"))
    # Sorted at the end, not per-directory: `walk` is top-down, so yielding as
    # we go would put a root-level `z.py` before `a/b.py`. Guards report
    # offenders by path, and an order that shifts with directory layout makes
    # the same defect print differently. The list is first-party only by now
    # (~400 entries), so this costs nothing.
    yield from sorted(found)
