"""`package_source_files` must exclude build output, not just look like it does.

Two source-scanning guards now depend on this helper to stay hermetic, so the
exclusion is load-bearing: if it silently stopped excluding, those guards would
go back to walking a 17,000-file virtualenv and failing with errors that name
the guard rather than the cause.
"""

from pathlib import Path

from reflexio.test_support.source_scan import package_source_files


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_yields_package_sources_and_skips_build_output(tmp_path: Path) -> None:
    wanted = {
        _write(tmp_path / "module.py"),
        _write(tmp_path / "server" / "service.py"),
        _write(tmp_path / "server" / "nested" / "deep.py"),
    }
    # The real shape that broke the aggregation guard: a gitignored plugin venv
    # carrying third-party sources far deeper than any first-party file.
    _write(
        tmp_path / "integrations" / "plugin" / ".venv" / "lib" / "sympy" / "polys.py"
    )
    _write(
        tmp_path
        / "integrations"
        / "plugin"
        / ".venv"
        / "lib"
        / "site-packages"
        / "j.py"
    )
    _write(tmp_path / "node_modules" / "pkg" / "shim.py")
    _write(tmp_path / "server" / "__pycache__" / "service.py")
    _write(tmp_path / "build" / "lib" / "copy.py")
    _write(tmp_path / ".tox" / "py312" / "thing.py")

    found = set(package_source_files(tmp_path))

    assert found == wanted


def test_is_sorted_so_guard_output_is_stable(tmp_path: Path) -> None:
    # Guards report offenders by path; unsorted output makes a failure message
    # differ run to run for the same defect.
    for name in ("z.py", "a.py", "m.py"):
        _write(tmp_path / name)

    found = list(package_source_files(tmp_path))

    assert found == sorted(found)


def test_a_dotfile_directory_at_the_root_is_skipped(tmp_path: Path) -> None:
    # `.`-prefixed exclusion is by DIRECTORY component, not by the file's own
    # name — a module legitimately named `_private.py` must still be yielded.
    _write(tmp_path / ".hidden" / "thing.py")
    kept = _write(tmp_path / "_private.py")

    assert set(package_source_files(tmp_path)) == {kept}
