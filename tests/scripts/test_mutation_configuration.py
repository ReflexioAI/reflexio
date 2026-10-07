"""The weekly engine must load actual source targets and serial, keyless pytest."""

from pathlib import Path


def test_mutmut_loads_current_configuration(monkeypatch):
    from mutmut.__main__ import load_config

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    config = load_config()
    assert config.paths_to_mutate
    assert all(path.is_file() for path in config.paths_to_mutate)
    assert config.tests_dir == ["tests/"]
    assert config.pytest_add_cli_args[:2] == ["-o", "addopts="]
    assert "not requires_credentials" in " ".join(config.pytest_add_cli_args)
    assert {Path("reflexio/"), Path("skills/"), Path("docs/lib/methods/")} <= set(
        config.also_copy
    )
    assert all(
        path.exists()
        for path in (Path("reflexio/"), Path("skills/"), Path("docs/lib/methods/"))
    )
