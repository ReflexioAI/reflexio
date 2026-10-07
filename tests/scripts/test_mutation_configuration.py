"""The weekly engine must load actual source targets and serial, keyless pytest."""

from pathlib import Path


def test_mutmut_loads_current_configuration(monkeypatch):
    from mutmut.configuration import config, reset_config

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    reset_config()
    settings = config()
    reset_config()
    assert settings.source_paths
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
