"""First-run provisioning of user and mode env files.

Every test runs with a temp CWD, a temp user env dir and a temp template
package on ``sys.path``, so nothing reads or writes the developer's
``~/.reflexio``. The process environment is snapshotted and restored because
``load_dotenv`` writes into ``os.environ`` directly.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from reflexio.cli import env_loader


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Temp CWD + user dir; restore ``os.environ`` after the test."""
    snapshot = dict(os.environ)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    home = tmp_path / "home" / ".reflexio"
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", home)
    monkeypatch.setattr(env_loader, "_USER_ENV_FILE", home / ".env")
    monkeypatch.delenv("REFLEXIO_ENV_FILE", raising=False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    yield home
    os.environ.clear()
    os.environ.update(snapshot)


def _template_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]
) -> str:
    """Create an importable package holding env templates; return its name."""
    name = f"env_templates_{uuid.uuid4().hex}"
    package = tmp_path / "pkgs" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for filename, content in files.items():
        (package / filename).write_text(content)
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    return name


def test_set_env_var_replaces_first_of_several_commented_lines(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# KEY=one\nOTHER=1\n# KEY=two\n")
    env_loader.set_env_var(env, "KEY", "v")
    assert env.read_text().splitlines() == ['KEY="v"', "OTHER=1", "# KEY=two"]


def test_first_run_creates_user_env_from_package_template(
    isolated_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    package = _template_package(
        tmp_path,
        monkeypatch,
        {".env.example": "PROV_SECRET=\nPROV_PLAIN=template\n"},
    )
    created = env_loader.load_reflexio_env(
        package_data_module=package, auto_generate_keys=["PROV_SECRET"]
    )
    assert created is not None and created == isolated_env / ".env"
    content = created.read_text()
    secret = content.split("PROV_SECRET=", 1)[1].splitlines()[0]
    assert len(secret) == 64 and int(secret, 16) >= 0
    assert "PROV_PLAIN=template" in content
    if os.name != "nt":  # Windows chmod only toggles the read-only bit
        assert (created.stat().st_mode & 0o777) == 0o600
    assert os.environ["PROV_SECRET"] == secret
    assert env_loader.get_loaded_env_path() == created.resolve()
    out = capsys.readouterr().out
    assert f"Created directory: {isolated_env}" in out
    assert "Auto-generated: PROV_SECRET" in out


def test_cwd_template_wins_and_existing_dir_is_not_announced(
    isolated_env: Path, capsys: pytest.CaptureFixture[str]
):
    isolated_env.mkdir(parents=True)
    Path(".env.example").write_text("PROV_FROM_CWD=yes\n")
    created = env_loader.load_reflexio_env(package_data_module="no_such_pkg_xyz")
    assert created is not None and "PROV_FROM_CWD=yes" in created.read_text()
    out = capsys.readouterr().out
    assert "Created directory" not in out
    assert "Auto-generated" not in out


def test_first_run_without_any_template_warns_and_creates_nothing(
    isolated_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    # Point the editable-install fallback at a tree with no template.
    import reflexio

    fake_pkg = tmp_path / "fake_root" / "reflexio" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    monkeypatch.setattr(reflexio, "__file__", str(fake_pkg))
    assert env_loader.load_reflexio_env(package_data_module="no_such_pkg_xyz") is None
    assert not isolated_env.exists()
    assert "no .env.example template available" in capsys.readouterr().out


def test_existing_env_backfills_only_missing_keys(
    isolated_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    isolated_env.mkdir(parents=True)
    env = isolated_env / ".env"
    env.write_text("PROV_PRESENT=kept\n")
    monkeypatch.delenv("PROV_PRESENT", raising=False)
    monkeypatch.delenv("PROV_MISSING", raising=False)
    assert (
        env_loader.load_reflexio_env(
            auto_generate_keys=["PROV_PRESENT", "PROV_MISSING"]
        )
        == env
    )
    content = env.read_text()
    assert "PROV_PRESENT=kept" in content
    assert content.count("PROV_MISSING=") == 1
    assert len(os.environ["PROV_MISSING"]) == 64
    assert "Auto-generated missing keys: PROV_MISSING" in capsys.readouterr().out


def test_ensure_user_env_for_setup_loads_existing_and_backfills(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
):
    isolated_env.mkdir(parents=True)
    env = isolated_env / ".env"
    env.write_text("PROV_SETUP_VALUE=from-file\n")
    monkeypatch.delenv("PROV_SETUP_VALUE", raising=False)
    monkeypatch.delenv("PROV_SETUP_SECRET", raising=False)
    assert (
        env_loader.ensure_user_env_for_setup(auto_generate_keys=["PROV_SETUP_SECRET"])
        == env
    )
    assert os.environ["PROV_SETUP_VALUE"] == "from-file"
    assert "PROV_SETUP_SECRET=" in env.read_text()
    assert env_loader.get_loaded_env_path() == env.resolve()


def test_ensure_user_env_for_setup_creates_from_template(
    isolated_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = _template_package(
        tmp_path, monkeypatch, {".env.example": "PROV_CREATED=1\n"}
    )
    created = env_loader.ensure_user_env_for_setup(package_data_module=package)
    assert created is not None and created == isolated_env / ".env"
    assert created.read_text() == "PROV_CREATED=1\n"
    assert env_loader.get_loaded_env_path() == created.resolve()


def test_no_mode_falls_back_to_plain_loader(isolated_env: Path):
    Path(".env").write_text("PROV_PLAIN_FALLBACK=1\n")
    assert env_loader.load_reflexio_env_for_mode() == Path(".env")
    assert os.environ["PROV_PLAIN_FALLBACK"] == "1"
    assert env_loader.load_existing_reflexio_env_for_mode() == Path(".env")


def test_mode_first_run_creates_home_mode_file_from_package_template(
    isolated_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = _template_package(
        tmp_path, monkeypatch, {".env.covmode": "PROV_MODE_VALUE=bundled\n"}
    )
    monkeypatch.setenv("DEPLOYMENT_MODE", "covmode")
    monkeypatch.delenv("PROV_MODE_VALUE", raising=False)
    created = env_loader.load_reflexio_env_for_mode(package_data_module=package)
    target = isolated_env / ".env.covmode"
    assert created == target
    assert target.read_text() == "PROV_MODE_VALUE=bundled\n"
    if os.name != "nt":  # Windows chmod only toggles the read-only bit
        assert (target.stat().st_mode & 0o777) == 0o600
    assert os.environ["PROV_MODE_VALUE"] == "bundled"
    assert env_loader.get_loaded_env_path() == target.resolve()
    # The created home file is found on the next start instead of recreated.
    target.write_text("PROV_MODE_VALUE=edited\n")
    monkeypatch.delenv("PROV_MODE_VALUE")
    assert env_loader.load_reflexio_env_for_mode(package_data_module=package) == target
    assert os.environ["PROV_MODE_VALUE"] == "edited"


def test_mode_without_template_creates_nothing(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DEPLOYMENT_MODE", "nonexistentcovmode")
    assert (
        env_loader.load_reflexio_env_for_mode(package_data_module="no_such_pkg_xyz")
        is None
    )
    assert not isolated_env.exists()
    assert (
        env_loader.load_existing_reflexio_env_for_mode(
            package_data_module="no_such_pkg_xyz"
        )
        is None
    )


def test_mode_template_from_cwd_and_editable_root(
    isolated_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    Path(".env.cwdmode").write_text("A=cwd\n")
    assert env_loader._find_mode_template("cwdmode", "no_such_pkg_xyz") == "A=cwd\n"

    import reflexio

    root = tmp_path / "editable"
    (root / "reflexio").mkdir(parents=True)
    (root / ".env.rootmode").write_text("A=root\n")
    monkeypatch.setattr(reflexio, "__file__", str(root / "reflexio" / "__init__.py"))
    assert env_loader._find_mode_template("rootmode", "no_such_pkg_xyz") == "A=root\n"
    assert env_loader._find_mode_template("othermode", "no_such_pkg_xyz") is None


def test_dotenv_disabled_files_supply_nothing(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
):
    env = Path(".env")
    env.write_text("PROV_DISABLED_VALUE=1\n")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    monkeypatch.delenv("PROV_DISABLED_VALUE", raising=False)
    assert env_loader.load_reflexio_env() == env
    assert "PROV_DISABLED_VALUE" not in os.environ
