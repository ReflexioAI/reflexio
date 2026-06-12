"""Shared .env discovery, loading, and mutation utility.

Searches for .env in multiple locations. On first run, auto-creates
~/.reflexio/.env from the bundled .env.example template.
"""

from __future__ import annotations

import importlib.resources
import logging
import re
import secrets
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from .paths import reflexio_home

_USER_ENV_DIR = reflexio_home()
_USER_ENV_FILE = _USER_ENV_DIR / ".env"


# Path to the .env file that load_reflexio_env last resolved — None until
# load_reflexio_env runs for the first time. Exposed via get_loaded_env_path
# so the startup banner can show the operator exactly which dotenv was
# picked (./.env vs ~/.reflexio/.env vs auto-created).
_loaded_env_path: Path | None = None


def get_env_path() -> Path:
    """Return the canonical path to the user-level .env file.

    Returns:
        Path: ``~/.reflexio/.env``
    """
    return _USER_ENV_FILE


def get_loaded_env_path() -> Path | None:
    """Return the .env path that the most recent ``load_reflexio_env`` call
    resolved, or None if the loader hasn't run yet.

    Used by the startup banner so operators can see at a glance which
    dotenv file was actually consumed (``./.env`` wins over
    ``~/.reflexio/.env`` when both exist).
    """
    return _loaded_env_path


def set_env_var(env_path: Path, key: str, value: str) -> None:
    """Write or update an environment variable in a .env file.

    If the key already exists (active or commented-out), the line is replaced
    in-place. Active (uncommented) lines are prioritized over commented ones.
    Values are always wrapped in double quotes for safe parsing.

    Args:
        env_path (Path): Path to the .env file.
        key (str): Environment variable name.
        value (str): Environment variable value.
    """
    content = env_path.read_text() if env_path.exists() else ""
    lines = content.splitlines()
    pattern = re.compile(rf"^#?\s*{re.escape(key)}=")
    active_idx: int | None = None
    commented_idx: int | None = None
    for i, line in enumerate(lines):
        if not pattern.match(line):
            continue
        if line.lstrip().startswith("#"):
            if commented_idx is None:
                commented_idx = i
        else:
            active_idx = i
            break
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    replacement = f'{key}="{escaped}"'
    target = active_idx if active_idx is not None else commented_idx
    if target is not None:
        lines[target] = replacement
    else:
        lines.append(replacement)
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)


_ENV_SEARCH_PATHS = [
    Path(".env"),  # 1. Current directory (local dev / project-level)
    _USER_ENV_FILE,  # 2. User home default (~/.reflexio/.env)
]


def load_reflexio_env(
    *,
    package_data_module: str = "reflexio.data",
    auto_generate_keys: list[str] | None = None,
) -> Path | None:
    """Load .env from the first location found, or auto-create on first run.

    Search order:
        1. ./.env (current directory)
        2. ~/.reflexio/.env (user home)
        3. Auto-create from bundled .env.example template

    Args:
        package_data_module: Module containing bundled .env.example
            (for importlib.resources). The OS package passes "reflexio.data";
            a downstream build may pass its own data module.
        auto_generate_keys: Env var names to auto-generate as hex tokens
            (e.g., ["JWT_SECRET_KEY"]).

    Returns:
        Path to the loaded .env file, or None if no .env was found/created.
    """
    global _loaded_env_path
    for env_path in _ENV_SEARCH_PATHS:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            resolved = env_path.resolve()
            _logger.debug("Loaded env from: %s", resolved)
            _loaded_env_path = resolved
            # Auto-generate any missing secret keys into the existing .env
            _backfill_missing_keys(env_path, auto_generate_keys or [])
            return env_path

    # No .env found — auto-create from bundled template
    created = _create_default_env(package_data_module, auto_generate_keys or [])
    if created is not None:
        _loaded_env_path = created.resolve()
    return created


def resolve_mode(cli_mode: str | None = None) -> str | None:
    """Resolve the active deployment mode for mode-aware env loading.

    Selection precedence:
        1. Explicit ``--mode`` flag (``cli_mode``)
        2. ``DEPLOYMENT_MODE`` environment variable
        3. None (caller falls back to the plain ``.env`` loader)

    The resolved mode is spliced into ``.env.<mode>`` file paths, so it is
    validated against a safe slug pattern: an empty/whitespace value resolves to
    None (fall back to the plain loader), and anything containing path
    characters (``/``, ``..``) or other non-slug characters raises ValueError
    rather than redirecting reads/writes/chmods outside ``~/.reflexio/``.

    Args:
        cli_mode: Mode passed explicitly via a CLI flag, if any.

    Returns:
        The normalized (stripped, lowercased) mode string, or None when no
        mode is selected.

    Raises:
        ValueError: If the selected mode is not a safe slug.
    """
    import os

    raw_mode = cli_mode if cli_mode is not None else os.environ.get("DEPLOYMENT_MODE")
    if raw_mode is None:
        return None
    mode = raw_mode.strip().lower()
    if not mode:
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", mode):
        raise ValueError(f"Invalid deployment mode: {raw_mode!r}")
    return mode


def load_reflexio_env_for_mode(
    *,
    cli_mode: str | None = None,
    package_data_module: str = "reflexio.data",
    auto_generate_keys: list[str] | None = None,
) -> Path | None:
    """Load ``.env.<mode>`` presets plus optional home secrets, mode-aware.

    Loads the committed ``.env.<mode>`` presets (from CWD or the user home dir)
    together with the optional gitignored ``~/.reflexio/.env.<mode>`` secrets
    file. Both are loaded with ``override=False`` so the process environment
    (task-def / shell ``export``) always wins, and home secrets take precedence
    over committed presets for any disjoint keys.

    When no mode is resolved, this falls back to :func:`load_reflexio_env`.

    Args:
        cli_mode: Mode passed explicitly via a CLI flag, if any.
        package_data_module: Module containing the bundled ``.env.<mode>``
            template (for ``importlib.resources``).
        auto_generate_keys: Env var names to auto-generate as hex tokens
            (e.g., ``["JWT_SECRET_KEY"]``) when missing. Generated secrets are
            written ONLY to the gitignored ``~/.reflexio/.env.<mode>`` home
            file — never the committed ``./.env.<mode>`` presets. Load-only
            callers (the uvicorn / migration entrypoints) pass ``None``.

    Returns:
        Path to the loaded mode env file, or None if nothing was found/created.
    """
    global _loaded_env_path
    mode = resolve_mode(cli_mode)
    if mode is None:
        return load_reflexio_env(
            package_data_module=package_data_module,
            auto_generate_keys=auto_generate_keys,
        )

    # Optional gitignored home secrets, loaded first so committed presets and
    # process env both still win where they define the same keys.
    home_secrets = _USER_ENV_DIR / f".env.{mode}"
    if home_secrets.exists():
        load_dotenv(dotenv_path=home_secrets, override=False)

    loaded: Path | None = None
    for env_path in (Path(f".env.{mode}"), home_secrets):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            _loaded_env_path = env_path.resolve()
            _logger.debug("Loaded mode env from: %s (mode=%s)", _loaded_env_path, mode)
            loaded = env_path
            break

    if loaded is None:
        loaded = _create_default_env_for_mode(mode, package_data_module)

    # Auto-generate any missing secret keys into the gitignored home file ONLY,
    # never the committed ./.env.<mode> presets file.
    if auto_generate_keys:
        _USER_ENV_DIR.mkdir(parents=True, exist_ok=True)
        _backfill_missing_keys(home_secrets, auto_generate_keys)

    return loaded


def _create_default_env_for_mode(mode: str, package_data_module: str) -> Path | None:
    """Create ``~/.reflexio/.env.<mode>`` from the bundled mode template.

    Args:
        mode: The resolved deployment mode (e.g. ``"platform"``).
        package_data_module: Module path for finding the ``.env.<mode>``
            template.

    Returns:
        Path to the newly created mode env file, or None if no template found.
    """
    global _loaded_env_path
    content = _find_mode_template(mode, package_data_module)
    if content is None:
        return None
    _USER_ENV_DIR.mkdir(parents=True, exist_ok=True)
    target = _USER_ENV_DIR / f".env.{mode}"
    target.write_text(content)
    target.chmod(0o600)
    load_dotenv(dotenv_path=target, override=False)
    _loaded_env_path = target.resolve()
    return target


def _find_mode_template(mode: str, package_data_module: str) -> str | None:
    """Find ``.env.<mode>`` template content from CWD or package data.

    Args:
        mode: The resolved deployment mode (e.g. ``"platform"``).
        package_data_module: Dotted module path for importlib.resources lookup.

    Returns:
        The template content as a string, or None if not found anywhere.
    """
    # 1. Current directory (local dev checkout)
    local = Path(f".env.{mode}")
    if local.exists():
        return local.read_text()

    # 2. Package data (installed package)
    try:
        ref = importlib.resources.files(package_data_module).joinpath(f".env.{mode}")
        return ref.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    # 3. Editable install: .env.<mode> lives at project root, two levels up.
    try:
        import reflexio as _pkg

        project_root = Path(_pkg.__file__).resolve().parent.parent
        candidate = project_root / f".env.{mode}"
        if candidate.is_file():
            return candidate.read_text()
    except Exception:  # noqa: BLE001, S110
        pass

    return None


def ensure_user_env_for_setup(
    *,
    package_data_module: str = "reflexio.data",
    auto_generate_keys: list[str] | None = None,
) -> Path | None:
    """Return ``~/.reflexio/.env``, creating it from the template if missing.

    Distinct from :func:`load_reflexio_env`: that function honors a CWD-local
    ``./.env`` (intentional for ``services start``, where a project-level
    override should take precedence). For ``setup init`` we never want to
    write to whatever ``.env`` happens to be in the user's CWD — they may be
    running from a worktree or a project root with an unrelated ``.env``,
    and the resulting writes pollute that file and break the documented
    ``~/.reflexio/.env`` invariant.

    Args:
        package_data_module: Module containing the bundled ``.env.example``
            template (for ``importlib.resources``). The OS package passes
            ``reflexio.data``; a downstream build may pass its own data module.
        auto_generate_keys: Env var names to auto-fill with random hex
            tokens when creating from the template.

    Returns:
        Path to ``~/.reflexio/.env`` (existing or newly created), or
        ``None`` if no template could be found.
    """
    global _loaded_env_path
    if _USER_ENV_FILE.exists():
        load_dotenv(dotenv_path=_USER_ENV_FILE)
        resolved = _USER_ENV_FILE.resolve()
        _logger.debug("Loaded user env from: %s", resolved)
        _loaded_env_path = resolved
        _backfill_missing_keys(_USER_ENV_FILE, auto_generate_keys or [])
        return _USER_ENV_FILE
    created = _create_default_env(package_data_module, auto_generate_keys or [])
    if created is not None:
        _loaded_env_path = created.resolve()
    return created


def _backfill_missing_keys(env_path: Path, keys: list[str]) -> None:
    """Generate and write any missing secret keys into an existing .env file.

    Called when ``load_reflexio_env`` finds a pre-existing .env (e.g. created
    by ``setup init``) that may be missing keys that ``services start``
    requires (like JWT_SECRET_KEY).

    Args:
        env_path: Path to the existing .env file.
        keys: Env var names to check/generate.
    """
    import os

    generated: list[str] = []
    for key in keys:
        if os.environ.get(key):
            continue
        token = secrets.token_hex(32)
        set_env_var(env_path, key, token)
        os.environ[key] = token
        generated.append(key)
    if generated:
        sys.stdout.write(f"  Auto-generated missing keys: {', '.join(generated)}\n")
        sys.stdout.flush()


def _find_env_example(package_data_module: str) -> str | None:
    """Find .env.example content from CWD or package data.

    Args:
        package_data_module: Dotted module path for importlib.resources lookup.

    Returns:
        The template content as a string, or None if not found anywhere.
    """
    # 1. Current directory (local dev checkout)
    local = Path(".env.example")
    if local.exists():
        return local.read_text()

    # 2. Package data (installed package)
    try:
        ref = importlib.resources.files(package_data_module).joinpath(".env.example")
        return ref.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):  # fmt: skip
        pass

    # 3. Editable install: .env.example lives at project root, two levels above reflexio/
    try:
        import reflexio as _pkg

        project_root = Path(_pkg.__file__).resolve().parent.parent
        candidate = project_root / ".env.example"
        if candidate.is_file():
            return candidate.read_text()
    except Exception:  # noqa: BLE001, S110
        pass

    return None


def _create_default_env(
    package_data_module: str,
    auto_generate_keys: list[str],
) -> Path | None:
    """Create ~/.reflexio/.env from .env.example with auto-generated secrets.

    Args:
        package_data_module: Module path for finding the .env.example template.
        auto_generate_keys: Env var names to auto-fill with random hex tokens.

    Returns:
        Path to the newly created .env file, or None if template not found.
    """
    content = _find_env_example(package_data_module)
    if content is None:
        sys.stdout.write(
            "Warning: no .env file found and no .env.example template available.\n"
            "  Set required environment variables manually.\n"
        )
        sys.stdout.flush()
        return None

    created_dir = not _USER_ENV_DIR.exists()
    _USER_ENV_DIR.mkdir(parents=True, exist_ok=True)
    if created_dir:
        sys.stdout.write(f"Created directory: {_USER_ENV_DIR}\n")

    # Auto-generate secret keys
    for key in auto_generate_keys:
        token = secrets.token_hex(32)
        content = re.sub(
            rf"^{re.escape(key)}=.*$",
            f"{key}={token}",
            content,
            count=1,
            flags=re.MULTILINE,
        )

    _USER_ENV_FILE.write_text(content)
    _USER_ENV_FILE.chmod(0o600)
    load_dotenv(dotenv_path=_USER_ENV_FILE)

    sys.stdout.write(f"Created env file: {_USER_ENV_FILE}\n")
    if auto_generate_keys:
        sys.stdout.write(f"  Auto-generated: {', '.join(auto_generate_keys)}\n")
    sys.stdout.write(f"  Edit {_USER_ENV_FILE} to add your API keys.\n\n")
    sys.stdout.flush()
    return _USER_ENV_FILE
