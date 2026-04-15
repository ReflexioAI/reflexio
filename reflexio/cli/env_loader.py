"""Shared .env discovery, loading, and mutation utility.

Searches for .env in multiple locations. On first run, auto-creates
~/.reflexio/.env from the bundled .env.example template.
"""

from __future__ import annotations

import importlib.resources
import re
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

_USER_ENV_DIR = Path.home() / ".reflexio"
_USER_ENV_FILE = _USER_ENV_DIR / ".env"


def get_env_path() -> Path:
    """Return the canonical path to the user-level .env file.

    Returns:
        Path: ``~/.reflexio/.env``
    """
    return _USER_ENV_FILE


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
            (for importlib.resources). OS package uses "reflexio.data",
            enterprise uses "reflexio_ext.data".
        auto_generate_keys: Env var names to auto-generate as hex tokens
            (e.g., ["JWT_SECRET_KEY"]).

    Returns:
        Path to the loaded .env file, or None if no .env was found/created.
    """
    for env_path in _ENV_SEARCH_PATHS:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            sys.stdout.write(f"  Loaded env from: {env_path.resolve()}\n")
            sys.stdout.flush()
            # Auto-generate any missing secret keys into the existing .env
            _backfill_missing_keys(env_path, auto_generate_keys or [])
            return env_path

    # No .env found — auto-create from bundled template
    return _create_default_env(package_data_module, auto_generate_keys or [])


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
