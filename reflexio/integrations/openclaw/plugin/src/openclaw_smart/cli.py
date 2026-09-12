"""User-facing CLI for openclaw-smart.

Exposes the following subcommands:

- ``show``: print current project-specific skills and project preferences
  (as markdown).
- ``learn``: publish unpublished interactions and force reflexio extraction
  now over the active session buffer.
- ``restart``: stop and restart the reflexio backend + dashboard services
  so local edits take effect without restarting openClaw.
- ``clear-all``: delete all local reflexio data and openclaw-smart session
  buffers.

Installation/uninstallation is intentionally NOT exposed here — those flows
live in ``reflexio setup openclaw`` (parent reflexio CLI) and are managed by
the openClaw plugin system directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openclaw_smart import context_format, ids, publish, state
from openclaw_smart.reflexio_adapter import Adapter

_REFLEXIO_ENV_PATH = Path.home() / ".reflexio" / ".env"
_REFLEXIO_UNREACHABLE_MSG = (
    "Failed to reach reflexio. Check ~/.openclaw-smart/backend.log "
    "or restart openClaw.\n"
)

# plugin/src/openclaw_smart/cli.py -> parents[2] is plugin/
_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parents[2]
_SCRIPTS_DIR = _PLUGIN_ROOT / "scripts"
_DASHBOARD_DIR = _PLUGIN_ROOT / "dashboard"
_BACKEND_SCRIPT = _SCRIPTS_DIR / "backend-service.sh"
_DASHBOARD_SCRIPT = _SCRIPTS_DIR / "dashboard-service.sh"
_REFLEXIO_DIR = Path.home() / ".reflexio"
_DEFAULT_STORAGE_ROOT = _REFLEXIO_DIR / "data"
_REFLEXIO_CONFIG_PATH = _REFLEXIO_DIR / "configs" / "config_self-host-org.json"
_LOCAL_STORAGE_ENV = "LOCAL_STORAGE_PATH"
_DEFAULT_ORG_ID_ENV = "REFLEXIO_DEFAULT_ORG_ID"

# Mirrors reflexio.cli.bootstrap_config._DEFAULT_ORG_ID.
_DEFAULT_ORG_ID = "self-host-org"

# Mirrors sqlite_storage._dataset_path. Duplicated rather than imported: that
# module is cheap on its own, but reaching it executes the storage package's
# __init__ and costs ~1.7s, and `openclaw-smart-hook` runs per session event.
# `test_derived_filename_matches_the_canonical_resolver` pins the two together.
_LEGACY_DB_FILENAME = "reflexio.db"
_IDENTITY_CLAIM_TABLE = "_dataset_identity"

# Files SQLite keeps beside the database it is given.
_SQLITE_SIDECAR_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _latest_session_id() -> str | None:
    """Most-recently-modified session JSONL in the state dir. None if none exist."""
    root = state.state_dir()
    if not root.is_dir():
        return None
    files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0].stem


def cmd_show(args: argparse.Namespace) -> int:
    """Print this project's skills + preferences, plus globally-shared skills.

    Args:
        args (argparse.Namespace): Parsed CLI args. Honors ``args.project``
            (override project id for the project-scoped fetches).

    Returns:
        int: 0 on success.
    """
    project_id = args.project or ids.resolve_project_id()
    adapter = Adapter()
    user_playbooks, agent_playbooks, profiles = adapter.fetch_all(
        project_id=project_id,
        user_playbook_top_k=3,
        agent_playbook_top_k=3,
        profile_top_k=3,
    )
    md = context_format.render(
        project_id=project_id,
        user_playbooks=user_playbooks,
        agent_playbooks=agent_playbooks,
        profiles=profiles,
    )
    sys.stdout.write(
        md or f"_No skills or preferences yet for project `{project_id}`._\n"
    )
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Force reflexio extraction over the active session's interactions.

    Args:
        args (argparse.Namespace): Parsed CLI args. Honors ``args.session``
            (defaults to most-recent), ``args.project`` (defaults to
            ``ids.resolve_project_id()``), and ``args.note`` (free-form
            text appended as a User turn before publish).

    Returns:
        int: 0 on success or no-op, 1 if reflexio is unreachable.
    """
    session_id = args.session or _latest_session_id()
    if not session_id:
        sys.stdout.write("No active openclaw-smart session buffer found.\n")
        return 0
    project_id = args.project or ids.resolve_project_id()

    note = (args.note or "").strip()
    if note:
        state.append(
            session_id,
            {
                "ts": int(time.time()),
                "role": "User",
                "content": note,
                "user_id": project_id,
            },
        )

    status, count = publish.publish_unpublished(
        session_id=session_id,
        project_id=project_id,
        force_extraction=True,
        skip_aggregation=False,
    )
    if status == "ok":
        suffix = " (including your note)" if note else ""
        sys.stdout.write(
            f"Forced extraction on session `{session_id}` over {count} interactions{suffix}.\n"
        )
        return 0
    if status == "nothing":
        sys.stdout.write(f"No unpublished interactions on session `{session_id}`.\n")
        return 0
    sys.stdout.write(_REFLEXIO_UNREACHABLE_MSG)
    return 1


def _run_service(script: Path, subcmd: str) -> int:
    """Invoke a service script (``backend-service.sh`` / ``dashboard-service.sh``).

    Args:
        script (Path): Absolute path to the service shell script.
        subcmd (str): Subcommand to pass (``start``, ``stop``, ``status``).

    Returns:
        int: The script's exit code, or 1 if the script is missing.
    """
    if not script.exists():
        sys.stderr.write(f"error: {script} not found\n")
        return 1
    try:
        subprocess.run([str(script), subcmd], check=True)
        return 0
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    except OSError as exc:
        # Script lost executable bit, permission denied, missing interpreter, etc.
        sys.stderr.write(f"error: failed to execute {script}: {exc}\n")
        return 1


def _service_status(script: Path, wait_ready_s: float = 3.0) -> str:
    """Return the one-line status string for a service script."""
    if not script.exists():
        return "script missing"
    deadline = time.monotonic() + wait_ready_s
    while True:
        try:
            result = subprocess.run(
                [str(script), "status"], capture_output=True, text=True, check=False
            )
        except OSError as exc:
            return f"script error: {exc}"
        status = result.stdout.strip() or "unknown"
        if status != "not running" or time.monotonic() >= deadline:
            return status
        time.sleep(0.2)


def _is_running_status(status: str) -> bool:
    return status.startswith("running on ")


def cmd_restart(args: argparse.Namespace) -> int:
    """Restart the reflexio backend and openclaw-smart dashboard services.

    Args:
        args (argparse.Namespace): Parsed CLI args. Honors
            ``args.skip_backend``, ``args.skip_dashboard``, and
            ``args.no_rebuild``.

    Returns:
        int: 0 on success, non-zero if the dashboard rebuild fails or
            either service's ``start`` subcommand exits non-zero.
    """
    do_backend = not args.skip_backend
    do_dashboard = not args.skip_dashboard

    if not (do_backend or do_dashboard):
        sys.stdout.write("Nothing to restart (both services skipped).\n")
        return 0

    if do_backend:
        sys.stdout.write("Stopping reflexio backend…\n")
        _run_service(_BACKEND_SCRIPT, "stop")
    if do_dashboard:
        sys.stdout.write("Stopping dashboard…\n")
        _run_service(_DASHBOARD_SCRIPT, "stop")

    if do_dashboard and not args.no_rebuild and _DASHBOARD_DIR.is_dir():
        rc = _rebuild_dashboard()
        if rc != 0:
            if do_backend:
                _run_service(_BACKEND_SCRIPT, "start")
            return rc

    start_rc = 0
    if do_backend:
        sys.stdout.write("Starting reflexio backend…\n")
        rc = _run_service(_BACKEND_SCRIPT, "start")
        if rc != 0:
            sys.stderr.write(f"error: reflexio backend failed to start (exit {rc})\n")
            start_rc = rc
    if do_dashboard:
        sys.stdout.write("Starting dashboard…\n")
        rc = _run_service(_DASHBOARD_SCRIPT, "start")
        if rc != 0:
            sys.stderr.write(f"error: dashboard failed to start (exit {rc})\n")
            start_rc = start_rc or rc

    sys.stdout.write("\n")
    if do_backend:
        sys.stdout.write(f"reflexio backend: {_service_status(_BACKEND_SCRIPT)}\n")
    if do_dashboard:
        sys.stdout.write(f"dashboard: {_service_status(_DASHBOARD_SCRIPT)}\n")
    return start_rc


def _rebuild_dashboard() -> int:
    """Rebuild the dashboard Next.js bundle, installing deps if needed."""
    if not shutil.which("npm"):
        sys.stderr.write("warning: npm not on PATH; serving previous build\n")
        return 0
    next_bin = _DASHBOARD_DIR / "node_modules" / ".bin" / "next"
    if not next_bin.exists():
        install_cmd = (
            ["npm", "ci", "--no-audit", "--no-fund"]
            if (_DASHBOARD_DIR / "package-lock.json").exists()
            else ["npm", "install", "--no-audit", "--no-fund"]
        )
        sys.stdout.write(
            f"Installing dashboard dependencies ({' '.join(install_cmd)})…\n"
        )
        try:
            subprocess.run(install_cmd, cwd=_DASHBOARD_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"error: dashboard install failed (exit {exc.returncode})\n"
            )
            return exc.returncode or 1
    sys.stdout.write("Rebuilding dashboard…\n")
    try:
        subprocess.run(["npm", "run", "build"], cwd=_DASHBOARD_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"error: dashboard build failed (exit {exc.returncode})\n")
        return exc.returncode or 1
    return 0


class _ClearAllError(RuntimeError):
    """Raised when a destructive clear-all target is unsafe or unsupported."""


@dataclass(frozen=True)
class _ClearAllTarget:
    """One filesystem target removed by ``clear-all``."""

    path: Path
    kind: str
    label: str


def _read_dotenv_value(env_path: Path, key: str) -> str | None:
    """Read one simple KEY=VALUE binding from a dotenv file."""
    if not env_path.is_file():
        return None
    try:
        lines = env_path.read_text().splitlines()
    except OSError:
        return None
    prefix = f"{key}="
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value.strip()
    return None


def _resolve_absolute_path(raw_path: str | Path, *, source: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise _ClearAllError(f"{source} must be an absolute path: {raw_path}")
    return path


def _effective_storage_root() -> Path:
    raw = os.environ.get(_LOCAL_STORAGE_ENV, "").strip()
    if not raw:
        raw = _read_dotenv_value(_REFLEXIO_ENV_PATH, _LOCAL_STORAGE_ENV) or ""
    return _resolve_absolute_path(
        raw or _DEFAULT_STORAGE_ROOT, source=_LOCAL_STORAGE_ENV
    )


def _load_reflexio_config() -> dict[str, object] | None:
    if not _REFLEXIO_CONFIG_PATH.is_file():
        return None
    try:
        loaded = json.loads(_REFLEXIO_CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _ClearAllError(
            f"could not read reflexio config {_REFLEXIO_CONFIG_PATH}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise _ClearAllError(
            f"reflexio config {_REFLEXIO_CONFIG_PATH} is not a JSON object"
        )
    return loaded


def _storage_config_kind(storage_config: dict[str, object]) -> str:
    if storage_config.get("managed_by") == "platform":
        return "remote"
    explicit_type = str(
        storage_config.get("type") or storage_config.get("storage_type") or ""
    ).lower()
    if explicit_type in {"supabase", "postgres"}:
        return "remote"
    if explicit_type == "sqlite":
        return "sqlite"
    if (
        "url" in storage_config
        and "key" in storage_config
        and "db_url" in storage_config
    ):
        return "remote"
    if "db_url" in storage_config:
        return "remote"
    if "db_path" in storage_config or not storage_config:
        return "sqlite"
    # Reached by any shape this build cannot interpret -- including a config
    # left over from the removed disk backend (#98). `validate_stored_config`
    # rejects those outright, so the server cannot load such an org either: we
    # do not know what storage it uses, and guessing is not an option for a
    # destructive command.
    raise _ClearAllError(
        "unsupported reflexio storage_config shape; refusing to delete local data"
    )


def _dangerous_clear_all_paths() -> set[Path]:
    home = Path.home().resolve(strict=False)
    reflexio_dir = _resolve_absolute_path(_REFLEXIO_DIR, source="reflexio dir")
    return {
        Path("/").resolve(strict=False),
        home,
        reflexio_dir.resolve(strict=False),
        _resolve_absolute_path(
            reflexio_dir / "configs", source="reflexio configs"
        ).resolve(strict=False),
        _PLUGIN_ROOT.resolve(strict=False),
        _PLUGIN_ROOT.parent.resolve(strict=False),
    }


def _validate_deletion_target(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise _ClearAllError(f"refusing to delete symlink target: {path}")
    resolved = path.resolve(strict=False)
    if resolved in _dangerous_clear_all_paths():
        raise _ClearAllError(f"refusing to delete dangerous path: {resolved}")


def _derive_db_filename(org_id: str) -> str:
    """Return the database filename *org_id* owns.

    Args:
        org_id (str): The dataset identity.

    Returns:
        str: The filename, e.g. ``reflexio_self-host-org.db``.
    """
    return f"reflexio_{org_id}.db"


def _effective_org_id() -> str:
    """Resolve the dataset identity this installation reads and writes.

    Same precedence the server and ``reset_db.py`` use, so ``clear-all`` clears
    the database the backend would actually open: ``REFLEXIO_DEFAULT_ORG_ID``
    from the environment, then from ``~/.reflexio/.env``, then the default.

    Returns:
        str: The dataset identity.
    """
    raw = os.environ.get(_DEFAULT_ORG_ID_ENV, "").strip()
    if not raw:
        raw = _read_dotenv_value(_REFLEXIO_ENV_PATH, _DEFAULT_ORG_ID_ENV) or ""
    return raw.strip() or _DEFAULT_ORG_ID


def _claimed_identity(path: Path) -> str | None:
    """Return the identity that claims *path*, if it records one.

    Strictly read-only -- opened ``mode=ro`` so inspecting a database never
    creates one, and never takes the write lock a running backend may hold.

    Args:
        path (Path): The database to inspect.

    Returns:
        str | None: The claiming identity, or ``None`` when the file is absent,
        unreadable, or carries no claim.
    """
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            f"SELECT org_id FROM {_IDENTITY_CLAIM_TABLE} WHERE k = 1"  # noqa: S608
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else None


def _sqlite_artifact_targets(db_path: Path, label: str) -> list[_ClearAllTarget]:
    """The database plus the sidecars SQLite keeps beside it."""
    return [
        _ClearAllTarget(Path(f"{db_path}{suffix}"), "file", label)
        for suffix in _SQLITE_SIDECAR_SUFFIXES
    ]


def _identity_owned_targets(root: Path, org_id: str) -> list[_ClearAllTarget]:
    """Targets under *root* that belong to *org_id*, and nothing else.

    The storage root is shared: after dataset isolation it holds one
    ``reflexio_<org>.db`` per identity, alongside artifacts this plugin does not
    own -- the enterprise ``sql_app.db``, for one. ``derive_db_path`` documents
    those siblings as untouched, so enumerate what we own instead of deleting
    the directory that contains them.

    Args:
        root (Path): The storage root.
        org_id (str): The dataset identity being cleared.

    Returns:
        list[_ClearAllTarget]: Targets owned by *org_id*.
    """
    if not root.is_dir():
        return []

    targets = _sqlite_artifact_targets(
        root / _derive_db_filename(org_id), "this dataset's SQLite data"
    )

    # A pre-isolation install still reads `reflexio.db`, adopted in place by its
    # first claimant. Clear it only when it is ours -- or when nobody has
    # claimed it yet, in which case we are the installation that would adopt it.
    legacy = root / _LEGACY_DB_FILENAME
    if legacy.is_file():
        owner = _claimed_identity(legacy)
        if owner in (None, org_id):
            targets.extend(_sqlite_artifact_targets(legacy, "legacy SQLite data"))

    return targets


def _resolve_clear_all_targets() -> list[_ClearAllTarget]:
    """Resolve exactly what ``clear-all`` may delete.

    Read-only: resolution never creates the root, a database, or an identity
    claim. (``resolve_sqlite_db_path`` upstream deliberately does all three, so
    it is not reusable here -- it would create a database in order to delete
    one.)

    Returns:
        list[_ClearAllTarget]: Deduplicated, validated targets.

    Raises:
        _ClearAllError: If storage is remote, misconfigured, or a target is unsafe.
    """
    root = _effective_storage_root()
    targets = _identity_owned_targets(root, _effective_org_id())

    config = _load_reflexio_config()
    storage_config = config.get("storage_config") if config else None
    if storage_config is not None:
        if not isinstance(storage_config, dict):
            raise _ClearAllError("reflexio storage_config is not a JSON object")
        kind = _storage_config_kind(storage_config)
        if kind == "remote":
            raise _ClearAllError(
                "clear-all only resets local Reflexio storage; "
                "remote Supabase/Postgres storage is not supported"
            )
        if kind == "sqlite":
            raw_db_path = storage_config.get("db_path")
            if isinstance(raw_db_path, str) and raw_db_path.strip():
                db_path = _resolve_absolute_path(
                    raw_db_path.strip(), source="configured SQLite db_path"
                )
                targets.extend(
                    _sqlite_artifact_targets(db_path, "configured SQLite data")
                )

    deduped: list[_ClearAllTarget] = []
    seen: set[Path] = set()
    for target in targets:
        _validate_deletion_target(target.path)
        resolved = target.path.resolve(strict=False)
        if resolved in seen:
            continue
        deduped.append(_ClearAllTarget(resolved, target.kind, target.label))
        seen.add(resolved)
    return deduped


def _remove_clear_all_target(target: _ClearAllTarget) -> bool:
    """Remove a target. Returns True when something was actually removed."""
    path = target.path
    if not path.exists():
        return False
    if target.kind == "dir":
        if not path.is_dir():
            raise _ClearAllError(
                f"expected directory target but found non-directory: {path}"
            )
        shutil.rmtree(path)
        return True
    if target.kind == "file":
        if not path.is_file():
            raise _ClearAllError(f"expected file target but found non-file: {path}")
        path.unlink()
        return True
    raise _ClearAllError(f"unknown clear-all target kind: {target.kind}")


def cmd_clear_all(args: argparse.Namespace) -> int:
    """Delete all local reflexio data and openclaw-smart session buffers.

    Stops the managed backend first when it is running, wipes local Reflexio
    data targets, removes local session JSONL buffers under ``state_dir()``,
    then restarts the backend if it was running before. Requires ``--yes``.

    Args:
        args (argparse.Namespace): Parsed CLI args. Honors ``args.yes``
            (skip the confirmation prompt).

    Returns:
        int: 0 on success, 1 if reset is unsafe, unsupported, or fails.
    """
    try:
        targets = _resolve_clear_all_targets()
    except _ClearAllError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    if not args.yes:
        target_lines = "\n".join(
            f"  - {target.path} ({target.label})" for target in targets
        )
        sys.stdout.write(
            "This will permanently delete ALL local reflexio data at:\n"
            f"{target_lines}\n"
            f"and local session buffers under {state.state_dir()}.\n"
            "If the reflexio backend is running, it will be stopped first "
            "and restarted afterward.\n"
            "Re-run with --yes to confirm.\n"
        )
        return 1

    was_running = _is_running_status(_service_status(_BACKEND_SCRIPT, wait_ready_s=0.0))
    if was_running:
        sys.stdout.write("Stopping reflexio backend…\n")
        stop_rc = _run_service(_BACKEND_SCRIPT, "stop")
        if stop_rc != 0:
            sys.stderr.write(
                f"error: reflexio backend failed to stop (exit {stop_rc})\n"
            )
            return stop_rc or 1

    clear_all_failed = False
    removed_targets = 0
    try:
        for target in targets:
            if _remove_clear_all_target(target):
                removed_targets += 1
    except (OSError, _ClearAllError) as exc:
        sys.stderr.write(f"error: could not remove reflexio data: {exc}\n")
        clear_all_failed = True

    removed_buffers = 0
    root = state.state_dir()
    if not clear_all_failed and root.is_dir():
        for buf in root.glob("*.jsonl"):
            try:
                buf.unlink()
                removed_buffers += 1
            except OSError as exc:
                sys.stderr.write(f"warning: could not remove {buf}: {exc}\n")
                clear_all_failed = True

    start_rc = 0
    if was_running:
        sys.stdout.write("Starting reflexio backend…\n")
        start_rc = _run_service(_BACKEND_SCRIPT, "start")

    target_summary = (
        f"removed {removed_targets} data target(s)"
        if removed_targets
        else "nothing to wipe"
    )
    sys.stdout.write(
        f"Cleared reflexio: {target_summary}. "
        f"Removed {removed_buffers} local session buffer(s).\n"
    )
    return start_rc or (1 if clear_all_failed else 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclaw-smart")
    sub = parser.add_subparsers(dest="command", required=True)

    sh = sub.add_parser(
        "show",
        help="Show current project-specific skills and project preferences",
    )
    sh.add_argument("--project", help="Override project id")
    sh.set_defaults(func=cmd_show)

    ln = sub.add_parser(
        "learn",
        help="Force reflexio extraction over the active session now",
    )
    ln.add_argument("--session", help="Session id (defaults to latest)")
    ln.add_argument("--project", help="Override project id")
    ln.add_argument(
        "--note",
        default=None,
        help=(
            "Free-form note to publish as a User turn before extraction "
            "(e.g. an insight, preference, or workflow rule)"
        ),
    )
    ln.set_defaults(func=cmd_learn)

    ca = sub.add_parser(
        "clear-all",
        help="Delete all local reflexio data and restart the backend",
    )
    ca.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive clear without prompting",
    )
    ca.set_defaults(func=cmd_clear_all)

    rs = sub.add_parser(
        "restart",
        help="Restart the reflexio backend and dashboard to pick up changes",
    )
    rs.add_argument(
        "--skip-backend",
        action="store_true",
        help="Do not stop/start the reflexio backend",
    )
    rs.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Do not stop/start the dashboard",
    )
    rs.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip the `npm run build` step before restarting the dashboard",
    )
    rs.set_defaults(func=cmd_restart)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
