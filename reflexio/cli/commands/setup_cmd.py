"""Interactive setup wizard for Reflexio integrations."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from reflexio.server.llm.model_defaults import EMBEDDING_CAPABLE_PROVIDERS


class InstallLocation(Enum):
    """Where to install the Claude Code integration files.

    CURRENT_PROJECT installs into ``<cwd>/.claude/`` — scoped to one project.
    ALL_PROJECTS installs into ``~/.claude/`` — active in every Claude Code session.
    """

    CURRENT_PROJECT = "current_project"
    ALL_PROJECTS = "all_projects"


app = typer.Typer(
    help="Configure Reflexio: run 'init' for plain CLI setup, or one of "
    "the integration commands (openclaw, claude-code) to also install "
    "their hooks."
)

_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"env_var": "OPENAI_API_KEY", "model": "gpt-5-mini", "display": "OpenAI"},
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
        "display": "Anthropic",
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "model": "gemini-3-flash-preview",
        "display": "Gemini",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "display": "DeepSeek",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "model": "gemini-3-flash-preview",
        "display": "OpenRouter",
    },
    "minimax": {
        "env_var": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
        "display": "MiniMax",
    },
    "dashscope": {
        "env_var": "DASHSCOPE_API_KEY",
        "model": "qwen-plus",
        "display": "DashScope",
    },
    "xai": {"env_var": "XAI_API_KEY", "model": "grok-3-mini", "display": "xAI"},
    "moonshot": {
        "env_var": "MOONSHOT_API_KEY",
        "model": "moonshot-v1-8k",
        "display": "Moonshot",
    },
    "zai": {"env_var": "ZAI_API_KEY", "model": "glm-4-flash", "display": "ZAI"},
}


def _set_env_var(env_path: Path, key: str, value: str) -> None:
    """Write or update an environment variable in a .env file.

    Thin wrapper around :func:`reflexio.cli.env_loader.set_env_var` kept
    for backward compatibility with tests that import this name.

    Args:
        env_path (Path): Path to the .env file.
        key (str): Environment variable name.
        value (str): Environment variable value.
    """
    from reflexio.cli.env_loader import set_env_var

    set_env_var(env_path, key, value)


def _prompt_llm_provider(env_path: Path) -> tuple[str, str, str]:
    """Interactively prompt the user to choose an LLM provider and API key.

    Args:
        env_path (Path): Path to the .env file for writing the key.

    Returns:
        tuple[str, str, str]: The display name, default model, and provider key
            for the chosen provider.
    """
    provider_keys = list(_PROVIDERS.keys())

    typer.echo("\nWhich LLM provider for feedback extraction?")
    for idx, key in enumerate(provider_keys, 1):
        display = _PROVIDERS[key]["display"]
        model = _PROVIDERS[key]["model"]
        typer.echo(f"  [{idx}] {display:<14s} ({model})")

    choice = typer.prompt("Choice", type=int)
    if not 1 <= choice <= len(provider_keys):
        typer.echo(f"Error: choice must be between 1 and {len(provider_keys)}")
        raise typer.Exit(1)

    selected_key = provider_keys[choice - 1]
    provider_info = _PROVIDERS[selected_key]
    env_var = provider_info["env_var"]
    model = provider_info["model"]
    display_name = provider_info["display"]

    api_key = typer.prompt(f"Enter your {env_var}")
    if not api_key.strip():
        typer.echo("Error: API key cannot be empty")
        raise typer.Exit(1)
    _set_env_var(env_path, env_var, api_key)

    return display_name, model, selected_key


def _prompt_install_location() -> InstallLocation:
    """Interactively prompt the user to choose where to install the integration.

    Returns:
        InstallLocation: The chosen install location.
    """
    typer.echo("\nWhere should the Claude Code integration be installed?")
    typer.echo("  [1] All projects (~/.claude/) — applies to every Claude Code session")
    typer.echo(
        "  [2] Current project only (./.claude/) — applies only when working in this directory"
    )

    choice = typer.prompt("Choice", type=int, default=1)
    if choice == 1:
        return InstallLocation.ALL_PROJECTS
    if choice == 2:
        return InstallLocation.CURRENT_PROJECT

    typer.echo("Error: choice must be 1 or 2")
    raise typer.Exit(1)


_EMBEDDING_CHOICES: list[tuple[str, str, str]] = [
    (
        "openai",
        "OPENAI_API_KEY",
        "OpenAI  (text-embedding-3-small — recommended, lowest cost)",
    ),
    ("gemini", "GEMINI_API_KEY", "Gemini  (text-embedding-004)"),
]


def _prompt_embedding_provider(env_path: Path, llm_provider_key: str) -> str | None:
    """Prompt for an embedding-capable API key if the LLM provider lacks embedding support.

    Skips the prompt only when the LLM provider already supports embeddings.

    Args:
        env_path (Path): Path to the .env file for writing the key.
        llm_provider_key (str): The provider key selected for LLM generation.

    Returns:
        str | None: Display name of the embedding provider, or None if the LLM
            provider already supports embeddings.
    """
    if llm_provider_key in EMBEDDING_CAPABLE_PROVIDERS:
        return None

    llm_display = _PROVIDERS[llm_provider_key]["display"]
    typer.echo(f"\nYour LLM provider ({llm_display}) doesn't support text embeddings.")
    typer.echo("Reflexio needs an embedding model for semantic search.\n")
    typer.echo("Which provider for embeddings?")
    for idx, (_, _, label) in enumerate(_EMBEDDING_CHOICES, 1):
        typer.echo(f"  [{idx}] {label}")

    choice = typer.prompt("Choice", type=int, default=1)
    if not 1 <= choice <= len(_EMBEDDING_CHOICES):
        typer.echo(f"Error: choice must be between 1 and {len(_EMBEDDING_CHOICES)}")
        raise typer.Exit(1)

    _, env_var, _ = _EMBEDDING_CHOICES[choice - 1]
    api_key = typer.prompt(f"Enter your {env_var}")
    if not api_key.strip():
        typer.echo("Error: API key cannot be empty")
        raise typer.Exit(1)
    _set_env_var(env_path, env_var, api_key)

    return _PROVIDERS[_EMBEDDING_CHOICES[choice - 1][0]]["display"]


_LOCAL_SERVER_URL = "http://localhost:8081"


def _prompt_local_sqlite(env_path: Path) -> str:
    """Option 1 — local SQLite with a local Reflexio server.

    Writes ``REFLEXIO_URL`` pointing at the local server so the CLI
    and Claude Code hooks know where to connect.

    Args:
        env_path (Path): Path to the .env file.

    Returns:
        str: Storage label for the wizard summary.
    """
    _set_env_var(env_path, "REFLEXIO_URL", _LOCAL_SERVER_URL)
    return "SQLite (local)"


def _prompt_managed_reflexio(env_path: Path) -> str:
    """Option 2 — point the CLI at reflexio.ai + verify via whoami.

    Prompts for a Reflexio API key, writes ``REFLEXIO_URL`` and
    ``REFLEXIO_API_KEY`` to ``.env``, then calls ``whoami()`` to
    verify the account and show resolved storage per-org.

    Args:
        env_path (Path): Path to the .env file.

    Returns:
        str: Storage label for the wizard summary.
    """
    reflexio_api_key = typer.prompt("Reflexio API key")
    if not reflexio_api_key.strip():
        typer.echo("Error: API key cannot be empty")
        raise typer.Exit(1)

    from reflexio.defaults import DEFAULT_SERVER_URL

    _set_env_var(env_path, "REFLEXIO_URL", DEFAULT_SERVER_URL)
    _set_env_var(env_path, "REFLEXIO_API_KEY", reflexio_api_key)

    try:
        from reflexio.client.client import ReflexioClient

        client = ReflexioClient(
            api_key=reflexio_api_key, url_endpoint=DEFAULT_SERVER_URL
        )
        resp = client.whoami()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"\n  (could not verify account — {type(exc).__name__}: {exc})")
        return "Managed Reflexio"

    if not resp.success:
        typer.echo(
            f"\n  (account verification failed: {resp.message or 'unknown error'})"
        )
        return "Managed Reflexio"

    # Detect the "server is in self-host mode" case. If the remote server
    # returns the canonical ``self-host-org`` default org id, it means
    # auth isn't being enforced — anyone hitting the endpoint without a
    # token would see the same response. The user's API key was not
    # actually validated, and any publishes will land in the server's
    # shared default storage instead of the user's per-org Supabase.
    if resp.org_id == "self-host-org":
        typer.echo(
            "\n  ⚠ The remote server returned the default 'self-host-org' "
            "identity instead of your real org."
        )
        typer.echo(
            "    Your API key was NOT validated. The server is running in "
            "self-host mode, which means:"
        )
        typer.echo(
            "      • All publishes will land in the server's shared "
            "storage, not your per-org Supabase."
        )
        typer.echo(
            "      • Other users hitting the same server share the same data namespace."
        )
        typer.echo(
            "    Contact the server operator to enable enterprise auth, "
            "or point REFLEXIO_URL at a deployment that enforces it."
        )
        return "Managed Reflexio"

    typer.echo("\n  Verified cloud account:")
    typer.echo(f"    Org ID:        {resp.org_id}")
    typer.echo(f"    Storage type:  {resp.storage_type or 'unconfigured'}")
    if resp.storage_label:
        marker = "[configured]" if resp.storage_configured else "[unconfigured]"
        typer.echo(f"    Storage:       {resp.storage_label}  {marker}")
    if not resp.storage_configured:
        typer.echo(
            "\n  ⚠ Your org has no storage configured at "
            f"{DEFAULT_SERVER_URL}/settings."
        )
        typer.echo(
            "    Publishes will succeed but no data will be written until "
            "you configure it."
        )

    return "Managed Reflexio"


def _prompt_self_hosted(env_path: Path) -> str:
    """Option 3 — point the CLI at a self-hosted Reflexio server.

    Prompts for a Reflexio API key and writes ``REFLEXIO_URL`` (defaulting
    to localhost) and ``REFLEXIO_API_KEY`` to ``.env``.

    Args:
        env_path (Path): Path to the .env file for writing credentials.

    Returns:
        str: Storage label for the wizard summary.
    """
    reflexio_url = typer.prompt("Reflexio server URL", default=_LOCAL_SERVER_URL)
    reflexio_api_key = typer.prompt("Reflexio API key")
    if not reflexio_api_key.strip():
        typer.echo("Error: API key cannot be empty")
        raise typer.Exit(1)

    _set_env_var(env_path, "REFLEXIO_URL", reflexio_url)
    _set_env_var(env_path, "REFLEXIO_API_KEY", reflexio_api_key)

    return "Self-hosted Reflexio"


def _prompt_user_id(env_path: Path, fallback: str = "claude-code") -> str:
    """
    Prompt for REFLEXIO_USER_ID. Press Enter to accept the default.

    Tags all interactions, profiles, and playbooks published by Claude Code.
    Customize when you want per-developer attribution or to match a
    managed/remote user account. If REFLEXIO_USER_ID is already set in
    the environment (e.g. from a previous setup run), that value is offered
    as the default.

    Args:
        env_path (Path): Path to the .env file to persist the value.
        fallback (str): Default user_id when neither the env nor user input supplies one.

    Returns:
        str: The resolved user_id (trimmed; fallback if empty).
    """
    current = (os.environ.get("REFLEXIO_USER_ID") or "").strip()
    default = current or fallback
    typer.echo("")
    typer.echo(
        "User ID tags interactions, profiles, and playbooks published by Claude Code."
    )
    typer.echo("Press Enter to keep the default.")
    user_id = (typer.prompt("User ID", default=default) or default).strip() or fallback
    _set_env_var(env_path, "REFLEXIO_USER_ID", user_id)
    return user_id


def _prompt_storage(env_path: Path) -> str:
    """Interactively prompt the user to choose a storage backend.

    Args:
        env_path (Path): Path to the .env file to update.

    Returns:
        str: The storage mode label for the wizard summary.
    """
    typer.echo("\nWhere should Reflexio store data?")
    typer.echo("  [1] Local SQLite (default, no setup needed)")
    typer.echo(
        "  [2] Managed Reflexio (reflexio.ai — storage managed at reflexio.ai/settings)"
    )
    typer.echo("  [3] Self-hosted Reflexio (connect to your own server)")

    choice = typer.prompt("Choice", type=int, default=1)
    if choice == 1:
        return _prompt_local_sqlite(env_path)
    if choice == 2:
        return _prompt_managed_reflexio(env_path)
    if choice == 3:
        return _prompt_self_hosted(env_path)

    typer.echo("Error: choice must be 1, 2, or 3")
    raise typer.Exit(1)


def _install_openclaw_integration() -> bool:
    """Install the Reflexio hook and skill into OpenClaw.

    Returns:
        bool: True if the hook was verified as registered.

    Raises:
        typer.Exit: If the openclaw CLI is not found on PATH.
    """
    if not shutil.which("openclaw"):
        typer.echo("Error: openclaw CLI not found. Install from https://openclaw.ai")
        raise typer.Exit(1)

    import reflexio

    pkg_dir = Path(reflexio.__file__).parent
    integration_dir = pkg_dir / "integrations" / "openclaw"
    hook_dir = integration_dir / "hook"
    skill_dir = integration_dir / "skill"
    rules_dir = integration_dir / "rules"
    commands_dir = integration_dir / "commands"

    # Install plugin and enable hook
    try:
        subprocess.run(
            ["openclaw", "plugins", "install", str(hook_dir), "--link"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["openclaw", "hooks", "enable", "reflexio-context"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: openclaw command failed: {exc.stderr or exc.stdout}")
        raise typer.Exit(1) from exc

    # Copy skill directory. ClawHub drops `_meta.json` at the skill root on
    # install — if that's present, don't clobber the user's ClawHub-installed
    # copy. Otherwise always refresh (so `pip install --upgrade reflexio-ai &&
    # reflexio setup openclaw` stays the normal upgrade flow).
    workspace_skills = Path.home() / ".openclaw" / "skills" / "reflexio"
    if (workspace_skills / "_meta.json").exists():
        typer.echo(f"ClawHub-installed skill at {workspace_skills} — skipping refresh")
    else:
        if workspace_skills.exists():
            shutil.rmtree(workspace_skills)
        shutil.copytree(skill_dir, workspace_skills)

    # Copy each command directory to ~/.openclaw/skills/<command-name>
    if commands_dir.exists():
        for cmd_subdir in commands_dir.iterdir():
            if cmd_subdir.is_dir():
                dest = Path.home() / ".openclaw" / "skills" / cmd_subdir.name
                shutil.copytree(cmd_subdir, dest, dirs_exist_ok=True)
                typer.echo(f"Command installed: {dest}")

    # Copy rules to default workspace (always-active behavioral constraints)
    if rules_dir.exists():
        workspace_dir = Path.home() / ".openclaw" / "workspace"
        for rule_file in rules_dir.glob("*.md"):
            dest = workspace_dir / rule_file.name
            workspace_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rule_file, dest)
            typer.echo(f"Rule installed: {dest}")

    # Verify
    result = subprocess.run(
        ["openclaw", "hooks", "list"],
        capture_output=True,
        text=True,
    )
    if "reflexio-context" in result.stdout:
        typer.echo("Hook installed and registered")
        return True

    typer.echo("Warning: Hook may not be registered -- check 'openclaw hooks list'")
    return False


def _uninstall_openclaw() -> None:
    """Remove the Reflexio integration from OpenClaw."""
    typer.confirm(
        "This will remove the Reflexio integration from OpenClaw. Continue?",
        abort=True,
    )
    if shutil.which("openclaw"):
        subprocess.run(
            ["openclaw", "hooks", "disable", "reflexio-context"],
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["openclaw", "plugins", "uninstall", "reflexio-context"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        typer.echo("Warning: openclaw CLI not found on PATH, skipping hook removal")
    workspace_skills = Path.home() / ".openclaw" / "skills" / "reflexio"
    if workspace_skills.exists():
        shutil.rmtree(workspace_skills)
        typer.echo(f"Removed skill: {workspace_skills}")

    import reflexio as _reflexio

    integration_dir = Path(_reflexio.__file__).parent / "integrations" / "openclaw"
    commands_dir = integration_dir / "commands"
    if commands_dir.exists():
        for cmd_subdir in commands_dir.iterdir():
            if cmd_subdir.is_dir():
                cmd_dir = Path.home() / ".openclaw" / "skills" / cmd_subdir.name
                if cmd_dir.exists():
                    shutil.rmtree(cmd_dir)
    # Remove rules from default workspace
    rules_file = Path.home() / ".openclaw" / "workspace" / "reflexio.md"
    if rules_file.exists():
        rules_file.unlink()
        typer.echo(f"Removed rule: {rules_file}")

    typer.echo("Reflexio integration fully removed from OpenClaw.")


@app.command("openclaw")
def openclaw(
    uninstall: Annotated[
        bool,
        typer.Option(
            "--uninstall", help="Remove the Reflexio integration from OpenClaw"
        ),
    ] = False,
) -> None:
    """Set up (or remove) the Reflexio integration for OpenClaw."""
    if uninstall:
        _uninstall_openclaw()
        return

    # Step 1: Load .env path
    from reflexio.cli.env_loader import load_reflexio_env

    env_path = load_reflexio_env()
    if env_path is None:
        typer.echo("Error: could not locate or create a .env file")
        raise typer.Exit(1)

    # Step 2: LLM provider
    display_name, model, provider_key = _prompt_llm_provider(env_path)

    # Step 2.5: Embedding provider (if LLM provider lacks embedding support)
    embedding_label = _prompt_embedding_provider(env_path, provider_key)

    # Step 3: Storage
    storage_label = _prompt_storage(env_path)

    # Step 4: Install OpenClaw integration
    typer.echo("")
    hook_ok = _install_openclaw_integration()

    # Step 5: Summary
    hook_status = "reflexio-context" if hook_ok else "reflexio-context (unverified)"
    skill_path = Path.home() / ".openclaw" / "skills" / "reflexio"

    typer.echo("")
    typer.echo("Setup complete!")
    typer.echo(f"  LLM Provider: {display_name} ({model})")
    if embedding_label:
        typer.echo(f"  Embedding Provider: {embedding_label}")
    typer.echo(f"  Storage: {storage_label}")
    typer.echo(f"  Hook: {hook_status}")
    typer.echo(f"  Skill: {skill_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Start Reflexio: reflexio services start")
    typer.echo("  2. Restart OpenClaw gateway: openclaw gateway restart")
    typer.echo(
        "  3. Start a conversation -- Reflexio will capture and learn automatically"
    )


# ---------------------------------------------------------------------------
# Generic (integration-less) setup
# ---------------------------------------------------------------------------


@app.command("init")
def init(
    skip_llm: Annotated[
        bool,
        typer.Option(
            "--skip-llm",
            help=(
                "Skip the LLM provider prompt (use when you're only "
                "going to publish to a managed Reflexio server, which "
                "manages its own LLM keys server-side)"
            ),
        ),
    ] = False,
) -> None:
    """Configure Reflexio without installing any integration.

    Writes ``REFLEXIO_URL`` / ``REFLEXIO_API_KEY`` / LLM provider keys
    / storage backend into ``~/.reflexio/.env``. This is the command
    to run if you're using the ``reflexio`` CLI directly from your
    shell and don't need the OpenClaw or Claude Code hook
    installation.

    Under the hood it reuses the same ``_prompt_storage`` +
    ``_prompt_llm_provider`` helpers the integration setup commands
    use, so the flow and the resulting ``.env`` are identical to what
    you'd get from those commands minus the hook-installation step.

    Args:
        skip_llm: When True, skip the LLM provider prompt. Useful if
            you're only going to point the CLI at a managed Reflexio
            server, which handles extraction with its own LLM keys.
    """
    from reflexio.cli.env_loader import load_reflexio_env

    env_path = load_reflexio_env()
    if env_path is None:
        typer.echo("Error: could not locate or create a .env file")
        raise typer.Exit(1)

    # Step 1: Storage (ask first — managed mode doesn't need an LLM key)
    storage_label = _prompt_storage(env_path)

    # Step 2: LLM provider (skipped for managed mode — the remote server
    # handles extraction so the local .env doesn't need an LLM key).
    # Also skipped when the user explicitly passes --skip-llm.
    is_managed = storage_label == "Managed Reflexio"
    display_name: str | None = None
    model: str | None = None
    embedding_label: str | None = None
    if is_managed:
        typer.echo(
            "\nSkipping LLM provider — Managed Reflexio handles "
            "extraction server-side with its own model keys."
        )
    elif skip_llm:
        typer.echo("\nSkipping LLM provider per --skip-llm.")
    else:
        display_name, model, provider_key = _prompt_llm_provider(env_path)
        embedding_label = _prompt_embedding_provider(env_path, provider_key)

    # Step 3: Summary — no integration to print
    typer.echo("")
    typer.echo("Setup complete!")
    if display_name and model:
        typer.echo(f"  LLM Provider: {display_name} ({model})")
    if embedding_label:
        typer.echo(f"  Embedding Provider: {embedding_label}")
    typer.echo(f"  Storage: {storage_label}")
    typer.echo(f"  .env: {env_path}")
    typer.echo("")
    typer.echo(
        "Next: run 'reflexio status whoami' to verify the connection "
        "(managed mode) or 'reflexio services start' to launch the "
        "local backend (SQLite / self-hosted mode)."
    )


# ---------------------------------------------------------------------------
# Claude Code integration
# ---------------------------------------------------------------------------


def _get_integration_dir() -> Path:
    """Locate the claude_code integration directory within the installed package."""
    import reflexio

    return Path(reflexio.__file__).parent / "integrations" / "claude_code"


def _upsert_hook(hooks: dict, event_name: str, hook_command: str) -> None:
    """Add or update a hook entry for the given event in the hooks dict.

    Args:
        hooks: The hooks dict from settings.json.
        event_name: The hook event name (e.g., "Stop", "UserPromptSubmit").
        hook_command: The shell command to run.
    """
    event_hooks: list[dict] = hooks.setdefault(event_name, [])
    hook_entry = {
        "matcher": "",
        "hooks": [{"type": "command", "command": hook_command}],
    }
    for existing in event_hooks:
        inner = existing.get("hooks", [])
        if any("reflexio" in h.get("command", "") for h in inner):
            existing["hooks"] = hook_entry["hooks"]
            return
    event_hooks.append(hook_entry)


def _merge_hook_config(settings_path: Path, handler_js_path: Path) -> None:
    """Add or update Reflexio hooks in .claude/settings.json.

    Installs two hooks:
    - SessionStart: checks if the Reflexio server is running and starts it in
      the background if not (~10ms, non-blocking).
    - UserPromptSubmit: runs `reflexio search` on every user prompt and injects
      results as context Claude sees.

    No Stop hook is installed — conversation capture is handled by the expert
    skill's mid-session publish or by an explicit `/reflexio-extract` command,
    giving the user control over when to extract learnings.

    Args:
        settings_path: Path to the project's .claude/settings.json.
        handler_js_path: Absolute path to handler.js in the installed package.
    """
    settings: dict = {}
    if settings_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            settings = json.loads(settings_path.read_text())

    hooks = settings.setdefault("hooks", {})

    # Session start hook (SessionStart) — checks/starts Reflexio server proactively
    session_start_hook_sh = handler_js_path.parent / "session_start_hook.sh"
    _upsert_hook(hooks, "SessionStart", f"bash {session_start_hook_sh}")

    # Search hook (UserPromptSubmit) — injects Reflexio context before Claude responds
    search_hook_js = handler_js_path.parent / "search_hook.js"
    _upsert_hook(hooks, "UserPromptSubmit", f"node {search_hook_js}")

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _remove_hook_config(settings_path: Path) -> None:
    """Remove all Reflexio hooks from .claude/settings.json.

    Args:
        settings_path: Path to the project's .claude/settings.json.
    """
    if not settings_path.exists():
        return
    try:
        settings = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    hooks = settings.get("hooks")
    if not hooks:
        return

    for event_name in ["Stop", "UserPromptSubmit", "SessionStart"]:
        event_hooks = hooks.get(event_name, [])
        hooks[event_name] = [
            entry
            for entry in event_hooks
            if not any(
                "reflexio" in h.get("command", "") for h in entry.get("hooks", [])
            )
        ]
        if not hooks[event_name]:
            del hooks[event_name]
    if not settings["hooks"]:
        del settings["hooks"]

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


_MARKER_FILENAME = ".installed-by-reflexio"


def _write_marker(marker_path: Path, location: InstallLocation) -> None:
    """Write a JSON marker file recording the install location and timestamp.

    Args:
        marker_path: Where to write the marker file.
        location: The install location enum value.
    """
    import datetime

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "location": location.value,
                "installed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


def _install_claude_code_integration(
    target_dir: Path,
    *,
    expert: bool = False,
    location: InstallLocation = InstallLocation.ALL_PROJECTS,
) -> tuple[Path, Path]:
    """Install the Reflexio skill and hook into a Claude Code project or user directory.

    Args:
        target_dir: Root directory — either the project root or ``Path.home()``.
        expert: If True, install the expert skill instead of the normal skill.
        location: Where to install (current project or all projects).

    Returns:
        tuple[Path, Path]: (skill_path, handler_js_path) for the summary.
    """
    integration_dir = _get_integration_dir()
    if not integration_dir.exists():
        typer.echo(f"Error: integration files not found at {integration_dir}")
        raise typer.Exit(1)

    claude_dir = target_dir / ".claude"

    # Copy skill
    skill_src = (
        integration_dir / "skill" / "SKILL-expert.md"
        if expert
        else integration_dir / "skill" / "SKILL.md"
    )
    skill_dest = claude_dir / "skills" / "reflexio" / "SKILL.md"
    skill_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skill_src, skill_dest)

    # Copy rules file (always-in-context instructions)
    rules_src = integration_dir / "rules" / "reflexio.md"
    rules_dest = claude_dir / "rules" / "reflexio.md"
    rules_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rules_src, rules_dest)

    # Expert mode: also install /reflexio-extract command
    if expert:
        cmd_src = integration_dir / "commands" / "reflexio-extract" / "SKILL.md"
        cmd_dest = claude_dir / "commands" / "reflexio-extract" / "SKILL.md"
        cmd_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cmd_src, cmd_dest)

    # Configure hook
    handler_js = integration_dir / "hook" / "handler.js"
    settings_path = claude_dir / "settings.json"
    _merge_hook_config(settings_path, handler_js)

    # Write marker for uninstall auto-detection
    marker_path = claude_dir / "skills" / "reflexio" / _MARKER_FILENAME
    _write_marker(marker_path, location)

    return skill_dest, handler_js


def _detect_install_locations(
    project_dir: Path,
) -> list[tuple[InstallLocation, Path]]:
    """Detect where Reflexio is installed by checking marker files.

    Args:
        project_dir: The project directory to check for project-level installs.

    Returns:
        list[tuple[InstallLocation, Path]]: List of (location, base_dir) pairs
            where the integration is installed.
    """
    locations: list[tuple[InstallLocation, Path]] = []
    for loc, base in [
        (InstallLocation.ALL_PROJECTS, Path.home()),
        (InstallLocation.CURRENT_PROJECT, project_dir),
    ]:
        marker = base / ".claude" / "skills" / "reflexio" / _MARKER_FILENAME
        if marker.exists():
            locations.append((loc, base))
    return locations


def _remove_from_dir(base_dir: Path) -> None:
    """Remove the Reflexio integration files from a .claude directory.

    Args:
        base_dir: The directory containing the .claude/ folder.
    """
    claude_dir = base_dir / ".claude"

    # Remove skill directory (includes marker file)
    skill_dir = claude_dir / "skills" / "reflexio"
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
        typer.echo(f"  Removed skill: {skill_dir}")

    # Remove rules file
    rules_file = claude_dir / "rules" / "reflexio.md"
    if rules_file.exists():
        rules_file.unlink()
        typer.echo(f"  Removed rules: {rules_file}")

    # Remove /reflexio-extract command
    cmd_dir = claude_dir / "commands" / "reflexio-extract"
    if cmd_dir.exists():
        shutil.rmtree(cmd_dir)
        typer.echo(f"  Removed command: {cmd_dir}")

    # Remove hook from settings
    settings_path = claude_dir / "settings.json"
    _remove_hook_config(settings_path)
    typer.echo(f"  Removed hook from: {settings_path}")


def _uninstall_claude_code(project_dir: Path, *, global_install: bool = False) -> None:
    """Remove the Reflexio integration from Claude Code.

    When ``--global`` or ``--project-dir`` is explicit, removes from that
    location directly. Otherwise auto-detects via marker files.

    Args:
        project_dir: Root directory of the Claude Code project.
        global_install: If True, only remove from ~/.claude/.
    """
    # When --global is explicit, skip detection and remove from ~/.claude/
    if global_install:
        home = Path.home()
        marker = home / ".claude" / "skills" / "reflexio" / _MARKER_FILENAME
        if not marker.exists():
            typer.confirm(
                "No Reflexio marker found in ~/.claude/. Remove anyway?",
                abort=True,
            )
        else:
            typer.confirm(
                "Remove Reflexio integration from ~/.claude/ (all projects)?",
                abort=True,
            )
        _remove_from_dir(home)
        typer.echo("Reflexio integration removed.")
        return

    locations = _detect_install_locations(project_dir)

    if not locations:
        typer.confirm(
            f"No Reflexio marker found. Remove integration from {project_dir}/.claude/?",
            abort=True,
        )
        _remove_from_dir(project_dir)
        typer.echo("Reflexio integration removed.")
        return

    if len(locations) == 1:
        loc, base = locations[0]
        loc_label = (
            "~/.claude/ (all projects)"
            if loc == InstallLocation.ALL_PROJECTS
            else f"{base}/.claude/ (current project)"
        )
        typer.confirm(
            f"Found Reflexio integration at {loc_label}. Remove it?",
            abort=True,
        )
        _remove_from_dir(base)
        typer.echo("Reflexio integration removed.")
        return

    # Both locations have installs
    typer.echo("\nReflexio is installed in multiple locations:")
    typer.echo("  [1] All projects (~/.claude/)")
    typer.echo(f"  [2] Current project ({project_dir}/.claude/)")
    typer.echo("  [3] Both")
    choice = typer.prompt("Which installation to remove?", type=int)

    targets: list[tuple[InstallLocation, Path]] = []
    if choice == 1:
        targets = [locations[0]]
    elif choice == 2:
        targets = [locations[1]]
    elif choice == 3:
        targets = locations
    else:
        typer.echo("Error: choice must be 1, 2, or 3")
        raise typer.Exit(1)

    for _, base in targets:
        _remove_from_dir(base)
    typer.echo("Reflexio integration removed.")


@app.command("claude-code")
def claude_code_setup(
    uninstall: Annotated[
        bool,
        typer.Option("--uninstall", help="Remove the Reflexio integration"),
    ] = False,
    expert: Annotated[
        bool,
        typer.Option(
            "--expert",
            help="Install the expert skill (search + summarize + publish)",
        ),
    ] = False,
    project_dir: Annotated[
        Path | None,
        typer.Option(
            "--project-dir",
            help="Target project directory (default: current directory)",
        ),
    ] = None,
    global_install: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Install to ~/.claude/ (user-level, applies to all projects)",
        ),
    ] = False,
) -> None:
    """Set up (or remove) the Reflexio integration for Claude Code."""
    # Resolve install location
    if global_install and project_dir is not None:
        typer.echo("Error: --global and --project-dir are mutually exclusive")
        raise typer.Exit(1)

    # Uninstall uses auto-detection — no need for the interactive location prompt
    if uninstall:
        target = (
            Path.home()
            if global_install
            else Path(project_dir)
            if project_dir is not None
            else Path.cwd()
        )
        _uninstall_claude_code(target, global_install=global_install)
        return

    if global_install:
        target = Path.home()
        location = InstallLocation.ALL_PROJECTS
    elif project_dir is not None:
        target = Path(project_dir)
        location = InstallLocation.CURRENT_PROJECT
    else:
        location = _prompt_install_location()
        target = Path.home() if location == InstallLocation.ALL_PROJECTS else Path.cwd()

    # Step 1: Load .env path
    from reflexio.cli.env_loader import load_reflexio_env

    env_path = load_reflexio_env()
    if env_path is None:
        typer.echo("Error: could not locate or create a .env file")
        raise typer.Exit(1)

    # Step 2: Storage (ask first — determines whether LLM key is needed)
    storage_label = _prompt_storage(env_path)

    # Step 3: LLM provider (only needed for local server — remote handles its own keys)
    is_remote = storage_label in {"Managed Reflexio", "Self-hosted Reflexio"}
    embedding_label: str | None = None
    display_name: str | None = None
    model: str | None = None
    if is_remote:
        typer.echo(
            "\nSkipping LLM provider — the remote Reflexio server handles extraction."
        )
    else:
        display_name, model, provider_key = _prompt_llm_provider(env_path)
        embedding_label = _prompt_embedding_provider(env_path, provider_key)

    # Step 3.5: Configure user_id for Claude Code
    user_id = _prompt_user_id(env_path)

    # Step 4: Install skill + hook
    typer.echo("")
    skill_path, _ = _install_claude_code_integration(
        target, expert=expert, location=location
    )
    skill_type = "expert" if expert else "normal"

    # Step 5: Summary
    location_label = (
        "All projects (~/.claude/)"
        if location == InstallLocation.ALL_PROJECTS
        else f"Current project ({target}/.claude/)"
    )
    typer.echo("")
    typer.echo("Setup complete!")
    typer.echo(f"  Install location: {location_label}")
    if is_remote:
        typer.echo("  LLM Provider: managed by remote server")
    else:
        typer.echo(f"  LLM Provider: {display_name} ({model})")
    if embedding_label:
        typer.echo(f"  Embedding Provider: {embedding_label}")
    typer.echo(f"  Storage: {storage_label}")
    typer.echo(f"  User ID: {user_id}")
    typer.echo(f"  Skill ({skill_type}): {skill_path}")
    typer.echo("  Hooks: SessionStart + UserPromptSubmit")
    if location == InstallLocation.ALL_PROJECTS:
        typer.echo("")
        typer.echo("Note: User-level hooks fire for ALL Claude Code sessions.")
    typer.echo("")
    if location == InstallLocation.ALL_PROJECTS:
        typer.echo(
            "Next: Start any Claude Code session — Reflexio is active in all projects."
        )
    else:
        typer.echo("Next: Start a Claude Code session in this project.")
    if is_remote:
        typer.echo("Reflexio will connect to the remote server automatically.")
    else:
        typer.echo(
            "The skill will guide Claude to check and start the Reflexio server automatically."
        )
