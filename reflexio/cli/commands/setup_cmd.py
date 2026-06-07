"""Interactive setup wizard for Reflexio integrations.

Note: a previous ``claude-code`` subcommand was removed; see the
submodule README migration notes for cleanup of legacy hook entries
in ``~/.claude/settings.json``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help=(
        "Configure Reflexio: run 'init' for plain CLI setup, use 'openclaw' "
        "to install host-tool hooks, or use 'openai-codex' to configure "
        "Codex OAuth tokens."
    )
)

_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"env_var": "OPENAI_API_KEY", "model": "gpt-5.4-mini", "display": "OpenAI"},
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

_EMBEDDING_PROVIDERS: frozenset[str] = frozenset({"openai", "gemini"})


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


# Map embedding provider keys to their canonical model names. Used by both
# the interactive prompt (writes the choice to org config) and the
# ``--embedding`` flag (skips the prompt entirely for CI). Kept here so the
# two call sites can't drift on what e.g. "gemini" actually means.
_EMBEDDING_MODEL_NAMES: dict[str, str] = {
    "local": "local/minilm-l6-v2",
    "openai": "text-embedding-3-small",
    "gemini": "gemini/gemini-embedding-001",
}


# Valid values for the ``--embedding`` flag across every setup command.
# Defined once here so a typo in one command surface (init, openclaw)
# doesn't silently fall through to "auto" in another.
_VALID_EMBEDDING_FLAGS: frozenset[str] = frozenset(
    {"auto", "local", "openai", "gemini"}
)


def _build_embedding_choices() -> list[tuple[str, str | None, str]]:
    """Build the interactive embedding-provider menu at call time.

    The local option is included only when ``chromadb`` is importable; built
    dynamically so the menu always reflects the current Python environment
    rather than a snapshot frozen at module load.

    Returns:
        list[tuple[str, str | None, str]]: ``(provider_key, env_var_name,
        display_label)`` rows. ``env_var_name`` is ``None`` for the local
        provider (no API key needed).
    """
    from reflexio.server.llm.providers.local_embedding_provider import (
        is_chromadb_importable,
    )

    choices: list[tuple[str, str | None, str]] = []
    if is_chromadb_importable():
        choices.append(
            (
                "local",
                None,
                "Local (in-process MiniLM-L6-v2; ~25 MB; no API key needed)",
            )
        )
    choices.extend(
        [
            ("openai", "OPENAI_API_KEY", "OpenAI (text-embedding-3-small)"),
            ("gemini", "GEMINI_API_KEY", "Gemini (gemini-embedding-001)"),
        ]
    )
    return choices


def _is_non_interactive() -> bool:
    """Return True when prompts must be skipped (CI / scripted use).

    Two triggers:

    - stdin is not a TTY (``nohup``, container, pipe-fed shell)
    - ``REFLEXIO_NONINTERACTIVE=1`` in the environment (explicit opt-out)
    """
    if os.environ.get("REFLEXIO_NONINTERACTIVE") == "1":
        return True
    return not sys.stdin.isatty()


def _prompt_embedding_provider(env_path: Path, llm_provider_key: str) -> str | None:
    """Prompt for an embedding-capable API key if the LLM provider lacks embedding support.

    Skips the prompt when the LLM provider already supports embeddings, or
    when the environment is non-interactive (no TTY or
    ``REFLEXIO_NONINTERACTIVE=1``). In the non-interactive case the first
    available choice is returned without writing to ``.env``.

    Args:
        env_path (Path): Path to the .env file for writing the key.
        llm_provider_key (str): The provider key selected for LLM generation.

    Returns:
        str | None: Display name of the embedding provider, or None if the LLM
            provider already supports embeddings.
    """
    if llm_provider_key in _EMBEDDING_PROVIDERS:
        return None

    choices = _build_embedding_choices()
    if not choices:
        return None

    # Non-interactive: pick the first available option without prompting. When
    # chromadb is importable that's local (no key required); otherwise it's
    # OpenAI or Gemini, which still won't have an API key but at least the
    # caller knows the wizard didn't block.
    if _is_non_interactive():
        provider_key, env_var, _ = choices[0]
        if env_var is None:
            # Local needs no key — nothing to write.
            return _provider_display(provider_key)
        return _provider_display(provider_key)

    llm_display = _PROVIDERS[llm_provider_key]["display"]
    typer.echo(f"\nYour LLM provider ({llm_display}) doesn't support text embeddings.")
    typer.echo("Reflexio needs an embedding model for semantic search.\n")
    typer.echo("Which provider for embeddings?")
    for idx, (_, _, label) in enumerate(choices, 1):
        typer.echo(f"  [{idx}] {label}")

    choice = typer.prompt("Choice", type=int, default=1)
    if not 1 <= choice <= len(choices):
        typer.echo(f"Error: choice must be between 1 and {len(choices)}")
        raise typer.Exit(1)

    provider_key, env_var, _ = choices[choice - 1]
    if env_var is None:
        # Local: no API key needed.
        return _provider_display(provider_key)

    api_key = typer.prompt(f"Enter your {env_var}")
    if not api_key.strip():
        typer.echo("Error: API key cannot be empty")
        raise typer.Exit(1)
    _set_env_var(env_path, env_var, api_key)

    return _provider_display(provider_key)


def _provider_display(provider_key: str) -> str:
    """Return the user-facing display name for a provider key.

    The local provider isn't in ``_PROVIDERS`` (which only lists LLM
    providers), so this helper centralizes the special case.
    """
    if provider_key == "local":
        return "Local (MiniLM-L6-v2)"
    return _PROVIDERS[provider_key]["display"]


def _write_embedding_model_to_org_config(model_name: str) -> None:
    """Persist the user's embedding choice to the default-org config file.

    Loads the existing config, updates ``llm_config.embedding_model_name``,
    and writes it back. All other fields are preserved. Reuses the same
    ``LocalFileConfigStorage`` round-trip pattern as ``save_storage_to_config``.

    Args:
        model_name: Canonical model name to persist
            (e.g. ``"local/minilm-l6-v2"``).
    """
    from reflexio.models.config_schema import LLMConfig
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    storage = LocalFileConfigStorage("self-host-org")
    config = storage.load_config()
    if config.llm_config is None:
        config.llm_config = LLMConfig(embedding_model_name=model_name)
    else:
        config.llm_config.embedding_model_name = model_name
    storage.save_config(config)


def _choose_embedding_provider(env_path: Path, *, embedding_flag: str) -> str | None:
    """Run the upfront embedding-provider step for ``setup init``.

    Behaviour matrix:

    +-----------------+----------------+-------------------------------------+
    | ``embedding``   | TTY?           | What happens                        |
    +=================+================+=====================================+
    | ``"local"`` /   | (any)          | Write the matching model to org     |
    | ``"openai"`` /  |                | config; no prompt.                  |
    | ``"gemini"``    |                |                                     |
    +-----------------+----------------+-------------------------------------+
    | ``"auto"``      | non-interactive| No prompt, no org-config write —    |
    |                 |                | runtime auto-detection (Layer A)    |
    |                 |                | picks the embedder.                 |
    +-----------------+----------------+-------------------------------------+
    | ``"auto"``      | interactive    | Show the menu (default = local      |
    |                 |                | when chromadb is importable). Write |
    |                 |                | the choice to org config; for       |
    |                 |                | OpenAI / Gemini also collect the    |
    |                 |                | API key inline.                     |
    +-----------------+----------------+-------------------------------------+

    Args:
        env_path: Path to the user's .env file. Used to write the cloud
            embedding API key when the user picks OpenAI / Gemini in the
            interactive prompt.
        embedding_flag: Value of the ``--embedding`` flag. ``"auto"`` is
            the default and means "ask interactively, or fall back to
            runtime auto-detection if there's no TTY."

    Returns:
        str | None: Display name of the chosen embedding provider, or None
            when no override was written (auto + non-interactive, or no
            choices available).
    """
    if embedding_flag in _EMBEDDING_MODEL_NAMES:
        # Explicit non-default flag wins over interactive / auto-detection.
        # ``--embedding=local`` requires chromadb at runtime, so refuse to
        # persist a broken override the same way the interactive flow
        # hides the option in that situation.
        if embedding_flag == "local":
            from reflexio.server.llm.providers.local_embedding_provider import (
                is_chromadb_importable,
            )

            if not is_chromadb_importable():
                typer.echo(
                    "Error: --embedding=local requires chromadb. "
                    "Install it with `pip install chromadb` or pick "
                    "openai/gemini/auto."
                )
                raise typer.Exit(1)
        _write_embedding_model_to_org_config(_EMBEDDING_MODEL_NAMES[embedding_flag])
        return _provider_display(embedding_flag)

    # embedding_flag == "auto" from here on.
    if _is_non_interactive():
        # No TTY → runtime auto-detection picks the embedder. No write.
        return None

    choices = _build_embedding_choices()
    if not choices:
        # No providers available (chromadb not importable AND no cloud
        # embedders in scope). Defer to runtime auto-detection, which will
        # raise a clear error if nothing matches.
        return None

    typer.echo("\nChoose embedding provider:")
    for idx, (_, _, label) in enumerate(choices, 1):
        suffix = " — recommended" if idx == 1 else ""
        typer.echo(f"  [{idx}] {label}{suffix}")

    choice = typer.prompt("Choice", type=int, default=1)
    if not 1 <= choice <= len(choices):
        typer.echo(f"Error: choice must be between 1 and {len(choices)}")
        raise typer.Exit(1)

    provider_key, env_var, _ = choices[choice - 1]

    # For cloud embedders the user clearly wants that provider, so collect
    # the API key inline if it isn't already set. Local has env_var=None.
    # Validation runs BEFORE the org-config write so a blank API key
    # leaves ``llm_config.embedding_model_name`` untouched rather than
    # mutating it to a provider that still isn't configured.
    if env_var is not None and not os.environ.get(env_var):
        api_key = typer.prompt(f"Enter your {env_var}")
        if not api_key.strip():
            typer.echo("Error: API key cannot be empty")
            raise typer.Exit(1)
        _set_env_var(env_path, env_var, api_key)

    _write_embedding_model_to_org_config(_EMBEDDING_MODEL_NAMES[provider_key])
    return _provider_display(provider_key)


_LOCAL_SERVER_URL = "http://localhost:8081"


def _prompt_local_sqlite(env_path: Path) -> str:
    """Option 1 — local SQLite with a local Reflexio server.

    Writes ``REFLEXIO_URL`` pointing at the local server so the CLI
    and any installed integration hooks (e.g., OpenClaw) know where
    to connect.

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
    """Install the Reflexio plugin into OpenClaw via the plugin system.

    Returns:
        bool: True if the plugin was verified as registered.

    Raises:
        typer.Exit: If the openclaw CLI is not found on PATH.
    """
    if not shutil.which("openclaw"):
        typer.echo("Error: openclaw CLI not found. Install from https://openclaw.ai")
        raise typer.Exit(1)

    import reflexio

    pkg_dir = Path(reflexio.__file__).parent
    plugin_dir = pkg_dir / "integrations" / "openclaw" / "plugin"

    if not plugin_dir.exists():
        typer.echo(f"Error: plugin directory not found at {plugin_dir}")
        raise typer.Exit(1)

    # Clean install: remove any existing installation and stale extension dir
    subprocess.run(
        ["openclaw", "plugins", "uninstall", "--force", "reflexio-federated"],
        check=False,
        capture_output=True,
        text=True,
    )
    stale_ext = Path.home() / ".openclaw" / "extensions" / "reflexio-federated"
    shutil.rmtree(stale_ext, ignore_errors=True)

    # Install plugin and restart gateway so inspect sees the new state
    try:
        subprocess.run(
            ["openclaw", "plugins", "install", str(plugin_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["openclaw", "plugins", "enable", "reflexio-federated"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["openclaw", "gateway", "restart"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Error: openclaw command failed: {exc.stderr or exc.stdout}")
        raise typer.Exit(1) from exc

    # Verify — match exact "Status: loaded" to avoid false positives from
    # "not loaded" or "unloaded"
    result = subprocess.run(
        ["openclaw", "plugins", "inspect", "reflexio-federated"],
        capture_output=True,
        text=True,
    )
    if re.search(r"Status:\s*loaded\b", result.stdout):
        typer.echo("Plugin installed and registered")
        return True

    typer.echo(
        "Error: Plugin not loaded -- check 'openclaw plugins inspect reflexio-federated'"
    )
    return False


def _uninstall_openclaw() -> None:
    """Remove the Reflexio integration from OpenClaw."""
    typer.confirm(
        "This will remove the Reflexio integration from OpenClaw. Continue?",
        abort=True,
    )
    if shutil.which("openclaw"):
        subprocess.run(
            ["openclaw", "plugins", "disable", "reflexio-federated"],
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["openclaw", "plugins", "uninstall", "--force", "reflexio-federated"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        typer.echo("Warning: openclaw CLI not found on PATH, skipping plugin removal")

    # Remove setup markers
    from reflexio.cli.paths import reflexio_home

    reflexio_dir = reflexio_home()
    if reflexio_dir.exists():
        for marker in reflexio_dir.glob(".setup_complete_*"):
            marker.unlink(missing_ok=True)
            typer.echo(f"Removed setup marker: {marker}")

    typer.echo("Reflexio integration fully removed from OpenClaw.")


@app.command("openclaw")
def openclaw(
    uninstall: Annotated[
        bool,
        typer.Option(
            "--uninstall", help="Remove the Reflexio integration from OpenClaw"
        ),
    ] = False,
    embedding: Annotated[
        str,
        typer.Option(
            "--embedding",
            help=(
                "Embedding provider: 'local' (in-process MiniLM), 'openai', "
                "'gemini', or 'auto' (default — let runtime auto-detection "
                "pick). Skips the interactive prompt."
            ),
        ),
    ] = "auto",
) -> None:
    """Set up (or remove) the Reflexio integration for OpenClaw."""
    if uninstall:
        _uninstall_openclaw()
        return

    if embedding not in _VALID_EMBEDDING_FLAGS:
        typer.echo(
            f"Error: --embedding must be one of "
            f"{sorted(_VALID_EMBEDDING_FLAGS)}, got {embedding!r}"
        )
        raise typer.Exit(1)

    # Step 1: Load .env path. Always target ~/.reflexio/.env — running setup
    # from a worktree or project root that happens to contain its own .env
    # would otherwise pollute that file via load_reflexio_env's CWD-first
    # search. Setup writes are user-global, not project-local.
    from reflexio.cli.env_loader import ensure_user_env_for_setup

    env_path = ensure_user_env_for_setup()
    if env_path is None:
        typer.echo("Error: could not locate or create a .env file")
        raise typer.Exit(1)

    # Step 2: LLM provider
    display_name, model, _provider_key = _prompt_llm_provider(env_path)

    # Step 3: Storage. Decided BEFORE the embedding step because Managed /
    # Self-hosted modes own their own embedding config server-side, and a
    # local override would just shadow the operator's choice.
    storage_label = _prompt_storage(env_path)

    # Step 3.5: Upfront embedding-provider step. Local is the default when
    # chromadb is importable; the choice persists to org config so it
    # survives later cloud-key changes. Skipped for remote storage modes
    # for the reason above.
    is_remote = storage_label in {"Managed Reflexio", "Self-hosted Reflexio"}
    embedding_label: str | None = None
    if not is_remote:
        embedding_label = _choose_embedding_provider(env_path, embedding_flag=embedding)

    # Step 4: Install OpenClaw integration
    typer.echo("")
    hook_ok = _install_openclaw_integration()

    # Step 5: Summary
    hook_status = "reflexio-federated" if hook_ok else "reflexio-federated (unverified)"

    typer.echo("")
    typer.echo("Setup complete!")
    typer.echo(f"  LLM Provider: {display_name} ({model})")
    if embedding_label:
        typer.echo(f"  Embedding Provider: {embedding_label}")
    typer.echo(f"  Storage: {storage_label}")
    typer.echo(f"  Plugin: {hook_status}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Restart OpenClaw gateway: openclaw gateway restart")
    typer.echo(
        "  2. Start a conversation -- Reflexio will capture and learn automatically"
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
    embedding: Annotated[
        str,
        typer.Option(
            "--embedding",
            help=(
                "Embedding provider: 'local' (in-process MiniLM), 'openai', "
                "'gemini', or 'auto' (default — let runtime auto-detection "
                "pick). Skips the interactive prompt."
            ),
        ),
    ] = "auto",
) -> None:
    """Configure Reflexio without installing any integration.

    Writes ``REFLEXIO_URL`` / ``REFLEXIO_API_KEY`` / LLM provider keys
    / storage backend into ``~/.reflexio/.env``. This is the command
    to run if you're using the ``reflexio`` CLI directly from your
    shell and don't need the OpenClaw hook installation.

    Under the hood it reuses the same ``_prompt_storage`` +
    ``_prompt_llm_provider`` helpers the integration setup commands
    use, so the flow and the resulting ``.env`` are identical to what
    you'd get from those commands minus the hook-installation step.

    Args:
        skip_llm: When True, skip the LLM provider prompt. Useful if
            you're only going to point the CLI at a managed Reflexio
            server, which handles extraction with its own LLM keys.
        embedding: Embedding-provider selector for non-interactive use.
            ``"auto"`` (default) leaves the choice to runtime
            auto-detection. ``"local"`` / ``"openai"`` / ``"gemini"``
            persist the corresponding model to org config and skip the
            prompt. Honors ``REFLEXIO_NONINTERACTIVE=1`` (skips the
            prompt and behaves as ``"auto"`` when the flag is left at
            its default).
    """
    if embedding not in _VALID_EMBEDDING_FLAGS:
        typer.echo(
            f"Error: --embedding must be one of "
            f"{sorted(_VALID_EMBEDDING_FLAGS)}, got {embedding!r}"
        )
        raise typer.Exit(1)

    # Always target ~/.reflexio/.env — see ensure_user_env_for_setup docstring
    # for why we don't honor a CWD-local .env in setup commands.
    from reflexio.cli.env_loader import ensure_user_env_for_setup

    env_path = ensure_user_env_for_setup()
    if env_path is None:
        typer.echo("Error: could not locate or create a .env file")
        raise typer.Exit(1)

    # Step 1: Storage (ask first — managed mode doesn't need an LLM key)
    storage_label = _prompt_storage(env_path)

    # Step 2: LLM provider (skipped for managed mode — the remote server
    # handles extraction so the local .env doesn't need an LLM key).
    # Also skipped when the user explicitly passes --skip-llm.
    is_managed = storage_label == "Managed Reflexio"
    is_remote = storage_label in {"Managed Reflexio", "Self-hosted Reflexio"}
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
        display_name, model, _ = _prompt_llm_provider(env_path)

    # Step 2.5: Upfront embedding-provider step. Local is the default when
    # chromadb is importable; the choice is persisted to org config so it
    # survives later cloud-key changes. Skipped for both Managed and
    # Self-hosted modes — the remote server owns its own model config and
    # a local override would just shadow whatever the operator set there.
    # Replaces the legacy ``_prompt_embedding_provider`` call here — that
    # helper still exists for the ``services start`` first-run wizard.
    if not is_remote:
        embedding_label = _choose_embedding_provider(env_path, embedding_flag=embedding)

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


@app.command("openai-codex")
def openai_codex_setup(
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't auto-open the browser; print the URL to copy/paste instead.",
        ),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Seconds to wait for the OAuth callback before failing.",
        ),
    ] = 300,
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="Print currently saved Codex token metadata and exit (no login).",
        ),
    ] = False,
    logout: Annotated[
        bool,
        typer.Option(
            "--logout",
            help="Delete the saved Codex token file and exit.",
        ),
    ] = False,
) -> None:
    """Sign in to OpenAI via your ChatGPT subscription (Codex OAuth).

    Stores access + refresh tokens at ``~/.reflexio/auth/openai-codex.json``.
    The codex proxy and any other reflexio component that needs OpenAI auth
    reads from this file directly — no dependency on OpenClaw or any other
    CLI. The proxy auto-refreshes the access token when it nears expiry.

    Run this once; the codex proxy (``codex_proxy.py``) then picks up the
    stored tokens automatically on start.

    Re-run this command if your subscription tier changes or the
    refresh_token gets revoked (rare).
    """
    # Imported here so plain `reflexio --help` doesn't require the OAuth
    # module to load (slight startup speedup; mostly cosmetic).
    from reflexio.cli.codex_auth import (
        REFLEXIO_CODEX_TOKENS_PATH,
        get_fresh_tokens,
        load_tokens_raw,
        login_interactive,
    )

    if logout:
        if REFLEXIO_CODEX_TOKENS_PATH.exists():
            REFLEXIO_CODEX_TOKENS_PATH.unlink()
            typer.echo(f"Removed {REFLEXIO_CODEX_TOKENS_PATH}")
        else:
            typer.echo("No saved Codex tokens to remove.")
        return

    if show:
        tokens = load_tokens_raw()
        if tokens is None:
            typer.echo(f"No tokens at {REFLEXIO_CODEX_TOKENS_PATH}.")
            typer.echo("Run `reflexio setup openai-codex` to sign in.")
            raise typer.Exit(1)
        typer.echo(f"  path:      {REFLEXIO_CODEX_TOKENS_PATH}")
        typer.echo(f"  email:     {tokens.email}")
        typer.echo(f"  plan_type: {tokens.plan_type}")
        typer.echo(
            f"  account_id ...{tokens.account_id[-8:]}"
            if tokens.account_id
            else "  account_id (empty)"
        )
        typer.echo(f"  expires_at: {tokens.expires_at} (unix epoch)")
        typer.echo(f"  expired:   {tokens.is_expired()}")
        return

    typer.echo("Starting OpenAI Codex OAuth flow...")
    try:
        tokens = login_interactive(
            open_browser=not no_browser,
            timeout_s=timeout,
        )
    except TimeoutError as e:
        typer.echo(f"Timed out: {e}")
        raise typer.Exit(1) from e
    except ValueError as e:
        typer.echo(f"Login failed: {e}")
        raise typer.Exit(1) from e

    typer.echo("")
    typer.echo("Sign-in successful.")
    typer.echo(f"  saved to:  {REFLEXIO_CODEX_TOKENS_PATH}")
    if tokens.email:
        typer.echo(f"  email:     {tokens.email}")
    typer.echo(f"  plan_type: {tokens.plan_type}")
    typer.echo("")
    typer.echo("Verify the token resolves cleanly via the proxy's health endpoint:")
    typer.echo("  curl -s http://127.0.0.1:11435/health | jq")
    typer.echo("")
    typer.echo(
        "If the saved plan_type doesn't match what you expect (e.g. shows "
        "'plus' instead of 'max-x20'), wait a minute for OpenAI to propagate "
        "the subscription change and re-run this command — the JWT is issued "
        "at sign-in time."
    )
    # Exercise the refresh path immediately so any clock skew between the
    # JWT's `exp` claim and our local clock is caught now, not at first use.
    _ = get_fresh_tokens()
