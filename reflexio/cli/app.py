"""Full CLI command registry, loaded after operator environment bootstrap."""

from __future__ import annotations

import typer

from reflexio.cli.root_app import create_root_app


def create_app(exclude: set[str] | None = None) -> typer.Typer:
    """Register commands, optionally excluding groups replaced by extensions."""
    app = create_root_app()

    # Register command groups
    from reflexio.cli.commands.admin_cmd import app as admin_app
    from reflexio.cli.commands.agent_playbooks import app as agent_playbooks_app
    from reflexio.cli.commands.api import app as api_app
    from reflexio.cli.commands.auth import app as auth_app
    from reflexio.cli.commands.config_cmd import app as config_app
    from reflexio.cli.commands.doctor import app as doctor_app
    from reflexio.cli.commands.embeddings import app as embeddings_app
    from reflexio.cli.commands.interactions import app as interactions_app
    from reflexio.cli.commands.profiles import app as profiles_app
    from reflexio.cli.commands.services import app as services_app
    from reflexio.cli.commands.setup_cmd import app as setup_app
    from reflexio.cli.commands.shortcuts import register_shortcuts
    from reflexio.cli.commands.status_cmd import app as status_app
    from reflexio.cli.commands.user_playbooks import app as user_playbooks_app

    _skip = exclude or set()

    _sub_apps: list[tuple[typer.Typer, str]] = [
        (services_app, "services"),
        (embeddings_app, "embeddings"),
        (interactions_app, "interactions"),
        (profiles_app, "user-profiles"),
        (agent_playbooks_app, "agent-playbooks"),
        (user_playbooks_app, "user-playbooks"),
        (config_app, "config"),
        (auth_app, "auth"),
        (status_app, "status"),
        (api_app, "api"),
        (doctor_app, "doctor"),
        (setup_app, "setup"),
        (admin_app, "admin"),
    ]
    for sub_app, name in _sub_apps:
        if name not in _skip:
            app.add_typer(sub_app, name=name)

    # Top-level shortcuts (publish, search, context)
    register_shortcuts(app)

    return app
