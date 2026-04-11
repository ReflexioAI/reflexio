"""Reflexio CLI entry point.

Usage:
    reflexio [--json] [--url URL] [--api-key KEY] COMMAND [OPTIONS]

Command groups:
    reflexio services         start|stop
    reflexio interactions     publish|list|search|delete|delete-all
    reflexio profiles         list|search|delete|delete-all|generate
    reflexio feedbacks        list|search|delete|delete-all|aggregate|regenerate
    reflexio raw-feedbacks    list|search|add|delete|delete-all
    reflexio config           show|set
    reflexio auth             login|status|logout|setup
    reflexio status           check
    reflexio api              METHOD PATH
    reflexio doctor           check

Shortcuts:
    reflexio publish      (alias for: interactions publish)
    reflexio search       (unified search across profiles and feedbacks)
    reflexio context      (fetch formatted context for agent injection)
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point.

    Loads ``~/.reflexio/.env`` so every CLI invocation (including those
    spawned by hooks) picks up ``REFLEXIO_URL`` and ``REFLEXIO_API_KEY``.

    Checks for registered CLI extensions (e.g. enterprise) via the
    ``reflexio.cli`` entry-point group before falling back to the
    default open-source app factory.
    """
    from reflexio.cli.env_loader import load_reflexio_env

    load_reflexio_env()

    from importlib.metadata import entry_points

    eps = entry_points(group="reflexio.cli")
    if eps:
        create_app = next(iter(eps)).load()
    else:
        from reflexio.cli.app import create_app

    app = create_app()
    app(argv if argv is not None else sys.argv[1:], standalone_mode=True)


if __name__ == "__main__":
    main()
