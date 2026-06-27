"""Tests for ``block_implicit_dotenv_walkup``.

The OSS launcher, run from the ``open_source/reflexio`` submodule, must NOT pick
up a parent-directory ``.env`` (e.g. the enterprise-root ``.env``). Reflexio's
own loader is already scoped to ``./.env`` + ``~/.reflexio/.env``; the leak comes
from third-party libraries (litellm) calling a path-less ``dotenv.load_dotenv()``
at import time, which walks UP the tree. The guard neutralizes that path-less
form while leaving explicit-path loads intact. These tests pin both halves.
"""

from __future__ import annotations

import os

import dotenv
import pytest

from reflexio.cli.env_loader import block_implicit_dotenv_walkup


@pytest.fixture
def restore_load_dotenv():
    """Restore the global ``dotenv.load_dotenv`` so the guard never leaks
    between tests."""
    original = dotenv.load_dotenv
    try:
        yield
    finally:
        dotenv.load_dotenv = original


def test_pathless_walkup_is_noop(
    tmp_path, monkeypatch: pytest.MonkeyPatch, restore_load_dotenv
) -> None:
    # A parent .env that a path-less load_dotenv() would discover by walking up.
    (tmp_path / ".env").write_text("BACKEND_PORT=8091\n")
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    monkeypatch.delenv("BACKEND_PORT", raising=False)

    block_implicit_dotenv_walkup()
    result = dotenv.load_dotenv()  # path-less -> would walk up to tmp_path/.env

    assert result is False
    assert "BACKEND_PORT" not in os.environ  # the parent .env did NOT leak


def test_explicit_path_still_loads(
    tmp_path, monkeypatch: pytest.MonkeyPatch, restore_load_dotenv
) -> None:
    env = tmp_path / "explicit.env"
    env.write_text("REFLEXIO_GUARD_EXPLICIT=yes\n")
    monkeypatch.delenv("REFLEXIO_GUARD_EXPLICIT", raising=False)

    block_implicit_dotenv_walkup()
    result = dotenv.load_dotenv(dotenv_path=env)

    assert result is True
    assert os.environ["REFLEXIO_GUARD_EXPLICIT"] == "yes"


def test_idempotent(restore_load_dotenv) -> None:
    block_implicit_dotenv_walkup()
    guarded = dotenv.load_dotenv
    block_implicit_dotenv_walkup()
    assert dotenv.load_dotenv is guarded  # not re-wrapped on the second call
