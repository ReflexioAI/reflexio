"""Test skip decorators shared between OS and enterprise test suites."""

from __future__ import annotations

import base64
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def skip_in_precommit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to skip tests during pre-commit hooks."""
    return pytest.mark.skipif(
        os.environ.get("PRECOMMIT") == "1", reason="Test skipped in pre-commit hook"
    )(func)


def skip_low_priority(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to skip low priority tests unless explicitly requested.

    These tests are skipped by default and only run when RUN_LOW_PRIORITY=1 is set.

    Usage::

        @skip_low_priority
        def test_something_low_priority():
            ...

    To run low priority tests::

        RUN_LOW_PRIORITY=1 pytest ...
    """
    return pytest.mark.skipif(
        os.environ.get("RUN_LOW_PRIORITY") != "1",
        reason="Low priority test - set RUN_LOW_PRIORITY=1 to run",
    )(func)


def encode_image_to_base64(image_fp: str) -> str:
    return base64.b64encode(Path(image_fp).read_bytes()).decode("utf-8")
