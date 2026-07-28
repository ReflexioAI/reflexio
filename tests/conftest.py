"""Test configuration — delegates to shared reflexio.test_support module."""

import os

# Must run before any other import in this module. The enterprise sibling of this
# conftest (`reflexio_ext/tests/conftest.py`) scrubs SENTRY_DSN inside a session-scoped
# pytest fixture, which runs too late: `reflexio_ext/server/api.py` calls
# `sentry_sdk.init()` at MODULE IMPORT TIME, and several test files import that module
# during collection — before any fixture, even a session-scoped autouse one, gets a
# chance to run (fixture setup fires at first-test setup, which is after collection has
# already imported every module). This package has no equivalent import-time
# `sentry_sdk.init()` call today, but the scrub is moved to module level here too for
# structural symmetry with the enterprise conftest and so it stays correct if one is
# ever added. A developer's shell or `.env` DSN otherwise reaches the real Sentry project
# and pollutes production triage with test-run events.
_PREVIOUS_SENTRY_DSN = os.environ.get("SENTRY_DSN")
os.environ["SENTRY_DSN"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent  # tests/
PROJECT_ROOT = _THIS_DIR.parent.parent  # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Env vars that change OSS code paths and must not leak in from a developer's
# `~/.reflexio/.env` or the enterprise worktree `.env`. CI sets none of these,
# so the suite passes there even without the cleanup. Cleared once per session
# before any test imports modules that read them at call time.
_OSS_TEST_POLLUTING_ENV_VARS = (
    "DEPLOYMENT_MODE",
    "REFLEXIO_STORAGE",
    "REFLEXIO_EMBEDDING_PROVIDER",
    "REFLEXIO_EMBEDDING_SERVICE_URL",
    "REFLEXIO_EMBEDDING_DAEMON_HOST",
    "REFLEXIO_EMBEDDING_LOCAL_SERVICE_PROBE_TIMEOUT_MS",
    "REFLEXIO_RERANK_SERVICE_URL",
    "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS",
    "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
)
for _var in _OSS_TEST_POLLUTING_ENV_VARS:
    os.environ.pop(_var, None)

# Load the developer's provider credentials without importing the server yet.
# The server configures file handlers during import, so the temporary paths
# below must be in place before that import occurs.
from reflexio.cli.env_loader import load_reflexio_env  # noqa: E402

load_reflexio_env()

# The loader intentionally imports provider credentials from the developer
# environment, but it can also reintroduce local service-routing variables from
# an enterprise checkout. Keep those routing choices out of OSS tests.
for _var in _OSS_TEST_POLLUTING_ENV_VARS:
    os.environ.pop(_var, None)

# Redirect `~/.reflexio` for the entire test session so tests that call
# `reflexio.cli.paths.reflexio_home()` (e.g. via `LocalFileConfigStorage`'s
# default `base_dir`) don't pick up the developer's existing
# `~/.reflexio/configs/config_<org>.json` files. Without this, any test that
# constructs `create_app()` against the default `self-host-org` org_id loads
# whatever leftover storage config the developer happens to have on disk —
# producing `No storage factory registered for StorageConfigSupabase` when
# the leftover was from a prior `--storage supabase` run.
_REFLEXIO_TEST_HOME = Path(tempfile.mkdtemp(prefix="reflexio-test-home-"))
os.environ["REFLEXIO_LOG_DIR"] = str(_REFLEXIO_TEST_HOME)
os.environ["LOCAL_STORAGE_PATH"] = str(_REFLEXIO_TEST_HOME / ".reflexio" / "data")
import reflexio.server as _test_server  # noqa: E402
from reflexio.server.extensions import reset_services  # noqa: E402

_test_server.LOCAL_STORAGE_PATH = os.environ["LOCAL_STORAGE_PATH"]

from reflexio.test_support.llm_credentials import (  # noqa: E402
    ensure_provider_credential,
)
from reflexio.test_support.llm_mock import cleanup_llm_mock, configure_llm_mock

# Service constructors resolve a default model eagerly, so a machine with no
# provider key errors out at fixture setup instead of running the suite. Runs
# after `load_reflexio_env()` and after the `reflexio.server` import (which
# pulls in litellm, whose import-time dotenv walk-up can also supply keys), so
# every credential source has had its chance before we decide to fill the gap.
ensure_provider_credential()


def pytest_configure(config):
    configure_llm_mock(config)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify path-based test tiers before ``-m`` selection is evaluated."""
    for item in items:
        path = Path(str(item.path))
        if "e2e_tests" in path.parts:
            item.add_marker(pytest.mark.e2e)
        elif path.name.endswith(("_integration.py", "_integration_test.py")):
            item.add_marker(pytest.mark.integration)


def pytest_unconfigure(config):
    cleanup_llm_mock(config)
    # Pair for the module-import-time scrub above: restore whatever SENTRY_DSN was
    # ambient before this conftest ran, now that the whole session is done with it.
    if _PREVIOUS_SENTRY_DSN is None:
        os.environ.pop("SENTRY_DSN", None)
    else:
        os.environ["SENTRY_DSN"] = _PREVIOUS_SENTRY_DSN


@pytest.fixture(autouse=True)
def _reset_runtime_services() -> Iterator[None]:
    """Clear process-global services and local routing before and after each test."""
    for var in _OSS_TEST_POLLUTING_ENV_VARS:
        os.environ.pop(var, None)
    reset_services()
    yield
    reset_services()
    for var in _OSS_TEST_POLLUTING_ENV_VARS:
        os.environ.pop(var, None)


@pytest.fixture
def tool_call_completion():
    """Factory helpers for mocking a tool-calling conversation.

    Yields:
        tuple: ``(make_tool_call_response, make_finish_response)`` —
            call the first to build an assistant turn that requests a
            tool, and the second to build the terminal stop turn.

    Usage::

        def test_my_loop(tool_call_completion):
            make_tc, make_stop = tool_call_completion
            responses = [make_tc("emit", {"v": 1}), make_stop()]
            with patch("litellm.completion", side_effect=responses):
                ...
    """
    from reflexio.test_support.llm_mock import (
        make_finish_response,
        make_tool_call_response,
    )

    return make_tool_call_response, make_finish_response
