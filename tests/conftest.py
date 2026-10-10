"""Test configuration — delegates to shared reflexio.test_support module."""

import errno
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent  # tests/
PROJECT_ROOT = _THIS_DIR.parent  # repo root, or mutmut's relocated source root

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
    "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS",
    "REFLEXIO_RERANK_ENABLED",
    "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
)
for _var in _OSS_TEST_POLLUTING_ENV_VARS:
    os.environ.pop(_var, None)

# Mutation collection must not auto-create an env file in the real user home.
# Both this loader and the server import load it, so isolate before either call.
_REFLEXIO_TEST_HOME = Path(tempfile.mkdtemp(prefix="reflexio-test-home-"))
if "MUTANT_UNDER_TEST" in os.environ:
    _mutation_env_file = _REFLEXIO_TEST_HOME / ".env"
    _mutation_env_file.touch()
    os.environ["REFLEXIO_ENV_FILE"] = str(_mutation_env_file)

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
os.environ["REFLEXIO_LOG_DIR"] = str(_REFLEXIO_TEST_HOME)
os.environ["LOCAL_STORAGE_PATH"] = str(_REFLEXIO_TEST_HOME / ".reflexio" / "data")

# The durable extraction scheduler discovers work on a 2s poll, tuned for a
# long-lived server. A test that publishes and then asserts on the extracted
# result pays that latency once per publish and does nothing with the time:
# `tests/lib/test_profile_workflows_unit.py` spent 58.9s of which 50s was this
# sleep (8.6s with the poll at 0.05). This shortens discovery only -- the
# extraction still has to run, and its correctness is what the tests assert.
# `setdefault`, so an explicit value still wins for anyone testing the real
# interval.
os.environ.setdefault("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "0.05")
import reflexio.server as _test_server  # noqa: E402
from reflexio.server.extensions import reset_services  # noqa: E402

_test_server.LOCAL_STORAGE_PATH = os.environ["LOCAL_STORAGE_PATH"]

from reflexio.test_support.embedding_mock import patch_embeddings  # noqa: E402
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


# ``addopts`` passes ``-n auto``, which xdist resolves to the machine's full CPU count.
# Every worker imports torch/onnxruntime/sentence-transformers and runs real embeddings, so on
# a developer laptop that saturates each core and freezes the desktop. CI runners have the
# box to themselves and want the full count, so the cap is local-only.
_LOCAL_MAX_XDIST_WORKERS = 4


@pytest.hookimpl
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Cap what ``-n auto``/``-n logical`` expand to locally; None in CI keeps xdist's default.

    xdist calls this for both modes and for neither when an explicit ``-n <N>`` is given
    (``xdist/plugin.py``: ``if config.option.numprocesses in ("auto", "logical")``), so an
    explicit count is already the caller's own choice. Capping ``logical`` too is
    deliberate — it asks for *more* workers than physical cores, so exempting it would
    reopen the all-cores local run this cap exists to prevent.
    """
    if os.environ.get("CI"):
        return None
    return min(_LOCAL_MAX_XDIST_WORKERS, os.cpu_count() or 1)


def pytest_configure(config):
    configure_llm_mock(config)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Include fixture costs in mutation scheduling, only during statistics."""
    if os.environ.get("MUTANT_UNDER_TEST") != "stats" or report.when not in (
        "setup",
        "teardown",
    ):
        return
    from mutmut.state import state

    state().duration_by_test[report.nodeid] += report.duration


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Iterator[None]:
    """Attribute calls made while fixtures drain to their own test, not the next."""
    yield
    if os.environ.get("MUTANT_UNDER_TEST") != "stats":
        return
    from mutmut.state import state

    stats = state()
    for function in stats._stats:
        stats.tests_by_mangled_function_name[function].add(
            item.nodeid.removeprefix("mutants/")
        )
    stats._stats.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Scratch exhaustion is infrastructure failure, never a mutation kill."""
    yield
    if "MUTANT_UNDER_TEST" not in os.environ or call.excinfo is None:
        return
    pending: list[BaseException | None] = [call.excinfo.value]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        if (isinstance(error, OSError) and error.errno == errno.ENOSPC) or (
            isinstance(error, sqlite3.OperationalError)
            and getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL
        ):
            pytest.exit("Mutation temporary storage exhausted", returncode=35)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
        pending.extend((error.__cause__, error.__context__))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> Iterator[None]:
    """Remove this session's owned scratch before mutant workers use os._exit."""
    yield
    if "MUTANT_UNDER_TEST" not in os.environ:
        return
    factory = getattr(session.config, "_tmp_path_factory", None)
    if not isinstance(factory, pytest.TempPathFactory):
        return
    owned = factory._basetemp
    if owned is not None:
        try:
            shutil.rmtree(owned)
        except OSError:
            pytest.exit("Mutation temporary storage cleanup failed", returncode=35)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify path-based test tiers before ``-m`` selection is evaluated."""
    for item in items:
        path = Path(str(item.path))
        if "e2e_tests" in path.parts:
            item.add_marker(pytest.mark.e2e)
        elif path.name.endswith(("_integration.py", "_integration_test.py")):
            item.add_marker(pytest.mark.integration)

    # Each mutant uses pytest -x. Run quick checks first without dropping any
    # selected tests; survivors still run the complete associated test set.
    if "__mutmut_" not in os.environ.get("MUTANT_UNDER_TEST", ""):
        return
    try:
        durations = json.loads((PROJECT_ROOT / "mutmut-stats.json").read_text())[
            "duration_by_test"
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return
    if not isinstance(durations, dict):
        return
    measured = [durations.get(item.nodeid, math.inf) for item in items]
    try:
        invalid = any(
            not isinstance(value, (int, float)) or value < 0 or math.isnan(value)
            for value in measured
        )
    except OverflowError:
        return
    if invalid:
        return
    items.sort(key=lambda item: (durations.get(item.nodeid, math.inf), item.nodeid))


def pytest_unconfigure(config):
    cleanup_llm_mock(config)


@pytest.fixture(autouse=True)
def _deterministic_embeddings(request: pytest.FixtureRequest) -> Iterator[None]:
    """Give every non-e2e test a vector source, so storage writes are real writes.

    Embeddings were computed in-process until local inference moved behind a
    service; nothing replaced that for tests. The suite kept passing only
    because the SQLite ingest path swallows EmbeddingUnavailableError and stores
    an empty vector -- so rows were written unsearchable, and assertions about
    vector search or clustering downstream of them passed vacuously.

    Two carve-outs, both "do not stub the thing under test":

    - ``e2e_tests`` exercise the real stack, the same reason they bypass the
      LLM mock.
    - ``server/llm`` owns the embedding dispatch this patches. Stubbing
      ``_embed_texts`` there would replace the unit under test, so those suites
      keep the real implementation and mock the provider beneath it instead.

    The vectors are deterministic but NOT semantic (hash-derived), so this makes
    write paths real without making similarity assertions meaningful -- place
    vectors explicitly for those.
    """
    parts = Path(str(request.node.path)).parts
    if "e2e_tests" in parts or ("llm" in parts and "server" in parts):
        yield
        return
    with patch_embeddings():
        yield


@pytest.fixture(autouse=True)
def _reset_runtime_services() -> Iterator[None]:
    """Clear process-global services and local routing before and after each test."""
    for var in _OSS_TEST_POLLUTING_ENV_VARS:
        os.environ.pop(var, None)
    reset_services()
    yield
    reset_services()
    # Local schedulers capture a test's context and storage. Stop them before
    # the next test; otherwise they keep polling deleted SQLite directories
    # and mutmut records their utility calls against unrelated tests.
    scheduler_module = sys.modules.get(
        "reflexio.server.services.playbook.aggregation_scheduler"
    )
    if scheduler_module is not None:
        for scheduler in list(scheduler_module._LOCAL_SCHEDULERS.values()):
            scheduler.stop(timeout_seconds=5)
            assert not scheduler.is_running(), (
                "test-owned aggregation scheduler did not stop"
            )
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


class TransientTestError(ConnectionError):
    """Stands in for a dropped database connection in failure-policy tests."""


@pytest.fixture
def transient_failure_classifier() -> Iterator[type[TransientTestError]]:
    """Classify ``TransientTestError`` as transient; reset policy state around it.

    Yields:
        type[TransientTestError]: Raise this to simulate a rollout drop.
    """
    from reflexio.server import background_work

    background_work.configure_transient_failure_classifier(
        lambda exc: isinstance(exc, TransientTestError)
    )
    background_work._streaks.clear()
    yield TransientTestError
    background_work.configure_transient_failure_classifier(None)
    background_work._streaks.clear()
