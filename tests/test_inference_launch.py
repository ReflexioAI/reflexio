"""Endpoint ownership is shared by clients, CLI and container adapters."""

import subprocess
import sys
from pathlib import Path

import pytest

from reflexio.config_migration import ConfigurationMigrationError
from reflexio.inference_launch import (
    container_inference_launch,
    inference_service_url,
    ordinary_inference_launch,
)


def test_launcher_reuses_parser_without_importing_server_or_site_packages():
    package_root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
sys.path.insert(0, {str(package_root)!r})
from reflexio.inference_launch import ordinary_inference_launch, container_inference_launch
from reflexio.config_env import ConfigurationMigrationError
from reflexio.env_utils import env_truthy
assert env_truthy(' ON ')
assert ordinary_inference_launch({{}}).mode == 'colocated'
assert container_inference_launch({{'REFLEXIO_INFERENCE_WORKLOAD':'query', 'REFLEXIO_QUERY_EMBEDDING_SERVICE_URL':'http://127.0.0.1:8093'}}).mode == 'query'
try:
    ordinary_inference_launch({{'REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON': 'true'}})
except ConfigurationMigrationError:
    pass
else:
    raise AssertionError('Legacy ownership settings must still fail without site packages')
assert not any(name == 'reflexio.server' or name.startswith('reflexio.server.')
               for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter and repository-owned probe
        [sys.executable, "-I", "-S", "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "url",
    [
        "https://remote.example/inference/",
        "http://127.0.0.1:9999",
        "http://localhost:9999",
    ],
)
def test_explicit_endpoint_attaches_without_losing_its_address(url):
    env = {"REFLEXIO_EMBEDDING_SERVICE_URL": url}
    assert inference_service_url(env) == url.rstrip("/")
    assert ordinary_inference_launch(env).mode == "attached"
    assert ordinary_inference_launch(env).health_url is None


@pytest.mark.parametrize("legacy", ["true", "TrUe", "1", "yes", "on"])
def test_disabling_without_an_endpoint_requires_migration(legacy):
    with pytest.raises(ConfigurationMigrationError, match="explicitly"):
        ordinary_inference_launch({"REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON": legacy})


@pytest.mark.parametrize("legacy", ["", "false", "0", "no", "off"])
def test_default_ownership_and_port_context(legacy):
    env = {"REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON": legacy, "EMBEDDING_PORT": " "}
    assert ordinary_inference_launch(env).health_url == "http://127.0.0.1:8072/health"
    assert container_inference_launch(env).health_url == "http://127.0.0.1:8089/health"
    env["EMBEDDING_PORT"] = "9999"
    assert ordinary_inference_launch(env).health_url == "http://127.0.0.1:9999/health"


def test_custom_legacy_host_is_not_silently_dropped_or_printed():
    secret = "private-customer-host.example"
    env = {"REFLEXIO_EMBEDDING_DAEMON_HOST": secret}
    with pytest.raises(ConfigurationMigrationError) as exc:
        inference_service_url(env)
    assert secret not in str(exc.value)
    assert "TIMEOUT_MS" in str(exc.value)
    env.update(
        REFLEXIO_EMBEDDING_SERVICE_URL="https://canonical.example",
        REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON="true",
    )
    assert inference_service_url(env) == "https://canonical.example"
    assert ordinary_inference_launch(env).mode == "attached"


def test_query_child_remains_independent_of_ordinary_disable_switch():
    env = {
        "REFLEXIO_INFERENCE_WORKLOAD": "query",
        "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
        "REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON": "true",
    }
    plan = container_inference_launch(env)
    assert plan.mode == "query"
    assert plan.health_url == "http://127.0.0.1:8093/health/model"
    env["REFLEXIO_QUERY_EMBEDDING_SERVICE_URL"] = "http://remote.example"
    with pytest.raises(ConfigurationMigrationError, match="reserved loopback"):
        container_inference_launch(env)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8093", " http://127.0.0.1:8093/ "])
def test_query_endpoint_normalization_matches_request_routing(url):
    plan = container_inference_launch(
        {
            "REFLEXIO_INFERENCE_WORKLOAD": "query",
            "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": url,
        }
    )
    assert plan.mode == "query"
    assert plan.health_url == "http://127.0.0.1:8093/health/model"


@pytest.mark.parametrize("value", ["query", " query "])
def test_explicit_query_workload_owns_only_the_reserved_child(value):
    plan = container_inference_launch(
        {
            "REFLEXIO_INFERENCE_WORKLOAD": value,
            "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
            "REFLEXIO_EMBEDDING_SERVICE_URL": "http://document-worker:8089",
        }
    )
    assert plan.mode == "query"
    assert plan.health_url == "http://127.0.0.1:8093/health/model"


@pytest.mark.parametrize("role", [None, "", "  ", "shared"])
def test_shared_role_keeps_caller_owned_loopback_endpoint_attached(role):
    env = {
        "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
        "REFLEXIO_EMBEDDING_SERVICE_URL": "http://document-worker:8089",
    }
    if role is not None:
        env["REFLEXIO_INFERENCE_WORKLOAD"] = role
    assert container_inference_launch(env).mode == "attached"


@pytest.mark.parametrize("value", ["True", "false", "query-only", "unknown"])
def test_unknown_role_rejected_before_attaching_to_external_endpoint(value):
    with pytest.raises(
        ConfigurationMigrationError, match="REFLEXIO_INFERENCE_WORKLOAD"
    ):
        container_inference_launch(
            {
                "REFLEXIO_INFERENCE_WORKLOAD": value,
                "REFLEXIO_EMBEDDING_SERVICE_URL": "http://document-worker:8089",
            }
        )


@pytest.mark.parametrize("old", ["", "false", "true", "True"])
def test_retired_api_query_boolean_is_never_a_second_role_authority(old):
    with pytest.raises(ValueError, match="REFLEXIO_API_QUERY_EMBEDDING_ENABLED"):
        container_inference_launch(
            {
                "REFLEXIO_API_QUERY_EMBEDDING_ENABLED": old,
                "REFLEXIO_INFERENCE_WORKLOAD": "query",
                "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
                "REFLEXIO_EMBEDDING_SERVICE_URL": "http://document-worker:8089",
            }
        )


@pytest.mark.parametrize(
    ("settings", "code", "output"),
    [
        (
            {
                "REFLEXIO_INFERENCE_WORKLOAD": "query",
                "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
            },
            0,
            "query http://127.0.0.1:8093/health/model",
        ),
        (
            {
                "REFLEXIO_INFERENCE_WORKLOAD": "shared",
                "REFLEXIO_EMBEDDING_SERVICE_URL": "http://127.0.0.1:8093",
            },
            0,
            "attached -",
        ),
        ({"REFLEXIO_INFERENCE_WORKLOAD": "query"}, 1, "reserved loopback URL"),
        ({"REFLEXIO_INFERENCE_WORKLOAD": "True"}, 1, "REFLEXIO_INFERENCE_WORKLOAD"),
        (
            {"REFLEXIO_API_QUERY_EMBEDDING_ENABLED": ""},
            1,
            "REFLEXIO_API_QUERY_EMBEDDING_ENABLED",
        ),
        (
            {"REFLEXIO_API_QUERY_EMBEDDING_ENABLED": "false"},
            1,
            "REFLEXIO_API_QUERY_EMBEDDING_ENABLED",
        ),
    ],
)
def test_actual_container_plan_cli(settings, code, output, tmp_path):
    import os

    root = Path(__file__).resolve().parents[1]
    empty = tmp_path / "empty.env"
    empty.touch()
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    env.update(
        HOME=str(tmp_path),
        REFLEXIO_ENV_FILE=str(empty),
        PYTHONPATH=str(root),
        **settings,
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter/module, isolated fixture env.
        [sys.executable, "-m", "reflexio.inference_launch"],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == code, result.stderr
    assert output in (result.stdout if code == 0 else result.stderr)
    assert "Traceback" not in result.stderr
