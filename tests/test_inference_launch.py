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
from reflexio.inference_launch import ordinary_inference_launch
from reflexio.env_utils import env_truthy
assert env_truthy(' ON ')
assert ordinary_inference_launch({{}}).mode == 'colocated'
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
        "REFLEXIO_API_QUERY_EMBEDDING_ENABLED": "true",
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
            "REFLEXIO_API_QUERY_EMBEDDING_ENABLED": "true",
            "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": url,
        }
    )
    assert plan.mode == "query"
    assert plan.health_url == "http://127.0.0.1:8093/health/model"
