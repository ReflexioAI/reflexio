"""Offline conversion exercises the real CLI without user files or network."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app
from reflexio.cli.config_upgrade import convert_config_file
from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import validate_stored_config


def _payload():
    return {
        "storage_config": {
            "db_url": "postgresql://user:native-secret@localhost/test",
            "key": "legacy-http-secret",
            "schema_name": "custom_schema",
        },
        "batch_size": 23,
        "batch_interval": 13,
        "agent_feedback_configs": [
            {
                "feedback_definition_prompt": "Preserve custom procedures",
                "feedback_aggregator_config": {"min_feedback_threshold": 9},
                "extension": {"batch_size": "opaque"},
            }
        ],
        "playbook_optimizer_config": {
            "optimize_user_playbooks": True,
            "auto_update_user_playbooks": False,
            "use_merge": False,
            "max_merge_invocations": 9,
        },
        "extension": {"batch_size": "opaque"},
    }


def test_document_conversion_preserves_policy_identity_and_opaque_metadata():
    payload = _payload()
    before = deepcopy(payload)
    result = canonicalize_config_document(payload)
    assert payload == before
    assert canonicalize_config_document(result) == result
    assert (
        validate_stored_config(result).model_dump_json()
        == validate_stored_config(payload).model_dump_json()
    )
    assert result["storage_config"] == {
        "type": "supabase",
        "schema": "custom_schema",
        "db_url": "postgresql://user:native-secret@localhost/test",
    }
    assert result["extension"] == {"batch_size": "opaque"}
    assert result["user_playbook_extractor_config"]["extension"] == {
        "batch_size": "opaque"
    }
    assert result["playbook_optimizer_config"]["user_playbooks_enabled"] is False
    assert result["playbook_optimizer_config"]["max_merge_invocations"] == 0


def test_real_cli_converts_copy_without_client_or_operator_env_writes(
    tmp_path, monkeypatch
):
    source, output = tmp_path / "old.json", tmp_path / "canonical.json"
    encoded = json.dumps(_payload()).encode()
    source.write_bytes(encoded)
    env = tmp_path / ".env"
    env.write_bytes(b"OPERATOR_SETTING=preserve\n")
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(env))
    output.write_bytes(b"old output")
    with patch(
        "reflexio.cli.commands.config_cmd.get_client",
        side_effect=AssertionError("offline operation"),
    ):
        result = CliRunner().invoke(
            create_app(),
            ["--json", "config", "upgrade-file", str(source), "--output", str(output)],
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True, "data": {"output": str(output)}}
    assert "native-secret" not in result.output
    assert "legacy-http-secret" not in result.output
    assert source.read_bytes() == encoded
    assert env.read_bytes() == b"OPERATOR_SETTING=preserve\n"
    assert json.loads(output.read_text()) == canonicalize_config_document(_payload())
    if os.name != "nt":
        assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "contents",
    ["not-json-secret", "[]", '{"storage_config":null,"batch_size":"invalid-secret"}'],
)
def test_invalid_input_never_replaces_output_or_leaks_values(tmp_path, contents):
    source, output = tmp_path / "old.json", tmp_path / "canonical.json"
    source.write_text(contents)
    output.write_bytes(b"preserve output")
    result = CliRunner().invoke(
        create_app(), ["config", "upgrade-file", str(source), "--output", str(output)]
    )
    assert result.exit_code != 0
    assert "invalid-secret" not in result.output
    assert "not-json-secret" not in result.output
    assert output.read_bytes() == b"preserve output"
    assert source.read_text() == contents


def test_input_cannot_be_selected_as_output(tmp_path):
    source = tmp_path / "old.json"
    encoded = json.dumps(_payload()).encode()
    source.write_bytes(encoded)
    with pytest.raises(ValueError, match="paths must differ"):
        convert_config_file(source, source)
    assert source.read_bytes() == encoded


def test_failed_publication_cleans_temporary_output_and_preserves_originals(tmp_path):
    source, output = tmp_path / "old.json", tmp_path / "canonical.json"
    encoded = json.dumps(_payload()).encode()
    source.write_bytes(encoded)
    output.write_bytes(b"preserve output")
    with (
        patch.object(Path, "replace", side_effect=OSError("publication failed")),
        pytest.raises(OSError),
    ):
        convert_config_file(source, output)
    assert source.read_bytes() == encoded
    assert output.read_bytes() == b"preserve output"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "canonical.json",
        "old.json",
    ]


@pytest.mark.parametrize(
    "storage",
    [
        None,
        {"db_path": "/tmp/custom.sqlite"},
        {"db_url": "postgresql://local:local@localhost/test"},
        {
            "type": "postgres",
            "schema_name": "custom",
            "db_url": "postgresql://local:local@localhost/test",
            "pool_size": 7,
            "read_db_url": "postgresql://local:local@localhost/replica",
            "read_pool_size": 3,
        },
        {"type": "postgres", "managed_by": "platform", "schema_present": False},
    ],
)
def test_conversion_keeps_storage_policy_without_materializing_defaults(storage):
    payload = {"storage_config": storage}
    result = canonicalize_config_document(payload)
    assert (
        validate_stored_config(result).model_dump_json()
        == validate_stored_config(payload).model_dump_json()
    )
    if storage is not None:
        assert "pool_acquire_timeout" not in result["storage_config"]
        assert "read_pool_acquire_timeout" not in result["storage_config"]


@pytest.mark.parametrize("selector", ["url", "key", "read_url", "read_key"])
def test_each_http_selector_is_retired_only_after_identity_is_explicit(selector):
    result = canonicalize_config_document(
        {
            "storage_config": {
                selector: "unused-http-value",
                "db_url": "postgresql://local:local@localhost/test",
            }
        }
    )
    assert result["storage_config"] == {
        "type": "supabase",
        "db_url": "postgresql://local:local@localhost/test",
    }


@pytest.mark.parametrize(
    "storage",
    [
        {
            "type": "postgres",
            "db_url": "postgresql://local:local@localhost/test",
            "key": "conflicting-secret",
        },
        {"url": "https://example.test", "key": "missing-dsn-secret"},
    ],
)
def test_storage_conflicts_never_become_a_different_backend(storage):
    with pytest.raises(ValueError, match="cannot be converted safely"):
        canonicalize_config_document({"storage_config": storage})


@pytest.mark.parametrize("has_operator_env", [False, True])
@pytest.mark.parametrize(
    "global_args",
    [
        [],
        ["--json"],
        ["--api-key", "config", "--json"],
        ["--server-url=upgrade-file", "--json"],
    ],
)
def test_module_entrypoint_does_not_bootstrap_operator_environment(
    tmp_path, has_operator_env, global_args
):
    """Exercise launcher imports in a fresh interpreter, not create_app()."""
    import subprocess
    import sys

    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "work"
    cwd.mkdir()
    dotenv = cwd / ".env"
    dotenv.write_text("REFLEXIO_STORAGE=supabase\nDATA_SUPABASE_KEY=do-not-load\n")
    user_dir = home / ".reflexio"
    if has_operator_env:
        user_dir.mkdir()
        (user_dir / ".env").write_bytes(dotenv.read_bytes())
    source = cwd / "source.json"
    source.write_text(json.dumps(_payload()))
    output = cwd / "converted.json"
    script = """
import importlib.abc
import runpy
import sys
class ForbidBootstrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"reflexio.cli.env_loader", "reflexio.server", "reflexio.client.client"}:
            raise AssertionError("Offline command imported bootstrap: " + fullname)
sys.meta_path.insert(0, ForbidBootstrap())
runpy.run_module("reflexio.cli.__main__", run_name="__main__")
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("REFLEXIO_", "DATA_"))
    }
    env.update(HOME=str(home), REFLEXIO_STORAGE="supabase")
    proc = subprocess.run(  # noqa: S603 — fixed interpreter/script and fixture paths
        [
            sys.executable,
            "-c",
            script,
            *global_args,
            "config",
            "upgrade-file",
            str(source),
            "--output",
            str(output),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    if "--json" in global_args:
        assert json.loads(proc.stdout) == {"ok": True, "data": {"output": str(output)}}
    assert json.loads(output.read_text())["window_size"] == 23
    assert json.loads(source.read_text()) == _payload()
    assert (
        dotenv.read_text()
        == "REFLEXIO_STORAGE=supabase\nDATA_SUPABASE_KEY=do-not-load\n"
    )
    if has_operator_env:
        assert (user_dir / ".env").read_bytes() == dotenv.read_bytes()
    else:
        assert not user_dir.exists()
    assert "do-not-load" not in proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--json", "config", "show"],
        ["--server-url", "config", "upgrade-file"],
        ["--api-key", "upgrade-file", "config", "show"],
    ],
)
def test_offline_probe_preserves_online_route_arguments(args):
    from reflexio.cli.offline import dispatch_offline

    original = list(args)
    assert dispatch_offline(args) is False
    assert args == original
