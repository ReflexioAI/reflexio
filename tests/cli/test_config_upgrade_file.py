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
    typed = validate_stored_config(result)
    assert (typed.window_size, typed.stride_size) == (23, 13)
    assert typed.user_playbook_extractor_config is not None
    assert typed.user_playbook_extractor_config.aggregation_config.min_cluster_size == 9
    assert typed.playbook_optimizer_config.max_merge_invocations == 0
    assert typed.playbook_optimizer_config.user_playbooks_enabled is False
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


@pytest.mark.parametrize(
    "payload",
    [
        _payload(),
        {
            "storage_config": {
                "storage_type": "postgres",
                "schema_name": "custom",
                "db_url": "postgresql://local:local@localhost/test",
                "read_db_url": "postgresql://local:local@localhost/replica",
                "pool_size": 7,
                "read_pool_size": 3,
            },
            "extraction_preset": "long_form",
        },
        {"storage_config": {"db_path": "/tmp/custom.sqlite"}, "batch_size": 31},
        {"storage_config": {"managed_by": "platform", "schema_present": False}},
    ],
)
def test_conversion_does_not_require_runtime_acceptance_of_legacy_input(
    monkeypatch, payload
):
    from reflexio.models import config_schema

    expected = canonicalize_config_document(payload)
    received = []

    def canonical_only(document):
        # Simulate the future strict reader: old names cannot reach validation.
        assert document == expected
        received.append(document)
        return config_schema.Config.model_validate(document, extra="ignore")

    # Explicit conversion must outlive removal of these runtime rewrite hooks.
    monkeypatch.delattr(
        config_schema, "canonicalize_stored_config_fields", raising=False
    )
    monkeypatch.delattr(config_schema, "_normalize_storage_backend", raising=False)
    monkeypatch.setattr(config_schema, "validate_stored_config", canonical_only)
    assert canonicalize_config_document(payload) == expected
    assert len(received) == 1


@pytest.mark.parametrize(
    "duplicate",
    [
        {"storage_type": "postgres", "type": "postgres"},
        {"storage_type": "supabase", "type": "postgres"},
        {"schema_name": "private-schema", "schema": "private-schema"},
        {"schema_name": "private-schema", "schema": "other-private-schema"},
    ],
)
def test_storage_spelling_conflicts_remain_invalid_and_value_free(duplicate):
    payload = {
        "storage_config": {
            "db_url": "postgresql://user:private-password@localhost/test",
            **duplicate,
        }
    }
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="cannot be converted safely") as error:
        canonicalize_config_document(payload)
    assert "private" not in str(error.value)
    assert payload == original


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
    "storage,expected",
    [
        (None, None),
        (
            {"db_path": "/tmp/custom.sqlite"},
            {"type": "sqlite", "db_path": "/tmp/custom.sqlite"},
        ),
        (
            {"db_url": "postgresql://local:local@localhost/test"},
            {"type": "postgres", "db_url": "postgresql://local:local@localhost/test"},
        ),
        (
            {
                "type": "postgres",
                "schema_name": "custom",
                "db_url": "postgresql://local:local@localhost/test",
                "pool_size": 7,
                "read_db_url": "postgresql://local:local@localhost/replica",
                "read_pool_size": 3,
            },
            {
                "type": "postgres",
                "schema": "custom",
                "db_url": "postgresql://local:local@localhost/test",
                "pool_size": 7,
                "read_db_url": "postgresql://local:local@localhost/replica",
                "read_pool_size": 3,
            },
        ),
        (
            {"type": "postgres", "managed_by": "platform", "schema_present": False},
            {"type": "postgres", "managed_by": "platform", "schema_present": False},
        ),
    ],
)
def test_conversion_keeps_storage_policy_without_materializing_defaults(
    storage, expected
):
    result = canonicalize_config_document({"storage_config": storage})
    assert result == {"storage_config": expected}
    loaded = validate_stored_config(result)
    if expected is not None:
        assert loaded.storage_config is not None
        assert loaded.storage_config.storage_type == expected["type"]
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


def test_real_cli_preserves_precise_opaque_numbers_and_source(tmp_path, monkeypatch):
    from decimal import Decimal

    source, output = tmp_path / "precise.json", tmp_path / "canonical.json"
    original = b'{"storage_config":{"type":"sqlite","db_path":"/tmp/offline-precision.sqlite"},"batch_size":23,"batch_interval":7,"extension":{"fraction":0.12345678901234567890123456789,"scaled":1.00,"huge":1e1000,"integer":123456789012345678901234567890,"boolean":true,"array":[0.12345678901234567890123456789]}}'
    source.write_bytes(original)
    source.chmod(0o640)
    monkeypatch.setenv("REFLEXIO_HOME", str(tmp_path / "unused-home"))
    result = CliRunner().invoke(
        create_app(),
        ["--json", "config", "upgrade-file", str(source), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    expected = json.loads(original, parse_float=Decimal)["extension"]
    converted = json.loads(output.read_text(), parse_float=Decimal)["extension"]
    assert converted == expected
    assert type(converted["boolean"]) is bool and converted["boolean"] is True
    assert type(converted["integer"]) is int
    assert converted["fraction"].as_tuple() == expected["fraction"].as_tuple()
    assert converted["scaled"].as_tuple() == expected["scaled"].as_tuple()
    assert converted["huge"] == Decimal("1e1000")
    assert source.read_bytes() == original
    assert source.stat().st_mode & 0o777 == 0o640
    assert output.stat().st_mode & 0o077 == 0
    assert output.read_text().startswith("{\n  ")
    assert not (tmp_path / "unused-home").exists()


def test_real_cli_refuses_fractional_legacy_integer_without_rounding(tmp_path):
    source, output = tmp_path / "invalid.json", tmp_path / "preserved.json"
    original = b'{"storage_config":{"type":"sqlite","db_path":"/tmp/precision.sqlite"},"batch_size":23.000000000000000000000001,"batch_interval":7,"extension":{"secret":"private-precision-value"}}'
    source.write_bytes(original)
    output.write_bytes(b"existing-output")
    result = CliRunner().invoke(
        create_app(),
        ["--json", "config", "upgrade-file", str(source), "--output", str(output)],
    )
    assert result.exit_code != 0
    assert "private-precision-value" not in result.output
    assert source.read_bytes() == original
    assert output.read_bytes() == b"existing-output"


@pytest.mark.parametrize(
    "raw",
    [
        '{"private-secret":1e9999999999999999999999999999999}',
        '{"private-secret":' + "[" * 2000 + "0" + "]" * 2000 + "}",
    ],
    ids=["oversized-exponent", "deep-nesting"],
)
def test_real_cli_sanitizes_parser_limits_without_writes(tmp_path, raw):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(raw)
    output.write_bytes(b"existing-output")
    import simplejson

    vars(simplejson)["_toggle_speedups"](False)
    try:
        result = CliRunner().invoke(
            create_app(),
            ["--json", "config", "upgrade-file", str(source), "--output", str(output)],
        )
    finally:
        vars(simplejson)["_toggle_speedups"](True)
    assert result.exit_code != 0
    assert "private-secret" not in result.output
    assert "Input is not a valid configuration for safe conversion." in result.output
    assert source.read_text() == raw
    assert output.read_bytes() == b"existing-output"
