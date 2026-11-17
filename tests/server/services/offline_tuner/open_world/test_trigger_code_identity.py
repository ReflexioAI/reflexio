"""Trigger authority invalidates on validation/sizing changes invisible to schemas.

All mutations are read-only resource proxies. Live source files and imported
classes stay unchanged, including while other test processes use the checkout.
"""

from __future__ import annotations

import importlib.resources
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server.services.offline_tuner.open_world import (
    rule_evidence_authority as authority,
)
from reflexio.server.services.playbook import rule_evidence
from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    TriggerBatch,
    TriggerGate,
)

# An independently listed, reviewed closure. Do not derive this expectation from
# the implementation: dropping a declared validator would otherwise pass quietly.
_EXPECTED_DEPENDENCIES = (
    (
        "portable",
        "reflexio.server.services.playbook.rule_evidence",
        "trigger_maintenance.py",
    ),
    ("portable_models", "reflexio.server.services.playbook.rule_evidence", "models.py"),
    (
        "portable_execution",
        "reflexio.server.services.playbook.rule_evidence",
        "__init__.py",
    ),
    ("wire_profile", "reflexio.server.services.playbook.rule_evidence", "wire.py"),
    ("revision", "reflexio.server.services.playbook", "revision.py"),
    ("canonical_publication", "reflexio.server.services.playbook", "publication.py"),
    (
        "canonical_manifests",
        "reflexio.server.services.offline_tuner.evidence",
        "manifests.py",
    ),
    ("structured_output", "reflexio.models", "structured_output.py"),
    (
        "runtime",
        "reflexio.server.services.offline_tuner.open_world",
        "rule_evidence_runtime.py",
    ),
    (
        "authority",
        "reflexio.server.services.offline_tuner.open_world",
        "rule_evidence_authority.py",
    ),
    (
        "qualification",
        "reflexio.server.services.offline_tuner.open_world",
        "trigger_qualification.py",
    ),
    (
        "v2_stages",
        "reflexio.server.services.offline_tuner.open_world",
        "rule_evidence_v2.py",
    ),
    (
        "source_validation",
        "reflexio.server.services.offline_tuner.open_world",
        "rule_evidence.py",
    ),
    ("identity", "reflexio.server.services.offline_tuner.open_world", "identity.py"),
    (
        "identity_models",
        "reflexio.server.services.offline_tuner.open_world",
        "models.py",
    ),
    (
        "native_transport",
        "reflexio.server.services.offline_tuner.open_world",
        "analyst.py",
    ),
)


class _ReadOnlyResourceProxy:
    def __init__(
        self,
        resource: Any,
        package: str,
        path: tuple[str, ...],
        reads: list[tuple[str, str]],
        changed: tuple[str, str] | None,
    ) -> None:
        self.resource = resource
        self.package = package
        self.path = path
        self.reads = reads
        self.changed = changed

    def joinpath(self, *paths: str) -> _ReadOnlyResourceProxy:
        return _ReadOnlyResourceProxy(
            self.resource.joinpath(*paths),
            self.package,
            (*self.path, *paths),
            self.reads,
            self.changed,
        )

    def read_bytes(self) -> bytes:
        key = (self.package, "/".join(self.path))
        self.reads.append(key)
        original = self.resource.read_bytes()
        if key == self.changed:
            return (
                original
                + b"\n# Read-only code-identity mutation; never written or executed.\n"
            )
        return original

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resource, name)


def _intercept_resources(
    monkeypatch: pytest.MonkeyPatch, *, changed: tuple[str, str] | None
) -> list[tuple[str, str]]:
    original_files = importlib.resources.files
    reads: list[tuple[str, str]] = []

    def files(package: str) -> _ReadOnlyResourceProxy:
        return _ReadOnlyResourceProxy(
            original_files(package), package, (), reads, changed
        )

    monkeypatch.setattr(importlib.resources, "files", files)
    return reads


def _method() -> authority.RuleEvidenceMethod:
    return authority.qualify_rule_evidence_method(
        SimpleNamespace(
            transport_identity=SimpleNamespace(
                provider="fixture-provider", model_revision="fixture-model"
            )
        )
    )


def _trigger_digests(method: authority.RuleEvidenceMethod) -> tuple[str, str]:
    assert method.trigger_gate is not None and method.trigger_builder is not None
    return method.trigger_gate.digest, method.trigger_builder.digest


def _emitted_contract() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, int | str], int
]:
    return (
        TriggerGate.model_json_schema(),
        TriggerBatch.model_json_schema(),
        rule_evidence.input_budget_policy(),
        rule_evidence.PAYLOAD_PROXY_TOKEN_LIMIT,
    )


def test_declared_trigger_closure_is_complete_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert authority.TRIGGER_CODE_DEPENDENCIES == _EXPECTED_DEPENDENCIES
    original = authority._trigger_implementation_digest()
    original_components = _trigger_digests(_method())
    reads = _intercept_resources(monkeypatch, changed=None)
    assert authority._trigger_implementation_digest() == original
    assert sorted(reads) == sorted(
        (package, filename) for _, package, filename in _EXPECTED_DEPENDENCIES
    )
    reads.clear()
    assert _trigger_digests(_method()) == original_components
    assert set(reads) == {
        (package, filename) for _, package, filename in _EXPECTED_DEPENDENCIES
    }
    assert authority._trigger_implementation_digest() == original


@pytest.mark.parametrize(
    "dependency",
    _EXPECTED_DEPENDENCIES,
    ids=[name for name, _, _ in _EXPECTED_DEPENDENCIES],
)
def test_each_owned_source_change_invalidates_trigger_identity_without_schema_change(
    monkeypatch: pytest.MonkeyPatch, dependency: tuple[str, str, str]
) -> None:
    _, package, filename = dependency
    real_resource = importlib.resources.files(package).joinpath(filename)
    real_bytes_digest = sha256(real_resource.read_bytes()).hexdigest()
    original_code = authority._trigger_implementation_digest()
    original_components = _trigger_digests(_method())
    emitted = _emitted_contract()
    reads = _intercept_resources(monkeypatch, changed=(package, filename))

    assert authority._trigger_implementation_digest() != original_code
    changed_components = _trigger_digests(_method())
    assert all(
        changed != original
        for changed, original in zip(
            changed_components, original_components, strict=True
        )
    )
    assert _emitted_contract() == emitted
    assert (package, filename) in reads
    assert sha256(real_resource.read_bytes()).hexdigest() == real_bytes_digest
