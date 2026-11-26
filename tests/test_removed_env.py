"""Removed-environment registry guards and byte-pinned rejection messages.

``tests/test_data/removed_env_messages.json`` was captured from the
pre-registry implementation of ``reflexio.removed_env``. Each case is a
single name, every name together, every ``canonical=`` scope, and the
caller-supplied alias shapes enterprise uses (extra names, single-entry maps).
After a deliberate message change, regenerate the file with
``uv run python -m tests.test_removed_env`` and review the diff.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

import reflexio
from reflexio.removed_env import REMOVED_ENV, REMOVED_ENV_ALIASES, reject_removed_env

_GOLDEN = Path(__file__).parent / "test_data" / "removed_env_messages.json"
_EXTRA = {
    "EXTRA_REMOVED_DEVICE": "REFLEXIO_EMBEDDING_DEVICE",
    "EXTRA_REMOVED_BLANK": "",
}
_NARROW = ("REFLEXIO_AGENT_VERSION", "REFLEXIO_LLM_SEED", "REFLEXIO_INSTANCE_ID")


def _message(
    env: dict[str, str],
    aliases: dict[str, str] | None = None,
    canonical: str | None = None,
) -> str | None:
    try:
        if aliases is None:
            reject_removed_env(env, canonical=canonical)
        else:
            reject_removed_env(env, aliases, canonical=canonical)
    except ValueError as error:
        return str(error)
    return None


def _build_cases() -> dict[str, str | None]:
    everything = dict.fromkeys(REMOVED_ENV_ALIASES, "private-value")
    cases = {
        f"single:{name}": _message({name: "private-value"})
        for name in REMOVED_ENV_ALIASES
    }
    cases["all"] = _message(everything)
    for scope in sorted(set(REMOVED_ENV_ALIASES.values())):
        cases[f"scope:{scope}"] = _message(everything, canonical=scope)
    cases["scope:<unknown>"] = _message(everything, canonical="NOT_A_SCOPE")
    custom = {**REMOVED_ENV_ALIASES, **_EXTRA}
    both = {**everything, **dict.fromkeys(_EXTRA, "private-value")}
    cases["custom:all"] = _message(both, custom)
    cases["custom:scope:REFLEXIO_EMBEDDING_DEVICE"] = _message(
        both, custom, "REFLEXIO_EMBEDDING_DEVICE"
    )
    cases["custom:scope:"] = _message(both, custom, "")
    for name in _EXTRA:
        cases[f"custom:single:{name}"] = _message({name: "x"}, custom)
    for name in _NARROW:
        cases[f"narrow:{name}"] = _message(everything, {name: ""})
    return cases


_EXPECTED: dict[str, str | None] = json.loads(_GOLDEN.read_text())
_ACTUAL = _build_cases()


def test_case_table_matches_golden_keys():
    assert set(_ACTUAL) == set(_EXPECTED)


@pytest.mark.parametrize("case", list(_EXPECTED))
def test_message_is_byte_identical(case):
    assert _ACTUAL.get(case) == _EXPECTED[case]


def test_messages_never_echo_values():
    assert not any("private-value" in (message or "") for message in _ACTUAL.values())


def test_every_record_has_a_unique_name_and_a_scope():
    names = [record.name for record in REMOVED_ENV]
    assert all(name and name == name.strip() for name in names)
    assert len(names) == len(set(names))
    # The literal scope column and the records describe the same names.
    assert set(names) == set(REMOVED_ENV_ALIASES)
    assert all(isinstance(scope, str) for scope in REMOVED_ENV_ALIASES.values())


def test_no_replacement_or_scope_is_dangling():
    removed = {record.name for record in REMOVED_ENV}
    for record in REMOVED_ENV:
        scope = REMOVED_ENV_ALIASES[record.name]
        # A printed replacement is the scope a canonical reader rejects under.
        assert record.replacement in (None, scope), record.name
        assert record.replacement != "", record.name
        # Pointing an upgrade at another removed name would loop operators.
        assert scope not in removed, record.name
        assert record.migration != "", record.name
        assert not record.guidance or record.guidance.startswith("; "), record.name


def _canonical_arguments() -> list[tuple[str, str | None]]:
    """Return (location, resolved literal) for each canonical= call argument."""
    found: list[tuple[str, str | None]] = []
    for path in sorted(Path(reflexio.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "reject_removed_env"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "canonical":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    literal: str | None = value.value
                else:
                    literal = constants.get(getattr(value, "id", ""))
                found.append((f"{path.name}:{node.lineno}", literal))
    return found


def test_every_scoped_reader_names_a_registered_scope():
    arguments = _canonical_arguments()
    assert arguments, "scan found no scoped readers"
    scopes = set(REMOVED_ENV_ALIASES.values())
    assert [(at, lit) for at, lit in arguments if lit not in scopes] == []


if __name__ == "__main__":
    rows = (f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in _build_cases().items())
    _GOLDEN.write_text("{\n" + ",\n".join(rows) + "\n}\n")
    sys.exit(0)
