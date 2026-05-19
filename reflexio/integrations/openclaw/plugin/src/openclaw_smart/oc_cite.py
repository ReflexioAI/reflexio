"""Support helpers for openclaw-smart citation tracking.

Context injected by before_prompt_build / before_tool_call tags each skill
and preference bullet with a rank-based id fingerprinted by the underlying
real id (``[oc:s1-1a2b]`` for the first skill whose ``user_playbook_id``
starts with ``1a2b``, ``[oc:p2-c3d4]`` for the second preference). The
injected instruction asks the assistant to end impactful replies with a
marker like::

    ✨ 1 openclaw-smart learning applied [oc:s1-1a2b]

The agent_end hook later scans the assistant text for those markers and
resolves the ids against a per-session registry persisted at
``~/.openclaw-smart/sessions/<session_id>.injected.jsonl``. Legacy
``oc-cite`` Bash tool calls are still accepted as a fallback for older
instructions.

Why rank + fingerprint: rank alone resets at every injection, so a later
injection's ``s1`` would silently overwrite an earlier entry in the
append-only registry — if openClaw cited ``s1`` across a turn boundary,
the resolver would pick the wrong skill. Appending the first four
alphanumeric chars of the real id makes the id stable across injections
in the common case (distinct real ids → distinct fingerprints), so
cross-injection collisions become rare.

This module holds:

- ``rank_id``: ``p{n}-{fp}`` / ``s{n}-{fp}`` tag for a given (kind, rank,
  real_id) tuple. Fingerprint is omitted when no real id is available.
  ``p`` is preference, ``s`` is skill.
- ``CITATION_CMD_RE``: regex matching a valid legacy ``oc-cite`` command line.
- ``ensure_installed``: idempotent copy of ``plugin/bin/oc-cite`` to
  ``~/.openclaw-smart/bin/oc-cite`` with the executable bit set.
- ``CITATION_INSTRUCTION``: the trailer text appended to injected context
  so the assistant knows when and how to emit the citation marker.
"""

from __future__ import annotations

import logging
import re
import shutil
import stat as stat_
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# plugin/src/openclaw_smart/oc_cite.py -> parents[2] is plugin/
_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parents[2]
_SOURCE_SCRIPT = _PLUGIN_ROOT / "bin" / "oc-cite"
_INSTALL_DIR = Path.home() / ".openclaw-smart" / "bin"
INSTALL_PATH = _INSTALL_DIR / "oc-cite"

_FINGERPRINT_LEN = 4

# Match a bare `oc-cite <ids>` invocation. Ids are rank tokens of the form
# `p<N>` (preference) or `s<N>` (skill) with an optional `-<fp>`
# fingerprint (1-4 alphanumeric chars), optionally `oc:`-prefixed (since
# bullets render as `[oc:p1-ab12]` and the model often copies the tag
# verbatim). The `(?i:...)` inline flags make the prefix, kind letter,
# and fingerprint case-insensitive so `OC:P1-AB12` is accepted. Tokens
# may be comma- and/or whitespace-separated. Chained commands (&&, |, ;)
# and extra trailing tokens remain rejected by the anchored `\s*$`
# terminator so accidental mentions don't register as citations.
_ID_TOKEN = r"(?i:oc:)?(?i:[ps])\d+(?:-(?i:[a-z0-9]){1,4})?"
_ID_SEP = r"[,\s]+"
CITATION_CMD_RE = re.compile(
    rf"^\s*(?:[^\s]*/)?oc-cite\s+({_ID_TOKEN}(?:{_ID_SEP}{_ID_TOKEN})*)\s*$"
)
_CLEAN_ID_RE = re.compile(r"^(?i:oc:)?((?i:[ps])\d+(?:-(?i:[a-z0-9]){1,4})?)$")
_SPLIT_RE = re.compile(_ID_SEP)
_TEXT_CITATION_LINE_RE = re.compile(
    r"(?im)^\s*✨\s+\d+\s+openclaw-smart learning(?:s)? applied\s+"
    r"\[oc:(?P<ids>[^\]]+)\]\s*$"
)

CITATION_INSTRUCTION = (
    "_First, fully answer the user — citation does not change what or how "
    "you reply. Then, as a final step, consider whether to cite: if — and "
    "only if — an injected `[oc:…]` item materially changed your reply "
    "(different wording, action, or conclusion than you would have produced "
    "without it), append exactly one final citation line after your answer. "
    "Do not call `oc-cite` or any other tool for citations. Ids come verbatim "
    "from the `[oc:…]` tags — keep the leading `p` (preference) or `s` "
    "(skill) and the `-<fp>` suffix. Use this exact format for one id: "
    "`✨ 1 openclaw-smart learning applied [oc:s1-ab12]`. Use this exact "
    "format for multiple ids: `✨ 2 openclaw-smart learnings applied "
    "[oc:s1-ab12,p2-cd34]`, where the number is the count of ids in the "
    "brackets. Never emit a standalone wrapper like `✨s1-ab12✨` or "
    "`✨abc123✨`; those are not openclaw-smart citations and cannot be "
    "resolved. Default is to skip. If an item is merely on-topic, confirms "
    "what you already planned, or your reply would read the same without it, "
    "do not cite — end the turn normally with your reply. When unsure, skip. "
    "Do not add any other text, tool calls, or role markers after the final "
    "citation line._"
)


def _fingerprint(real_id: Any) -> str:
    """Return the first ``_FINGERPRINT_LEN`` alphanumeric chars of ``real_id``."""
    if real_id is None:
        return ""
    return "".join(c for c in str(real_id).lower() if c.isalnum())[:_FINGERPRINT_LEN]


def rank_id(kind: str, rank: int, real_id: Any = None) -> str:
    """Return the citation id for a skill or preference item.

    Format is ``{letter}{rank}-{fingerprint}`` where ``letter`` is ``p`` for
    preferences and ``s`` for skills, ``rank`` is the 1-based position within
    the current retrieval batch, and ``fingerprint`` is up to 4 alphanumeric
    chars derived from ``real_id``. The fingerprint is omitted when no real
    id is available (falling back to the rank form ``s1`` / ``p1``).

    Raises:
        ValueError: If ``kind`` is not ``"profile"`` or ``"playbook"``.
    """
    if kind == "profile":
        prefix = "p"
    elif kind == "playbook":
        prefix = "s"
    else:
        raise ValueError(f"unknown citation kind: {kind!r}")
    fp = _fingerprint(real_id)
    return f"{prefix}{rank}-{fp}" if fp else f"{prefix}{rank}"


def parse_citation_command(command: str) -> list[str]:
    """Extract citation ids from an ``oc-cite`` Bash command string.

    Returns an empty list when the command does not match the expected shape
    (chained commands, extra arguments, or anything other than a bare
    ``oc-cite <ids>`` invocation are rejected to avoid false positives from
    accidental mentions).
    """
    match = CITATION_CMD_RE.match(command or "")
    if not match:
        return []
    return _parse_id_tokens(match.group(1))


def parse_text_citations(text: str) -> list[str]:
    """Extract text-only citation ids from a final learning marker line.

    The parser intentionally only accepts lines containing the visual
    ``openclaw-smart learning(s) applied`` marker, so ordinary references to
    injected ``[oc:...]`` ids inside an answer do not count as citations.
    When multiple matching lines exist, the last one wins because the
    instruction requires the citation marker to be final.
    """
    matches = list(_TEXT_CITATION_LINE_RE.finditer(text or ""))
    if not matches:
        return []
    return _parse_id_tokens(matches[-1].group("ids"))


def _parse_id_tokens(raw_ids: str) -> list[str]:
    ids: list[str] = []
    for tok in _SPLIT_RE.split(raw_ids.strip()):
        if clean := _CLEAN_ID_RE.match(tok):
            ids.append(clean.group(1).lower())
    return ids


def ensure_installed() -> Path:
    """Idempotently install ``oc-cite`` into ``~/.openclaw-smart/bin/``.

    Called from every before_tool_call / before_prompt_build inject, so we
    short-circuit when the target file already exists with the executable
    bit set — the steady-state path is one ``stat`` syscall instead of
    mkdir + copy + stat + chmod. Keying on filesystem state (rather than a
    module-level boolean) keeps test isolation working when tests
    monkeypatch ``INSTALL_PATH`` to a fresh tmpdir.

    Never raises — filesystem errors are logged at DEBUG and the caller
    proceeds with injection regardless (the citation feature degrades to
    silent if the script is unreachable).
    """
    try:
        if (
            INSTALL_PATH.is_file()
            and INSTALL_PATH.stat().st_mode & stat_.S_IXUSR
            and _SOURCE_SCRIPT.is_file()
            and INSTALL_PATH.read_bytes() == _SOURCE_SCRIPT.read_bytes()
        ):
            return INSTALL_PATH
        _INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        if _SOURCE_SCRIPT.is_file():
            shutil.copy2(_SOURCE_SCRIPT, INSTALL_PATH)
            mode = INSTALL_PATH.stat().st_mode
            INSTALL_PATH.chmod(mode | stat_.S_IXUSR | stat_.S_IXGRP | stat_.S_IXOTH)
    except OSError as exc:
        _LOGGER.debug("oc-cite install failed: %s", exc)
    return INSTALL_PATH
